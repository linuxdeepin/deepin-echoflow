"""
CrispASR — server worker-pool concurrency: CUDA proof (improvements Phase 4b).

Question this kernel answers (needs a real NVIDIA card — on M1 the memory-bound
CPU model contends, so concurrency showed no throughput win): does
CRISPASR_SERVER_WORKERS=N give real request concurrency on a GPU where a single
request under-utilises the card, with identical transcripts?

The server pools N independent backend instances; a pure-ASR request (explicit
language, no aligner / no post-processing) runs on a pooled worker so up to N run
concurrently, while the primary backend + model_mutex serialise everything else.
Each worker owns its own CUDA context (separate command queue).

Design (CUDA build from CRISPASR_REF; a small ASR model so one request doesn't
saturate the GPU):
  A. WORKERS=1 (control) — 2 concurrent requests must SERIALISE (~2× single).
  B. WORKERS=2 — 2 SERIAL requests (~2× single) vs 2 CONCURRENT requests
     (~1× single if GPU concurrency works). speedup = serial/concurrent.
All requests must return IDENTICAL transcripts (correctness gate).

Acceptance: transcripts identical across every request; with WORKERS=2 the
concurrent pair is meaningfully faster than the serial pair (speedup > 1.3 =
real concurrency); with WORKERS=1 it is not (~1.0). A speedup ~1.0 at WORKERS=2
means the GPU is already saturated by one request (use a smaller model / report
honestly), not a bug.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
CRISPASR = BUILD / "bin" / "crispasr"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
MODEL_REPO = os.environ.get("MODEL_REPO", "cstr/moonshine-tiny-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "moonshine-tiny-q8_0.gguf")
BACKEND = os.environ.get("BACKEND", "moonshine")
PORT = int(os.environ.get("PORT", "8799"))
REPS = int(os.environ.get("REPS", "3"))  # median over REPS


def _sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


print(f"[pre-clone] cloning CrispASR @ {CRISPASR_REF}", flush=True)
if not REPO.exists():
    _sh(f"git clone --depth 1 --branch {CRISPASR_REF} --recursive https://github.com/CrispStrobe/CrispASR {REPO}")

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
if kh.resolve_hf_token():
    print("[auth] HF token resolved", flush=True)
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("script.start", ref=CRISPASR_REF, sha=sha)

# ── Build (CUDA) ───────────────────────────────────────────────────────────
kh.step("build.begin")
kh.install_build_toolchain()
cmake_cmd = (f"cmake {REPO} -B{BUILD} -GNinja -DCMAKE_BUILD_TYPE=Release "
             + " ".join(kh.cuda_build_flags()) + " " + " ".join(kh.cache_and_link_flags()))
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{kh.safe_build_jobs(gpu=True)}")
assert CRISPASR.is_file(), "crispasr binary missing"
kh.step("build.done")

# ── Model ──────────────────────────────────────────────────────────────────
kh.step("download.begin")
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer requests")
from huggingface_hub import hf_hub_download  # noqa: E402
import requests  # noqa: E402

model_path = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=str(MODELS),
                                  local_dir_use_symlinks=False))
JFK = REPO / "samples" / "jfk.wav"
assert JFK.is_file()
kh.step("download.done", model=str(model_path))

URL = f"http://127.0.0.1:{PORT}/v1/audio/transcriptions"
HEALTH = f"http://127.0.0.1:{PORT}/health"


def start_server(workers: int):
    env = {**os.environ, "CRISPASR_SERVER_WORKERS": str(workers), "CRISPASR_NO_WARMUP": "1"}
    proc = subprocess.Popen(
        [str(CRISPASR), "--server", "--host", "127.0.0.1", "--port", str(PORT),
         "-m", str(model_path), "--backend", BACKEND],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(120):
        try:
            if requests.get(HEALTH, timeout=2).ok:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup:\n" + (proc.stdout.read() if proc.stdout else ""))
        time.sleep(1)
    raise RuntimeError("server did not become ready")


def one_request() -> tuple[str, float]:
    t0 = time.time()
    with open(JFK, "rb") as f:
        r = requests.post(URL, files={"file": ("jfk.wav", f, "audio/wav")},
                          data={"language": "en", "response_format": "json"}, timeout=300)
    text = r.json().get("text", "") if r.ok else f"<HTTP {r.status_code}>"
    return text.strip(), time.time() - t0


def timed_serial() -> tuple[float, list[str]]:
    texts = []
    t0 = time.time()
    for _ in range(2):
        txt, _ = one_request()
        texts.append(txt)
    return time.time() - t0, texts


def timed_concurrent() -> tuple[float, list[str]]:
    texts: list[str] = [None, None]  # type: ignore

    def worker(i):
        texts[i], _ = one_request()

    t0 = time.time()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return time.time() - t0, texts


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def bench(workers: int) -> dict:
    kh.step(f"bench.workers{workers}.begin")
    proc = start_server(workers)
    try:
        one_request()  # warm
        single = median([one_request()[1] for _ in range(REPS)])
        ser_t, ser_txt = min((timed_serial() for _ in range(REPS)), key=lambda x: x[0])
        con_t, con_txt = min((timed_concurrent() for _ in range(REPS)), key=lambda x: x[0])
        all_txt = ser_txt + con_txt
        identical = len(set(t for t in all_txt if not t.startswith("<HTTP"))) == 1 and all(
            not t.startswith("<HTTP") for t in all_txt)
        speedup = round(ser_t / con_t, 2) if con_t else None
        res = dict(workers=workers, single_s=round(single, 2), serial2_s=round(ser_t, 2),
                   concurrent2_s=round(con_t, 2), speedup=speedup, transcripts_identical=identical,
                   sample_text=(all_txt[0][:60] if all_txt else ""))
        kh.step(f"bench.workers{workers}.done", **res)
        return res
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        time.sleep(2)


kh.step("bench.section.begin", reps=REPS)
w1 = bench(1)
w2 = bench(2)

print("\n" + "=" * 72)
print(f"SUMMARY — server worker-pool CUDA concurrency ({sha[:8]}, {BACKEND})")
print("=" * 72)
for r in (w1, w2):
    print(f"[WORKERS={r['workers']}] single={r['single_s']}s  serial×2={r['serial2_s']}s  "
          f"concurrent×2={r['concurrent2_s']}s  speedup(serial/concurrent)={r['speedup']}x  "
          f"identical={r['transcripts_identical']}")
verdict = "CONCURRENCY CONFIRMED" if (w2["transcripts_identical"] and (w2["speedup"] or 0) > 1.3
                                      and (w1["speedup"] or 0) < 1.3) else (
    "NO GPU-CONCURRENCY WIN (single request saturates the card — honest null)"
    if w2["transcripts_identical"] else "CORRECTNESS FAIL")
print(f"VERDICT: {verdict}")
kh._push_progress_to_hf(force=True)
kh.step("script.end", verdict=verdict, w1_speedup=w1["speedup"], w2_speedup=w2["speedup"])

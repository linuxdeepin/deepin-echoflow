"""
CrispASR — issue #220: chatterbox T3 CUDA illegal-memory-access A/B.

Issue #220: on a non-Metal GPU build (ghcr .../crispasr:main-cuda) chatterbox
aborts out of the box with

    CUDA error: an illegal memory access was encountered
    ... ggml_backend_cuda_synchronize ... (right after "CUDA Graph id N reused")

at AR decode step ~2. Root cause: the §186 Lk-bucketed T3 decode allocated its
step scheduler ONCE per bucket and then only re-ran ggml_backend_sched_graph_
compute each step (the M1 "reuse-shortcut"). ggml-cuda keys a captured CUDA-graph
executable on the split graph's uid and, on a uid match, replays it verbatim with
the device pointers baked in at capture time. A fresh split uid is minted only
inside ggml_backend_sched_alloc_graph — so skipping the per-step reset+alloc keeps
the uid constant and the 2nd step replays a STALE capture -> illegal access.

Fix (branch fix/chatterbox-t3-cuda-illegal-access): on a non-Metal GPU backend,
reset+alloc the bucket step sched EVERY step (granite/outetts CUDA-graph-bucket
pattern, docs/contributing.md §210). Old path kept behind
CRISPASR_CHATTERBOX_T3_BUCKET_REUSE=1 for A/B.

  A/B here:
    * fix_default : T3 on GPU, reset-per-step (the fix; default on CUDA build)
    * old_reuse   : T3 on GPU, CRISPASR_CHATTERBOX_T3_BUCKET_REUSE=1 (pre-fix path)
    * cpu_ref     : CRISPASR_CHATTERBOX_FULL_CPU=1 (known-good reference audio)

  Each is ASR-roundtripped through parakeet to prove intelligibility, not just
  "a WAV was written".

!!! IMPORTANT ARCH CAVEAT !!!
  ggml-cuda hard-gates CUDA-graph capture to sm_80+ (Ampere): ggml-cuda.cu
  `cc < GGML_CUDA_CC_AMPERE -> disable_due_to_gpu_arch`. The reporter's card is an
  RTX 3090 Ti (sm_86) where capture ENGAGES and the bug fires. Kaggle free GPUs are
  T4 (sm_75) / P100 (sm_60), BELOW sm_80 -> capture never engages -> the #220 crash
  path is NOT exercised here. On such a card BOTH old_reuse and fix_default are
  EXPECTED to pass; that only proves the fix does not REGRESS normal CUDA decode.
  A true crash-repro needs an sm_80+ device. The verdict below is arch-aware.

Build/report plumbing from the shared harness tools/kaggle/kaggle_harness.py.
enable_gpu=true in kernel-metadata.json.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "fix/chatterbox-t3-cuda-illegal-access")
CRISPASR_REPO = os.environ.get(
    "CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git"
)
TTS_TEXT = "Please call Stella. Ask her to bring these things with her from the store."


def run(cmd, check=True, env=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + CUDA build ──────────────────────────────────────────────
print(f"[start] ref={CRISPASR_REF}", flush=True)
print(f"  disk: {shutil.disk_usage('/kaggle/working')}", flush=True)
Path("/kaggle/working/started.txt").write_text("started\n")

if REPO.exists():
    shutil.rmtree(REPO)
run(
    [
        "git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
        CRISPASR_REPO, str(REPO),
    ]
)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
try:
    import kaggle_harness as kh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
kh.step("cloned", sha=sha, ref=CRISPASR_REF)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
arch = kh.detect_cuda_arch()
try:
    arch_int = int(arch)
except ValueError:
    arch_int = 0
capture_capable = arch_int >= 80
kh.step("gpu", gpu_name=gpu_name, arch=arch, capture_capable=capture_capable)
print(
    f"[arch] {gpu_name} sm_{arch}  CUDA-graph capture "
    f"{'ENGAGES (>= sm_80: #220 crash path IS exercised)' if capture_capable else 'DISABLED (< sm_80: #220 crash path NOT exercised)'}",
    flush=True,
)

kh.install_build_toolchain()
kh.step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    [
        "cmake", "-S", str(REPO), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
        "-DCRISPASR_BUILD_TESTS=OFF",
    ]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
kh.step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli"
        f" -j{kh.safe_build_jobs(gpu=True)}"
    )

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    cands = [
        c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)
    ]
    assert cands, "crispasr binary not found after build"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = (
    f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)
kh.step("build_done", cli=str(CLI))

# ── Download chatterbox (t3 + s3gen q8_0) + parakeet for ASR roundtrip ──
kh.step("downloading models")
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

t3_model = Path(hf_hub_download(
    "cstr/chatterbox-GGUF", "chatterbox-t3-q8_0.gguf",
    cache_dir=str(MODELS), token=token,
))
s3gen_model = Path(hf_hub_download(
    "cstr/chatterbox-GGUF", "chatterbox-s3gen-q8_0.gguf",
    cache_dir=str(MODELS), token=token,
))
asr_model = Path(hf_hub_download(
    "cstr/parakeet-tdt-0.6b-v2-GGUF", "parakeet-tdt-0.6b-v2-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
kh.step("models_downloaded")


# ── Run chatterbox TTS under a given env config ────────────────────
def run_tts(label, extra_env=None, timeout=420):
    kh.step(f"{label}.start")
    out_wav = WORK / f"tts-{label}.wav"
    if out_wav.exists():
        out_wav.unlink()

    env = {"GGML_CUDA_DEBUG": "1"}  # mirror the issue's env; surfaces "CUDA Graph id N reused"
    if extra_env:
        env.update(extra_env)

    cmd = [
        str(CLI), "--backend", "chatterbox",
        "-m", str(t3_model),
        "--codec-model", str(s3gen_model),
        "--tts", TTS_TEXT,
        "--tts-output", str(out_wav),
        "--seed", "42",
        "-v",
    ]
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, env={**os.environ, **env},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
        rc, stdout, stderr = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as ex:
        rc = -1
        stdout = (ex.stdout or b"").decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
        stderr = (ex.stderr or b"").decode(errors="replace") if isinstance(ex.stderr, bytes) else (ex.stderr or "")
    elapsed = round(time.time() - t0, 1)
    combined = stdout + "\n" + stderr
    (RESULTS / f"{label}_log.txt").write_text(combined)

    wav_exists = out_wav.exists() and out_wav.stat().st_size > 1000
    wav_size = out_wav.stat().st_size if out_wav.exists() else 0

    crash = ("illegal memory access" in combined) or ("CUDA error" in combined) or ("GGML_ASSERT" in combined)
    graph_reused = "CUDA Graph id" in combined and "reused" in combined
    t3_line = next((ln for ln in combined.splitlines() if "chatterbox: T3 →" in ln), "")

    print(f"\n{'=' * 64}", flush=True)
    print(
        f"Run: {label}  rc={rc}  elapsed={elapsed}s  "
        f"wav={'OK' if wav_exists else 'MISSING'}  size={wav_size}  "
        f"crash={'YES' if crash else 'no'}  graph_reused={'YES' if graph_reused else 'no'}",
        flush=True,
    )
    if t3_line:
        print(f"  {t3_line.strip()}", flush=True)
    if rc != 0 or crash:
        print("  --- log tail ---", flush=True)
        for ln in combined.splitlines()[-35:]:
            print(f"   {ln}", flush=True)

    kh.step(
        f"{label}.done", rc=rc, elapsed=elapsed, wav_ok=wav_exists,
        wav_size=wav_size, crash=crash, graph_reused=graph_reused,
    )
    return {
        "label": label, "rc": rc, "wav_ok": wav_exists, "wav_size": wav_size,
        "crash": crash, "graph_reused": graph_reused, "wav_path": str(out_wav),
        "elapsed": elapsed,
    }


old = run_tts("old_reuse", {"CRISPASR_CHATTERBOX_T3_BUCKET_REUSE": "1"})
fix = run_tts("fix_default", None)
cpu = run_tts("cpu_ref", {"CRISPASR_CHATTERBOX_FULL_CPU": "1"})


# ── ASR roundtrip each WAV with parakeet ───────────────────────────
def asr_roundtrip(label, wav_path, timeout=180):
    kh.step(f"asr_{label}.start")
    out_stem = WORK / f"asr-{label}"
    cmd = [
        str(CLI), "--backend", "parakeet", "-m", str(asr_model),
        "-f", wav_path, "-of", str(out_stem), "-otxt", "--no-prints",
    ]
    try:
        subprocess.run(cmd, env=os.environ, capture_output=True, text=True, timeout=timeout)
        txt_path = out_stem.with_suffix(".txt")
        text = txt_path.read_text().strip() if txt_path.exists() and txt_path.stat().st_size > 0 else ""
    except subprocess.TimeoutExpired:
        text = ""
    kh.step(f"asr_{label}.done", chars=len(text))
    return text


asr_old = asr_roundtrip("old_reuse", old["wav_path"]) if old["wav_ok"] else ""
asr_fix = asr_roundtrip("fix_default", fix["wav_path"]) if fix["wav_ok"] else ""
asr_cpu = asr_roundtrip("cpu_ref", cpu["wav_path"]) if cpu["wav_ok"] else ""

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 64, flush=True)
print(f"SUMMARY — issue #220 chatterbox T3 CUDA — {sha[:8]} on {gpu_name} (sm_{arch})", flush=True)
print("=" * 64, flush=True)
for res, asr in [(old, asr_old), (fix, asr_fix), (cpu, asr_cpu)]:
    print(
        f"  {res['label']:<12} rc={res['rc']:<4} "
        f"wav={'OK' if res['wav_ok'] else 'FAIL':<4} crash={'YES' if res['crash'] else 'no':<3} "
        f"reuse-capture={'YES' if res['graph_reused'] else 'no':<3} {res['elapsed']}s",
        flush=True,
    )
    print(f"               ASR: {asr[:90]!r}", flush=True)

fix_ok = fix["rc"] == 0 and fix["wav_ok"] and not fix["crash"]
old_crashed = old["crash"] or old["rc"] != 0 or not old["wav_ok"]

print("\n  VERDICT:", flush=True)
if not capture_capable:
    print(
        f"  sm_{arch} < sm_80 -> CUDA-graph *capture* is DISABLED (no "
        f"'CUDA Graph id N reused' stale-replay path here).", flush=True)
    if fix_ok and old_crashed:
        print(
            f"  CONFIRMED: old reuse-shortcut CRASHES (crash={old['crash']} "
            f"rc={old['rc']}) even on sm_{arch} while the fix produces valid audio "
            f"-> the crash is CAPTURE-INDEPENDENT (the reuse-shortcut is unsafe on "
            f"CUDA regardless of arch; sm_80+ capture is only an extra aggravator). "
            f"#220 fixed and verified end-to-end on real CUDA.", flush=True)
    elif fix_ok and not old_crashed:
        print(
            "  Both paths produced valid audio: no capture-independent crash on "
            "this card; the fix does not regress. (sm_80+ needed to exercise the "
            "capture-replay path too.)", flush=True)
    else:
        print("  fix_default FAILED on CUDA — regression, do NOT ship.", flush=True)
else:
    # sm_80+: the real test.
    if fix_ok and old_crashed:
        print(
            f"  CONFIRMED on sm_{arch}: old reuse-shortcut CRASHES "
            f"(crash={old['crash']} rc={old['rc']}) while the fix produces valid "
            f"audio. #220 fixed.", flush=True)
    elif fix_ok and not old_crashed:
        print(
            f"  Both paths passed on sm_{arch}: capture engaged "
            f"(reuse-capture={fix['graph_reused']}) but old path did not crash here "
            f"— fix is safe; crash may be card/driver specific.", flush=True)
    else:
        print(f"  fix_default FAILED on sm_{arch} — investigate before shipping.", flush=True)

kh.step(
    "summary", sha=sha, gpu=gpu_name, arch=arch, capture_capable=capture_capable,
    fix_ok=fix_ok, old_crashed=old_crashed,
    old_crash=old["crash"], old_rc=old["rc"], fix_rc=fix["rc"],
    asr_fix=asr_fix[:80], asr_old=asr_old[:80], asr_cpu=asr_cpu[:80],
)
kh._push_progress_to_hf(force=True)
kh.step("script.end")

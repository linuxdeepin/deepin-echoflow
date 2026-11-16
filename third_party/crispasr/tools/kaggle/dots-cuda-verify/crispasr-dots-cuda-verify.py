# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — dots.tts CUDA verify (PLAN #200)
#
# The dots.tts-soar backend (Qwen2.5-1.5B LLM + PatchEncoder + DiT flow-match
# head + BigVGAN vocoder) was developed and validated on CPU and Apple Metal.
# Its GPU path is backend-agnostic — every graph runs on a single backend via
# raw `ggml_gallocr` (no `ggml_backend_sched`), so there are no cross-backend
# copy hazards — but it had never run on CUDA. This kernel closes that gap.
#
# What it does:
# 1. Clone CrispASR @ env CRISPASR_REF (default: main — dots work is merged).
# 2. Build with -DGGML_CUDA=ON.
# 3. Run `crispasr --backend dots-tts -m auto --auto-download --tts ...` on
#    CUDA (GPU is the CLI default via crispasr_backend_should_use_gpu):
#      a. "Hello world." — short single-patch smoke test.
#      b. A multi-sentence paragraph — exercises the incremental streaming
#         PatchEncoder (O(N)) and EOS over many AR steps on the GPU.
# 4. Verify each output WAV is produced, non-silent, and of plausible length;
#    best-effort ASR round-trip each with whisper to confirm intelligibility
#    (recorded, non-gating).
# 5. Report PASS/FAIL per case + the full summary.json.
#
# NOTE: voice cloning is NOT exercised here — the CAM++ speaker encoder sibling
# is not yet auto-downloadable (registry has no companion entry for it; PLAN
# #200 item #3). Default-voice synthesis is the CUDA correctness target.
#
# Requirements: Kaggle GPU notebook (T4, P100, L4 — anything CUDA-capable).
# Disk: ~7 GB (build ~1 GB + dots f16 core ~4.6 GB + BigVGAN vocoder ~0.36 GB
# + whisper-base ~0.15 GB + deps).

# ─────────────────────────── cell 1 (code) ───────────────────────────
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = REPO / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")

PROGRESS = RESULTS / "progress.jsonl"
_T0 = time.time()


def step(name, **kv):
    payload = {"t": round(time.time() - _T0, 2), "step": name, **kv}
    line = json.dumps(payload)
    print(f"[step] {line}", flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def run(cmd, check=True, capture=False, env=None, cwd=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    if capture:
        r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        print(r.stdout, flush=True)
        if check and r.returncode != 0:
            raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
        return r
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ─────────────────────────── cell 2 (code) — clone + build ───────────────
step("start", ref=CRISPASR_REF)

if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)])

# Harness lives in the cloned repo — import it now that the clone succeeded.
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

step("cloned", sha=subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

# Inspect GPU
run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
step("gpu", gpu=gpu_name)

# Build with CUDA. Install ninja/ccache/mold first so the cache + linker
# launchers are available, then auto-detect the GPU arch (T4→75, A100→80…).
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = [
    "cmake", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
run(cmake_args)
step("cmake_done")

# Just need crispasr CLI + libs; skip examples we don't use. Stream the
# build line-by-line with a heartbeat so a hang/OOM is visible mid-run.
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}")
step("build_done")

CLI = BUILD / "examples" / "cli" / "crispasr"
if not CLI.exists():
    # Older layouts:
    candidates = list(BUILD.rglob("crispasr"))
    candidates = [c for c in candidates if c.is_file() and os.access(c, os.X_OK)]
    if not candidates:
        raise SystemExit("crispasr binary not found after build")
    CLI = candidates[0]
print(f"crispasr binary: {CLI}", flush=True)
step("cli_found", path=str(CLI))

# Ensure libcrispasr.so is loadable.
LIB_DIR = BUILD / "src"
os.environ["LD_LIBRARY_PATH"] = f"{LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"


# ─────────────────────────── cell 3 (code) — helpers ─────────────────────
def wav_summary(path: Path) -> dict:
    """Return {'duration_s': float, 'rms': float, 'n_samples': int, 'sr': int}."""
    if not path.exists():
        return {"error": "missing"}
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(n)
    if sw != 2:
        return {"error": f"sw={sw}"}
    fmt = f"<{n * ch}h"
    pcm = struct.unpack(fmt, raw)
    if ch > 1:
        pcm = pcm[::ch]
    if not pcm:
        return {"duration_s": 0.0, "rms": 0.0, "n_samples": 0, "sr": sr}
    s2 = sum(int(x) * int(x) for x in pcm) / max(1, len(pcm))
    rms = (s2 ** 0.5) / 32768.0
    return {
        "duration_s": round(len(pcm) / sr, 3),
        "rms": round(rms, 6),
        "n_samples": len(pcm),
        "sr": sr,
    }


def asr_roundtrip(wav: Path, timeout=600) -> str:
    """Best-effort transcribe with whisper to confirm intelligibility.

    Non-gating: any failure (download, model, crash) returns "" and the CUDA
    verdict is unaffected — this is a confidence signal, not a pass condition.
    """
    if not wav.exists():
        return ""
    cmd = [str(CLI), "--backend", "whisper", "-m", "auto", "--auto-download",
           "-f", str(wav), "--no-prints"]
    try:
        r = subprocess.run(cmd, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        (RESULTS / f"{wav.stem}.asr.log").write_text(r.stdout)
        # The transcript is the non-diagnostic stdout; keep the last non-empty
        # lines that don't look like progress/log noise.
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        text = " ".join(ln for ln in lines
                        if not ln.startswith(("[", "whisper", "ggml", "load")))
        return text[-400:]
    except Exception as ex:  # noqa: BLE001
        return f"<asr-error: {type(ex).__name__}>"


def try_synthesize(name: str, text: str, out_wav: Path,
                   extra_args=None, timeout=1800) -> dict:
    extra_args = extra_args or []
    cmd = [
        str(CLI),
        "--backend", "dots-tts",
        "-m", "auto",
        "--auto-download",
        "--tts", text,
        "--tts-output", str(out_wav),
        "--no-prints",
    ] + extra_args
    t0 = time.time()
    rc = None
    err_excerpt = ""
    try:
        r = subprocess.run(cmd, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        rc = r.returncode
        out = r.stdout
        (RESULTS / f"{name}.log").write_text(out)
        err_excerpt = out[-2048:]
        print(out, flush=True)
    except subprocess.TimeoutExpired as ex:
        rc = -1
        err_excerpt = f"TIMEOUT after {timeout}s\n{ex.stdout or ''}"
        print(err_excerpt, flush=True)

    dt = round(time.time() - t0, 2)
    wav = wav_summary(out_wav) if out_wav.exists() else {"error": "no-wav"}
    return {
        "name": name,
        "backend": "dots-tts",
        "text": text,
        "rc": rc,
        "elapsed_s": dt,
        "wav": wav,
        "err_excerpt": err_excerpt[-1024:],
    }


def verdict(res: dict, min_dur=0.2) -> str:
    if res["rc"] != 0:
        return "FAIL: crashed/non-zero rc"
    w = res["wav"]
    if "error" in w:
        return f"FAIL: wav {w['error']}"
    if w["duration_s"] < min_dur:
        return f"FAIL: wav too short ({w['duration_s']}s)"
    if w["rms"] < 1e-4:
        return f"FAIL: wav silent (rms={w['rms']})"
    return "PASS"


# ─────────────────────────── cell 4 (code) — run dots on CUDA ─────────────
results = []

# (a) Short single-patch smoke test.
step("hw_start")
r = try_synthesize("dots_hw", "Hello world.", RESULTS / "dots_hw.wav")
r["verdict"] = verdict(r)
r["asr"] = asr_roundtrip(RESULTS / "dots_hw.wav")
results.append(r)
step("hw_done", verdict=r["verdict"], rc=r["rc"], wav=r["wav"], asr=r["asr"])

# (b) Multi-sentence paragraph — incremental PatchEncoder + EOS over many steps.
LONG = ("The quick brown fox jumps over the lazy dog. "
        "Speech synthesis on the graphics processor should match the result "
        "on the central processor. This is a longer passage to exercise the "
        "streaming patch encoder and end of sequence detection.")
step("long_start")
r = try_synthesize("dots_long", LONG, RESULTS / "dots_long.wav")
r["verdict"] = verdict(r, min_dur=2.0)
r["asr"] = asr_roundtrip(RESULTS / "dots_long.wav")
results.append(r)
step("long_done", verdict=r["verdict"], rc=r["rc"], wav=r["wav"], asr=r["asr"])


# ─────────────────────────── cell 5 (code) — summary ──────────────────────
summary = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "ref": CRISPASR_REF,
    "gpu": gpu_name,
    "results": results,
    "verdicts": {r["name"]: r["verdict"] for r in results},
    "all_pass": all(r["verdict"] == "PASS" for r in results),
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
print("\n" + "=" * 60)
print(json.dumps(summary, indent=2))
print("=" * 60)

if not summary["all_pass"]:
    print("\nSOME CASES FAILED — see logs in /kaggle/working/results/", flush=True)
    sys.exit(1)
print("\ndots.tts PASSES on CUDA:", gpu_name, flush=True)

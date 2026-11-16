# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — firered encoder GPU A/B on CUDA (§224)
#
# The firered encoder used to rely on ggml's sched auto-copying CPU-resident
# weights to the GPU per layer; ggml removed that resolution, silently making
# the encoder CPU-only. d6e2ad85 added CRISPASR_FIRERED_ENC_GPU=1, which
# split-loads enc.* onto the GPU backend (dec.* stays CPU for the Q4_K SIMD
# vecmat decode path). Verified transcript-identical on Metal (2.3×) and
# Vulkan/MoltenVK (2.1×). This kernel closes the CUDA gap so the default can
# flip.
#
# What it does:
# 1. Clone CrispASR @ env CRISPASR_REF (default: main).
# 2. Build with -DGGML_CUDA=ON.
# 3. Auto-download firered-asr2-aed-q4_k (~918 MB), transcribe samples/jfk.wav
#    twice: (a) CPU-pinned encoder weights (CRISPASR_FIRERED_ENC_CPU=1 —
#    the default DID flip after this A/B passed, so the GPU arm is now
#    the no-env default and the old opt-in CRISPASR_FIRERED_ENC_GPU is
#    no longer read) and (b) the GPU-encoder default, both with
#    FIRERED_BENCH=1.
# 4. PASS iff both transcripts are non-empty and identical, and the GPU
#    encoder is at least as fast as baseline. Encoder timings reported.
#
# Requirements: Kaggle GPU notebook (T4/P100/L4). Disk ~3 GB.

# ─────────────────────────── cell 1 (code) ───────────────────────────
import json
import os
import re
import shutil
import subprocess
import sys
import time
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

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

step("cloned", sha=subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
step("gpu", gpu=gpu_name)

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

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}")
step("build_done")

CLI = BUILD / "examples" / "cli" / "crispasr"
if not CLI.exists():
    candidates = [c for c in BUILD.rglob("crispasr")
                  if c.is_file() and os.access(c, os.X_OK)]
    if not candidates:
        raise SystemExit("crispasr binary not found after build")
    CLI = candidates[0]
print(f"crispasr binary: {CLI}", flush=True)
step("cli_found", path=str(CLI))

LIB_DIR = BUILD / "src"
os.environ["LD_LIBRARY_PATH"] = f"{LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# ─────────────────────────── cell 3 (code) — A/B runs ─────────────────────
AUDIO = REPO / "samples" / "jfk.wav"
CACHE = WORK / "model-cache"
CACHE.mkdir(parents=True, exist_ok=True)

MODEL = CACHE / "firered-asr2-aed-q4_k.gguf"
if not MODEL.exists():
    run(["curl", "-sL", "-o", str(MODEL),
         "https://huggingface.co/cstr/firered-asr2-aed-GGUF/resolve/main/firered-asr2-aed-q4_k.gguf"],
        timeout=1800)
step("model_ready", bytes=MODEL.stat().st_size)

ENC_RE = re.compile(r"firered_bench:\s+encoder\s+([0-9.]+) ms")
DEC_RE = re.compile(r"firered_bench:\s+decoder\s+([0-9.]+) ms")


def firered_run(tag: str, extra_env: dict) -> dict:
    env = {"FIRERED_BENCH": "1", **extra_env}
    t0 = time.time()
    r = run([str(CLI), "--backend", "firered", "-m", str(MODEL),
             "--cache-dir", str(CACHE), "-f", str(AUDIO)],
            capture=True, env=env, timeout=1800)
    wall = time.time() - t0
    out = r.stdout
    enc = ENC_RE.search(out)
    dec = DEC_RE.search(out)
    # Transcript = last non-log line.
    transcript = ""
    for line in reversed(out.strip().splitlines()):
        low = line.strip()
        if not low:
            continue
        if any(low.startswith(p) for p in
               ("crispasr", "firered", "whisper", "ggml", "[step]", "$", "  firered_bench")):
            continue
        transcript = low
        break
    res = {
        "tag": tag,
        "wall_s": round(wall, 2),
        "encoder_ms": float(enc.group(1)) if enc else None,
        "decoder_ms": float(dec.group(1)) if dec else None,
        "transcript": transcript,
    }
    step("run_done", **res)
    return res


baseline = firered_run("cpu_pinned_enc", {"CRISPASR_FIRERED_ENC_CPU": "1"})
gpu_enc = firered_run("enc_gpu", {})
# Second GPU run: warm CUDA context / autotuned kernels.
gpu_enc2 = firered_run("enc_gpu_warm", {})

# ─────────────────────────── cell 4 (code) — verdict ─────────────────────
ok_text = (baseline["transcript"] and
           baseline["transcript"] == gpu_enc["transcript"] == gpu_enc2["transcript"])
ok_speed = (baseline["encoder_ms"] and gpu_enc2["encoder_ms"] and
            gpu_enc2["encoder_ms"] <= baseline["encoder_ms"])
verdict = "PASS" if (ok_text and ok_speed) else "FAIL"

summary = {
    "verdict": verdict,
    "gpu": gpu_name,
    "transcripts_identical": bool(ok_text),
    "encoder_speedup_warm": (round(baseline["encoder_ms"] / gpu_enc2["encoder_ms"], 2)
                             if ok_speed else None),
    "runs": [baseline, gpu_enc, gpu_enc2],
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
step("verdict", verdict=verdict)
if verdict != "PASS":
    raise SystemExit(1)

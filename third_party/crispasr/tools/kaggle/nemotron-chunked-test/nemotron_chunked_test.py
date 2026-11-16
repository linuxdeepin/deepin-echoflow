"""
CrispASR — Nemotron chunked encoder test on Kaggle

Build CrispASR, download the nemotron F16 GGUF, run on JFK audio with
the cache-aware chunked encoder. Report transcript + timing.

GPU worker (for internet + fast build via ccache).
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = REPO / "build"
MODEL_GGUF = WORK / "nemotron-3.5-asr-streaming-0.6b-f16.gguf"
BRANCH = os.environ.get("CRISPASR_REF", "main")

# ── Clone + harness ──
print(f"[1/5] Cloning CrispASR ({BRANCH})...", flush=True)
Path("/kaggle/working/status.txt").write_text("step1: cloning\n")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call(
    ["git", "clone", "--depth", "1", "--branch", BRANCH, "--recursive",
     "https://github.com/CrispStrobe/CrispASR.git", str(REPO)], timeout=120)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.step("cloned")

# ── Build ──
print("[2/5] Building...", flush=True)
Path("/kaggle/working/status.txt").write_text("step2: building\n")
kh.install_build_toolchain()

arch = kh.detect_cuda_arch()
flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()

subprocess.check_call([
    "cmake", "-G", "Ninja", "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
] + flags, cwd=str(REPO), timeout=120)

with kh.build_heartbeat("cmake.build"):
    subprocess.check_call([
        "cmake", "--build", str(BUILD),
        "--target", "crispasr",
        f"-j{kh.safe_build_jobs(gpu=True)}",
    ], cwd=str(REPO), timeout=2400)
kh.step("built")

# ── Download model ──
print("[3/5] Downloading model...", flush=True)
Path("/kaggle/working/status.txt").write_text("step3: downloading model\n")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="cstr/nemotron-3.5-asr-streaming-GGUF",
    filename="nemotron-3.5-asr-streaming-0.6b-f16.gguf",
    cache_dir=str(WORK / "hf_cache"),
)
# Copy to working dir for easier access
shutil.copy2(model_path, str(MODEL_GGUF))
print(f"  Model: {MODEL_GGUF} ({MODEL_GGUF.stat().st_size / 1e6:.0f} MB)", flush=True)
kh.step("model_downloaded")

# ── Test: chunked encoder (default) ──
print("[4/5] Running chunked encoder test on JFK...", flush=True)
Path("/kaggle/working/status.txt").write_text("step4: chunked encoder test\n")

t0 = time.time()
r = subprocess.run([
    str(BUILD / "bin" / "crispasr"),
    "--backend", "nemotron",
    "-m", str(MODEL_GGUF),
    "-f", str(REPO / "samples" / "jfk.wav"),
    "-l", "en-US",
    "--no-prints",
    "--no-gpu",        # CPU — avoids CUDA flash_attn crash on P100 sm_60
    "--no-flash-attn", # belt and suspenders
], capture_output=True, text=True, timeout=1800)  # 30 min for chunked encoder on CPU
dt = time.time() - t0

print(f"  rc={r.returncode}  time={dt:.1f}s", flush=True)
print(f"  stdout: '{r.stdout.strip()}'", flush=True)
if r.stderr:
    print(f"  stderr (last 2000 chars):\n{r.stderr[-2000:]}", flush=True)

transcript = r.stdout.strip()
Path("/kaggle/working/status.txt").write_text(
    f"step4: done rc={r.returncode} time={dt:.1f}s\n"
    f"transcript: {transcript[:200]}\n"
)
kh.step("chunked_test_done", rc=r.returncode, time_s=round(dt, 1),
        transcript_len=len(transcript))

# ── Test: batch encoder (A/B comparison) ──
print("[5/5] Running batch encoder (A/B) on JFK...", flush=True)
t0 = time.time()
env = os.environ.copy()
env["NEMOTRON_BATCH"] = "1"
r2 = subprocess.run([
    str(BUILD / "bin" / "crispasr"),
    "--backend", "nemotron",
    "-m", str(MODEL_GGUF),
    "-f", str(REPO / "samples" / "jfk.wav"),
    "-l", "en-US",
    "--no-prints",
    "--no-gpu",
    "--no-flash-attn",
], capture_output=True, text=True, timeout=600, env=env)
dt2 = time.time() - t0
transcript2 = r2.stdout.strip()
print(f"  batch rc={r2.returncode}  time={dt2:.1f}s", flush=True)
print(f"  batch stdout: '{transcript2}'", flush=True)
kh.step("batch_test_done", rc=r2.returncode, time_s=round(dt2, 1))

# ── Summary ──
print("\n=== SUMMARY ===", flush=True)
print(f"Chunked encoder: '{transcript}' ({dt:.1f}s)", flush=True)
print(f"Batch encoder:   '{transcript2}' ({dt2:.1f}s)", flush=True)

Path("/kaggle/working/status.txt").write_text(
    f"DONE\nchunked: '{transcript}' ({dt:.1f}s)\nbatch: '{transcript2}' ({dt2:.1f}s)\n"
)
kh.step("done")

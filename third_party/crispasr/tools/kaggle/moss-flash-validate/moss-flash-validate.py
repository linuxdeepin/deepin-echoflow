"""
MOSS-Audio flash-attn validation (CPU kernel).

Validates that the encoder flash-attn change (5e222198) produces
correct output on MOSS-Audio-4B-Instruct Q4_K. CPU-only (no GPU
quota cost). Compares ASR transcript + QA mode answer.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

Path("/kaggle/working/started.txt").write_text("started\n")


def run(cmd, check=True, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"failed (rc={r.returncode}): {cmd}")
    return r


# Clone + CPU build
import shutil
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--recursive",
     "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
try:
    import kaggle_harness as kh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("cloned", sha=sha)

kh.install_build_toolchain()
kh.step("toolchain_done")

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = [
    "cmake", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
    "-DCRISPASR_BUILD_TESTS=OFF",
] + kh.cache_and_link_flags()
run(cmake_args)
kh.step("cmake_done")

with kh.build_heartbeat("build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli"
        f" -j{kh.safe_build_jobs(gpu=False)}")

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr")
             if c.is_file() and os.access(c, os.X_OK)]
    assert cands, "crispasr binary not found"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = (
    f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
kh.step("build_done", cli=str(CLI))

# Download MOSS model
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

moss_model = Path(hf_hub_download(
    "cstr/MOSS-Audio-4B-Instruct-GGUF",
    "moss-audio-4b-instruct-q4_k.gguf",
    cache_dir=str(MODELS), token=token))
kh.step("model_downloaded")

jfk = REPO / "samples" / "jfk.wav"
assert jfk.exists()

# Test 1: ASR transcription
print("\n=== TEST 1: MOSS ASR ===", flush=True)
kh.step("asr.start")
t0 = time.time()
r = subprocess.run(
    [str(CLI), "--backend", "moss-audio", "-m", str(moss_model),
     "-f", str(jfk), "-np"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
asr_elapsed = round(time.time() - t0, 1)
asr_transcript = r.stdout.decode("utf-8", errors="replace").strip()
asr_stderr = r.stderr.decode("utf-8", errors="replace")
print(f"  rc={r.returncode} elapsed={asr_elapsed}s", flush=True)
print(f"  transcript ({len(asr_transcript)} chars): {asr_transcript[:200]}", flush=True)
for line in asr_stderr.strip().splitlines()[-3:]:
    print(f"  [stderr] {line}", flush=True)
kh.step("asr.done", rc=r.returncode, elapsed=asr_elapsed,
        chars=len(asr_transcript), preview=asr_transcript[:200])

# Test 2: QA mode
print("\n=== TEST 2: MOSS QA ===", flush=True)
kh.step("qa.start")
t0 = time.time()
r2 = subprocess.run(
    [str(CLI), "--backend", "moss-audio", "-m", str(moss_model),
     "-f", str(jfk), "--prompt", "What is the speaker talking about?", "-np"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
qa_elapsed = round(time.time() - t0, 1)
qa_answer = r2.stdout.decode("utf-8", errors="replace").strip()
print(f"  rc={r2.returncode} elapsed={qa_elapsed}s", flush=True)
print(f"  answer ({len(qa_answer)} chars): {qa_answer[:300]}", flush=True)
kh.step("qa.done", rc=r2.returncode, elapsed=qa_elapsed,
        chars=len(qa_answer), preview=qa_answer[:300])

# Summary
print("\n" + "=" * 60, flush=True)
print(f"SUMMARY — MOSS flash-attn validation — {sha[:8]} (CPU)", flush=True)
print(f"  ASR: rc={r.returncode} {len(asr_transcript)} chars {asr_elapsed}s", flush=True)
print(f"  QA:  rc={r2.returncode} {len(qa_answer)} chars {qa_elapsed}s", flush=True)
asr_ok = r.returncode == 0 and len(asr_transcript) > 20
qa_ok = r2.returncode == 0 and len(qa_answer) > 10
print(f"  VERDICT: {'PASS' if asr_ok and qa_ok else 'FAIL'}", flush=True)
kh.step("summary", asr_ok=asr_ok, qa_ok=qa_ok)
kh._push_progress_to_hf(force=True)
kh.step("script.end")

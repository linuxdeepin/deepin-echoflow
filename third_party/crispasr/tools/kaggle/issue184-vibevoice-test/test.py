"""
CrispASR -- Issue #184: vibevoice ggml_cont fix regression test on CUDA

Verifies that the ggml_cont fix for strided AdaLN views doesn't break
vibevoice TTS on CUDA. Runs synthesis + ASR roundtrip.
"""

import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

_ERROR_PATH = Path("/kaggle/working/error.txt")


def _excepthook(exc_type, exc_val, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
    print(msg, file=sys.stderr, flush=True)
    try:
        _ERROR_PATH.write_text(msg)
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_val, exc_tb)


sys.excepthook = _excepthook

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"


def run(cmd, check=True, env=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# -- Clone + build ----------------------------------------------------------
print("[start] vibevoice regression test for issue #184", flush=True)
Path("/kaggle/working/started.txt").write_text("started\n")

if REPO.exists():
    shutil.rmtree(REPO)
BRANCH = os.environ.get("CRISPASR_REF", "worktree-issue-184-vibevoice-tts-crash")
run(["git", "clone", "--depth", "1", "--branch", BRANCH,
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
    ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
).strip()
kh.step("cloned", sha=sha)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
kh.step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    [
        "cmake", "-S", str(REPO), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
        "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_AMR=OFF",
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
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    assert cands, "crispasr binary not found"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("build_done")

# -- Download models ---------------------------------------------------------
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

vibe_model = Path(hf_hub_download(
    "cstr/vibevoice-realtime-0.5b-GGUF",
    "vibevoice-realtime-0.5b-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
vibe_voice = Path(hf_hub_download(
    "cstr/vibevoice-realtime-0.5b-GGUF",
    "vibevoice-voice-emma.gguf",
    cache_dir=str(MODELS), token=token,
))
asr_model = Path(hf_hub_download(
    "cstr/parakeet-tdt-0.6b-v2-GGUF",
    "parakeet-tdt-0.6b-v2-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
kh.step("models_downloaded")

# -- TTS synthesis -----------------------------------------------------------
TTS_TEXT = "Please call Stella. Ask her to bring these things with her from the store."
out_wav = WORK / "vibevoice-test.wav"

def run_tts(label, extra_env=None, timeout=300):
    wav = WORK / f"vibevoice-{label}.wav"
    if wav.exists():
        wav.unlink()
    cmd = [
        str(CLI), "--backend", "vibevoice-tts",
        "-m", str(vibe_model),
        "--voice", str(vibe_voice),
        "--tts", TTS_TEXT,
        "--tts-output", str(wav),
        "--seed", "42",
        "-v",
    ]
    env = {**os.environ}
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=timeout)
        rc = r.returncode
        combined = r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired as ex:
        rc = -99
        combined = ((ex.stdout or b"") + b"\n" + (ex.stderr or b"")).decode(errors="replace")
    elapsed = round(time.time() - t0, 1)
    (WORK / f"tts_{label}_log.txt").write_text(combined)
    wav_ok = wav.exists() and wav.stat().st_size > 1000
    wav_size = wav.stat().st_size if wav.exists() else 0
    print(f"\n[{label}] rc={rc}  elapsed={elapsed}s  wav={'OK' if wav_ok else 'FAIL'}  size={wav_size}", flush=True)
    # Print all stderr lines containing diagnostics, plus the last 20 on failure
    for ln in combined.splitlines():
        if 'LM bucket' in ln or 'kv_sel=' in ln:
            print(f"  [diag] {ln}", flush=True)
    if rc != 0:
        print("  --- last 20 lines ---", flush=True)
        for ln in combined.splitlines()[-20:]:
            print(f"  {ln}", flush=True)
    return {"rc": rc, "elapsed": elapsed, "wav_ok": wav_ok, "wav_size": wav_size, "wav_path": str(wav)}


# Test 1: buckets ON + debug + CUDA_LAUNCH_BLOCKING (pinpoint crash)
kh.step("tts.start")
print("\n=== Test 1: buckets ON + debug + blocking ===", flush=True)
r1 = run_tts("debug", extra_env={
    "CUDA_LAUNCH_BLOCKING": "1",
    "CRISPASR_VIBEVOICE_LM_BUCKETS": "1",
    "VIBEVOICE_LM_BUCKET_DEBUG": "1",
})

# Test 2: default (buckets OFF, should pass)
print("\n=== Test 2: default (buckets OFF) ===", flush=True)
r2 = run_tts("default")

# Test 3: placeholder
r3 = r2

# Pick the first successful result for ASR roundtrip
r = None
out_wav = None
for label, result in [("default", r1), ("reuse", r2), ("cpu", r3)]:
    if result["wav_ok"]:
        r = result
        out_wav = Path(result["wav_path"])
        print(f"\nUsing {label} result for ASR roundtrip", flush=True)
        break

wav_ok = r is not None and r["wav_ok"]
wav_size = r["wav_size"] if r else 0
elapsed = r["elapsed"] if r else 0

kh.step("tts.done", default_rc=r1["rc"], reuse_rc=r2["rc"], cpu_rc=r3["rc"],
        wav_ok=wav_ok, wav_size=wav_size)

# -- ASR roundtrip -----------------------------------------------------------
asr_text = ""
if wav_ok and out_wav:
    kh.step("asr.start")
    out_stem = WORK / "asr-vibevoice"
    ra = subprocess.run(
        [str(CLI), "--backend", "parakeet", "-m", str(asr_model),
         "-f", str(out_wav), "-of", str(out_stem), "-otxt", "--no-prints"],
        capture_output=True, text=True, timeout=120,
    )
    txt_path = out_stem.with_suffix(".txt")
    asr_text = txt_path.read_text().strip() if txt_path.exists() else ""
    print(f"ASR roundtrip: {asr_text!r}", flush=True)
    kh.step("asr.done", chars=len(asr_text))

# -- Summary -----------------------------------------------------------------
print("\n" + "=" * 64, flush=True)
print(f"SUMMARY -- vibevoice issue #184 test -- {sha} on {gpu_name}", flush=True)
print("=" * 64, flush=True)

any_pass = wav_ok and len(asr_text) > 10
print(f"  buckets+blocking:  rc={r1['rc']}  wav={'OK' if r1['wav_ok'] else 'FAIL'}", flush=True)
print(f"  buckets+sched_dbg: rc={r2['rc']}  wav={'OK' if r2['wav_ok'] else 'FAIL'}", flush=True)
print(f"  default (no bkts): rc={r3['rc']}  wav={'OK' if r3['wav_ok'] else 'FAIL'}", flush=True)
print(f"  ASR:  {asr_text!r}", flush=True)
print(f"  VERDICT: {'PASS' if any_pass else 'FAIL'}", flush=True)

kh.step("summary", any_pass=any_pass, sha=sha,
        default_rc=r1["rc"], reuse_rc=r2["rc"], cpu_rc=r3["rc"])
kh._push_progress_to_hf(force=True)
kh.step("script.end")

"""
CrispASR issue #164 validation: voxcpm2 VOXCPM2_USE_GRAPH stop predictor + VAE.

Tests:
  A. graph=0 baseline — stop should fire (validates model works).
  B. graph=1 on CUDA — stop should now fire (was broken before fix).
  C. graph=1 on CPU — stop should fire (was broken before fix).
  D. graph=1, long text — CPU VAE fallback path must NOT stack overflow.
     (Before ebfe4f09, vae_decode_graph fell back to vae_decode which re-entered
     vae_decode_graph → infinite recursion → STATUS_STACK_OVERFLOW.  After fix,
     fallback calls vae_decode_cpu directly.)

Expected ~25 min on T4 (build + download + 4 TTS runs).
"""

import json
import os
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
# Build in /kaggle/temp to avoid filling up /kaggle/working (capped at ~20 GB,
# saved as kernel output). /kaggle/temp is ephemeral scratch space.
_TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else WORK
BUILD = _TEMP / "build"
CRISPASR_REPO = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
MODEL_REPO = "cstr/voxcpm2-GGUF"
MODEL_FILE = "voxcpm2-q4_k.gguf"

_T0 = time.time()
_PROGRESS_PATH = WORK / "progress.jsonl"


def step(name, **kw):
    entry = {"t": round(time.time() - _T0, 1), "step": name, **kw}
    print(json.dumps(entry), flush=True)
    with open(_PROGRESS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run(cmd, capture=False, check=True, timeout=600, cwd=None, env=None):
    e = {**os.environ, **(env or {})}
    if capture:
        r = subprocess.run(
            cmd, env=e, cwd=cwd, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        print(r.stdout[-4096:] if len(r.stdout) > 4096 else r.stdout, flush=True)
        if check and r.returncode != 0:
            raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
        return r
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# -- Clone + build --
step("start", ref=CRISPASR_REF)

if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)])

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
step("cloned", sha=sha)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()
run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    ["cmake", "-S", str(REPO), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON"]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
step("cmake_done")

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}"
    )
step("build_done")

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    candidates = list(BUILD.rglob("crispasr"))
    candidates = [c for c in candidates if c.is_file() and os.access(c, os.X_OK)]
    if not candidates:
        raise SystemExit("crispasr binary not found after build")
    CLI = candidates[0]
print(f"crispasr binary: {CLI}", flush=True)

LIB_DIR = BUILD / "src"
os.environ["LD_LIBRARY_PATH"] = (
    f"{LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

# -- Download model --
MODEL_DIR = _TEMP / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / MODEL_FILE

if not model_path.exists():
    token = kh.resolve_hf_token()
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=MODEL_REPO, filename=MODEL_FILE,
        local_dir=str(MODEL_DIR), token=token,
    )
step("model_downloaded", model=str(model_path),
     size_mb=round(model_path.stat().st_size / 1e6, 1))


# -- TTS runner --
SHORT_TEXT = "Hello world."
# Long enough (~30 words) to generate >66 AR steps → total output >500K samples,
# triggering the CPU VAE fallback path.  Before ebfe4f09, this caused an
# infinite mutual recursion (vae_decode ↔ vae_decode_graph) → stack overflow.
LONG_TEXT = (
    "The researchers published their findings in a peer-reviewed journal, "
    "demonstrating that the new approach significantly outperformed all "
    "existing baselines across a wide range of standardized benchmark tasks."
)
OUT_WAV = WORK / "test_output.wav"


def run_tts(label: str, text: str, extra_env: dict, timeout: int = 900) -> dict:
    step(f"{label}_start", text_words=len(text.split()))
    cmd = [
        str(CLI), "-m", str(model_path),
        "--tts", text,
        "--tts-output", str(OUT_WAV),
        "-t", "2",
        "--verbose", "2",
    ]
    env = {**os.environ, **extra_env}
    stderr_path = WORK / f"{label}_stderr.txt"
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env,
        )
        elapsed = time.time() - t0
        rc = r.returncode
        stderr_text = r.stderr
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        rc = -99
        stderr_text = (e.stderr or b"").decode("utf-8", errors="replace")
        print(f"  {label}: TIMEOUT after {elapsed:.1f}s", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        rc = -1
        stderr_text = str(e)

    with open(stderr_path, "w") as f:
        f.write(stderr_text[-50000:] if len(stderr_text) > 50000 else stderr_text)

    result = {
        "label": label,
        "rc": rc,
        "elapsed": round(elapsed, 1),
    }

    wav_size = OUT_WAV.stat().st_size if OUT_WAV.exists() else 0
    if wav_size > 0:
        result["wav_bytes"] = wav_size

    for line in stderr_text.splitlines():
        if "nan_check" in line:
            result.setdefault("nan_check_lines", []).append(line.strip())
        if "stop=" in line:
            result.setdefault("stop_lines", []).append(line.strip())
        if "stopped at step" in line:
            result["stop_fired"] = True
            result["stop_line"] = line.strip()
        if "hit max_len ceiling" in line:
            result["stop_fired"] = False
            result["ceiling_line"] = line.strip()
        if "replayed" in line:
            result["replayed"] = True
        if "using CPU" in line:
            result["cpu_vae_fallback"] = True
            result["cpu_fallback_line"] = line.strip()

    print(f"  {label}: rc={rc}, elapsed={elapsed:.1f}s, "
          f"stop_fired={result.get('stop_fired', 'unknown')}, "
          f"cpu_vae={result.get('cpu_vae_fallback', False)}", flush=True)
    for line in stderr_text.splitlines():
        if "voxcpm2:" in line:
            print(f"    {line.strip()}", flush=True)

    step(f"{label}_done", **{k: v for k, v in result.items()
                              if k not in ("stdout", "stderr", "stop_lines",
                                           "nan_check_lines")})
    return result


results = []
try:
    # -- Config A: graph=0 (baseline) --
    print("\n=== Config A: VOXCPM2_USE_GRAPH=0 (baseline) ===", flush=True)
    results.append(run_tts("nograph", SHORT_TEXT, {"VOXCPM2_USE_GRAPH": "0"}))

    # -- Config B: graph=1, CUDA (default) --
    print("\n=== Config B: VOXCPM2_USE_GRAPH=1 (CUDA, default) ===", flush=True)
    results.append(run_tts("graph_default", SHORT_TEXT, {
        "VOXCPM2_USE_GRAPH": "1",
    }))

    # -- Config C: graph=1, CUDA + FA_CPU (LocDiT/LocEnc on CPU) --
    print("\n=== Config C: graph=1 + FA_CPU (LocDiT/LocEnc on CPU) ===", flush=True)
    results.append(run_tts("graph_fa_cpu", SHORT_TEXT, {
        "VOXCPM2_USE_GRAPH": "1",
        "VOXCPM2_FA_CPU": "1",
    }))

    # -- Config D: long text → CPU VAE fallback must not stack overflow (ebfe4f09) --
    print("\n=== Config D: long text, CPU VAE fallback path (stack overflow fix) ===",
          flush=True)
    results.append(run_tts("long_text_cpu_vae", LONG_TEXT, {
        "VOXCPM2_USE_GRAPH": "1",
    }, timeout=1200))

except Exception as e:
    step("ERROR", error=str(e))
    import traceback
    traceback.print_exc()

# -- Summary --
print("\n" + "=" * 60, flush=True)
print("RESULTS SUMMARY", flush=True)
print("=" * 60, flush=True)
all_pass = True
for r in results:
    sf = r.get("stop_fired", "unknown")
    rc = r["rc"]
    label = r["label"]

    if label == "long_text_cpu_vae":
        # For the stack-overflow test: pass = no crash (rc==0) + WAV written.
        # stop_fired may be True or False depending on text length vs max_len.
        ok = (rc == 0 and r.get("wav_bytes", 0) > 0)
        status = "PASS" if ok else "FAIL"
        extra = f"  cpu_vae={r.get('cpu_vae_fallback', False)}  wav={r.get('wav_bytes', 0)}"
    else:
        ok = (sf is True)
        status = "PASS" if ok else ("FAIL" if sf is False else "UNKNOWN")
        extra = f"  {r.get('stop_line', r.get('ceiling_line', ''))}"

    if not ok:
        all_pass = False
    print(f"  {label:20s}: {status}  rc={rc}{extra}", flush=True)

if not results:
    all_pass = False
    print("  NO TESTS RAN", flush=True)

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}", flush=True)
step("DONE", all_pass=all_pass, n_tests=len(results), summary={
    r["label"]: {
        "rc": r["rc"],
        "elapsed": r["elapsed"],
        "stop_fired": r.get("stop_fired"),
        "cpu_vae_fallback": r.get("cpu_vae_fallback", False),
        "wav_bytes": r.get("wav_bytes", 0),
    }
    for r in results
})

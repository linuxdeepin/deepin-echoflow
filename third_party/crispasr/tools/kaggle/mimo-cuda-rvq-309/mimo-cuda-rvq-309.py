"""
CrispASR — #309 independent CUDA verification: MiMo tokenizer GPU RVQ.

PR #309 (W1nge, merged as 2c21990e) moved MiMo's 8-stage Euclidean RVQ onto CUDA
(mul_mat -> argmax -> get_rows -> residual per stage), gated to CUDA-resident
weights with a CPU fallback. The author validated 0/2208 mismatches on an
RTX 2080 Ti. This kernel re-confirms it INDEPENDENTLY on Kaggle CUDA (P100/T4)
before the change ships in a release.

Method:
  1. Build main (contains the merged RVQ) with CUDA — target mimo-tokenizer-smoke.
  2. Download mimo-tokenizer-q4_k.gguf (cstr/mimo-tokenizer-GGUF).
  3. Run the smoke with CRISPASR_MIMO_SMOKE_GPU=1 (weights on CUDA -> CUDA RVQ)
     and CRISPASR_MIMO_TOK_VERIFY_RVQ=1 (runs GPU then forces the CPU path and
     compares every code id).
  4. Assert: (a) "RVQ backend=CUDA" (proves the GPU path is actually active, not
     a silent CPU fallback), (b) "rvq_cpu_gpu_compare ... mismatches=0",
     (c) "smoke: N ok, 0 fail".

Expected ~15-20 min on T4/P100 (build dominates).
"""

import json
import os
import re
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

WORK = Path("/kaggle/working")   # keep tiny (progress only) — gotcha #22
TMP = Path("/kaggle/temp")       # clone + build + model here
REPO = TMP / "CrispASR"
BUILD = TMP / "build"
CRISPASR_REPO = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
MODEL_REPO = "cstr/mimo-tokenizer-GGUF"
MODEL_FILE = "mimo-tokenizer-q4_k.gguf"

_T0 = time.time()
_PROGRESS_PATH = WORK / "progress.jsonl"


def step(name, **kw):
    entry = {"t": round(time.time() - _T0, 1), "step": name, **kw}
    print(json.dumps(entry), flush=True)
    with open(_PROGRESS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run(cmd, check=True, timeout=2400, cwd=None, env=None):
    r = subprocess.run(cmd, env={**os.environ, **(env or {})}, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + CUDA build ─────────────────────────────────────────────────────
step("start", ref=CRISPASR_REF)
if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "5", "--branch", CRISPASR_REF, "--recursive", CRISPASR_REPO, str(REPO)])
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
has_rvq = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", "2c21990e", "HEAD"]).returncode == 0
step("cloned", sha=sha, has_309_rvq=has_rvq)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    ["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON"]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
step("cmake_done", arch=arch)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target mimo-tokenizer-smoke -j{kh.safe_build_jobs(gpu=True)}"
    )
step("build_done")

SMOKE = BUILD / "bin" / "mimo-tokenizer-smoke"
if not SMOKE.exists():
    cands = [c for c in BUILD.rglob("mimo-tokenizer-smoke") if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit("mimo-tokenizer-smoke binary not found after build")
    SMOKE = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
step("smoke_found", path=str(SMOKE))

# ── Download model ─────────────────────────────────────────────────────────
MODEL_DIR = TMP / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / MODEL_FILE
if not model_path.exists():
    from huggingface_hub import hf_hub_download
    try:
        token = kh.resolve_hf_token()
    except Exception:
        token = None
    hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=str(MODEL_DIR), token=token)
step("model_downloaded", size_mb=round(model_path.stat().st_size / 1e6, 1))

AUDIO = REPO / "samples" / "jfk.wav"
if not AUDIO.exists():
    raise SystemExit(f"test audio not found: {AUDIO}")

# ── Run smoke on the CUDA path with the CPU/CUDA RVQ compare ────────────────
env = dict(os.environ)
env["CRISPASR_MIMO_SMOKE_GPU"] = "1"       # load weights on CUDA -> CUDA RVQ active
env["CRISPASR_MIMO_TOK_VERIFY_RVQ"] = "1"  # run GPU then CPU, compare every code id
step("smoke_start", model=str(model_path))
r = subprocess.run([str(SMOKE), str(model_path), str(AUDIO)], timeout=600,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
out = r.stdout
print(out, flush=True)
step("smoke_done", rc=r.returncode)

# ── Assertions ─────────────────────────────────────────────────────────────
failures = []
if r.returncode != 0:
    failures.append(f"smoke exited rc={r.returncode}")
if "RVQ backend=CUDA" not in out:
    failures.append("did NOT report 'RVQ backend=CUDA' — GPU path not active (silent CPU fallback?)")
m = re.search(r"rvq_cpu_gpu_compare\s+codes=(\d+)\s+mismatches=(\d+)", out)
if not m:
    # accept the FAIL variant too, to surface the mismatch count
    m2 = re.search(r"rvq_cpu_gpu_compare.*mismatches=(\d+)", out)
    if not m2:
        failures.append("no rvq_cpu_gpu_compare line found (VERIFY did not run)")
    elif int(m2.group(1)) != 0:
        failures.append(f"CPU/CUDA RVQ mismatch: {m2.group(1)} code ids differ")
elif int(m.group(2)) != 0:
    failures.append(f"CPU/CUDA RVQ mismatch: {m.group(2)} of {m.group(1)} code ids differ")
if re.search(r"smoke:\s+\d+\s+ok,\s+([1-9]\d*)\s+fail", out):
    failures.append("smoke reported >0 failing checks")

print("\n=== #309 CUDA RVQ VERDICT ===", flush=True)
if failures:
    for f in failures:
        print("FAIL: " + f, flush=True)
    step("FAILED", failures=failures)
    raise SystemExit(f"{len(failures)} assertion(s) failed")
codes = m.group(1) if m else "?"
print(f"PASS — CUDA RVQ active, byte-exact vs CPU ({codes} code ids, 0 mismatches).", flush=True)
step("PASSED", gpu=gpu_name, codes=codes)

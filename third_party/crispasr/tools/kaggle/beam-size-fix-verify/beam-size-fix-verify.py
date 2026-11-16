"""
CrispASR — beam_size default fix verification (issue #161).

Verifies that the greedy-by-default change works correctly on CUDA:
  1. Build CrispASR from the fix branch with CUDA.
  2. Download parakeet-tdt-0.6b-v3-q4_k.gguf (~250 MB).
  3. Run three configurations on jfk.wav:
     a) default (should now be greedy)
     b) -bs 1  (explicit greedy)
     c) -bs 5  (explicit beam-5)
  4. Assert: default and -bs 1 produce identical transcripts.
  5. Assert: default is within 1.5x of -bs 1 wall time (not 4-5x).
  6. Report timings so we can confirm beam-5 is the slow path.

Expected ~15 min on T4, ~10 min on P100 (build dominates).
"""

import json
import os
import re
import subprocess
import sys
import time
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
BUILD = WORK / "build"
CRISPASR_REPO = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "worktree-fix-beam-size-defaults")
MODEL_REPO = "cstr/parakeet-tdt-0.6b-v3-GGUF"
MODEL_FILE = "parakeet-tdt-0.6b-v3-q4_k.gguf"

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


# ── Clone + build ──────────────────────────────────────────────────────────
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
step("cli_found", path=str(CLI))

LIB_DIR = BUILD / "src"
os.environ["LD_LIBRARY_PATH"] = (
    f"{LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

# ── Download model ────────────────────────────────────────────────────────
MODEL_DIR = WORK / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / MODEL_FILE

if not model_path.exists():
    token = kh.resolve_hf_token()
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=MODEL_REPO, filename=MODEL_FILE,
        local_dir=str(MODEL_DIR), token=token,
    )
step("model_downloaded", model=str(model_path), size_mb=round(model_path.stat().st_size / 1e6, 1))

# ── Audio file ────────────────────────────────────────────────────────────
AUDIO = REPO / "samples" / "jfk.wav"
if not AUDIO.exists():
    raise SystemExit(f"test audio not found: {AUDIO}")


# ── Run helper ────────────────────────────────────────────────────────────
def run_transcribe(label: str, extra_args: list, n_runs: int = 3) -> dict:
    """Run crispasr n_runs times, return {label, transcript, timings}."""
    step(f"{label}_start")
    timings = []
    transcript = None
    for i in range(n_runs):
        cmd = [
            str(CLI), "-m", str(model_path), "-f", str(AUDIO),
            "--backend", "parakeet", "-t", "2", "-l", "en",
        ] + extra_args
        t0 = time.time()
        r = subprocess.run(
            cmd, timeout=120,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=os.environ,
        )
        elapsed = time.time() - t0
        timings.append(round(elapsed, 3))
        if r.returncode != 0:
            print(f"  {label} run {i}: FAILED rc={r.returncode}", flush=True)
            print(f"  stderr: {r.stderr[-2048:]}", flush=True)
            continue
        # Extract transcript (non-timestamp lines from stdout)
        lines = [l.strip() for l in r.stdout.strip().splitlines()
                 if l.strip() and not l.strip().startswith("[")]
        text = " ".join(lines).strip()
        if transcript is None:
            transcript = text
        print(f"  {label} run {i}: {elapsed:.3f}s", flush=True)

    result = {
        "label": label,
        "transcript": transcript or "",
        "timings": timings,
        "median": sorted(timings)[len(timings) // 2] if timings else 999,
    }
    step(f"{label}_done", **result)
    return result


# ── Run the three configurations ──────────────────────────────────────────
print("\n=== Configuration A: default (should be greedy) ===", flush=True)
res_default = run_transcribe("default", [])

print("\n=== Configuration B: explicit greedy (-bs 1) ===", flush=True)
res_greedy = run_transcribe("greedy_bs1", ["-bs", "1"])

print("\n=== Configuration C: beam-5 (-bs 5) ===", flush=True)
res_beam5 = run_transcribe("beam5", ["-bs", "5"])


# ── Assertions ────────────────────────────────────────────────────────────
print("\n=== Results ===", flush=True)
for r in [res_default, res_greedy, res_beam5]:
    print(f"  {r['label']:15s}: median={r['median']:.3f}s  transcript={r['transcript'][:80]!r}", flush=True)

failures = []

# 1. Default and greedy must produce identical transcripts
if res_default["transcript"] != res_greedy["transcript"]:
    failures.append(
        f"FAIL: default transcript differs from -bs 1!\n"
        f"  default: {res_default['transcript']!r}\n"
        f"  greedy:  {res_greedy['transcript']!r}"
    )

# 2. Default should not be wildly slower than greedy (< 1.5x)
if res_greedy["median"] > 0.01:
    ratio = res_default["median"] / res_greedy["median"]
    if ratio > 1.5:
        failures.append(
            f"FAIL: default is {ratio:.1f}x slower than -bs 1 "
            f"({res_default['median']:.3f}s vs {res_greedy['median']:.3f}s) — "
            f"beam_size default not applied?"
        )
    print(f"  default/greedy ratio: {ratio:.2f}x", flush=True)

# 3. Beam-5 should be noticeably slower than greedy (sanity check)
if res_greedy["median"] > 0.01:
    beam_ratio = res_beam5["median"] / res_greedy["median"]
    print(f"  beam5/greedy ratio: {beam_ratio:.2f}x", flush=True)
    if beam_ratio < 1.3:
        print("  NOTE: beam-5 not much slower than greedy on this clip — "
              "may be too short to show the difference clearly", flush=True)

# 4. All transcripts should be non-empty
for r in [res_default, res_greedy, res_beam5]:
    if not r["transcript"]:
        failures.append(f"FAIL: {r['label']} produced empty transcript")

if failures:
    for f in failures:
        print(f"\n{f}", flush=True)
    step("FAILED", failures=failures)
    raise SystemExit(f"{len(failures)} assertion(s) failed")
else:
    print("\nAll assertions passed.", flush=True)
    step("PASSED", summary={
        "default_median": res_default["median"],
        "greedy_median": res_greedy["median"],
        "beam5_median": res_beam5["median"],
    })

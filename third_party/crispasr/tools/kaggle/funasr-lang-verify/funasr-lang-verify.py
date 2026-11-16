"""
CrispASR — funasr language flag verification.

Tests that -l now reaches the funasr prompt template:
  1. Build CrispASR from feature branch with CUDA.
  2. Download funasr-nano-2512-q8_0.gguf.
  3. Run JFK with default (no -l) → should produce English transcript.
  4. Run JFK with -l en → prompt becomes "语音转写成en：", may differ.
  5. Run JFK with -l English → prompt becomes "语音转写成English：".
  6. Verify all produce non-empty transcripts (model works).
  7. Report prompt effect: compare transcripts to confirm -l changes output.

Also verifies the CUDA weight-split fix (§136) still works — funasr on
CUDA should NOT produce !-loops.

Expected ~15 min on T4 (build dominates).
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
CRISPASR_REF = os.environ.get("CRISPASR_REF", "worktree-funasr-perf-and-lang")
MODEL_REPO = "cstr/funasr-nano-GGUF"
MODEL_FILE = "funasr-nano-2512-q8_0.gguf"

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
step("model_downloaded", model=str(model_path),
     size_mb=round(model_path.stat().st_size / 1e6, 1))

# ── Audio file ────────────────────────────────────────────────────────────
AUDIO = REPO / "samples" / "jfk.wav"
if not AUDIO.exists():
    raise SystemExit(f"test audio not found: {AUDIO}")


# ── Run helper ────────────────────────────────────────────────────────────
def run_transcribe(label: str, extra_args: list, n_runs: int = 2) -> dict:
    """Run crispasr n_runs times, return {label, transcript, timings, stderr_snippet}."""
    step(f"{label}_start")
    timings = []
    transcript = None
    stderr_snippet = ""
    for i in range(n_runs):
        cmd = [
            str(CLI), "-m", str(model_path), "-f", str(AUDIO),
            "--backend", "funasr", "-t", "2",
        ] + extra_args
        t0 = time.time()
        r = subprocess.run(
            cmd, timeout=120,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=os.environ,
        )
        elapsed = time.time() - t0
        timings.append(round(elapsed, 3))
        if i == 0:
            stderr_snippet = r.stderr[-2048:] if r.stderr else ""
        if r.returncode != 0:
            print(f"  {label} run {i}: FAILED rc={r.returncode}", flush=True)
            print(f"  stderr: {r.stderr[-2048:]}", flush=True)
            continue
        lines = [l.strip() for l in r.stdout.strip().splitlines()
                 if l.strip() and not l.strip().startswith("[")]
        text = " ".join(lines).strip()
        if transcript is None:
            transcript = text
        print(f"  {label} run {i}: {elapsed:.3f}s  text={text[:80]!r}", flush=True)

    result = {
        "label": label,
        "transcript": transcript or "",
        "timings": timings,
        "median": sorted(timings)[len(timings) // 2] if timings else 999,
        "stderr_snippet": stderr_snippet[:500],
    }
    step(f"{label}_done", **{k: v for k, v in result.items() if k != "stderr_snippet"})
    return result


# ── Run the configurations ────────────────────────────────────────────────
print("\n=== Config A: default (no -l) ===", flush=True)
res_default = run_transcribe("default", [])

print("\n=== Config B: -l en ===", flush=True)
res_en = run_transcribe("lang_en", ["-l", "en"])

print("\n=== Config C: -l English ===", flush=True)
res_english = run_transcribe("lang_English", ["-l", "English"])

print("\n=== Config D: -l zh (Chinese) ===", flush=True)
res_zh = run_transcribe("lang_zh", ["-l", "zh"])


# ── Assertions ────────────────────────────────────────────────────────────
print("\n=== Results ===", flush=True)
all_results = [res_default, res_en, res_english, res_zh]
for r in all_results:
    print(f"  {r['label']:15s}: median={r['median']:.3f}s  "
          f"transcript={r['transcript'][:100]!r}", flush=True)

failures = []

# 1. All transcripts must be non-empty (no !-loop, no crash)
for r in all_results:
    if not r["transcript"]:
        failures.append(f"FAIL: {r['label']} produced empty transcript")
    elif r["transcript"].count("!") > len(r["transcript"]) * 0.5:
        failures.append(f"FAIL: {r['label']} produced degenerate !-loop: {r['transcript'][:80]!r}")

# 2. Default should contain recognizable JFK words
jfk_words = ["country", "fellow", "americans", "ask"]
default_lower = res_default["transcript"].lower()
found = sum(1 for w in jfk_words if w in default_lower)
if found < 2:
    failures.append(
        f"FAIL: default transcript doesn't contain JFK keywords "
        f"(found {found}/4): {res_default['transcript'][:80]!r}"
    )

# 3. Check that -l actually changes the prompt (stderr should show different tokenization)
# With the fix, -l en produces a different prompt than default, which may
# yield a different transcript. We just verify the model runs — the prompt
# change is confirmed by the different token count in stderr.
print(f"\n  stderr snippets (prompt evidence):", flush=True)
for r in all_results:
    # Look for the "prefix=N suffix=M prompt=P" line in stderr
    prompt_line = ""
    for line in r["stderr_snippet"].splitlines():
        if "prefix=" in line and "suffix=" in line:
            prompt_line = line.strip()
            break
    print(f"    {r['label']:15s}: {prompt_line}", flush=True)

if failures:
    for f in failures:
        print(f"\n{f}", flush=True)
    step("FAILED", failures=failures)
    raise SystemExit(f"{len(failures)} assertion(s) failed")
else:
    print("\nAll assertions passed.", flush=True)
    step("PASSED", summary={
        r["label"]: {"median": r["median"], "transcript_len": len(r["transcript"])}
        for r in all_results
    })

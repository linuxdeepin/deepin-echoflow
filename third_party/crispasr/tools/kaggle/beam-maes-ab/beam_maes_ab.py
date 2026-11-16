# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — beam search / MAES A/B benchmark (§167)
#
# Tests beam search and MAES on backends that are too large for the VPS
# (8 GB RAM). Runs on Kaggle T4/P100 with CUDA.
#
# Backends tested:
#   1. moss-audio (4B, core_beam_decode replay)   — greedy vs beam=4
#   2. sensevoice (encoder CTC, core_ctc beam)    — greedy vs beam=8+gamma
#   3. granite-nle (BPE CTC, core_ctc beam)       — greedy vs beam=8+gamma
#   4. nemotron (RNNT beam + MAES, GPU validation) — greedy vs beam=4 vs MAES
#
# Each test: 3 runs, median wall time, text output comparison.
# Results written to /kaggle/working/beam_ab_results.json

# ─────────────────────────── cell 1 (code) ───────────────────────────
import json
import os
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
RESULTS = []

def step(msg):
    elapsed = time.time() - _T0
    print(f"\n{'='*60}\n[{elapsed:.0f}s] {msg}\n{'='*60}", flush=True)

_T0 = time.time()

# ─────────────────────────── cell 2 (code) ───────────────────────────
# ── Clone + build CrispASR with CUDA ──
step("Clone CrispASR")
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_DIR = WORK / "CrispASR"

if not CRISPASR_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", CRISPASR_URL, str(CRISPASR_DIR)])

sys.path.insert(0, str(CRISPASR_DIR / "tools" / "kaggle"))
# Fallback to bundled harness (same dir as this script)
try:
    _script_dir = str(Path(__file__).resolve().parent)
except NameError:
    _script_dir = str(WORK)
sys.path.insert(0, _script_dir)

import kaggle_harness as kh
kh.init_progress()

step("Build CrispASR (CUDA)")
kh.install_build_toolchain()

build_dir = CRISPASR_DIR / "build"
os.makedirs(build_dir, exist_ok=True)
os.chdir(str(CRISPASR_DIR))

# Configure with CUDA
subprocess.check_call([
    "cmake", "-G", "Ninja", "-B", str(build_dir),
    "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON",
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
], env={**os.environ, "CCACHE_DIR": str(WORK / ".ccache")})

# Build (heartbeat to avoid Kaggle timeout)
with kh.build_heartbeat("cuda-build"):
    subprocess.check_call(
        ["cmake", "--build", str(build_dir), "-j4"],
        env={**os.environ, "CCACHE_DIR": str(WORK / ".ccache")}
    )

CLI = str(build_dir / "bin" / "crispasr")
AUDIO = str(CRISPASR_DIR / "samples" / "jfk.wav")
assert os.path.isfile(CLI), f"CLI not found: {CLI}"
assert os.path.isfile(AUDIO), f"Audio not found: {AUDIO}"

# ─────────────────────────── cell 3 (code) ───────────────────────────
# ── Helper: run a backend with given flags, return text + timing ──
def run_backend(backend, model_path, extra_args=None, env_extra=None, n_runs=3):
    """Run CLI n_runs times, return dict with text, times, median_time."""
    cmd = [CLI, "-m", model_path, "--backend", backend, "-f", AUDIO, "--no-prints"]
    if extra_args:
        cmd.extend(extra_args)

    run_env = {**os.environ}
    if env_extra:
        run_env.update(env_extra)

    texts = []
    times = []
    for i in range(n_runs):
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, env=run_env, timeout=600)
        elapsed = time.time() - t0
        text = result.stdout.strip()
        texts.append(text)
        times.append(elapsed)
        print(f"  run {i+1}/{n_runs}: {elapsed:.1f}s  text={text[:80]}...", flush=True)

    median_time = sorted(times)[len(times)//2]
    return {
        "text": texts[0],
        "texts_identical": all(t == texts[0] for t in texts),
        "times": times,
        "median_time": median_time,
    }

def ab_test(backend, model_path, label_a, args_a, label_b, args_b,
            env_a=None, env_b=None):
    """Run A/B comparison and log results."""
    print(f"\n--- {backend}: {label_a} ---", flush=True)
    a = run_backend(backend, model_path, args_a, env_a)
    print(f"\n--- {backend}: {label_b} ---", flush=True)
    b = run_backend(backend, model_path, args_b, env_b)

    same_text = a["text"] == b["text"]
    result = {
        "backend": backend,
        "label_a": label_a,
        "label_b": label_b,
        "text_a": a["text"],
        "text_b": b["text"],
        "same_text": same_text,
        "median_a": a["median_time"],
        "median_b": b["median_time"],
        "speedup": a["median_time"] / b["median_time"] if b["median_time"] > 0 else 0,
    }
    RESULTS.append(result)
    print(f"\n  RESULT: {label_a}={a['median_time']:.1f}s  {label_b}={b['median_time']:.1f}s  "
          f"same_text={same_text}", flush=True)
    return result

# ─────────────────────────── cell 4 (code) ───────────────────────────
# ── Download models ──
step("Download models via auto-download")

# Each backend gets its model via -m auto (uses registry)
# But we need to pre-download to avoid timeout in timed runs.
# Use the CLI's auto-download by running a tiny warmup.

MODELS = {}

def download_model(backend, extra_args=None):
    """Download model by running a tiny warmup transcription."""
    silence = WORK / "silence_100ms.wav"
    if not silence.exists():
        # Generate 100ms silence WAV
        import struct
        sr = 16000
        n = sr // 10  # 100ms
        data = struct.pack(f"<{n}h", *([0]*n))
        with open(silence, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + len(data)))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr*2, 2, 16))
            f.write(b"data")
            f.write(struct.pack("<I", len(data)))
            f.write(data)

    cmd = [CLI, "-m", "auto", "--backend", backend, "-f", str(silence), "--no-prints"]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Downloading {backend}...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    # Find the model path from the verbose output
    # The model is now cached; subsequent runs use the cache
    print(f"  {backend}: exit={r.returncode}", flush=True)
    return "auto"

# Download each model
for backend in ["nemotron", "sensevoice", "moss-audio"]:
    try:
        MODELS[backend] = download_model(backend)
        print(f"  {backend}: ready", flush=True)
    except Exception as e:
        print(f"  {backend}: FAILED ({e})", flush=True)

# granite-nle needs explicit model path
try:
    MODELS["granite-4.1-nar"] = download_model("granite-4.1-nar")
    print(f"  granite-4.1-nar: ready", flush=True)
except Exception as e:
    print(f"  granite-4.1-nar: FAILED ({e})", flush=True)

# ─────────────────────────── cell 5 (code) ───────────────────────────
# ── A/B tests (each wrapped in try/except for partial results) ──

def save_partial():
    """Write whatever results we have so far."""
    rp = WORK / "beam_ab_results.json"
    with open(rp, "w") as f:
        json.dump(RESULTS, f, indent=2)

# 1. Nemotron: greedy vs beam=4 vs MAES
if "nemotron" in MODELS:
    try:
        step("Nemotron: greedy vs beam=4")
        ab_test("nemotron", "auto",
                "greedy", ["--beam-size", "1"],
                "beam=4", ["--beam-size", "4"])
        save_partial()
    except Exception as e:
        print(f"FAILED: nemotron beam: {e}", flush=True)

    try:
        step("Nemotron: beam=4 vs MAES(4,2,2.3)")
        ab_test("nemotron", "auto",
                "beam=4", ["--beam-size", "4"],
                "MAES", ["--beam-size", "4"],
                env_b={"CRISPASR_NEMOTRON_MAES": "1"})
        save_partial()
    except Exception as e:
        print(f"FAILED: nemotron MAES: {e}", flush=True)

# 2. Sensevoice: greedy vs beam=8
if "sensevoice" in MODELS:
    try:
        step("Sensevoice: greedy vs beam=8+gamma")
        ab_test("sensevoice", "auto",
                "greedy", ["--beam-size", "1"],
                "beam=8", ["--beam-size", "8"])
        save_partial()
    except Exception as e:
        print(f"FAILED: sensevoice beam: {e}", flush=True)

# 3. Granite-NLE: greedy vs beam=8
if "granite-4.1-nar" in MODELS:
    try:
        step("Granite-NLE: greedy vs beam=8")
        ab_test("granite-4.1-nar", "auto",
                "greedy", ["--beam-size", "1"],
                "beam=8", ["--beam-size", "8"])
        save_partial()
    except Exception as e:
        print(f"FAILED: granite-nle beam: {e}", flush=True)

# 4. Moss-Audio: greedy vs beam=4
if "moss-audio" in MODELS:
    try:
        step("Moss-Audio: greedy vs beam=4")
        ab_test("moss-audio", "auto",
                "greedy", ["--beam-size", "1"],
                "beam=4", ["--beam-size", "4"])
        save_partial()
    except Exception as e:
        print(f"FAILED: moss-audio beam: {e}", flush=True)

# ─────────────────────────── cell 6 (code) ───────────────────────────
# ── Summary ──
step("Summary")

results_path = WORK / "beam_ab_results.json"
with open(results_path, "w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nResults written to {results_path}", flush=True)

print("\n" + "="*80)
print("BEAM / MAES A/B RESULTS (§167)")
print("="*80)
print(f"{'Backend':20s} {'A':15s} {'B':15s} {'A time':>8s} {'B time':>8s} {'Same?':>6s}")
print("-"*80)
for r in RESULTS:
    print(f"{r['backend']:20s} {r['label_a']:15s} {r['label_b']:15s} "
          f"{r['median_a']:7.1f}s {r['median_b']:7.1f}s {'YES' if r['same_text'] else 'NO':>6s}")
print("-"*80)
print()
for r in RESULTS:
    if not r["same_text"]:
        print(f"TEXT DIFF: {r['backend']} {r['label_a']} vs {r['label_b']}")
        print(f"  A: {r['text_a'][:200]}")
        print(f"  B: {r['text_b'][:200]}")
        print()

total_time = time.time() - _T0
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")

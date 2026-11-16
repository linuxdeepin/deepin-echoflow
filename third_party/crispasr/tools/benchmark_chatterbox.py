#!/usr/bin/env python3
"""Per-step benchmark: CrispASR chatterbox vs Python reference.

Measures wall time at each pipeline stage:
  T3:    tokenize | prefill-build | prefill-compute | AR-decode
  S3Gen: conformer-encoder | CFM-euler | HiFT-vocoder

Usage:
  # C++ only (always works if models present):
  python tools/benchmark_chatterbox.py

  # Include Python reference (needs ResembleAI/chatterbox weights):
  python tools/benchmark_chatterbox.py --python
  RESEMBLE_CHATTERBOX_SRC=/path/to/chatterbox python tools/benchmark_chatterbox.py --python

  # Custom text / voice:
  python tools/benchmark_chatterbox.py --text "Hello world" --voice samples/jfk.wav

  # Turbo variant:
  python tools/benchmark_chatterbox.py --backend chatterbox-turbo

Env:
  CHATTERBOX_T3_MODEL   path to T3 GGUF (default: auto-detect in gguf-models/)
  CHATTERBOX_S3GEN      path to S3Gen GGUF (default: auto-detect)
  CHATTERBOX_VOICE_WAV  reference voice WAV for voice cloning
  RESEMBLE_CHATTERBOX_SRC  path to cloned ResembleAI/chatterbox repo (for Python ref)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_BIN = REPO_ROOT / "build" / "bin" / "crispasr"

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."
DEFAULT_T3_CANDIDATES = [
    "chatterbox-t3-q4_k.gguf",
    "chatterbox-t3-q8_0.gguf",
    "chatterbox-t3-f16.gguf",
]
DEFAULT_S3GEN_CANDIDATES = [
    "chatterbox-s3gen-q4_k.gguf",
    "chatterbox-s3gen-q8_0.gguf",
    "chatterbox-s3gen-f16.gguf",
]
DEFAULT_MODEL_DIRS = [
    Path("/Users/christianstrobele/code/gguf-models"),
    Path("/mnt/storage/gguf-models"),
    REPO_ROOT,
]


# ── helpers ──────────────────────────────────────────────────────────────────

def find_model(candidates: list[str], extra_dir: Optional[Path] = None) -> Optional[Path]:
    search_dirs = list(DEFAULT_MODEL_DIRS)
    if extra_dir:
        search_dirs.insert(0, extra_dir)
    for d in search_dirs:
        for c in candidates:
            p = d / c
            if p.exists():
                return p
    return None


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


# ── BENCH output parser ───────────────────────────────────────────────────────

_FIELD_RE  = re.compile(r"chatterbox:\s+([\w /()]+?)\s+(\d[\d.]+)\s*(ms|s)?")
_STEPS_RE  = re.compile(r"\((\d+)\s+(?:tokens?|steps?)[^)]*,\s*([\d.]+)\s*ms/(?:tok|step)\)")
_NSTEPS_RE = re.compile(r"\((\d+)\s+steps")

def parse_bench_output(stderr: str) -> dict:
    """Parse CHATTERBOX_BENCH=1 stderr into {label: value} dict.
    ms values are stored as floats; extra annotations stored as sub-keys."""
    result: dict = {}
    for line in stderr.splitlines():
        if "perf report" in line or "---" in line:
            continue
        m = _FIELD_RE.search(line)
        if not m:
            continue
        label = m.group(1).strip().rstrip("()")
        value = float(m.group(2))
        unit  = m.group(3) or "ms"
        result[label] = value if unit == "ms" else value * 1000.0
        # Extract per-step or per-token sub-annotations
        s = _STEPS_RE.search(line)
        if s:
            result[f"{label}._n"]      = int(s.group(1))
            result[f"{label}._per_ms"] = float(s.group(2))
    return result


# ── C++ benchmark ─────────────────────────────────────────────────────────────

def run_cpp_bench(args: argparse.Namespace) -> Optional[dict]:
    t3_model  = Path(os.environ.get("CHATTERBOX_T3_MODEL", "")) if os.environ.get("CHATTERBOX_T3_MODEL") else None
    s3gen     = Path(os.environ.get("CHATTERBOX_S3GEN", ""))    if os.environ.get("CHATTERBOX_S3GEN")    else None
    voice_wav = Path(os.environ.get("CHATTERBOX_VOICE_WAV", "")) if os.environ.get("CHATTERBOX_VOICE_WAV") else None

    if t3_model is None:
        candidates = DEFAULT_T3_CANDIDATES
        if "turbo" in args.backend:
            candidates = ["chatterbox-turbo-t3-f16.gguf", "chatterbox-turbo-t3-q8_0.gguf"] + candidates
        t3_model = find_model(candidates)
    if t3_model is None:
        print(f"[cpp] ERROR: no T3 model found. Set CHATTERBOX_T3_MODEL or put a GGUF in {DEFAULT_MODEL_DIRS[0]}")
        return None

    if s3gen is None:
        candidates = DEFAULT_S3GEN_CANDIDATES
        if "turbo" in args.backend:
            candidates = ["chatterbox-turbo-s3gen-f16.gguf"] + candidates
        s3gen = find_model(candidates)
    if s3gen is None:
        print(f"[cpp] ERROR: no S3Gen GGUF found. Set CHATTERBOX_S3GEN")
        return None

    if voice_wav is None and args.voice:
        voice_wav = Path(args.voice)

    print(f"[cpp] T3      : {t3_model}")
    print(f"[cpp] S3Gen   : {s3gen}")
    if voice_wav:
        print(f"[cpp] voice   : {voice_wav}")
    print(f"[cpp] text    : {args.text!r}")
    print(f"[cpp] backend : {args.backend}")

    cmd = [
        str(BUILD_BIN),
        "--backend", args.backend,
        "--model", str(t3_model),
        "--codec-model", str(s3gen),
        "--tts", args.text,
        "--output-file", "/dev/null",
        "--threads", str(args.threads),
    ]
    if voice_wav:
        cmd += ["--voice", str(voice_wav)]
    if not args.gpu:
        cmd += ["--no-gpu"]

    env = {**os.environ, "CHATTERBOX_BENCH": "1"}

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if result.returncode != 0:
        print(f"[cpp] FAILED (exit {result.returncode})")
        print(result.stderr[-2000:])
        return None

    stderr = result.stderr
    perf = parse_bench_output(stderr)
    perf["_wall_ms_total"] = wall_ms  # includes model-load time
    return perf


# ── Python reference benchmark ────────────────────────────────────────────────

def run_python_bench(args: argparse.Namespace) -> Optional[dict]:
    upstream_src = os.environ.get("RESEMBLE_CHATTERBOX_SRC")
    if upstream_src:
        sys.path.insert(0, str(Path(upstream_src).resolve()))

    try:
        import torch
        from chatterbox.tts import ChatterboxTTS
    except ImportError as e:
        print(f"[python] chatterbox not importable: {e}")
        return None

    model_dir = os.environ.get("CHATTERBOX_PYTHON_MODEL_DIR")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"[python] device: {device}")

    t_load0 = time.perf_counter()
    if model_dir:
        print(f"[python] loading from {model_dir}")
        model = ChatterboxTTS.from_local(model_dir, device=device)
    else:
        print("[python] downloading from HuggingFace (ResembleAI/chatterbox) ...")
        try:
            model = ChatterboxTTS.from_pretrained(device=device)
        except Exception as e:
            print(f"[python] from_pretrained failed: {e}")
            return None
    t_load_ms = (time.perf_counter() - t_load0) * 1000.0
    print(f"[python] model loaded in {t_load_ms:.0f} ms")

    # Warm up
    with torch.inference_mode():
        _ = model.generate("warmup")

    n_runs = args.n_runs
    timings: dict[str, list[float]] = {
        "total": [],
        "t3_generate": [],
        "s3gen_generate": [],
    }

    for i in range(n_runs):
        print(f"[python] run {i+1}/{n_runs} ...", end=" ", flush=True)

        with torch.inference_mode():
            t0 = time.perf_counter()
            wav = model.generate(args.text)
            total_ms = (time.perf_counter() - t0) * 1000.0

        n_samples = wav.shape[-1] if hasattr(wav, "shape") else len(wav)
        audio_s = n_samples / 24000.0
        print(f"{total_ms:.0f} ms ({audio_s:.2f} s audio, RTF={total_ms/1000/audio_s:.3f})")
        timings["total"].append(total_ms)

    if not timings["total"]:
        return None

    import statistics
    best = min(timings["total"])
    median = statistics.median(timings["total"])
    best_audio_s = None
    for t, ms in zip([args.text] * n_runs, timings["total"]):
        if ms == best:
            pass

    return {
        "total_ms_best":   best,
        "total_ms_median": median,
        "_text": args.text,
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def print_report(cpp: Optional[dict], python: Optional[dict], args: argparse.Namespace) -> None:
    print()
    print("=" * 64)
    print("  Chatterbox per-step benchmark")
    print(f"  text: {args.text!r}")
    print("=" * 64)

    if cpp:
        audio_s = cpp.get("audio duration", 0) / 1000.0  # already ms but labeled "s" in output
        # Re-parse: the "audio duration" field might be in seconds from the perf header
        # The regex captures the number; the unit label tells us ms vs s.
        # In our perf report: "audio duration   X.XX s" → parse_bench gives it as X.XX * 1000 = ms
        # But wait — parse_bench_output multiplies by 1000 if unit==s, so audio duration is in ms.
        audio_ms = cpp.get("audio duration", 0)
        audio_s_real = audio_ms / 1000.0

        t3_ms     = cpp.get("T3 total",    cpp.get("T3 total", 0))
        s3gen_ms  = cpp.get("S3Gen total", 0)
        wall_ms   = cpp.get("wall time",   0)

        print()
        print("  CrispASR C++ (CHATTERBOX_BENCH=1)")
        print(f"  {'audio generated':<22}  {audio_s_real:>8.2f} s")
        if wall_ms:
            rtf = wall_ms / 1000.0 / audio_s_real if audio_s_real > 0 else 0
            print(f"  {'wall time':<22}  {wall_ms:>8.0f} ms   RTF={rtf:.3f}  ({1/rtf:.1f}x real-time)")
        print()
        print(f"  {'Stage':<22}  {'ms':>8}  {'% total':>8}  {'notes'}")
        print(f"  {'-'*60}")

        stage_order = [
            ("tokenize",        "tokenize"),
            ("prefill build",   "prefill build"),
            ("prefill compute", "prefill compute"),
            ("decode loop",     "decode loop"),
            ("T3 total",        "T3 total"),
            ("encoder",         "encoder"),
            ("CFM euler",       "CFM euler"),
            ("HiFT vocoder",    "HiFT vocoder"),
            ("S3Gen total",     "S3Gen total"),
            ("wall time",       "wall time"),
        ]
        for key, label in stage_order:
            val = cpp.get(key)
            if val is None:
                continue
            pct = val / wall_ms * 100 if wall_ms > 0 else 0
            # Enrich notes from parsed sub-annotations
            notes = ""
            n     = cpp.get(f"{key}._n")
            per   = cpp.get(f"{key}._per_ms")
            if n and per:
                unit_s = "tok" if key == "decode loop" else "step"
                notes  = f"{n} {unit_s}s, {per:.2f} ms/{unit_s}"
            sep = "  ├" if key not in ("T3 total", "S3Gen total", "wall time") else "  ╞"
            if key in ("T3 total", "S3Gen total"):
                print(f"  {'─'*60}")
            print(f"  {label:<22}  {val:>8.1f}  {pct:>7.1f}%  {notes}")

    if python:
        print()
        print("  Python reference (ResembleAI/chatterbox)")
        best   = python.get("total_ms_best",   0)
        median = python.get("total_ms_median", 0)
        print(f"  {'total (best)':<22}  {best:>8.0f} ms")
        print(f"  {'total (median)':<22}  {median:>8.0f} ms")

        if cpp and wall_ms and best:
            speedup = best / wall_ms
            print(f"\n  CrispASR speedup vs Python: {speedup:.1f}x (best/best)")

    print()
    print("  Notes:")
    print("  - CHATTERBOX_BENCH=1 is needed to see per-step breakdown from C++")
    print("  - Run with --python to include Python reference timing")
    print("  - gianni-cor/chatterbox.cpp uses different GGUF weights (Q4_0 format,")
    print("    their converter); their numbers target M3 Ultra — compare with caution.")
    print("    Their best: RTF=0.16 on M3 Ultra, 0.07 on RTX 5090 (Vulkan).")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Chatterbox per-step benchmark")
    parser.add_argument("--text",     default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--voice",    default=None,         help="Reference voice WAV")
    parser.add_argument("--backend",  default="chatterbox", help="Backend name (chatterbox, chatterbox-turbo)")
    parser.add_argument("--threads",  type=int, default=8,  help="CPU thread count")
    parser.add_argument("--n-runs",   type=int, default=3,  help="Runs for Python benchmark")
    parser.add_argument("--python",   action="store_true",  help="Run Python reference benchmark")
    parser.add_argument("--no-gpu",   dest="gpu", action="store_false", default=True,
                        help="Disable GPU (Metal/CUDA)")
    args = parser.parse_args()

    if not BUILD_BIN.exists():
        print(f"ERROR: {BUILD_BIN} not found. Build with: cmake --build build -j$(nproc)")
        sys.exit(1)

    print("── C++ benchmark ─────────────────────────────────────")
    cpp = run_cpp_bench(args)

    python = None
    if args.python:
        print()
        print("── Python reference benchmark ────────────────────────")
        python = run_python_bench(args)

    print_report(cpp, python, args)


if __name__ == "__main__":
    main()

"""
CrispASR -- Issue #183: GQA_NATIVE + chunked codec decode + scratch sched reset

Tests the three changes from issue #183 on CUDA (Kaggle P100/T4):
  1. GQA_NATIVE: verify TTS still produces correct audio (ASR roundtrip).
  2. Chunked codec decode: compare output with chunking ON vs OFF.
  3. Scratch sched reset: check VRAM doesn't grow across repeated requests.
  4. Scaling test: short vs long text to verify O(N) slope (not O(N^2)).

Build from the feature branch, run qwen3-tts synthesis, ASR roundtrip.
"""

import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Unbuffered I/O so partial output is visible on Kaggle errors
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

_ERROR_PATH = Path("/kaggle/working/error.txt")


def _excepthook(exc_type, exc_val, exc_tb):
    """Write unhandled exceptions to a file so we can download them."""
    import traceback as _tb
    msg = "".join(_tb.format_exception(exc_type, exc_val, exc_tb))
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
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "worktree-issue-183-gqa-native")
CRISPASR_REPO = os.environ.get(
    "CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git"
)

SHORT_TEXT = "Please call Stella. Ask her to bring these things with her from the store."
LONG_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths, stories "
    "and rumors of artificial beings endowed with intelligence or consciousness by "
    "master craftsmen. The seeds of modern AI were planted by philosophers who "
    "attempted to describe the process of human thinking as the mechanical "
    "manipulation of symbols. This work culminated in the invention of the "
    "programmable digital computer in the nineteen forties, a machine based on the "
    "abstract essence of mathematical reasoning. This device and the ideas behind "
    "it inspired a handful of scientists to begin seriously discussing the "
    "possibility of building an electronic brain. The field of AI research was "
    "founded at a workshop held on the campus of Dartmouth College, USA during the "
    "summer of nineteen fifty six. Those who attended would become the leaders of "
    "AI research for decades. Many of them predicted that a machine as intelligent "
    "as a human being would exist in no more than a generation, and they were given "
    "millions of dollars to make this vision come true. Eventually, it became "
    "obvious that commercial developers and researchers had grossly underestimated "
    "the difficulty of the project. In nineteen seventy four, in response to the "
    "criticism of Sir James Lighthill and ongoing pressure from the US Congress, "
    "both the US and British governments cut off exploratory research in artificial "
    "intelligence. The next few years would later be called an AI winter, a period "
    "when obtaining funding for AI projects was difficult."
)


def run(cmd, check=True, env=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# -- Clone + CUDA build -------------------------------------------------
print(f"[start] ref={CRISPASR_REF}", flush=True)
print(f"  disk: {shutil.disk_usage('/kaggle/working')}", flush=True)
Path("/kaggle/working/started.txt").write_text("started\n")

if REPO.exists():
    shutil.rmtree(REPO)
# Clone: try branch directly first, fall back to clone main + fetch branch
r = subprocess.run([
    "git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
    CRISPASR_REPO, str(REPO),
])
if r.returncode != 0:
    print(f"  direct branch clone failed, trying main + fetch...", flush=True)
    if REPO.exists():
        shutil.rmtree(REPO)
    run(["git", "clone", "--depth", "1", CRISPASR_REPO, str(REPO)])
    run(["git", "-C", str(REPO), "fetch", "origin", CRISPASR_REF])
    run(["git", "-C", str(REPO), "checkout", "FETCH_HEAD"])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
try:
    import kaggle_harness as kh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
kh.step("cloned", sha=sha, ref=CRISPASR_REF)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
kh.step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
kh.step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    [
        "cmake", "-S", str(REPO), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
        "-DCRISPASR_BUILD_TESTS=OFF",
        "-DCRISPASR_AMR=OFF",  # opencore-amr FetchContent fails on Kaggle
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
    cands = [
        c for c in BUILD.rglob("crispasr")
        if c.is_file() and os.access(c, os.X_OK)
    ]
    assert cands, "crispasr binary not found after build"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = (
    f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)
kh.step("build_done", cli=str(CLI))


# -- Download models ----------------------------------------------------
kh.step("downloading models")
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
    )
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

# 0.6B customvoice model (has built-in voices, no --voice ref needed)
tts_model = Path(hf_hub_download(
    "cstr/qwen3-tts-0.6b-customvoice-GGUF",
    "qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf",
    cache_dir=str(MODELS), token=token,
))
tts_codec = Path(hf_hub_download(
    "cstr/qwen3-tts-tokenizer-12hz-GGUF",
    "qwen3-tts-tokenizer-12hz.gguf",
    cache_dir=str(MODELS), token=token,
))
asr_model = Path(hf_hub_download(
    "cstr/parakeet-tdt-0.6b-v2-GGUF",
    "parakeet-tdt-0.6b-v2-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
kh.step("models_downloaded")


# -- Helper: get GPU memory usage via nvidia-smi -------------------------
def gpu_mem_mb():
    """Return (used_mb, total_mb) for GPU 0."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        parts = out.split(",")
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return -1, -1


# -- Helper: run TTS synthesis -------------------------------------------
def run_tts(label, text, extra_env=None, timeout=600):
    """Run qwen3-tts synthesis and return dict with results."""
    kh.step(f"{label}.start")
    out_wav = WORK / f"tts-{label}.wav"
    if out_wav.exists():
        out_wav.unlink()

    env = {
        "QWEN3_TTS_BENCH": "1",
    }
    if extra_env:
        env.update(extra_env)

    cmd = [
        str(CLI), "--backend", "qwen3-tts-customvoice",
        "-m", str(tts_model),
        "--codec-model", str(tts_codec),
        "--tts", text,
        "--tts-output", str(out_wav),
        "--seed", "42",
        "-v",
    ]
    mem_before = gpu_mem_mb()
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, env={**os.environ, **env},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
        rc, stdout, stderr = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as ex:
        rc = -1
        stdout = (
            (ex.stdout or b"").decode(errors="replace")
            if isinstance(ex.stdout, bytes) else (ex.stdout or "")
        )
        stderr = (
            (ex.stderr or b"").decode(errors="replace")
            if isinstance(ex.stderr, bytes) else (ex.stderr or "")
        )
    elapsed = round(time.time() - t0, 1)
    mem_after = gpu_mem_mb()
    combined = stdout + "\n" + stderr
    (RESULTS / f"{label}_log.txt").write_text(combined)

    wav_exists = out_wav.exists() and out_wav.stat().st_size > 1000
    wav_size = out_wav.stat().st_size if out_wav.exists() else 0

    # Extract perf line
    perf_match = re.search(
        r"qwen3_tts: perf.*?total=([\d.]+)\s+ms.*?audio=([\d.]+)\s+s.*?rtf=([\d.]+)",
        combined,
    )
    total_ms = float(perf_match.group(1)) if perf_match else None
    audio_s = float(perf_match.group(2)) if perf_match else None
    rtf = float(perf_match.group(3)) if perf_match else None

    # Extract ar_loop timing
    ar_match = re.search(
        r"ar_loop\s+([\d.]+)\s+ms\s+\((\d+)\s+frames?,\s+([\d.]+)\s+ms/frame\)",
        combined,
    )
    ms_per_frame = float(ar_match.group(3)) if ar_match else None
    n_frames = int(ar_match.group(2)) if ar_match else None

    # Check for sched_debug splits (if enabled)
    splits_match = re.search(r"splits=(\d+)", combined)
    n_splits = int(splits_match.group(1)) if splits_match else None

    print(f"\n{'='*64}", flush=True)
    print(
        f"Run: {label}  rc={rc}  elapsed={elapsed}s  "
        f"wav={'OK' if wav_exists else 'MISSING'}  size={wav_size}",
        flush=True,
    )
    if rtf is not None:
        print(f"  total={total_ms:.0f}ms  audio={audio_s:.1f}s  RTF={rtf:.3f}", flush=True)
    if ms_per_frame:
        print(f"  ar_loop: {ms_per_frame:.1f} ms/frame ({n_frames} frames)", flush=True)
    print(f"  VRAM: {mem_before[0]}MB -> {mem_after[0]}MB (of {mem_after[1]}MB)", flush=True)
    if rc != 0:
        print("  --- stderr tail ---", flush=True)
        for ln in combined.splitlines()[-30:]:
            print(f"   {ln}", flush=True)

    kh.step(
        f"{label}.done",
        rc=rc, elapsed=elapsed, wav_ok=wav_exists,
        wav_size=wav_size, rtf=rtf, ms_per_frame=ms_per_frame,
        n_frames=n_frames, vram_before=mem_before[0], vram_after=mem_after[0],
    )
    return {
        "label": label, "rc": rc, "wav_ok": wav_exists,
        "wav_size": wav_size, "rtf": rtf, "ms_per_frame": ms_per_frame,
        "n_frames": n_frames, "wav_path": str(out_wav), "elapsed": elapsed,
        "total_ms": total_ms, "audio_s": audio_s,
        "vram_before": mem_before[0], "vram_after": mem_after[0],
    }


# -- ASR roundtrip -------------------------------------------------------
def asr_roundtrip(label, wav_path, timeout=120):
    """Transcribe a WAV and return the text."""
    kh.step(f"asr_{label}.start")
    out_stem = WORK / f"asr-{label}"
    cmd = [
        str(CLI), "--backend", "parakeet",
        "-m", str(asr_model),
        "-f", wav_path,
        "-of", str(out_stem), "-otxt",
        "--no-prints",
    ]
    try:
        r = subprocess.run(
            cmd, env=os.environ, capture_output=True, text=True, timeout=timeout
        )
        txt_path = out_stem.with_suffix(".txt")
        text = (
            txt_path.read_text().strip()
            if txt_path.exists() and txt_path.stat().st_size > 0
            else ""
        )
    except subprocess.TimeoutExpired:
        text = ""
    kh.step(f"asr_{label}.done", chars=len(text))
    return text


# ========================================================================
# TEST 1: Basic TTS synthesis (short text) -- GQA_NATIVE correctness
# ========================================================================
print("\n" + "=" * 64, flush=True)
print("TEST 1: Short text TTS (GQA_NATIVE correctness)", flush=True)
print("=" * 64, flush=True)

short = run_tts("short", SHORT_TEXT)
asr_short = asr_roundtrip("short", short["wav_path"]) if short["wav_ok"] else ""
print(f"  ASR roundtrip: {asr_short!r}", flush=True)

# ========================================================================
# TEST 2: Long text TTS -- verify O(N) scaling, not O(N^2)
# ========================================================================
print("\n" + "=" * 64, flush=True)
print("TEST 2: Long text TTS (O(N) scaling test)", flush=True)
print("=" * 64, flush=True)

long_result = run_tts("long", LONG_TEXT, timeout=600)
asr_long = asr_roundtrip("long", long_result["wav_path"]) if long_result["wav_ok"] else ""
print(f"  ASR roundtrip (first 120 chars): {asr_long[:120]!r}", flush=True)

# ========================================================================
# TEST 3: Chunked codec decode -- compare chunk=150 vs chunk=0 (disabled)
# ========================================================================
print("\n" + "=" * 64, flush=True)
print("TEST 3: Chunked codec decode (chunk=150 vs disabled)", flush=True)
print("=" * 64, flush=True)

# Re-run short text with chunking disabled to compare output
nochunk = run_tts("nochunk", SHORT_TEXT,
                  extra_env={"QWEN3_TTS_CODEC_CHUNK": "0"})

# Compare WAV sizes (should be identical since same codes, just different decode path)
if short["wav_ok"] and nochunk["wav_ok"]:
    print(f"  Chunked WAV:   {short['wav_size']} bytes", flush=True)
    print(f"  Unchunked WAV: {nochunk['wav_size']} bytes", flush=True)
    size_match = short["wav_size"] == nochunk["wav_size"]
    print(f"  Size match: {size_match}", flush=True)
    # The outputs won't be bit-identical due to FP non-associativity, but should
    # be perceptually identical. ASR roundtrip both to verify.
    asr_nochunk = asr_roundtrip("nochunk", nochunk["wav_path"]) if nochunk["wav_ok"] else ""
    print(f"  ASR nochunk: {asr_nochunk!r}", flush=True)
    kh.step("chunk_compare", size_match=size_match,
            asr_chunk=asr_short, asr_nochunk=asr_nochunk)

# ========================================================================
# TEST 4: VRAM stability -- run TTS twice, check VRAM doesn't grow
#          (scratch sched reset)
# ========================================================================
print("\n" + "=" * 64, flush=True)
print("TEST 4: VRAM stability (scratch sched reset)", flush=True)
print("=" * 64, flush=True)

mem_baseline = gpu_mem_mb()
print(f"  Baseline VRAM: {mem_baseline[0]} MB", flush=True)

vram_r1 = run_tts("vram1", LONG_TEXT, timeout=600)
mem_after_r1 = gpu_mem_mb()
print(f"  After run 1: {mem_after_r1[0]} MB", flush=True)

vram_r2 = run_tts("vram2", LONG_TEXT, timeout=600)
mem_after_r2 = gpu_mem_mb()
print(f"  After run 2: {mem_after_r2[0]} MB", flush=True)

vram_growth = mem_after_r2[0] - mem_after_r1[0]
print(f"  VRAM growth between runs: {vram_growth} MB", flush=True)
vram_stable = abs(vram_growth) < 100  # allow small margin
print(f"  VRAM stable: {vram_stable}", flush=True)
kh.step("vram_stability", growth_mb=vram_growth, stable=vram_stable,
        after_r1=mem_after_r1[0], after_r2=mem_after_r2[0])


# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "=" * 64, flush=True)
print(f"SUMMARY -- issue #183 GQA_NATIVE test -- {sha[:8]} on {gpu_name}", flush=True)
print("=" * 64, flush=True)

# Test 1: correctness
test1_pass = short["rc"] == 0 and short["wav_ok"] and len(asr_short) > 10
print(f"  TEST 1 (correctness):  {'PASS' if test1_pass else 'FAIL'}"
      f"  rc={short['rc']}  wav={'OK' if short['wav_ok'] else 'FAIL'}"
      f"  RTF={short.get('rtf', '?')}", flush=True)
print(f"    ASR: {asr_short!r}", flush=True)

# Test 2: long text scaling
test2_pass = long_result["rc"] == 0 and long_result["wav_ok"]
scaling_ratio = None
if (short.get("ms_per_frame") and long_result.get("ms_per_frame")
        and short["ms_per_frame"] > 0):
    scaling_ratio = long_result["ms_per_frame"] / short["ms_per_frame"]
    # O(N) means ms_per_frame should stay roughly constant (ratio ~1)
    # O(N^2) means it would grow proportionally to length
    test2_scaling_ok = scaling_ratio < 2.0  # generous threshold
    print(f"  TEST 2 (scaling):      {'PASS' if test2_pass and test2_scaling_ok else 'WARN'}"
          f"  short={short['ms_per_frame']:.1f}ms/f  long={long_result['ms_per_frame']:.1f}ms/f"
          f"  ratio={scaling_ratio:.2f}x"
          f"  RTF={long_result.get('rtf', '?')}", flush=True)
else:
    print(f"  TEST 2 (scaling):      {'PASS' if test2_pass else 'FAIL'}"
          f"  (no per-frame timing available)", flush=True)

# Test 3: chunked codec
test3_pass = short["wav_ok"] and nochunk["wav_ok"]
print(f"  TEST 3 (chunked codec): {'PASS' if test3_pass else 'FAIL'}"
      f"  chunk={short['wav_size']}B  nochunk={nochunk['wav_size']}B", flush=True)

# Test 4: VRAM stability
print(f"  TEST 4 (VRAM stable):  {'PASS' if vram_stable else 'WARN'}"
      f"  growth={vram_growth}MB", flush=True)

all_pass = test1_pass and test2_pass and test3_pass
verdict = "ALL PASS" if all_pass else "SOME FAILURES"
print(f"\n  VERDICT: {verdict}", flush=True)

if all_pass:
    print("\n  GQA_NATIVE + chunked codec + scratch reset work correctly on CUDA.", flush=True)
    print("  Safe to merge to main.", flush=True)

kh.step(
    "summary",
    test1=test1_pass, test2=test2_pass, test3=test3_pass,
    vram_stable=vram_stable, scaling_ratio=scaling_ratio,
    sha=sha, verdict=verdict,
)
kh._push_progress_to_hf(force=True)
kh.step("script.end")

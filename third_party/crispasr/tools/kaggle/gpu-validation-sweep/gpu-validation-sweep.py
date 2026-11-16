"""
CrispASR — GPU validation sweep (2026-06-13)

Single kernel covering multiple open PLAN items that need CUDA validation:

1. #114 chunk-context audit: qwen3-asr / granite / omniasr-llm long-form
   chunking quality on GPU (30s vs 60s clips).
2. #72 gemma4/mimo GPU-residency: A/B timing CPU vs GPU weight residency.
3. #58 MOSS-Audio GPU: basic dGPU smoke (encoder + LLM decode).
4. #125 P0 mimo-asr CUDA: retest sched src-mutation fix (a5a518c8).

Runs on $KAGGLE_ACCOUNT account. Attaches crispasr-ccache for fast builds.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass


def run(cmd, check=True, env=None, timeout=None, capture=False):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    kw = dict(env=e, timeout=timeout)
    if capture:
        kw["stdout"] = subprocess.PIPE
        kw["stderr"] = subprocess.PIPE
        kw["text"] = True
    r = subprocess.run(cmd, **kw)
    if check and r.returncode != 0:
        if capture:
            print(r.stdout[-2000:] if r.stdout else "", flush=True)
            print(r.stderr[-2000:] if r.stderr else "", flush=True)
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


def download_model(repo, filename, token=None):
    """Download a model file from HF, return local path."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, cache_dir=str(MODELS), token=token))


def run_crispasr(args, env_extra=None, timeout=300):
    """Run crispasr CLI with -np, return (rc, transcript, elapsed_s).

    Always passes -np (no-prints) to suppress log output to stderr.
    Transcript is read from stdout only — clean, no log-line pollution.
    Stderr is printed for diagnostics but not mixed into the transcript.
    """
    e = os.environ.copy()
    if env_extra:
        e.update(env_extra)
    # Ensure -np is present so stdout has only the transcript
    if "-np" not in args and "--no-prints" not in args:
        args = list(args) + ["-np"]
    cmd = [str(CLI)] + args
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=e, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
        rc = r.returncode
        transcript = r.stdout.decode("utf-8", errors="replace").strip()
        stderr_text = r.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as ex:
        rc = -99
        transcript = (ex.stdout or b"").decode("utf-8", errors="replace").strip()
        stderr_text = (ex.stderr or b"").decode("utf-8", errors="replace")
    elapsed = round(time.time() - t0, 1)
    # Print last few stderr lines for diagnostics
    for line in stderr_text.strip().splitlines()[-3:]:
        print(f"  [stderr] {line}", flush=True)
    return rc, transcript, elapsed


# ── Early diagnostics (before anything can fail) ──────────────────
print("[start] GPU validation sweep", flush=True)
print(f"  disk: {shutil.disk_usage('/kaggle/working')}", flush=True)
print(f"  cwd: {os.getcwd()}", flush=True)
# Write a breadcrumb immediately so we know the script started
Path("/kaggle/working/started.txt").write_text("started\n")

# ── Clone + CUDA build ─────────────────────────────────────────────
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--recursive",
     "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
# Bundled fallback if clone path doesn't have it
if os.path.join(str(REPO), "tools", "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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
kh.step("cloned", sha=sha)

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
    ["cmake", "-S", str(REPO), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
     "-DCRISPASR_BUILD_TESTS=OFF"]
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
    cands = [c for c in BUILD.rglob("crispasr")
             if c.is_file() and os.access(c, os.X_OK)]
    assert cands, "crispasr binary not found"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = (
    f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)
kh.step("build_done", cli=str(CLI))

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
    )
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")

# Keep a running summary dict
summary = {}

# Download JFK audio (11s EN reference) — used across all tests
jfk_wav = REPO / "samples" / "jfk.wav"
assert jfk_wav.exists(), "samples/jfk.wav missing"

# ── Fetch a long-form audio clip for chunk-context tests ──────────
# Use the built-in hp0.wav (25s) if available, otherwise synthesize
# a 60s clip by concatenating jfk.wav 5x
long_wav = WORK / "long_60s.wav"
if not long_wav.exists():
    try:
        import wave
        with wave.open(str(jfk_wav), "rb") as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())
        with wave.open(str(long_wav), "wb") as out:
            out.setparams(params)
            # ~55s = 5 repeats of 11s
            for _ in range(5):
                out.writeframes(frames)
        print(f"[info] created {long_wav} ({long_wav.stat().st_size} bytes)", flush=True)
    except Exception as ex:
        print(f"[warn] failed to create long wav: {ex}", flush=True)

# =====================================================================
# TEST 1: #125 P0 — mimo-asr CUDA smoke (sched src-mutation fix)
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST 1: #125 P0 — mimo-asr CUDA smoke", flush=True)
print("=" * 70, flush=True)
kh.step("test1.start")

try:
    mimo_model = download_model("cstr/mimo-asr-GGUF", "mimo-asr-q4_k.gguf", token)
    mimo_tok = download_model("cstr/mimo-tokenizer-GGUF", "mimo-tokenizer-q4_k.gguf", token)
    kh.step("test1.model_downloaded")

    # Basic JFK transcribe — should not crash
    rc, out, elapsed = run_crispasr([
        "-m", str(mimo_model), "--codec-model", str(mimo_tok),
        "-f", str(jfk_wav), "-v",
    ], timeout=300)

    transcript = out

    mimo_ok = rc == 0 and len(transcript) > 10
    summary["mimo_cuda_smoke"] = {
        "rc": rc, "elapsed": elapsed, "ok": mimo_ok,
        "transcript_len": len(transcript),
        "transcript_preview": transcript[:120],
    }
    print(f"\n  mimo-asr CUDA: rc={rc} elapsed={elapsed}s ok={mimo_ok}", flush=True)
    print(f"  transcript: {transcript[:120]}", flush=True)
    kh.step("test1.done", **summary["mimo_cuda_smoke"])

    # Also test with long audio to check for the segfault
    if long_wav.exists():
        kh.step("test1.long.start")
        rc2, out2, elapsed2 = run_crispasr([
            "-m", str(mimo_model), "--codec-model", str(mimo_tok),
            "-f", str(long_wav), "-v",
        ], timeout=600)
        t2 = out2
        summary["mimo_cuda_long"] = {"rc": rc2, "elapsed": elapsed2, "ok": rc2 == 0, "chars": len(t2)}
        print(f"  mimo-asr CUDA long: rc={rc2} elapsed={elapsed2}s chars={len(t2)}", flush=True)
        kh.step("test1.long.done", **summary["mimo_cuda_long"])

except Exception as ex:
    print(f"  ERROR: {ex}", flush=True)
    summary["mimo_cuda_smoke"] = {"error": str(ex)}
    kh.step("test1.error", error=str(ex))

# =====================================================================
# TEST 2: #72 — gemma4-e2b GPU-residency A/B
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST 2: #72 — gemma4-e2b GPU-residency A/B", flush=True)
print("=" * 70, flush=True)
kh.step("test2.start")

try:
    gemma_model = download_model("cstr/gemma4-e2b-it-GGUF", "gemma4-e2b-it-q4_k.gguf", token)
    kh.step("test2.model_downloaded")

    # A: default (GPU residency)
    rc_a, out_a, elapsed_a = run_crispasr([
        "-m", str(gemma_model), "-f", str(jfk_wav), "-v",
    ], timeout=300)
    ta = out_a

    # B: force CPU residency
    rc_b, out_b, elapsed_b = run_crispasr([
        "-m", str(gemma_model), "-f", str(jfk_wav), "-v",
    ], env_extra={"CRISPASR_N_GPU_LAYERS": "0"}, timeout=300)
    tb = out_b

    speedup = round(elapsed_b / elapsed_a, 2) if elapsed_a > 0 else 0
    summary["gemma4_gpu_residency"] = {
        "gpu_elapsed": elapsed_a, "cpu_elapsed": elapsed_b,
        "speedup": speedup,
        "gpu_rc": rc_a, "cpu_rc": rc_b,
        "gpu_transcript": ta[:100], "cpu_transcript": tb[:100],
    }
    print(f"\n  gemma4-e2b GPU: {elapsed_a}s  CPU: {elapsed_b}s  speedup: {speedup}x", flush=True)
    print(f"  GPU transcript: {ta[:100]}", flush=True)
    print(f"  CPU transcript: {tb[:100]}", flush=True)
    kh.step("test2.done", **summary["gemma4_gpu_residency"])

except Exception as ex:
    print(f"  ERROR: {ex}", flush=True)
    summary["gemma4_gpu_residency"] = {"error": str(ex)}
    kh.step("test2.error", error=str(ex))

# =====================================================================
# TEST 3: #58 — MOSS-Audio GPU smoke
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST 3: #58 — MOSS-Audio GPU smoke", flush=True)
print("=" * 70, flush=True)
kh.step("test3.start")

try:
    moss_model = download_model(
        "cstr/MOSS-Audio-4B-Instruct-GGUF",
        "moss-audio-4b-instruct-q4_k.gguf", token
    )
    kh.step("test3.model_downloaded")

    rc, out, elapsed = run_crispasr([
        "-m", str(moss_model), "-f", str(jfk_wav), "-v",
    ], timeout=300)
    transcript = out

    # Also test with --prompt for audio understanding mode
    rc2, out2, elapsed2 = run_crispasr([
        "-m", str(moss_model), "-f", str(jfk_wav),
        "--prompt", "What is the speaker talking about?",
        "-v",
    ], timeout=300)
    answer = out2

    summary["moss_audio_gpu"] = {
        "asr_rc": rc, "asr_elapsed": elapsed,
        "asr_transcript": transcript[:120],
        "qa_rc": rc2, "qa_elapsed": elapsed2,
        "qa_answer": answer[:200],
    }
    print(f"\n  MOSS ASR: rc={rc} elapsed={elapsed}s", flush=True)
    print(f"  transcript: {transcript[:120]}", flush=True)
    print(f"  MOSS QA: rc={rc2} elapsed={elapsed2}s", flush=True)
    print(f"  answer: {answer[:200]}", flush=True)
    kh.step("test3.done", **summary["moss_audio_gpu"])

except Exception as ex:
    print(f"  ERROR: {ex}", flush=True)
    summary["moss_audio_gpu"] = {"error": str(ex)}
    kh.step("test3.error", error=str(ex))

# =====================================================================
# TEST 4: #114 — chunk-context audit (qwen3-asr, granite, omniasr-llm)
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST 4: #114 — chunk-context audit", flush=True)
print("=" * 70, flush=True)
kh.step("test4.start")

chunk_results = {}

# Models to test for chunk-context
chunk_backends = [
    ("qwen3-asr", "cstr/qwen3-asr-0.6b-GGUF", "qwen3-asr-0.6b-q4_k.gguf", None),
    ("granite-speech", "cstr/granite-speech-4.0-1b-GGUF", "granite-speech-4.0-1b-q4_k.gguf", None),
    ("omniasr-llm", "cstr/omniasr-llm-300m-v2-GGUF", "omniasr-llm-300m-v2-q4_k.gguf", None),
]

for backend_name, repo, fname, extra in chunk_backends:
    print(f"\n--- {backend_name} ---", flush=True)
    kh.step(f"test4.{backend_name}.start")
    try:
        model_path = download_model(repo, fname, token)

        # A: short audio (JFK 11s) — baseline, should always work
        rc_short, out_short, el_short = run_crispasr([
            "-m", str(model_path), "-f", str(jfk_wav), "-v",
        ], timeout=300)
        t_short = out_short

        # B: long audio (60s concat) — tests chunking
        if long_wav.exists():
            rc_long, out_long, el_long = run_crispasr([
                "-m", str(model_path), "-f", str(long_wav), "-v",
            ], timeout=600)
        else:
            rc_long, out_long, el_long = -1, "", 0
        t_long = out_long

        # C: long audio with explicit chunk-seconds
        if long_wav.exists():
            rc_chunk, out_chunk, el_chunk = run_crispasr([
                "-m", str(model_path), "-f", str(long_wav),
                "--chunk-seconds", "30", "-v",
            ], timeout=600)
        else:
            rc_chunk, out_chunk, el_chunk = -1, "", 0
        t_chunk = out_chunk

        result = {
            "short_rc": rc_short, "short_chars": len(t_short),
            "short_elapsed": el_short,
            "long_rc": rc_long, "long_chars": len(t_long),
            "long_elapsed": el_long,
            "chunk_rc": rc_chunk, "chunk_chars": len(t_chunk),
            "chunk_elapsed": el_chunk,
            "short_preview": t_short[:80],
            "long_preview": t_long[:80],
            "chunk_preview": t_chunk[:80],
        }
        chunk_results[backend_name] = result
        print(f"  short: rc={rc_short} {len(t_short)} chars {el_short}s", flush=True)
        print(f"  long:  rc={rc_long} {len(t_long)} chars {el_long}s", flush=True)
        print(f"  chunk: rc={rc_chunk} {len(t_chunk)} chars {el_chunk}s", flush=True)
        kh.step(f"test4.{backend_name}.done", **result)

    except Exception as ex:
        print(f"  ERROR: {ex}", flush=True)
        chunk_results[backend_name] = {"error": str(ex)}
        kh.step(f"test4.{backend_name}.error", error=str(ex))

summary["chunk_context_audit"] = chunk_results

# =====================================================================
# TEST 5: #125 — gemma4-e2b long-audio with CRISPASR_GEMMA4_AUTO_CHUNK=1
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST 5: #125 — gemma4-e2b auto-chunk long audio", flush=True)
print("=" * 70, flush=True)
kh.step("test5.start")

try:
    # Reuse gemma_model from test 2 if available
    if "gemma_model" not in dir():
        gemma_model = download_model("cstr/gemma4-e2b-it-GGUF", "gemma4-e2b-it-q4_k.gguf", token)

    if long_wav.exists():
        # A: without auto-chunk (default — may degrade on long audio)
        rc_a, out_a, el_a = run_crispasr([
            "-m", str(gemma_model), "-f", str(long_wav), "-v",
        ], timeout=600)
        ta = out_a

        # B: with CRISPASR_GEMMA4_AUTO_CHUNK=1
        rc_b, out_b, el_b = run_crispasr([
            "-m", str(gemma_model), "-f", str(long_wav), "-v",
        ], env_extra={"CRISPASR_GEMMA4_AUTO_CHUNK": "1"}, timeout=600)
        tb = out_b

        summary["gemma4_auto_chunk"] = {
            "default_rc": rc_a, "default_chars": len(ta), "default_elapsed": el_a,
            "autochunk_rc": rc_b, "autochunk_chars": len(tb), "autochunk_elapsed": el_b,
            "default_preview": ta[:120],
            "autochunk_preview": tb[:120],
        }
        print(f"  default:   rc={rc_a} {len(ta)} chars {el_a}s", flush=True)
        print(f"  autochunk: rc={rc_b} {len(tb)} chars {el_b}s", flush=True)
        kh.step("test5.done", **summary["gemma4_auto_chunk"])
    else:
        print("  SKIP: no long wav", flush=True)
        summary["gemma4_auto_chunk"] = {"skip": "no long wav"}

except Exception as ex:
    print(f"  ERROR: {ex}", flush=True)
    summary["gemma4_auto_chunk"] = {"error": str(ex)}
    kh.step("test5.error", error=str(ex))

# =====================================================================
# TEST 6: #127c — cohere-asr-ja perf (if model available)
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST 6: #127c — cohere-asr-ja smoke", flush=True)
print("=" * 70, flush=True)
kh.step("test6.start")

try:
    cohere_ja_model = download_model("CKHO/cohere-asr-ja-GGUF", "cohere-asr-ja-q4_k.gguf", token)
    kh.step("test6.model_downloaded")

    # JFK EN (should still produce something — model is JA-tuned but EN is in the base)
    rc_en, out_en, el_en = run_crispasr([
        "-m", str(cohere_ja_model), "-f", str(jfk_wav), "-v",
    ], timeout=300)
    t_en = out_en

    summary["cohere_asr_ja"] = {
        "en_rc": rc_en, "en_chars": len(t_en), "en_elapsed": el_en,
        "en_preview": t_en[:120],
    }
    print(f"  cohere-ja EN: rc={rc_en} {len(t_en)} chars {el_en}s", flush=True)
    print(f"  transcript: {t_en[:120]}", flush=True)
    kh.step("test6.done", **summary["cohere_asr_ja"])

except Exception as ex:
    print(f"  ERROR: {ex}", flush=True)
    summary["cohere_asr_ja"] = {"error": str(ex)}
    kh.step("test6.error", error=str(ex))

# =====================================================================
# SUMMARY
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print(f"SUMMARY — GPU validation sweep — {sha[:8]} on {gpu_name}", flush=True)
print("=" * 70, flush=True)

# Test 1: mimo-asr CUDA
t1 = summary.get("mimo_cuda_smoke", {})
print(f"\n#125 mimo-asr CUDA: {'PASS' if t1.get('ok') else 'FAIL'}"
      f"  rc={t1.get('rc')} elapsed={t1.get('elapsed')}s", flush=True)
t1l = summary.get("mimo_cuda_long", {})
if t1l:
    print(f"  long audio: rc={t1l.get('rc')} chars={t1l.get('chars')}", flush=True)

# Test 2: gemma4 GPU residency
t2 = summary.get("gemma4_gpu_residency", {})
print(f"\n#72 gemma4 GPU residency: GPU={t2.get('gpu_elapsed')}s "
      f"CPU={t2.get('cpu_elapsed')}s speedup={t2.get('speedup')}x", flush=True)

# Test 3: MOSS-Audio GPU
t3 = summary.get("moss_audio_gpu", {})
print(f"\n#58 MOSS-Audio GPU: ASR rc={t3.get('asr_rc')} "
      f"elapsed={t3.get('asr_elapsed')}s", flush=True)
print(f"  QA rc={t3.get('qa_rc')} elapsed={t3.get('qa_elapsed')}s", flush=True)

# Test 4: chunk-context
print(f"\n#114 chunk-context audit:", flush=True)
for bname, bdata in chunk_results.items():
    if "error" in bdata:
        print(f"  {bname}: ERROR — {bdata['error']}", flush=True)
    else:
        short_c = bdata.get("short_chars", 0)
        long_c = bdata.get("long_chars", 0)
        chunk_c = bdata.get("chunk_chars", 0)
        ratio = round(long_c / short_c, 2) if short_c > 0 else 0
        print(f"  {bname}: short={short_c} long={long_c} chunk={chunk_c} "
              f"(long/short={ratio}x)", flush=True)

# Test 5: gemma4 auto-chunk
t5 = summary.get("gemma4_auto_chunk", {})
if "error" not in t5 and "skip" not in t5:
    print(f"\n#125 gemma4 auto-chunk: default={t5.get('default_chars')} "
          f"autochunk={t5.get('autochunk_chars')} chars", flush=True)

# Test 6: cohere-asr-ja
t6 = summary.get("cohere_asr_ja", {})
if "error" not in t6:
    print(f"\n#127c cohere-asr-ja: EN rc={t6.get('en_rc')} "
          f"{t6.get('en_chars')} chars", flush=True)

# Save full summary
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nFull results saved to {RESULTS / 'summary.json'}", flush=True)

kh.step("summary", **{k: str(v)[:200] for k, v in summary.items()})
kh._push_progress_to_hf(force=True)
kh.step("script.end")

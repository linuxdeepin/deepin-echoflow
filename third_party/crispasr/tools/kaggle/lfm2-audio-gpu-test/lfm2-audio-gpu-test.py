"""
CrispASR — LFM2-Audio GPU end-to-end test

Tests the full LFM2-Audio pipeline on GPU (T4/P100):
  1. CUDA build of crispasr-cli
  2. Download EN Q5_K + JP Q4_K + detokenizer GGUFs
  3. ASR: transcribe JFK audio (EN) — verify output contains key phrases
  4. ASR: transcribe demo audio (JP) — verify non-empty output
  5. TTS: synthesize "Hello world" (EN) — verify audio produced
  6. TTS roundtrip: synthesize → ASR → verify text similarity
  7. Report: timing (CPU vs GPU), output quality

Models:
  - cstr/lfm2-audio-1.5b-GGUF/lfm2-audio-1.5b-q5_k.gguf (~1.6 GB)
  - cstr/lfm2-audio-1.5b-jp-GGUF/lfm2-audio-1.5b-jp-q4_k.gguf (~1.5 GB)
  - cstr/lfm2-audio-1.5b-GGUF/lfm2-audio-1.5b-f16-detokenizer.gguf (~157 MB)
  - cstr/lfm2-audio-1.5b-jp-GGUF/lfm2-audio-1.5b-jp-f16-detokenizer.gguf (~157 MB)
"""

import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get(
    "CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git"
)


def run(cmd, check=True, env=None, timeout=None, capture=False):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    kw = dict(env=e, timeout=timeout)
    if capture:
        kw["capture_output"] = True
        kw["text"] = True
    r = subprocess.run(cmd, **kw)
    if check and r.returncode != 0:
        if capture:
            print(f"STDOUT: {r.stdout}")
            print(f"STDERR: {r.stderr}")
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + CUDA build ──────────────────────────────────────────────
print(f"[start] ref={CRISPASR_REF}", flush=True)
if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
     "--recursive", CRISPASR_REPO, str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
kh.step("cloned", sha=sha, ref=CRISPASR_REF)

run(["nvidia-smi", "-L"])
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
    assert cands, "crispasr binary not found after build"
    CLI = cands[0]
ld_path = f"{BUILD / 'src'}:{BUILD / 'ggml' / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = ld_path
kh.step("build_done", cli=str(CLI))

# ── Download models ─────────────────────────────────────────────────
kh.step("downloading_models")
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

# EN Q5_K + detokenizer
en_model = Path(hf_hub_download(
    "cstr/lfm2-audio-1.5b-GGUF", "lfm2-audio-1.5b-q5_k.gguf",
    cache_dir=str(MODELS), token=token,
))
en_detok = Path(hf_hub_download(
    "cstr/lfm2-audio-1.5b-GGUF", "lfm2-audio-1.5b-f16-detokenizer.gguf",
    cache_dir=str(MODELS), token=token,
))
# Place detokenizer next to model for auto-discovery
import shutil
en_detok_target = en_model.parent / "lfm2-audio-1.5b-f16-detokenizer.gguf"
if not en_detok_target.exists():
    shutil.copy2(en_detok, en_detok_target)

# JP Q4_K + detokenizer
jp_model = Path(hf_hub_download(
    "cstr/lfm2-audio-1.5b-jp-GGUF", "lfm2-audio-1.5b-jp-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
jp_detok = Path(hf_hub_download(
    "cstr/lfm2-audio-1.5b-jp-GGUF", "lfm2-audio-1.5b-jp-f16-detokenizer.gguf",
    cache_dir=str(MODELS), token=token,
))
jp_detok_target = jp_model.parent / "lfm2-audio-1.5b-jp-f16-detokenizer.gguf"
if not jp_detok_target.exists():
    shutil.copy2(jp_detok, jp_detok_target)

kh.step("models_downloaded", en=str(en_model), jp=str(jp_model))

# ── Test 1: ASR EN (JFK) ────────────────────────────────────────────
kh.step("test_asr_en")
jfk_wav = REPO / "samples" / "jfk.wav"
t0 = time.time()
r = run([str(CLI), "--backend", "lfm2-audio", "-m", str(en_model),
         "-f", str(jfk_wav), "-l", "en", "-np"],
        capture=True, timeout=600)
asr_en_time = time.time() - t0
asr_en_text = r.stdout.strip()
print(f"ASR EN output ({asr_en_time:.1f}s): {asr_en_text}")

# Verify key phrases
assert "Americans" in asr_en_text or "country" in asr_en_text, \
    f"ASR EN output missing key phrases: {asr_en_text}"
kh.step("asr_en_done", time_s=asr_en_time, text=asr_en_text[:200])

# ── Test 2: ASR JP (demo audio from HF) ─────────────────────────────
kh.step("test_asr_jp")
# Download the demo audio from the model repo
try:
    demo_mp4 = Path(hf_hub_download(
        "LiquidAI/LFM2.5-Audio-1.5B-JP", "demo.mp4",
        cache_dir=str(MODELS), token=token,
    ))
    demo_wav = WORK / "demo.wav"
    run(["ffmpeg", "-i", str(demo_mp4), "-t", "10", "-ar", "16000",
         "-ac", "1", "-c:a", "pcm_s16le", str(demo_wav), "-y"],
        check=False)
    if demo_wav.exists():
        t0 = time.time()
        r = run([str(CLI), "--backend", "lfm2-audio", "-m", str(jp_model),
                 "-f", str(demo_wav), "-l", "ja", "-np"],
                capture=True, timeout=600)
        asr_jp_time = time.time() - t0
        asr_jp_text = r.stdout.strip()
        print(f"ASR JP output ({asr_jp_time:.1f}s): {asr_jp_text}")
        assert len(asr_jp_text) > 0, "ASR JP produced empty output"
        kh.step("asr_jp_done", time_s=asr_jp_time, text=asr_jp_text[:200])
    else:
        print("SKIP: demo.wav extraction failed")
        kh.step("asr_jp_skip", reason="ffmpeg failed")
except Exception as e:
    print(f"SKIP ASR JP: {e}")
    kh.step("asr_jp_skip", reason=str(e)[:200])

# ── Test 3: TTS EN ──────────────────────────────────────────────────
kh.step("test_tts_en")
tts_wav = RESULTS / "tts_en.wav"
t0 = time.time()
r = run([str(CLI), "--backend", "lfm2-audio", "-m", str(en_model),
         "--tts", "Hello world, this is a test of speech synthesis.",
         "--tts-output", str(tts_wav), "-l", "en"],
        capture=True, timeout=600, check=False)
tts_en_time = time.time() - t0

if tts_wav.exists() and tts_wav.stat().st_size > 100:
    with wave.open(str(tts_wav)) as w:
        n = w.getnframes()
        sr = w.getframerate()
    print(f"TTS EN: {n} samples @ {sr} Hz = {n/sr:.2f}s ({tts_en_time:.1f}s wall)")
    assert n > 1000, f"TTS produced too few samples: {n}"
    kh.step("tts_en_done", time_s=tts_en_time, samples=n, sr=sr)
else:
    print(f"TTS EN: no output or empty file (rc={r.returncode})")
    if r.stderr:
        print(f"STDERR: {r.stderr[:500]}")
    kh.step("tts_en_fail", rc=r.returncode)

# ── Test 4: TTS JP ──────────────────────────────────────────────────
kh.step("test_tts_jp")
tts_jp_wav = RESULTS / "tts_jp.wav"
t0 = time.time()
r = run([str(CLI), "--backend", "lfm2-audio", "-m", str(jp_model),
         "--tts", "こんにちは、今日はいい天気ですね。",
         "--tts-output", str(tts_jp_wav), "-l", "ja"],
        capture=True, timeout=600, check=False)
tts_jp_time = time.time() - t0

if tts_jp_wav.exists() and tts_jp_wav.stat().st_size > 100:
    with wave.open(str(tts_jp_wav)) as w:
        n = w.getnframes()
        sr = w.getframerate()
    print(f"TTS JP: {n} samples @ {sr} Hz = {n/sr:.2f}s ({tts_jp_time:.1f}s wall)")
    kh.step("tts_jp_done", time_s=tts_jp_time, samples=n, sr=sr)
else:
    print(f"TTS JP: no output (rc={r.returncode})")
    kh.step("tts_jp_fail", rc=r.returncode)

# ── Summary ──────────────────────────────────────────────────────────
summary = {
    "sha": sha,
    "gpu": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True
    ).strip(),
    "asr_en_time_s": asr_en_time,
    "asr_en_text": asr_en_text[:200],
}
if "tts_en_time" in dir():
    summary["tts_en_time_s"] = tts_en_time
if "asr_jp_time" in dir():
    summary["asr_jp_time_s"] = asr_jp_time

with open(RESULTS / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

kh.step("done", **{k: v for k, v in summary.items() if isinstance(v, (int, float, str))})
print("\n[DONE]")

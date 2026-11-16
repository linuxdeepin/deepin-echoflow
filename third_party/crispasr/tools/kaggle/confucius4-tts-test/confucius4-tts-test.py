#!/usr/bin/env python3
"""Kaggle kernel: Confucius4-TTS end-to-end T2S decode test.

Builds CrispASR from main, downloads the Q4_K T2S GGUF, tokenizes a test
string with the HF tokenizer, and runs the T2S decode loop to generate
semantic codes. Validates the full pipeline: model load → text projector
MLP → GPT-2 prefill → autoregressive decode → EOS/max.

Push (under $KAGGLE_ACCOUNT):
  # Kaggle CLI auth: ~/.kaggle/access_token (never commit a token; see tests/test_no_secrets.py)
  python -m kaggle kernels push -p tools/kaggle/confucius4-tts-test
"""

import os
import sys
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")

# ── Phase 0: Clone repo ─────────────────────────────────────────────────────
print("=== Phase 0: clone repo ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "main",
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])

# Init ALL submodules
subprocess.check_call(
    ["git", "submodule", "update", "--init", "--recursive"],
    cwd=str(REPO),
)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

# ── Phase 1: Install deps ───────────────────────────────────────────────────
kh.step("install deps")
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer tokenizers")

# ── Phase 2: Resolve HF token ───────────────────────────────────────────────
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# ── Phase 3: Download T2S Q4_K GGUF ─────────────────────────────────────────
kh.step("download T2S GGUF")
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    "cstr/confucius4-tts-GGUF",
    "confucius4-tts-t2s-q4_k.gguf",
    local_dir=str(TEMP / "models"),
    token=hf_token,
)
print(f"  T2S model: {model_path} ({os.path.getsize(model_path) / 1024**2:.0f} MB)")

# Also download S2A companion model
s2a_path = hf_hub_download(
    "cstr/confucius4-tts-GGUF",
    "confucius4-tts-s2a-q4_k.gguf",
    local_dir=str(TEMP / "models"),
    token=hf_token,
)
print(f"  S2A model: {s2a_path} ({os.path.getsize(s2a_path) / 1024**2:.0f} MB)")

# Download BigVGAN vocoder (auto-discovered by CLI as sibling of S2A)
try:
    voc_path = hf_hub_download(
        "cstr/confucius4-tts-GGUF",
        "confucius4-tts-bigvgan-22k-f16.gguf",
        local_dir=str(TEMP / "models"),
        token=hf_token,
    )
    print(f"  BigVGAN vocoder: {voc_path} ({os.path.getsize(voc_path) / 1024**2:.0f} MB)")
except Exception as e:
    print(f"  BigVGAN vocoder: not available yet ({e})")

# ── Phase 4: Download tokenizer ──────────────────────────────────────────────
kh.step("download tokenizer")
tok_path = hf_hub_download(
    "netease-youdao/Confucius4-TTS",
    "tokenizer.json",
    local_dir=str(TEMP / "tokenizer"),
    token=hf_token,
)
print(f"  tokenizer: {tok_path}")

# ── Phase 5: Tokenize test string ────────────────────────────────────────────
kh.step("tokenize")
from tokenizers import Tokenizer

tok = Tokenizer.from_file(tok_path)
# Match the Python inference format exactly:
# formatted = "You are a helpful assistant. {lang_token}:{text}"
# lang_token for English = "Please read the following English text"
test_text = "Hello world, this is a test of the Confucius four text to speech system."
formatted = f"You are a helpful assistant. Please read the following English text:{test_text}"
enc = tok.encode(formatted)
token_ids_str = ",".join(str(x) for x in enc.ids)
print(f"  text: {test_text}")
print(f"  formatted: {formatted[:80]}...")
print(f"  token IDs ({len(enc.ids)}): {token_ids_str[:100]}...")
print(f"  tokens: {enc.tokens[:15]}...")

# ── Phase 6: Build CrispASR ─────────────────────────────────────────────────
kh.step("build CrispASR")
BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} "
    f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF " + " ".join(flags),
)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=False)} --target crispasr-cli"
    )
crispasr_bin = BUILD / "bin" / "crispasr"
print(f"  binary: {crispasr_bin} ({os.path.getsize(str(crispasr_bin)) / 1024**2:.0f} MB)")

# ── Phase 7: Verify model loads ──────────────────────────────────────────────
kh.step("verify model load")
result = subprocess.run(
    [str(crispasr_bin), "--backend", "confucius4-tts",
     "-m", model_path, "--list-backends"],
    capture_output=True, text=True, timeout=30,
)
print(f"  rc={result.returncode}")
for line in result.stderr.split("\n"):
    if "confucius4" in line.lower() or "T2S" in line:
        print(f"  {line}")

# ── Phase 8: Run T2S decode ──────────────────────────────────────────────────
kh.step("T2S decode")
env = os.environ.copy()
env["CRISPASR_CONFUCIUS4_TEXT_IDS"] = token_ids_str
# gallocr is now the default for GPT-2 step (sched has index corruption)

tts_wav = TEMP / "confucius4_output.wav"
# Use 10 ODE steps instead of 25 for faster Kaggle test; no WaveNet yet
result = subprocess.run(
    [str(crispasr_bin), "--backend", "confucius4-tts",
     "-m", model_path, "--codec-model", s2a_path,
     "--tts", test_text, "--tts-output", str(tts_wav),
     "--tts-steps", "10", "-v"],
    capture_output=True, text=True, timeout=600, env=env,
)
print(f"  TTS rc={result.returncode}")
for line in result.stderr.split("\n"):
    if "confucius4:" in line or "output:" in line or "DiT" in line or "ODE" in line:
        print(f"  {line.strip()}")
# Also print last 20 lines of stderr for debugging
stderr_lines = [l for l in result.stderr.split("\n") if l.strip()]
if len(stderr_lines) > 20:
    print("  --- last 20 stderr lines ---")
for line in stderr_lines[-20:]:
    print(f"  {line.strip()}")

if tts_wav.exists():
    print(f"  WAV: {tts_wav} ({os.path.getsize(str(tts_wav))} bytes)")
else:
    print("  WAV: not produced")

# ── Phase 9: ASR roundtrip ───────────────────────────────────────────────────
kh.step("ASR roundtrip")
if tts_wav.exists() and os.path.getsize(str(tts_wav)) > 100:
    # Download a small whisper model for ASR
    import urllib.request
    whisper_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
    whisper_model = str(TEMP / "models" / "ggml-tiny.en.bin")
    os.makedirs(os.path.dirname(whisper_model), exist_ok=True)
    if not os.path.exists(whisper_model):
        urllib.request.urlretrieve(whisper_url, whisper_model)
    print(f"  ASR model: {whisper_model}")

    asr_result = subprocess.run(
        [str(crispasr_bin), "-m", whisper_model, "-f", str(tts_wav), "--no-prints"],
        capture_output=True, text=True, timeout=60,
    )
    print(f"  ASR rc={asr_result.returncode}")
    transcript = asr_result.stdout.strip()
    print(f"  ASR transcript: '{transcript}'")

    if transcript:
        # Check word overlap with original text
        orig_words = set(test_text.lower().split())
        asr_words = set(transcript.lower().split())
        overlap = orig_words & asr_words
        print(f"  Word overlap: {len(overlap)}/{len(orig_words)} "
              f"({100*len(overlap)/max(len(orig_words),1):.0f}%)")
    else:
        print("  ASR: empty transcript (expected — audio is silence/stub)")
else:
    print("  Skipping ASR — no WAV file produced")

# ── Phase 10: Summary ────────────────────────────────────────────────────────
kh.step("summary")
import re
if "prefill done" in result.stderr:
    print("  PREFILL: OK")
if "generated" in result.stderr and "semantic codes" in result.stderr:
    m = re.search(r"generated (\d+) semantic codes", result.stderr)
    if m:
        print(f"  CODES: {m.group(1)}")
if "output:" in result.stderr:
    m = re.search(r"output: (\d+) mel frames.*\(([0-9.]+)s", result.stderr)
    if m:
        print(f"  MEL: {m.group(1)} frames, {m.group(2)}s")
if tts_wav.exists():
    print(f"  WAV: {os.path.getsize(str(tts_wav))} bytes")
print(f"  TTS rc={result.returncode}")

print("\n=== Done ===", flush=True)

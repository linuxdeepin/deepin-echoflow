"""lahgtna-chatterbox A/B: Python reference vs CrispASR GGUF (#170).

Compare speech token sequences and audio from the original Python
ChatterboxMultilingualTTS vs our re-converted GGUF on Arabic text.
"""
import json
import os
import struct
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
_T0 = time.time()

def step(msg):
    print(f"\n{'='*60}\n[{time.time()-_T0:.0f}s] {msg}\n{'='*60}", flush=True)

RESULTS = {}

# ── Test sentences ──
TEST_SENTENCES = [
    ("ar", "مرحبا بالعالم"),
    ("ar", "كيف حالك اليوم؟ أنا بخير شكرا لك."),
    ("ar", "هذا اختبار للنموذج العربي."),
]

# ═══════════════════════════════════════════════════════════════════
# Part 1: Python reference (original ChatterboxMultilingualTTS)
# ═══════════════════════════════════════════════════════════════════

step("Install chatterbox-tts Python package")
# Kaggle's torchvision conflicts with chatterbox's torch pin; remove it
# before installing (chatterbox doesn't need torchvision).
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"],
               capture_output=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "chatterbox-tts", "soundfile"])

step("Load Python model (oddadmix/lahgtna-chatterbox-v1)")
import torch
import soundfile as sf
from huggingface_hub import snapshot_download
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Download lahgtna model files
lahgtna_dir = Path(snapshot_download(
    "oddadmix/lahgtna-chatterbox-v1",
    allow_patterns=["*.json", "*.safetensors", "*.pt", "ve.*"],
    token=os.environ.get("HF_TOKEN"),
))
print(f"Model dir: {lahgtna_dir}")
for f in sorted(os.listdir(lahgtna_dir)):
    if not f.startswith('.'):
        print(f"  {f} ({os.path.getsize(lahgtna_dir/f)/1e6:.1f} MB)")

# Load with ChatterboxMultilingualTTS.from_local
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model = ChatterboxMultilingualTTS.from_local(lahgtna_dir, device)
print("Model loaded")

step("Generate Python reference audio")
py_results = []
for lang, text in TEST_SENTENCES:
    print(f"\n--- Python: [{lang}] {text} ---", flush=True)
    t0 = time.time()
    wav = model.generate(text, language_id=lang)
    elapsed = time.time() - t0

    # Save WAV
    fname = f"py_{lang}_{len(py_results)}.wav"
    out_path = WORK / fname
    sf.write(str(out_path), wav.squeeze().cpu().numpy(), 24000)
    dur = len(wav.squeeze()) / 24000
    print(f"  Generated: {dur:.2f}s in {elapsed:.1f}s", flush=True)
    py_results.append({
        "lang": lang, "text": text, "wav": fname,
        "duration": dur, "time": elapsed,
    })

# Also capture speech tokens from the T3 stage
step("Capture Python T3 speech tokens")
py_tokens = []
for i, (lang, text) in enumerate(TEST_SENTENCES):
    print(f"\n--- T3 tokens: [{lang}] {text} ---", flush=True)
    try:
        # Access T3 internals to get text token IDs
        tok_text = model.tokenizer.encode(text, language_id=lang)
        print(f"  Text token IDs ({len(tok_text)}): {tok_text[:20]}...")
        py_tokens.append({"lang": lang, "text": text, "text_tokens": tok_text[:50]})
    except Exception as e:
        print(f"  Token capture failed: {e}")
        py_tokens.append({"lang": lang, "text": text, "error": str(e)})

# Free GPU memory before CrispASR build
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
import gc; gc.collect()

# ═══════════════════════════════════════════════════════════════════
# Part 2: CrispASR GGUF
# ═══════════════════════════════════════════════════════════════════

step("Clone + build CrispASR")
CRISPASR_DIR = WORK / "CrispASR"
if not CRISPASR_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/CrispStrobe/CrispASR.git",
                           str(CRISPASR_DIR)])

sys.path.insert(0, str(CRISPASR_DIR / "tools" / "kaggle"))
try:
    import kaggle_harness as kh
    # Full harness regime: authenticate HF pulls from the attached token
    # dataset (anon pulls get rate-limited); hf_transfer wedges multi-GB
    # Kaggle downloads, so keep the plain resumable downloader.
    kh.resolve_hf_token()
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    kh.init_progress()
    kh.install_build_toolchain()
except Exception as e:
    print(f"Harness: {e}")
    subprocess.run("apt-get update -qq && apt-get install -y --no-install-recommends cmake ninja-build g++ ccache",
                   shell=True, capture_output=True)

build_dir = CRISPASR_DIR / "build"
os.makedirs(build_dir, exist_ok=True)

subprocess.check_call([
    "cmake", "-G", "Ninja", "-B", str(build_dir),
    "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON",
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
], cwd=str(CRISPASR_DIR),
   env={**os.environ, "CCACHE_DIR": str(WORK / ".ccache")})

try:
    with kh.build_heartbeat("cuda-build"):
        subprocess.check_call(
            ["cmake", "--build", str(build_dir), "-j4"],
            env={**os.environ, "CCACHE_DIR": str(WORK / ".ccache")})
except:
    subprocess.check_call(
        ["cmake", "--build", str(build_dir), "-j4"],
        env={**os.environ, "CCACHE_DIR": str(WORK / ".ccache")})

CLI = str(build_dir / "bin" / "crispasr")

step("Download GGUF models")
# T3 (lahgtna, re-converted with mtl tokenizer)
subprocess.run([CLI, "--backend", "lahgtna-chatterbox", "-m", "auto",
                "--codec-model", "auto", "--auto-download",
                "--tts", "test", "--tts-output", "/dev/null"],
               capture_output=True, timeout=300)
print("Models downloaded/cached", flush=True)

step("Generate CrispASR GGUF audio")
gguf_results = []
for i, (lang, text) in enumerate(TEST_SENTENCES):
    print(f"\n--- GGUF: [{lang}] {text} ---", flush=True)
    fname = f"gguf_{lang}_{i}.wav"
    out_path = WORK / fname
    t0 = time.time()
    r = subprocess.run([
        CLI, "--backend", "lahgtna-chatterbox", "-m", "auto",
        "--codec-model", "auto", "--auto-download",
        "-l", lang,
        "--tts", text,
        "--tts-output", str(out_path),
    ], capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    print(f"  stdout: {r.stdout[-200:]}", flush=True)
    print(f"  stderr tail: {r.stderr[-500:]}", flush=True)
    if out_path.exists():
        sz = out_path.stat().st_size
        # Read WAV to get duration
        with open(out_path, "rb") as f:
            f.read(4)  # RIFF
            f.read(4)  # size
            f.read(4)  # WAVE
            f.read(4)  # fmt
            chunk_size = struct.unpack("<I", f.read(4))[0]
            f.read(chunk_size)  # fmt data
            f.read(4)  # data
            data_size = struct.unpack("<I", f.read(4))[0]
        dur = data_size / (24000 * 2)  # 16-bit mono
        print(f"  Generated: {dur:.2f}s in {elapsed:.1f}s ({sz} bytes)", flush=True)
        gguf_results.append({
            "lang": lang, "text": text, "wav": fname,
            "duration": dur, "time": elapsed,
        })
    else:
        print(f"  FAILED: no output", flush=True)
        gguf_results.append({
            "lang": lang, "text": text, "error": "no output",
            "stderr": r.stderr[-500:],
        })

# ═══════════════════════════════════════════════════════════════════
# Part 3: ASR roundtrip (whisper-tiny on both outputs)
# ═══════════════════════════════════════════════════════════════════

step("ASR roundtrip (whisper-tiny)")
for label, results in [("Python", py_results), ("GGUF", gguf_results)]:
    print(f"\n--- {label} outputs ---", flush=True)
    for res in results:
        if "error" in res:
            print(f"  [{res['lang']}] SKIPPED (error)", flush=True)
            continue
        wav_path = WORK / res["wav"]
        r = subprocess.run([CLI, "-m", "auto", "-l", "ar", "-f", str(wav_path), "--no-prints"],
                          capture_output=True, text=True, timeout=120)
        asr_text = r.stdout.strip()
        res["asr"] = asr_text
        print(f"  [{res['lang']}] TTS: {res['text'][:50]}", flush=True)
        print(f"       ASR: {asr_text}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

step("Summary")
RESULTS = {
    "python": py_results,
    "gguf": gguf_results,
    "tokens": py_tokens,
}
with open(WORK / "lahgtna_ab_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print("LAHGTNA A/B RESULTS (#170)")
print("="*70)
print(f"{'':5s} {'Text':40s} {'Py dur':>8s} {'GGUF dur':>8s} {'Py ASR':>20s} {'GGUF ASR':>20s}")
print("-"*110)
for i in range(len(TEST_SENTENCES)):
    lang, text = TEST_SENTENCES[i]
    py = py_results[i] if i < len(py_results) else {}
    gg = gguf_results[i] if i < len(gguf_results) else {}
    py_d = f"{py.get('duration', 0):.1f}s" if 'duration' in py else "ERR"
    gg_d = f"{gg.get('duration', 0):.1f}s" if 'duration' in gg else "ERR"
    py_a = py.get('asr', 'N/A')[:20]
    gg_a = gg.get('asr', 'N/A')[:20]
    print(f"[{lang}] {text[:38]:38s} {py_d:>8s} {gg_d:>8s} {py_a:>20s} {gg_a:>20s}")
print("-"*110)

total = time.time() - _T0
print(f"\nTotal: {total:.0f}s ({total/60:.1f} min)")

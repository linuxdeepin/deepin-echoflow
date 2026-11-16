"""
CrispASR — Zonos TTS diff harness (PLAN #130)

Stage-by-stage comparison of Python reference vs C++ runtime:
  1. Run Python Zonos reference -> dump conditioning, codes, PCM
  2. Build C++ crispasr-cli with CUDA
  3. Run C++ Zonos -> dump codes
  4. Compare: conditioning prefix, AR codes, DAC PCM

This identifies exactly where the C++ diverges from the Python reference.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
REF_DIR = WORK / "ref_dump"
CPP_DIR = WORK / "cpp_dump"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get(
    "CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git"
)
TEXT = "Hello world"
SEED = 42


def run(cmd, check=True, env=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.stdout:
        print(r.stdout[-2000:], flush=True)
    if r.stderr:
        print(r.stderr[-3000:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode})")
    return r


# ── Install deps ────────────────────────────────────────────────────
print("[1/6] Installing dependencies...", flush=True)
subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "--no-install-recommends",
                "espeak-ng", "cmake", "ninja-build", "g++", "ccache", "mold"],
               capture_output=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "zonos", "soundfile", "huggingface_hub"])

# ── Step 1: Python reference ───────────────────────────────────────
print("\n[2/6] Running Python Zonos reference...", flush=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

# Clone repo (for the reference script)
if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
subprocess.run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
                "--recursive", CRISPASR_REPO, str(REPO)])

ref_wav = WORK / "ref_output.wav"
ref_codes = WORK / "ref_codes.txt"

r = run([
    sys.executable,
    str(REPO / "tools" / "reference_backends" / "zonos_tts_reference.py"),
    "--text", TEXT,
    "--output", str(ref_wav),
    "--dump-dir", str(REF_DIR),
    "--dump-codes", str(ref_codes),
    "--seed", str(SEED),
    "--language", "en-us",
    "--max-tokens", "200",  # Short for diff testing
], check=False, timeout=300)

ref_ok = ref_wav.exists() and ref_wav.stat().st_size > 1000

if ref_ok:
    print(f"  Reference WAV: {ref_wav.stat().st_size} bytes", flush=True)
    # List dumped activations
    if REF_DIR.exists():
        npy_files = sorted(REF_DIR.glob("*.npy"))
        print(f"  Activation dumps: {len(npy_files)} files", flush=True)
        for f in npy_files[:10]:
            arr = np.load(f)
            print(f"    {f.name}: shape={arr.shape} dtype={arr.dtype} "
                  f"mean={arr.mean():.4f} std={arr.std():.4f}", flush=True)
    # Show reference codes
    if ref_codes.exists():
        print(f"  Reference codes:", flush=True)
        for ln in ref_codes.read_text().splitlines()[:3]:
            print(f"    {ln[:100]}", flush=True)
else:
    print(f"  Reference FAILED (rc={r.returncode})", flush=True)
    print("  Continuing with C++ build to check C++ side independently...", flush=True)

# ── Step 2: Extract speaker embedding from reference ───────────────
print("\n[3/6] Extracting speaker embedding...", flush=True)
spk_emb_path = WORK / "speaker_emb.bin"

# Try to get speaker embedding from the reference run
try:
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    spk_emb_path = Path(hf_hub_download(
        "cstr/zonos-v0.1-transformer-GGUF", "jfk_speaker_emb.bin",
        cache_dir=str(WORK / "models"), token=token,
    ))
    print(f"  Speaker embedding: {spk_emb_path}", flush=True)
except Exception as e:
    print(f"  Speaker embedding download failed: {e}", flush=True)
    # Write a random one as fallback
    import struct
    emb = np.random.RandomState(42).randn(128).astype(np.float32)
    with open(str(spk_emb_path), "wb") as f:
        f.write(struct.pack("<i", 128))
        f.write(emb.tobytes())

# ── Step 3: CUDA build ─────────────────────────────────────────────
print("\n[4/6] Building C++ with CUDA...", flush=True)
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()

run(["nvidia-smi", "-L"], check=False)
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    ["cmake", "-S", str(REPO), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
     "-DCRISPASR_BUILD_TESTS=OFF"]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
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

# ── Step 4: Download GGUFs ─────────────────────────────────────────
print("\n[5/6] Downloading models...", flush=True)
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

zonos_gguf = Path(hf_hub_download(
    "cstr/zonos-v0.1-transformer-GGUF",
    "zonos-v0.1-transformer-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
dac_gguf = Path(hf_hub_download(
    "cstr/dac-44khz-GGUF",
    "dac-44khz-f16.gguf",
    cache_dir=str(MODELS), token=token,
))

# ── Step 5: Run C++ Zonos ──────────────────────────────────────────
print("\n[6/6] Running C++ Zonos...", flush=True)
cpp_wav = WORK / "cpp_output.wav"

env = {"ZONOS_SPEAKER_EMB_PATH": str(spk_emb_path)}

r = run([
    str(CLI), "--backend", "zonos-tts",
    "-m", str(zonos_gguf),
    "--codec-model", str(dac_gguf),
    "--tts", TEXT,
    "--tts-output", str(cpp_wav),
    "--seed", str(SEED),
    "-v",
], check=False, env=env, timeout=300)

cpp_ok = cpp_wav.exists() and cpp_wav.stat().st_size > 1000

# ── Compare ─────────────────────────────────────────────────────────
print(f"\n{'='*64}", flush=True)
print(f"DIFF RESULTS — Zonos TTS", flush=True)
print(f"{'='*64}", flush=True)
print(f"  Text: {TEXT!r}", flush=True)
print(f"  Reference: {'OK' if ref_ok else 'FAILED'}", flush=True)
print(f"  C++:       {'OK' if cpp_ok else 'FAILED'} (rc={r.returncode})", flush=True)

# Compare codes if both produced them
cpp_codes_path = Path("/mnt/storage/zonos-tts/cpp_codes.txt")
if not cpp_codes_path.exists():
    cpp_codes_path = WORK / "cpp_codes.txt"  # fallback

if ref_codes.exists() and cpp_codes_path.exists():
    print(f"\n  --- Code comparison ---", flush=True)
    ref_lines = ref_codes.read_text().strip().splitlines()
    cpp_text = cpp_codes_path.read_text().strip()
    # C++ format: first line "n_codes n_codebooks", then code matrix
    cpp_lines = cpp_text.splitlines()

    for i, rl in enumerate(ref_lines[:3]):
        print(f"  REF cb{i}: {rl[:80]}", flush=True)
    print(f"  ...", flush=True)
    print(f"  CPP first line: {cpp_lines[0][:80] if cpp_lines else '(empty)'}", flush=True)
    if len(cpp_lines) > 1:
        print(f"  CPP row 1: {cpp_lines[1][:80]}", flush=True)

# Compare conditioning prefix if both exist
ref_cond = REF_DIR / "conditioning_prefix.npy"
if ref_cond.exists():
    arr = np.load(ref_cond)
    print(f"\n  --- Conditioning prefix (reference) ---", flush=True)
    print(f"  Shape: {arr.shape}", flush=True)
    print(f"  Mean: {arr.mean():.6f}  Std: {arr.std():.6f}", flush=True)
    print(f"  First 5: {arr.flatten()[:5]}", flush=True)

# ASR roundtrip both
asr_model = None
try:
    asr_model = Path(hf_hub_download(
        "cstr/parakeet-tdt-0.6b-v2-GGUF",
        "parakeet-tdt-0.6b-v2-q4_k.gguf",
        cache_dir=str(MODELS), token=token,
    ))
except Exception:
    pass

for label, wav_path in [("ref", ref_wav), ("cpp", cpp_wav)]:
    if not wav_path.exists() or wav_path.stat().st_size < 1000:
        print(f"\n  ASR {label}: (no audio)", flush=True)
        continue
    if not asr_model:
        print(f"\n  ASR {label}: (no ASR model)", flush=True)
        continue
    out_stem = WORK / f"asr_{label}"
    try:
        subprocess.run([
            str(CLI), "--backend", "parakeet",
            "-m", str(asr_model),
            "-f", str(wav_path),
            "-of", str(out_stem), "-otxt",
            "--no-prints",
        ], env=os.environ, capture_output=True, text=True, timeout=120)
        txt_path = out_stem.with_suffix(".txt")
        text = txt_path.read_text().strip() if txt_path.exists() else ""
        print(f"\n  ASR {label}: {text[:150]!r}", flush=True)
    except Exception as e:
        print(f"\n  ASR {label}: error {e}", flush=True)

print(f"\nDone.", flush=True)

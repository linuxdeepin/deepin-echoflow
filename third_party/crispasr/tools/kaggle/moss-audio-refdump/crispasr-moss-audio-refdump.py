# %% [markdown]
# # CrispASR — MOSS-Audio-4B-Instruct reference activation dump
#
# Run the Python reference model on samples/jfk.wav and dump per-stage
# activations as a GGUF tensor archive for crispasr-diff validation.
# Upload to `cstr/MOSS-Audio-4B-Instruct-GGUF` as the ref GGUF.

# %% [code]
import os, subprocess, sys, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
MOSS_GITHUB = WORK / "MOSS-Audio-github"
OUT_REF = WORK / "moss-audio-4b-instruct-ref.gguf"
BRANCH = os.environ.get("CRISPASR_REF", "feature/moss-audio")

print(f"[1] cloning CrispASR {BRANCH}", flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", BRANCH,
    "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
])

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.step("cloned", branch=BRANCH, hf_token_ok=bool(hf_token))

# %% [code]
print("[2] cloning MOSS-Audio GitHub source", flush=True)
if MOSS_GITHUB.exists():
    shutil.rmtree(MOSS_GITHUB)
subprocess.check_call([
    "git", "clone", "--depth", "1",
    "https://github.com/OpenMOSS/MOSS-Audio.git", str(MOSS_GITHUB),
])
kh.step("moss_github_cloned")

# %% [code]
print("[3] installing deps", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torch", "transformers==4.57.1", "safetensors", "gguf",
    "huggingface_hub", "hf_transfer",
])
kh.step("deps_installed")

# %% [code]
print("[4] downloading MOSS-Audio model", flush=True)
from huggingface_hub import snapshot_download

for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "moss-audio-src"
        break
scratch.mkdir(parents=True, exist_ok=True)
print(f"  scratch: {scratch} (free: {shutil.disk_usage(scratch).free / (1024**3):.1f} GiB)", flush=True)

src = snapshot_download(
    repo_id="OpenMOSS-Team/MOSS-Audio-4B-Instruct",
    cache_dir=str(scratch),
)
print(f"  source: {src}", flush=True)
kh.step("model_downloaded")

# %% [code]
print("[5] running reference dump inline", flush=True)
os.environ["MOSS_AUDIO_DIR"] = src
os.environ["MOSS_AUDIO_GITHUB"] = str(MOSS_GITHUB)
os.environ["MOSS_AUDIO_PROMPT"] = "Transcribe this audio."
os.environ["MOSS_AUDIO_MAX_NEW"] = "128"

# Run dump inline (not subprocess) so tracebacks are visible in the notebook
sys.path.insert(0, str(REPO / "tools"))
import importlib
import reference_backends.moss_audio as moss_mod
import numpy as np
import wave

# Load audio (16kHz mono PCM)
wav_path = str(REPO / "samples" / "jfk.wav")
with wave.open(wav_path, "rb") as w:
    raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if w.getnchannels() > 1:
        audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)

print(f"  audio: {len(audio)} samples ({len(audio)/16000:.1f}s)", flush=True)

captures = moss_mod.dump(
    model_dir=Path(src),
    audio=audio,
    stages=set(moss_mod.DEFAULT_STAGES),
    max_new_tokens=128,
)

print(f"  captured {len(captures)} stages: {list(captures.keys())}", flush=True)

# Write GGUF
from gguf import GGUFWriter, GGMLQuantizationType
writer = GGUFWriter(str(OUT_REF), "moss_audio_ref")
writer.add_name("MOSS-Audio-4B-Instruct reference dump (jfk.wav)")
for name, arr in captures.items():
    if isinstance(arr, str):
        writer.add_string(f"ref.{name}", arr)
    elif isinstance(arr, np.ndarray):
        arr_f32 = arr.astype(np.float32) if arr.dtype != np.float32 else arr
        writer.add_tensor(f"ref.{name}", arr_f32, raw_dtype=GGMLQuantizationType.F32)
        print(f"  {name}: shape={arr.shape} dtype={arr.dtype}", flush=True)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

print(f"  ref GGUF: {OUT_REF} ({OUT_REF.stat().st_size / (1024**2):.1f} MiB)", flush=True)
kh.step("refdump_done", size_mib=round(OUT_REF.stat().st_size / (1024**2), 1))

# %% [code]
HF_REPO = "cstr/MOSS-Audio-4B-Instruct-GGUF"
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    if OUT_REF.exists():
        print(f"[6] uploading ref GGUF ({OUT_REF.stat().st_size / (1024**2):.1f} MiB)", flush=True)
        api.upload_file(
            path_or_fileobj=str(OUT_REF),
            path_in_repo="moss-audio-4b-instruct-ref.gguf",
            repo_id=HF_REPO, repo_type="model",
            commit_message="Add reference activation dump (jfk.wav, 15 stages)",
        )
        print("[6] uploaded ref GGUF", flush=True)
    kh.step("uploaded")
else:
    print("[6] no HF_TOKEN — staged locally", flush=True)
    if OUT_REF.exists():
        print(f"  {OUT_REF} ({OUT_REF.stat().st_size / (1024**2):.1f} MiB)")

kh.step("done")
print("[DONE]", flush=True)

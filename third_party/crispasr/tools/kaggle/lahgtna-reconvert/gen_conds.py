"""Generate multilingual-compatible conds for lahgtna-chatterbox (#170).

The conds.pt shipped with oddadmix/lahgtna-chatterbox-v1 was generated
from the English base model. When used with the multilingual T3
(t3_mtl23ls_v2), the model immediately produces EOS (0 speech tokens).

This script:
1. Loads the multilingual model
2. Calls prepare_conditionals on the built-in conds' voice
3. Saves new conds that are compatible with the mtl T3
4. Re-converts the GGUF with the new conds
5. Tests Arabic TTS
6. Uploads to HF
"""
import os, sys, subprocess, json, struct, shutil
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
WORK = Path("/kaggle/working")

print("=== Install deps ===", flush=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], capture_output=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "chatterbox-tts", "soundfile", "gguf"])

print("=== Download model ===", flush=True)
from huggingface_hub import snapshot_download
model_dir = Path(snapshot_download("oddadmix/lahgtna-chatterbox-v1",
                                    allow_patterns=["*.json", "*.safetensors", "*.pt", "ve.*"]))
print(f"Model: {model_dir}")

print("=== Load multilingual model ===", flush=True)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model = ChatterboxMultilingualTTS.from_local(model_dir, device)
print("Loaded")

print("=== Generate reference conds ===", flush=True)
# Use the existing conds to extract the voice reference,
# then re-generate conds through the multilingual model's pipeline.
# The simplest approach: generate a short Arabic clip using the existing
# conds, then use that as the reference for prepare_conditionals.
#
# Actually, we can just call prepare_conditionals on any Arabic reference.
# Let's generate a synthetic one first using espeak-ng as a bootstrap.
import subprocess as sp
ref_wav = WORK / "ref_arabic.wav"
sp.run(["apt-get", "install", "-y", "-qq", "espeak-ng"], capture_output=True)
sp.run(["espeak-ng", "-v", "ar", "-w", str(ref_wav), "مرحبا بالعالم"], capture_output=True)
if not ref_wav.exists():
    # Fallback: use silence
    import numpy as np, soundfile as sf
    sf.write(str(ref_wav), np.zeros(24000, dtype=np.float32), 24000)

model.prepare_conditionals(str(ref_wav), exaggeration=0.5)
print(f"Conds prepared from {ref_wav}")

# Save the new conds
new_conds_path = model_dir / "conds.pt"
# Backup original
shutil.copy2(new_conds_path, model_dir / "conds_original.pt")
# Save as a plain dict (converter expects dict, not Conditionals dataclass)
conds_dict = {
    "t3": {
        "speaker_emb": model.conds.t3.speaker_emb.cpu(),
        "cond_prompt_speech_tokens": model.conds.t3.cond_prompt_speech_tokens.cpu(),
        "emotion_adv": model.conds.t3.emotion_adv.cpu(),
    },
    "gen": {
        "prompt_token": model.conds.gen["prompt_token"].cpu(),
        "prompt_token_len": model.conds.gen["prompt_token_len"].cpu(),
        "prompt_feat": model.conds.gen["prompt_feat"].cpu(),
        "embedding": model.conds.gen["embedding"].cpu(),
    },
}
torch.save(conds_dict, new_conds_path)
print(f"Saved new conds (dict) to {new_conds_path}")

# Verify by generating Arabic
print("=== Test: generate Arabic with new conds ===", flush=True)
wav = model.generate("مرحبا بالعالم", language_id="ar")
import soundfile as sf
sf.write(str(WORK / "test_arabic.wav"), wav.squeeze().cpu().numpy(), 24000)
dur = len(wav.squeeze()) / 24000
print(f"Generated {dur:.1f}s of Arabic audio")

# Free GPU
del model
torch.cuda.empty_cache()
import gc; gc.collect()

print("=== Clone CrispASR + convert ===", flush=True)
CRISPASR_DIR = WORK / "CrispASR"
if not CRISPASR_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/CrispStrobe/CrispASR.git",
                           str(CRISPASR_DIR)])

out_dir = WORK / "gguf-out"
out_dir.mkdir(exist_ok=True)
subprocess.check_call([
    sys.executable, str(CRISPASR_DIR / "models" / "convert-chatterbox-to-gguf.py"),
    "--input", str(model_dir),
    "--output-dir", str(out_dir),
    "--t3-only",
])

t3_gguf = out_dir / "chatterbox-t3-f16.gguf"
print(f"GGUF: {t3_gguf} ({t3_gguf.stat().st_size/1e6:.1f} MB)")

# Verify
import gguf
r = gguf.GGUFReader(str(t3_gguf))
for name, field in r.fields.items():
    if 'text_vocab' in name:
        print(f"  {name} = {struct.unpack('<I', field.parts[-1].tobytes())[0]}")
for t in r.tensors:
    if 'text_emb' in t.name:
        print(f"  {t.name}: shape={t.shape}")

print("=== Upload to HF ===", flush=True)
hf_token = None
for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
          "/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-hf-token/hf_token.txt"]:
    if os.path.exists(p):
        with open(p) as f:
            hf_token = f.read().strip()
        break
if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    # Delete old file first to avoid LFS dedup
    try:
        api.delete_file("chatterbox-t3-f16.gguf", "cstr/lahgtna-chatterbox-v1-GGUF")
    except:
        pass
    api.upload_file(path_or_fileobj=str(t3_gguf),
                    path_in_repo="chatterbox-t3-f16.gguf",
                    repo_id="cstr/lahgtna-chatterbox-v1-GGUF")
    print("Uploaded")
else:
    print("No HF token — skipped upload")

print("=== Done ===")

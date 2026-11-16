#!/usr/bin/env python3
"""Quick end-to-end test: does Python Irodori-TTS produce speech without speaker conditioning?"""
import os, sys, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(str(WORK))

def log(msg): print(f"[e2e] {msg}", flush=True)

# Clone Irodori-TTS source
irodori_src = WORK / "irodori-tts-src"
if not irodori_src.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
        "https://github.com/Aratako/Irodori-TTS.git", str(irodori_src)])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "safetensors", "transformers", "sentencepiece", "huggingface_hub", "soundfile"])

# HF token
for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
          "/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-hf-token/hf_token.txt"]:
    if os.path.exists(p):
        os.environ["HF_TOKEN"] = open(p).read().strip()
        break

sys.path.insert(0, str(irodori_src))
import torch, json, numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.rf import sample_euler_rf_cfg
from transformers import AutoTokenizer

# Load model
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("Aratako/Irodori-TTS-500M-v3", "model.safetensors")
with safe_open(ckpt, framework="pt", device="cpu") as f:
    config = json.loads((f.metadata() or {}).get("config_json", "{}"))

cfg = ModelConfig(**{k: v for k, v in config.items() if hasattr(ModelConfig, k)})
log(f"Config: text_dim={cfg.text_dim}, model_dim={cfg.model_dim}, layers={cfg.num_layers}")

state = load_file(ckpt, device="cpu")
model = TextToLatentRFDiT(cfg)
model.load_state_dict(state, strict=False)
model.eval()
del state
import gc; gc.collect()
log("Model loaded")

# Tokenize
tok = AutoTokenizer.from_pretrained(cfg.text_tokenizer_repo, use_fast=True)
text = "こんにちは、世界。テストです。"
ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)
log(f"Text: '{text}' → {len(ids)} tokens: {ids}")

text_ids = torch.tensor([ids], dtype=torch.long)
text_mask = torch.ones_like(text_ids, dtype=torch.bool)

# No speaker (no_ref=True)
ref_latent = torch.zeros(1, 1, cfg.latent_dim * cfg.latent_patch_size)
ref_mask = torch.zeros(1, 1, dtype=torch.bool)

# Sample
log("Running RF sampling (40 steps, no CFG)...")
with torch.inference_mode():
    z = sample_euler_rf_cfg(
        model, text_ids, text_mask,
        ref_latent=ref_latent, ref_mask=ref_mask,
        sequence_length=63,
        num_steps=40,
        cfg_scale_text=0.0, cfg_scale_speaker=0.0, cfg_scale_caption=0.0,
        seed=42,
    )
log(f"Latent: {z.shape}")

# Save latent for external decode
z_np = z[0].float().numpy()
np.save(str(WORK / "python_nocfg_latent.npy"), z_np)
z_np.tofile(str(WORK / "python_nocfg_latent.bin"))
log(f"Saved latent: {z_np.shape}")

# Also try with CFG
log("Running with CFG (cfg_text=3.0)...")
with torch.inference_mode():
    z_cfg = sample_euler_rf_cfg(
        model, text_ids, text_mask,
        ref_latent=ref_latent, ref_mask=ref_mask,
        sequence_length=63,
        num_steps=40,
        cfg_scale_text=3.0, cfg_scale_speaker=0.0, cfg_scale_caption=0.0,
        seed=42,
    )
z_cfg_np = z_cfg[0].float().numpy()
np.save(str(WORK / "python_cfg_latent.npy"), z_cfg_np)
z_cfg_np.tofile(str(WORK / "python_cfg_latent.bin"))
log(f"CFG latent: {z_cfg_np.shape}")

# Decode with DACVAE
log("Loading DACVAE for decode...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/facebookresearch/dacvae.git", "audiotools"])
    from dacvae import DACVAE
    dacvae_path = hf_hub_download("Aratako/Semantic-DACVAE-Japanese-32dim", "weights.pth")
    codec = DACVAE.load(dacvae_path).eval()
    codec.decoder.alpha = 0.0
    if hasattr(codec.decoder, 'wm_model'):
        def _wm_pass(x, msg=None, _d=codec.decoder):
            return _d.wm_model.encoder_block.forward_no_conv(x)
        codec.decoder.watermark = _wm_pass

    import soundfile as sf
    with torch.inference_mode():
        # No CFG
        audio_nocfg = codec.decode(z.transpose(1,2)).squeeze().cpu().numpy()
        sf.write(str(WORK / "python_nocfg_audio.wav"), audio_nocfg, codec.sample_rate)
        log(f"No-CFG audio: {len(audio_nocfg)} samples ({len(audio_nocfg)/codec.sample_rate:.2f}s)")

        # With CFG
        audio_cfg = codec.decode(z_cfg.transpose(1,2)).squeeze().cpu().numpy()
        sf.write(str(WORK / "python_cfg_audio.wav"), audio_cfg, codec.sample_rate)
        log(f"CFG audio: {len(audio_cfg)} samples ({len(audio_cfg)/codec.sample_rate:.2f}s)")
except Exception as e:
    log(f"DACVAE decode failed: {e}")

# Upload to HF
try:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for fname in ["python_nocfg_latent.bin", "python_cfg_latent.bin",
                   "python_nocfg_audio.wav", "python_cfg_audio.wav"]:
        fpath = WORK / fname
        if fpath.exists():
            api.upload_file(path_or_fileobj=str(fpath), path_in_repo=fname,
                          repo_id="cstr/irodori-tts-GGUF", repo_type="model")
            log(f"Uploaded {fname}")
except Exception as e:
    log(f"Upload failed: {e}")

log("DONE")

#!/usr/bin/env python3
"""Kaggle kernel: convert facebook/w2v-bert-2.0 (layers 0..16) into an
encoder-only sidon-layout GGUF for Confucius4-TTS speaker conditioning.

The reference conditions T2S on `Wav2Vec2BertModel(...).hidden_states[17]`
(z-normalised with the baked stats). sidon.cpp already implements the exact
SeamlessM4T frontend + conformer stack; with `sidon.encoder_only=1` it loads
without the DAC decoder and `sidon_extract_hidden()` returns the layer-17
hidden states.

Push ($KAGGLE_ACCOUNT is the default account):
  python -m kaggle kernels push -p tools/kaggle/confucius4-w2v-convert
"""

import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = "main"
HF_REPO = "cstr/confucius4-tts-GGUF"

print(f"=== clone {BRANCH} ===", flush=True)
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

kh.step("install deps")
kh.sh_with_progress("pip install -q safetensors gguf huggingface_hub hf_transfer transformers")

kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

kh.step("download facebook/w2v-bert-2.0")
from huggingface_hub import hf_hub_download, HfApi

base = TEMP / "w2v-bert-2.0"
base.mkdir(parents=True, exist_ok=True)
for fname in ("model.safetensors", "config.json", "preprocessor_config.json"):
    p = hf_hub_download("facebook/w2v-bert-2.0", fname, local_dir=str(base), token=hf_token)
    print(f"  {fname}: {os.path.getsize(p) / 1024**2:.0f} MB")

kh.step("convert (encoder-only, 17 layers)")
out = TEMP / "confucius4-tts-w2v-f16.gguf"
subprocess.check_call([sys.executable, str(REPO / "models" / "convert-sidon-to-gguf.py"),
                       "--base", str(base), "--output", str(out),
                       "--encoder-only", "--layers", "17", "--dtype", "f16"])
print(f"  {out} ({os.path.getsize(out) / 1024**3:.2f} GiB)")

kh.step("upload to HF")
api = HfApi(token=hf_token)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api.upload_file(path_or_fileobj=str(out), path_in_repo=out.name,
                repo_id=HF_REPO, repo_type="model",
                commit_message="Add encoder-only w2v-BERT 2.0 (17L) GGUF for native speaker conditioning")
print("=== Done ===", flush=True)

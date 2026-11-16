#!/usr/bin/env python3
"""
Kaggle kernel: Irodori-TTS-500M-v3 → GGUF F16 + Q4_K + reference dump.

Downloads Aratako/Irodori-TTS-500M-v3, converts to GGUF (F16),
builds crispasr-quantize to produce Q4_K, runs Python reference
dump for crispasr-diff parity validation, uploads to HuggingFace.

Outputs:
  - irodori-tts-500m-v3-f16.gguf     (~1 GB)
  - irodori-tts-500m-v3-q4_k.gguf    (~250 MB)
  - irodori-tts-ref.gguf              (reference activations for diff)

Run under $KAGGLE_ACCOUNT account with GPU enabled.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Prevent tensorflow from messing with protobuf
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

WORK = Path("/kaggle/working")
os.chdir(str(WORK))

PROGRESS = WORK / "progress.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


log("Kernel started")

# ── Clone CrispASR (feat/irodori-tts branch) ─────────────────────────

REPO = WORK / "CrispASR"
if not REPO.exists():
    log("Cloning CrispASR (feat/irodori-tts branch)...")
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--branch", "feat/irodori-tts",
        "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
    ])

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
# Fallback: if clone path doesn't have harness, use bundled copy
if not (REPO / "tools" / "kaggle" / "kaggle_harness.py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import kaggle_harness as kh
    kh.init_progress()
    log("kaggle_harness imported OK")
except Exception as e:
    log(f"kaggle_harness import failed: {e} — continuing without harness")
    # Minimal stubs
    class _KH:
        def step(self, msg): log(f"[step] {msg}")
        def resolve_hf_token(self):
            import os
            for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
                      "/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-hf-token/hf_token.txt",
                      "/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-hf-token/hf_token.txt"]:
                if os.path.exists(p):
                    return open(p).read().strip()
            return os.environ.get("HF_TOKEN")
        def install_build_toolchain(self): pass
        def sh_with_progress(self, cmd):
            subprocess.check_call(cmd, shell=True)
        def safe_build_jobs(self, gpu=False): return 2
        def build_heartbeat(self, msg=""):
            from contextlib import nullcontext
            return nullcontext()
    kh = _KH()

# ── Install deps ─────────────────────────────────────────────────────

kh.step("installing dependencies")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "gguf", "safetensors", "transformers", "sentencepiece",
    "huggingface_hub", "hf_transfer",
])
log("deps installed")

# ── Authenticate HF ─────────────────────────────────────────────────

token = kh.resolve_hf_token()
if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    log("HF token resolved")
else:
    log("WARNING: no HF token found — downloads may fail for gated models")

# ── Download model ───────────────────────────────────────────────────

kh.step("downloading Irodori-TTS-500M-v3")

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Aratako/Irodori-TTS-500M-v3"

files = list_repo_files(REPO_ID)
st_files = [f for f in files if f.endswith(".safetensors")]
log(f"Repo files: {files}")
log(f"Safetensors: {st_files}")

if not st_files:
    log("ERROR: No .safetensors found in repo")
    sys.exit(1)

ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=st_files[0], token=token)
log(f"Downloaded: {ckpt_path} ({os.path.getsize(ckpt_path) / 1024 / 1024:.1f} MB)")

# ── Convert to GGUF (F16) ───────────────────────────────────────────

kh.step("converting to GGUF (F16)")

converter_path = REPO / "models" / "convert-irodori-tts-to-gguf.py"

output_f16 = WORK / "irodori-tts-500m-v3-f16.gguf"

cmd = [
    sys.executable, str(converter_path),
    "--checkpoint", ckpt_path,
    "--output", str(output_f16),
    "--tokenizer-repo", "sbintuitions/sarashina2.2-0.5b",
]
log(f"Running: {' '.join(cmd)}")
t0 = time.time()
subprocess.check_call(cmd)
elapsed = time.time() - t0
log(f"F16 GGUF done in {elapsed:.1f}s: {output_f16.stat().st_size / 1024 / 1024:.1f} MB")

# ── Build crispasr-quantize ──────────────────────────────────────────

kh.step("building crispasr-quantize")
kh.install_build_toolchain()

build_dir = Path("/kaggle/temp/quant-build")
cmake_env = os.environ.copy()
cmake_env["CCACHE_DIR"] = "/kaggle/working/.ccache"

kh.sh_with_progress(
    f"cmake -G Ninja -B {build_dir} -S {REPO}"
    f" -DCMAKE_BUILD_TYPE=Release -DCRISPASR_NO_C2PA_NATIVE=ON -DGGML_CUDA=OFF"
    f" -DCMAKE_C_COMPILER_LAUNCHER=ccache"
    f" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
)
n_jobs = kh.safe_build_jobs(gpu=False)
kh.sh_with_progress(
    f"cmake --build {build_dir} -j{n_jobs} --target crispasr-quantize"
)

# Find the binary
quantize_bin = None
import glob
for candidate in [
    build_dir / "bin" / "crispasr-quantize",
    build_dir / "examples" / "crispasr-quantize" / "crispasr-quantize",
]:
    if candidate.exists():
        quantize_bin = candidate
        break
if quantize_bin is None:
    hits = glob.glob(str(build_dir / "**" / "crispasr-quantize"), recursive=True)
    if hits:
        quantize_bin = Path(hits[0])

if quantize_bin is None or not quantize_bin.exists():
    log("ERROR: crispasr-quantize not found")
    kh.sh_with_progress(f"find {build_dir} -name 'crispasr*' -type f | head -20")
    sys.exit(1)

log(f"quantize binary: {quantize_bin}")

# ── Quantize to Q4_K ────────────────────────────────────────────────

kh.step("quantizing to Q4_K")
output_q4k = WORK / "irodori-tts-500m-v3-q4_k.gguf"

t0 = time.time()
kh.sh_with_progress(f"{quantize_bin} {output_f16} {output_q4k} q4_k")
elapsed = time.time() - t0
log(f"Q4_K done in {elapsed:.1f}s: {output_q4k.stat().st_size / 1024 / 1024:.1f} MB")

# Also Q8_0
output_q8 = WORK / "irodori-tts-500m-v3-q8_0.gguf"
kh.sh_with_progress(f"{quantize_bin} {output_f16} {output_q8} q8_0")
log(f"Q8_0: {output_q8.stat().st_size / 1024 / 1024:.1f} MB")

# ── Convert DACVAE decoder ───────────────────────────────────────────

kh.step("converting DACVAE decoder")

try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "git+https://github.com/facebookresearch/dacvae.git",
    ])

    dacvae_gguf = WORK / "dacvae-ja-32dim-f16.gguf"
    dacvae_converter = REPO / "models" / "convert-dacvae-to-gguf.py"
    subprocess.check_call([
        sys.executable, str(dacvae_converter),
        "--model", "Aratako/Semantic-DACVAE-Japanese-32dim",
        "--output", str(dacvae_gguf),
    ])
    log(f"DACVAE GGUF: {dacvae_gguf.stat().st_size / 1024 / 1024:.1f} MB")
except Exception as e:
    log(f"DACVAE conversion failed (non-fatal): {e}")
    dacvae_gguf = None

# ── Python reference dump (intermediates for crispasr-diff) ──────────

kh.step("running reference dump")

try:
    import json
    import numpy as np
    import torch
    from safetensors import safe_open

    # Read model config from checkpoint metadata
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        config_json = metadata.get("config_json")
    config = json.loads(config_json) if config_json else {}
    log(f"Model config: latent_dim={config.get('latent_dim', 128)}, "
        f"model_dim={config.get('model_dim', 2048)}, "
        f"num_layers={config.get('num_layers', 24)}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sbintuitions/sarashina2.2-0.5b", use_fast=True)

    # Encode test text
    test_text = "こんにちは、世界。テストです。"
    token_ids = tok.encode(test_text, add_special_tokens=False)
    token_ids = [tok.bos_token_id] + token_ids
    log(f"Test text: '{test_text}' → {len(token_ids)} tokens")

    # Load model weights and run forward passes for intermediate tensors
    # We'll load the Irodori-TTS model using safetensors lazy loading
    from safetensors.torch import load_file

    log("Loading model weights...")
    state = load_file(ckpt_path, device="cpu")
    log(f"  {len(state)} tensors loaded")

    # Build a minimal model to capture intermediates
    # We need: text_encoder output, speaker_encoder output, first DiT block output
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # Try to import Irodori-TTS source if available
    irodori_src = None
    for candidate in [
        Path("/kaggle/working") / "irodori-tts-src",
        Path("/tmp") / "irodori-tts-src",
    ]:
        if candidate.exists():
            irodori_src = candidate
            break

    if irodori_src is None:
        log("Cloning Irodori-TTS source for reference forward pass...")
        irodori_src = Path("/kaggle/temp") / "irodori-tts-src"
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/Aratako/Irodori-TTS.git", str(irodori_src),
        ])

    sys.path.insert(0, str(irodori_src))
    from irodori_tts.config import ModelConfig
    from irodori_tts.model import TextToLatentRFDiT

    # Instantiate model and load weights
    model_cfg = ModelConfig(**{k: v for k, v in config.items()
                              if k in [f.name for f in ModelConfig.__dataclass_fields__.values()]
                              if hasattr(ModelConfig, k)})
    log(f"Instantiating model (text_dim={model_cfg.text_dim}, model_dim={model_cfg.model_dim})...")
    model = TextToLatentRFDiT(model_cfg)
    model.load_state_dict(state, strict=False)
    model.eval()
    del state
    import gc; gc.collect()
    log("Model loaded, running forward pass...")

    # Tokenize and run text encoder
    text_ids = torch.tensor([token_ids], dtype=torch.long)
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)

    with torch.inference_mode():
        text_state = model.text_encoder(text_ids, text_mask)
        text_state = model.text_norm(text_state)
        log(f"  text_state: {text_state.shape} (first 4 values: {text_state[0, 0, :4].tolist()})")

    # Save reference as GGUF
    try:
        from gguf import GGUFWriter
        ref_output = WORK / "irodori-tts-ref.gguf"
        writer = GGUFWriter(str(ref_output), "irodori-tts-ref")
        writer.add_string("irodori.ref.test_text", test_text)
        writer.add_uint32("irodori.ref.n_tokens", len(token_ids))

        # Save text encoder output
        arr = text_state[0].detach().float().numpy()
        writer.add_tensor("ref.text_state", arr)
        log(f"  ref.text_state: {arr.shape}")

        # Try speaker encoder with zeros (unconditional)
        if model_cfg.use_speaker_condition_resolved and model.speaker_encoder is not None:
            latent_dim = model_cfg.latent_dim * model_cfg.latent_patch_size
            dummy_ref = torch.zeros(1, 10, latent_dim)
            dummy_mask = torch.ones(1, 10, dtype=torch.bool)
            with torch.inference_mode():
                spk_state = model.speaker_encoder(dummy_ref, dummy_mask)
                spk_state = model.speaker_norm(spk_state)
            arr_spk = spk_state[0].detach().float().numpy()
            writer.add_tensor("ref.spk_state_zeros", arr_spk)
            log(f"  ref.spk_state_zeros: {arr_spk.shape}")

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        log(f"Reference GGUF: {ref_output} ({ref_output.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        log(f"  GGUF reference write failed: {e}")

    del model
    gc.collect()

except Exception as e:
    import traceback
    log(f"Reference dump failed: {e}")
    log(traceback.format_exc())
    log("(non-fatal — GGUF conversion still succeeded)")

# ── Upload to HuggingFace ────────────────────────────────────────────

kh.step("uploading to HuggingFace")
try:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    hf_repo = "cstr/irodori-tts-GGUF"

    api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="model")

    upload_files = []
    for fpath in [output_f16, output_q4k, output_q8]:
        if fpath.exists() and fpath.stat().st_size > 0:
            upload_files.append(fpath)

    ref_output = WORK / "irodori-tts-ref.gguf"
    if ref_output.exists() and ref_output.stat().st_size > 0:
        upload_files.append(ref_output)
    dacvae_gguf_path = WORK / "dacvae-ja-32dim-f16.gguf"
    if dacvae_gguf_path.exists() and dacvae_gguf_path.stat().st_size > 0:
        upload_files.append(dacvae_gguf_path)

    for fpath in upload_files:
        log(f"  uploading {fpath.name} ({fpath.stat().st_size / 1024 / 1024:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fpath.name,
            repo_id=hf_repo,
            repo_type="model",
        )
        log(f"  ✓ {fpath.name}")

    log(f"Upload complete: {hf_repo}")
except Exception as e:
    log(f"HF upload failed (non-fatal): {e}")
    log("Files staged at /kaggle/working/ for manual download")

# ── Cleanup ──────────────────────────────────────────────────────────

import shutil
for d in [REPO, Path("/kaggle/temp/quant-build"), WORK / ".ccache"]:
    if d.exists():
        shutil.rmtree(str(d), ignore_errors=True)
        log(f"Cleaned up {d.name}")

# Remove F16 if Q4_K exists (save output space — Kaggle limits to 20 GB)
if output_q4k.exists() and output_f16.exists():
    output_f16.unlink()
    log("Removed F16 (Q4_K available on HF)")

kh.step("done")
log("\n=== ALL DONE ===")
for fpath in [output_f16, output_q4k, output_q8, WORK / "irodori-tts-ref.gguf"]:
    if fpath.exists():
        log(f"  {fpath.name}: {fpath.stat().st_size / 1024 / 1024:.1f} MB")

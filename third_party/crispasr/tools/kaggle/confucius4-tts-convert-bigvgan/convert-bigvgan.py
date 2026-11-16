#!/usr/bin/env python3
"""Convert nvidia/bigvgan_v2_22khz_80band_256x to GGUF for Confucius4-TTS vocoder.

Reuses the IndexTTS BigVGAN GGUF format: fuses weight_norm at convert time,
writes with the same shortened tensor names so indextts_voc.cpp can load it
directly with different hparams.

Push (under $KAGGLE_ACCOUNT):
  export KAGGLE_API_TOKEN=...
  python -m kaggle kernels push -p tools/kaggle/confucius4-tts-convert-bigvgan
"""

import os
import sys
import json
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")

HF_REPO = "cstr/confucius4-tts-GGUF"

# ── Phase 0: Clone + deps ───────────────────────────────────────────────────
print("=== Phase 0: setup ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "main",
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])
subprocess.check_call(
    ["git", "submodule", "update", "--init", "--recursive"],
    cwd=str(REPO),
)
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

kh.step("install deps")
kh.sh_with_progress("pip install -q gguf huggingface_hub hf_transfer torch")

kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# ── Phase 1: Download BigVGAN model ─────────────────────────────────────────
kh.step("download BigVGAN")
from huggingface_hub import hf_hub_download, HfApi
import torch
import numpy as np

# Download the generator checkpoint
gen_path = hf_hub_download(
    "nvidia/bigvgan_v2_22khz_80band_256x",
    "bigvgan_generator.pt",
    local_dir=str(TEMP / "bigvgan"),
    token=hf_token,
)
print(f"  generator: {gen_path} ({os.path.getsize(gen_path) / 1024**2:.0f} MB)")

# Also download config for reference
cfg_path = hf_hub_download(
    "nvidia/bigvgan_v2_22khz_80band_256x",
    "config.json",
    local_dir=str(TEMP / "bigvgan"),
    token=hf_token,
)
with open(cfg_path) as f:
    config = json.load(f)
print(f"  config: {json.dumps({k: config[k] for k in ['upsample_rates', 'upsample_kernel_sizes', 'upsample_initial_channel', 'num_mels', 'sampling_rate', 'activation']}, indent=2)}")

# ── Phase 2: Load and inspect ───────────────────────────────────────────────
kh.step("inspect model")
raw = torch.load(gen_path, map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "generator" in raw:
    sd = raw["generator"]
elif isinstance(raw, dict) and "state_dict" in raw:
    sd = raw["state_dict"]
else:
    sd = raw

print(f"  state_dict keys: {len(sd)}")
for name in sorted(sd.keys())[:20]:
    t = sd[name]
    print(f"  {name:60s} {str(list(t.shape)):25s} {t.dtype}")
if len(sd) > 20:
    print(f"  ... ({len(sd) - 20} more)")

# ── Phase 3: Fuse weight_norm ───────────────────────────────────────────────
kh.step("fuse weight_norm")

def fuse_weight_norm(sd):
    out = {}
    pairs = {}
    for k, v in sd.items():
        if k.endswith(".weight_g"):
            pairs.setdefault(k[:-len(".weight_g")], {})["g"] = v
        elif k.endswith(".weight_v"):
            pairs.setdefault(k[:-len(".weight_v")], {})["v"] = v
        else:
            out[k] = v
    fused = 0
    for stem, parts in pairs.items():
        if "g" not in parts or "v" not in parts:
            for suf, t in parts.items():
                out[f"{stem}.weight_{suf}"] = t
            continue
        g = parts["g"].to(torch.float32)
        v = parts["v"].to(torch.float32)
        flat = v.reshape(v.shape[0], -1)
        norm = flat.norm(p=2, dim=1).reshape(g.shape)
        out[f"{stem}.weight"] = v * (g / norm.clamp_min(1e-12))
        fused += 1
    print(f"  weight_norm fused: {fused} pairs")
    return out

sd = fuse_weight_norm(sd)

# ── Phase 4: Write GGUF ────────────────────────────────────────────────────
kh.step("write GGUF")
import gguf

def shorten(name):
    """Same shortener as IndexTTS converter for compatibility."""
    name = name.replace("speaker_encoder.", "se.")
    name = name.replace("res2net_block.", "r2n.")
    name = name.replace("resblocks.", "resb.")
    name = name.replace("blocks.", "b.")
    name = name.replace("conv.conv.", "c.")
    name = name.replace("norm.norm.", "n.")
    name = name.replace("activations.", "act.")
    name = name.replace("downsample.lowpass.", "ds.")
    name = name.replace("upsample.", "us.")
    name = name.replace("running_mean", "rm")
    name = name.replace("running_var", "rv")
    return name

SKIP_PREFIXES = ("logit_scale",)
SKIP_SUFFIXES = (".num_batches_tracked",)

out_path = TEMP / "confucius4-tts-bigvgan-22k-f16.gguf"
w = gguf.GGUFWriter(str(out_path), arch="indextts.bigvgan", use_temp_file=True)
w.add_name("confucius4-bigvgan-22khz")

# Hparams — using indextts.bigvgan.* keys for compatibility with indextts_voc.cpp
w.add_uint32("indextts.bigvgan.gpt_dim", 80)  # mel input, not GPT hidden
w.add_uint32("indextts.bigvgan.upsample_initial_channel", 1536)
w.add_uint32("indextts.bigvgan.num_upsamples", 6)
w.add_uint32("indextts.bigvgan.num_kernels", 3)
w.add_uint32("indextts.bigvgan.speaker_embedding_dim", 0)  # no speaker conditioning
w.add_uint32("indextts.bigvgan.num_mels", 80)
w.add_uint32("indextts.sampling_rate", 22050)
w.add_uint32("indextts.bigvgan.hop_size", 256)
w.add_array("indextts.bigvgan.upsample_rates", [4, 4, 2, 2, 2, 2])
w.add_array("indextts.bigvgan.upsample_kernel_sizes", [8, 8, 4, 4, 4, 4])
w.add_array("indextts.bigvgan.resblock_kernel_sizes", [3, 7, 11])
w.add_array("indextts.bigvgan.resblock_dilation_sizes", [1, 3, 5, 1, 3, 5, 1, 3, 5])
w.add_string("indextts.bigvgan.activation", "snakebeta")
w.add_bool("indextts.bigvgan.cond_in_each_up_layer", False)
w.add_bool("indextts.bigvgan.use_tanh_at_final", False)

n_written = 0
n_skipped = 0
for name in sorted(sd.keys()):
    if any(name.startswith(p) for p in SKIP_PREFIXES):
        n_skipped += 1
        continue
    if any(name.endswith(s) for s in SKIP_SUFFIXES):
        n_skipped += 1
        continue

    t = sd[name].to(torch.float32).clamp_(-65504.0, 65504.0).numpy()
    short = shorten(name)

    # 1D tensors and special params stay F32; multi-dim → F16
    if t.ndim <= 1 or name.endswith(".bias") or "norm" in name \
            or "alpha" in name or "beta" in name or "filter" in name \
            or "running_mean" in name or "running_var" in name:
        w.add_tensor(short, t)
    else:
        w.add_tensor(short, t.astype(np.float16))
    n_written += 1
    if n_written % 20 == 0:
        print(f"  wrote {n_written} tensors...", flush=True)

print(f"  wrote {n_written} tensors, skipped {n_skipped}")
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"  GGUF: {out_path} ({os.path.getsize(str(out_path)) / 1024**2:.1f} MB)")

# ── Phase 5: Upload ─────────────────────────────────────────────────────────
kh.step("upload to HF")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi(token=hf_token)
api.upload_file(
    path_or_fileobj=str(out_path),
    path_in_repo="confucius4-tts-bigvgan-22k-f16.gguf",
    repo_id=HF_REPO, repo_type="model",
    commit_message="Add BigVGAN v2 22kHz F16 GGUF (80-mel vocoder for Confucius4-TTS)",
)
print(f"  uploaded to {HF_REPO}")

# ── Phase 6: Build quantizer + quantize ──────────────────────────────────────
kh.step("build quantizer")
BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} "
    f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF " + " ".join(flags),
)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=False)} --target crispasr-quantize"
    )
quantize_bin = BUILD / "bin" / "crispasr-quantize"

for quant in ("q8_0",):
    kh.step(f"quantize {quant}")
    qout = TEMP / f"confucius4-tts-bigvgan-22k-{quant}.gguf"
    kh.sh_with_progress(f"{quantize_bin} {out_path} {qout} {quant}")
    print(f"  {quant}: {qout} ({os.path.getsize(str(qout)) / 1024**2:.1f} MB)")
    kh.step(f"upload {quant}")
    api.upload_file(
        path_or_fileobj=str(qout),
        path_in_repo=f"confucius4-tts-bigvgan-22k-{quant}.gguf",
        repo_id=HF_REPO, repo_type="model",
        commit_message=f"Add BigVGAN v2 22kHz {quant.upper()} GGUF",
    )
    print(f"  uploaded {quant}")

print(f"\n=== Done — BigVGAN 22kHz GGUFs uploaded to {HF_REPO} ===", flush=True)

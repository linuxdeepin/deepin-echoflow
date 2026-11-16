#!/usr/bin/env python3
"""
Convert HumeAI/tada-codec decoder subfolder → GGUF for the CrispASR
`tada-tts` backend's codec decoder stage.

The TADA codec decoder converts expanded acoustic feature sequences
(512-d vectors at 50 Hz frame rate) into 24 kHz mono PCM audio via:

  1. decoder_proj: Linear(512, 1024)
  2. local_attention_decoder: 6-layer transformer encoder
     - LocalSelfAttention (8 heads, RoPE, 1024-d) with post-norm
     - FFN (1024 → 4096 → 1024, GELU) with post-norm
     - Final LayerNorm
  3. wav_decoder: DAC-style upsampler
     - WNConv1d(1024, 1536, k=7, p=3)
     - 4x DecoderBlock (strides [4, 4, 5, 6]):
         Snake1d → WNConvTranspose1d → 3x ResidualUnit(d=1,3,9)
         ResidualUnit: Snake1d → WNConv1d(k=7,dil) → Snake1d → WNConv1d(k=1)
     - Snake1d → WNConv1d(out, 1, k=7, p=3) → Tanh

Output: 24000 Hz mono (strides 4*4*5*6 = 480 upsample factor, 50 Hz→24 kHz)

Usage:
    python models/convert-tada-codec-to-gguf.py \\
        --input HumeAI/tada-codec \\
        --output tada-codec.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub")


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}…", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "decoder/*.safetensors", "decoder/*.json",
    ]))


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    n = hf_name

    # decoder_proj → codec.proj
    n = n.replace("decoder_proj.", "codec.proj.")

    # local_attention_decoder → codec.attn.*
    n = n.replace("local_attention_decoder.input_proj.", "codec.attn.input_proj.")
    n = n.replace("local_attention_decoder.final_norm.", "codec.attn.final_norm.")

    # Attention layers
    # local_attention_decoder.layers.{i}.self_attn.* → codec.attn.blk.{i}.*
    n = n.replace("local_attention_decoder.layers.", "codec.attn.blk.")
    n = n.replace(".self_attn.qkv.", ".attn_qkv.")
    n = n.replace(".self_attn.out_proj.", ".attn_output.")
    n = n.replace(".self_attn.layer_norm.", ".attn_norm.")
    n = n.replace(".ffn.0.", ".ffn_up.")    # Linear(1024, 4096)
    n = n.replace(".ffn.3.", ".ffn_down.")  # Linear(4096, 1024)
    # Note: ffn.1 = GELU (no params), ffn.2 = Dropout (no params), ffn.4 = Dropout
    n = n.replace(".norm.", ".ffn_norm.")

    # wav_decoder → codec.dac.*
    n = n.replace("wav_decoder.model.", "codec.dac.")

    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert TADA codec decoder to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. HumeAI/tada-codec) or local dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    # The decoder lives in a subfolder
    decoder_dir = model_dir / "decoder"
    if not decoder_dir.exists():
        decoder_dir = model_dir  # might be pointed directly at the decoder dir

    config_path = decoder_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"  Config: {json.dumps(cfg, indent=2)}")
    else:
        cfg = {}
        print("  No config.json found, using defaults")

    embed_dim = int(cfg.get("embed_dim", 512))
    hidden_dim = int(cfg.get("hidden_dim", 1024))
    num_attn_layers = int(cfg.get("num_attn_layers", 6))
    num_attn_heads = int(cfg.get("num_attn_heads", 8))
    attn_dim_ff = int(cfg.get("attn_dim_feedforward", 4096))
    wav_channels = int(cfg.get("wav_decoder_channels", 1536))
    strides = cfg.get("strides", [4, 4, 5, 6])

    print(f"\nTADA Codec Decoder")
    print(f"  Input:         embed_dim={embed_dim}")
    print(f"  Attention:     hidden={hidden_dim}  layers={num_attn_layers}  "
          f"heads={num_attn_heads}  ff={attn_dim_ff}")
    print(f"  WAV decoder:   channels={wav_channels}  strides={strides}")
    print(f"  Upsample:      {'x'.join(map(str, strides))} = "
          f"{np.prod(strides)}x → {50*np.prod(strides)} Hz")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    st_files = sorted(decoder_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"no safetensors in {decoder_dir}")
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_idx = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            name_to_idx[k] = i
    print(f"  Safetensors:   {len(name_to_idx)} tensors in {len(st_files)} file(s)")

    # ── Write GGUF ──
    out_path = Path(args.output)
    w = GGUFWriter(str(out_path), arch="tada-codec", use_temp_file=False)

    w.add_name("tada-codec-decoder")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))

    u32("tada_codec.embed_dim", embed_dim)
    u32("tada_codec.hidden_dim", hidden_dim)
    u32("tada_codec.num_attn_layers", num_attn_layers)
    u32("tada_codec.num_attn_heads", num_attn_heads)
    u32("tada_codec.attn_dim_ff", attn_dim_ff)
    u32("tada_codec.wav_channels", wav_channels)
    w.add_array("tada_codec.strides", strides)
    u32("tada_codec.sample_rate", 24000)

    # ── Pre-materialize weight-normed convolutions ──
    # PyTorch weight_norm stores parametrizations.weight.original0 (g, magnitude)
    # and parametrizations.weight.original1 (v, direction). The TADA modules call
    # torch.nn.utils.parametrizations.weight_norm(module) without overriding dim,
    # so dim=0 for both Conv1d weights (out, in, k) and ConvTranspose1d weights
    # (in, out, k). The actual weight is w = g * v / ||v|| over every dimension
    # except dim 0. We compute this at conversion time and store a single .weight
    # tensor in the GGUF.
    wn_pairs = {}  # prefix → {"g": tensor, "v": tensor}
    for hf_name in sorted(name_to_idx.keys()):
        if ".parametrizations.weight.original0" in hf_name:
            prefix = hf_name.replace(".parametrizations.weight.original0", "")
            wn_pairs.setdefault(prefix, {})["g"] = hf_name
        elif ".parametrizations.weight.original1" in hf_name:
            prefix = hf_name.replace(".parametrizations.weight.original1", "")
            wn_pairs.setdefault(prefix, {})["v"] = hf_name

    materialized = {}  # hf_prefix → numpy weight
    for prefix, pair in wn_pairs.items():
        if "g" in pair and "v" in pair:
            g = handles[name_to_idx[pair["g"]]].get_tensor(pair["g"]).to(torch.float32)
            v = handles[name_to_idx[pair["v"]]].get_tensor(pair["v"]).to(torch.float32)
            norm_dims = tuple(range(1, v.ndim))
            v_norm = torch.linalg.vector_norm(v, dim=norm_dims, keepdim=True)
            wt = v * (g / (v_norm + 1e-12))
            materialized[prefix] = wt.numpy()
            print(f"  WN materialized: {prefix}  g={list(g.shape)}  v={list(v.shape)} → w={list(wt.shape)}")

    # ── Map tensors ──
    n_mapped = 0
    n_skipped = 0
    skip_wn_raw = set()
    for prefix in wn_pairs:
        if "g" in wn_pairs[prefix]:
            skip_wn_raw.add(wn_pairs[prefix]["g"])
        if "v" in wn_pairs[prefix]:
            skip_wn_raw.add(wn_pairs[prefix]["v"])

    for hf_name in sorted(name_to_idx.keys()):
        # Skip raw weight-norm tensors (replaced by materialized weights)
        if hf_name in skip_wn_raw:
            n_skipped += 1
            continue

        # Skip the per-layer precomputed block-attention masks: 6 × (8192, 8192)
        # F16 = ~805 MB (76% of the codec!) of dead weight. The runtime never
        # binds them — it rebuilds attn_mask from token_masks per decode (see
        # tada_codec.cpp build_decode_graph). Dropping them shrinks the codec
        # GGUF ~1055 → ~250 MB with byte-identical output. (rope_freqs IS used —
        # don't skip it.)
        if hf_name.endswith("_precomputed_mask"):
            n_skipped += 1
            continue

        gn = map_tensor_name(hf_name)
        if gn is None:
            n_skipped += 1
            continue

        t = handles[name_to_idx[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()

        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        if n_mapped <= 30 or n_mapped % 50 == 0:
            print(f"  [{n_mapped}] {gn:60s} {t.shape}  {t.dtype}")

    # Add materialized weight-norm tensors
    for prefix, wt in materialized.items():
        # Map the prefix through the same name mapping, adding ".weight"
        gn = map_tensor_name(prefix + ".weight")
        if gn is None:
            gn = map_tensor_name(prefix + ".") # try with trailing dot
            if gn:
                gn = gn.rstrip(".") + ".weight"
        if gn is None:
            print(f"  WARN: could not map WN prefix {prefix}", file=sys.stderr)
            continue
        t = np.ascontiguousarray(wt.astype(out_dtype))
        w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        print(f"  [{n_mapped}] {gn:60s} {t.shape}  {t.dtype}  (WN materialized)")

    print(f"\nMapped: {n_mapped}, skipped: {n_skipped}")
    print(f"Writing {out_path}…")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / 1e9
    print(f"Done: {out_path} ({sz:.2f} GB)")


if __name__ == "__main__":
    main()

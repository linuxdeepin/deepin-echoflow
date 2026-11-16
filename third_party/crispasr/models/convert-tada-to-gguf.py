#!/usr/bin/env python3
"""
Convert any HumeAI/tada-* HuggingFace safetensors → GGUF F16 for the
CrispASR `tada-tts` backend.

Supports both TADA variants:
  tada-3b-ml: Llama-3.2-3B backbone, num_time_classes=1024, head_layers=4
  tada-1b:    Llama-3.2-1B backbone, num_time_classes=256,  head_layers=6,
              tie_word_embeddings=True (lm_head shares embed_tokens weight)

Common TADA-specific layers (all variants):
  - acoustic_proj: Linear(512, hidden_size)
  - time_start_embed: Embedding(num_time_classes, hidden_size)
  - time_end_embed: Embedding(num_time_classes, hidden_size)
  - acoustic_mask_emb: Embedding(2, hidden_size)
  - bottleneck_proj: Linear(hidden_size, bottleneck_dim) or Identity
  - prediction_head: VibeVoiceDiffusionHead (flow matching)

Usage:
    python models/convert-tada-to-gguf.py \\
        --input HumeAI/tada-3b-ml \\
        --output tada-tts-3b-ml-f16.gguf

    python models/convert-tada-to-gguf.py \\
        --input HumeAI/tada-1b \\
        --output tada-tts-1b-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import math
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
        "*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model",
    ]))


# ---------------------------------------------------------------------------
# Tensor name remapping — HF → GGUF
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    if hf_name.endswith("num_batches_tracked"):
        return None
    if hf_name.endswith(".rotary_emb.inv_freq"):
        return None

    n = hf_name

    # ── Llama backbone → talker.* ──
    n = n.replace("model.embed_tokens.", "talker.token_embd.")
    n = n.replace("model.norm.", "talker.output_norm.")
    n = n.replace("lm_head.", "talker.output.")
    n = n.replace("model.layers.", "talker.blk.")

    # Per-layer renames
    n = n.replace(".self_attn.q_proj.", ".attn_q.")
    n = n.replace(".self_attn.k_proj.", ".attn_k.")
    n = n.replace(".self_attn.v_proj.", ".attn_v.")
    n = n.replace(".self_attn.o_proj.", ".attn_output.")
    n = n.replace(".input_layernorm.", ".attn_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".ffn_gate.")
    n = n.replace(".mlp.up_proj.", ".ffn_up.")
    n = n.replace(".mlp.down_proj.", ".ffn_down.")

    # ── TADA-specific layers ──
    n = n.replace("acoustic_proj.", "tada.acoustic_proj.")
    n = n.replace("time_start_embed.", "tada.time_start_embd.")
    n = n.replace("time_end_embed.", "tada.time_end_embd.")
    n = n.replace("acoustic_mask_emb.", "tada.acoustic_mask_embd.")
    n = n.replace("bottleneck_proj.", "tada.bottleneck_proj.")

    # ── FM prediction head → tada.fm_head.* ──
    n = n.replace("prediction_head.noisy_images_proj.", "tada.fm_head.noisy_proj.")
    n = n.replace("prediction_head.cond_proj.", "tada.fm_head.cond_proj.")
    n = n.replace("prediction_head.t_embedder.mlp.0.", "tada.fm_head.t_emb_mlp0.")
    n = n.replace("prediction_head.t_embedder.mlp.2.", "tada.fm_head.t_emb_mlp1.")

    # FM final layer — MUST come before generic .adaLN_modulation.1. rule
    n = n.replace("prediction_head.final_layer.norm_final.", "tada.fm_head.final_norm.")
    n = n.replace("prediction_head.final_layer.linear.", "tada.fm_head.final_proj.")
    n = n.replace("prediction_head.final_layer.adaLN_modulation.1.", "tada.fm_head.final_adaln.")

    # FM head layers: prediction_head.layers.{i}.* → tada.fm_head.blk.{i}.*
    n = n.replace("prediction_head.layers.", "tada.fm_head.blk.")
    n = n.replace(".ffn.gate_proj.", ".ffn_gate.")
    n = n.replace(".ffn.up_proj.", ".ffn_up.")
    n = n.replace(".ffn.down_proj.", ".ffn_down.")
    n = n.replace(".norm.", ".norm.")
    n = n.replace(".adaLN_modulation.1.", ".adaln.")

    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert HumeAI TADA model to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. HumeAI/tada-3b-ml or HumeAI/tada-1b) or local dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16"])
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    # Derive a clean model name from the input (last path component, strip HumeAI/ prefix)
    model_slug = Path(args.input).name.replace("_", "-").lower()

    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    # ── Llama backbone config ──
    n_layers = int(cfg["num_hidden_layers"])
    d_model = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = int(cfg.get("head_dim", d_model // n_heads))
    ff_dim = int(cfg["intermediate_size"])
    vocab_size = int(cfg["vocab_size"])
    rope_theta = float(cfg.get("rope_theta", 500000.0))
    rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-5))
    max_pos = int(cfg.get("max_position_embeddings", 131072))

    # ── TADA-specific config ──
    acoustic_dim = int(cfg.get("acoustic_dim", 512))
    num_time_classes = int(cfg.get("num_time_classes", 1024))
    num_time_bits = math.ceil(math.log2(num_time_classes))
    time_dim = 2 * num_time_bits
    shift_acoustic = int(cfg.get("shift_acoustic", 5))
    head_layers = int(cfg.get("head_layers", 4))
    head_ffn_ratio = float(cfg.get("head_ffn_ratio", 3.0))
    bottleneck_dim = cfg.get("bottleneck_dim", None)
    if bottleneck_dim is not None:
        bottleneck_dim = int(bottleneck_dim)
    acoustic_mean = float(cfg.get("acoustic_mean", 0.0))
    acoustic_std = float(cfg.get("acoustic_std", 1.5))
    context_window = int(cfg.get("context_window", 8))

    # FM head hidden_size (bottleneck or full hidden_size)
    fm_hidden = bottleneck_dim if bottleneck_dim is not None else d_model
    fm_latent = acoustic_dim + time_dim  # 532

    tie_word_embeddings = bool(cfg.get("tie_word_embeddings", False))

    print(f"\n{model_slug.upper()}")
    print(f"  Talker:        {n_layers}L  hidden={d_model}  "
          f"heads={n_heads}/{n_kv_heads}  head_dim={head_dim}  "
          f"ff={ff_dim}  vocab={vocab_size}")
    print(f"  RoPE theta:    {rope_theta}  max_pos={max_pos}")
    print(f"  RMS eps:       {rms_norm_eps}")
    print(f"  Acoustic:      dim={acoustic_dim}  mean={acoustic_mean}  "
          f"std={acoustic_std}")
    print(f"  Time:          classes={num_time_classes}  bits={num_time_bits}  "
          f"dim={time_dim}")
    print(f"  FM head:       layers={head_layers}  ffn_ratio={head_ffn_ratio}  "
          f"hidden={fm_hidden}  latent={fm_latent}")
    print(f"  Shift:         {shift_acoustic}")
    print(f"  Bottleneck:    {bottleneck_dim}")
    print(f"  Tied embeds:   {tie_word_embeddings}")

    if args.outtype == "f16":
        out_dtype = np.float16
        out_qt = GGMLQuantizationType.F16
    elif args.outtype == "bf16":
        out_dtype = None
        out_qt = GGMLQuantizationType.BF16
    else:
        out_dtype = np.float32
        out_qt = GGMLQuantizationType.F32

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"no safetensors in {model_dir}")
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_idx = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            name_to_idx[k] = i
    print(f"  Safetensors:   {len(name_to_idx)} tensors in {len(st_files)} file(s)")

    # ── Tokenizer ──
    # TADA model doesn't ship its own tokenizer — it uses Llama-3.2's.
    # Try local, then download from unsloth mirror (public, ungated).
    tj = model_dir / "tokenizer.json"
    if not tj.exists():
        print("  tokenizer.json not in model dir, downloading from unsloth/Llama-3.2-1B...")
        tok_dir = Path(snapshot_download("unsloth/Llama-3.2-1B", allow_patterns=[
            "tokenizer.json", "tokenizer_config.json",
        ]))
        tj = tok_dir / "tokenizer.json"
        if not tj.exists():
            sys.exit("no tokenizer.json found — even unsloth mirror failed")
    with open(tj, encoding="utf-8") as f:
        tjd = json.load(f)

    model_block = tjd.get("model", {})
    base_vocab = model_block.get("vocab", {})
    added = tjd.get("added_tokens", [])
    max_id = 0
    if base_vocab:
        max_id = max(int(v) for v in base_vocab.values())
    if added:
        max_id = max(max_id, max(int(it["id"]) for it in added))

    toks = [""] * (max_id + 1)
    for tok, idx in base_vocab.items():
        idx = int(idx)
        if idx < len(toks):
            toks[idx] = tok
    for it in added:
        idx = int(it["id"])
        if idx < len(toks):
            toks[idx] = str(it["content"])

    merges = model_block.get("merges", [])
    if merges and isinstance(merges[0], list):
        merges = [" ".join(p) for p in merges]

    # ── Write GGUF ──
    out_path = Path(args.output)
    w = GGUFWriter(str(out_path), arch="tada-tts", use_temp_file=False)

    w.add_name(f"tada-tts-{model_slug}")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))

    # Llama backbone metadata
    u32("tada.talker.n_layers", n_layers)
    u32("tada.talker.d_model", d_model)
    u32("tada.talker.n_heads", n_heads)
    u32("tada.talker.n_kv_heads", n_kv_heads)
    u32("tada.talker.head_dim", head_dim)
    u32("tada.talker.ff_dim", ff_dim)
    u32("tada.talker.vocab_size", vocab_size)
    u32("tada.talker.max_pos", max_pos)
    f32("tada.talker.rope_theta", rope_theta)
    f32("tada.talker.rms_norm_eps", rms_norm_eps)

    # TADA-specific metadata
    u32("tada.acoustic_dim", acoustic_dim)
    u32("tada.num_time_classes", num_time_classes)
    u32("tada.num_time_bits", num_time_bits)
    u32("tada.time_dim", time_dim)
    u32("tada.shift_acoustic", shift_acoustic)
    u32("tada.head_layers", head_layers)
    f32("tada.head_ffn_ratio", head_ffn_ratio)
    u32("tada.fm_hidden", fm_hidden)
    u32("tada.fm_latent", fm_latent)
    f32("tada.acoustic_mean", acoustic_mean)
    f32("tada.acoustic_std", acoustic_std)
    u32("tada.context_window", context_window)
    if bottleneck_dim is not None:
        u32("tada.bottleneck_dim", bottleneck_dim)

    # Tokenizer
    w.add_token_list(toks)
    print(f"  Tokens:        {len(toks)} entries")
    if merges:
        w.add_token_merges(merges)
        print(f"  Merges:        {len(merges)} entries")

    # ── Map tensors ──
    n_mapped = 0
    n_skipped = 0
    unmapped = []
    for hf_name in sorted(name_to_idx.keys()):
        gn = map_tensor_name(hf_name)
        if gn is None:
            n_skipped += 1
            continue
        # With tie_word_embeddings=True there is no lm_head.weight in the safetensors;
        # the C++ runtime falls back to token_embd.weight automatically.
        if tie_word_embeddings and hf_name == "lm_head.weight":
            n_skipped += 1
            continue

        # Verify it maps to our expected prefixes
        valid_prefixes = ("talker.", "tada.")
        if not any(gn.startswith(p) for p in valid_prefixes):
            unmapped.append(f"  [WARN unmapped] {hf_name} → {gn}")
            n_skipped += 1
            continue

        t_pt = handles[name_to_idx[hf_name]].get_tensor(hf_name)
        if t_pt.ndim <= 1:
            t = t_pt.to(torch.float32).numpy()
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        elif args.outtype == "bf16":
            if t_pt.dtype == torch.bfloat16:
                t = np.ascontiguousarray(t_pt.view(torch.uint16).numpy())
            else:
                t = np.ascontiguousarray(t_pt.to(torch.bfloat16).view(torch.uint16).numpy())
            w.add_tensor(gn, t, raw_dtype=out_qt)
        else:
            t = t_pt.to(torch.float32).numpy()
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        if n_mapped <= 30 or n_mapped % 50 == 0:
            print(f"  [{n_mapped}] {gn:55s} {t.shape}  {t.dtype}")

    if unmapped:
        for msg in unmapped[:20]:
            print(msg, file=sys.stderr)
        print(f"\n  WARNING: {len(unmapped)} unmapped tensors", file=sys.stderr)

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

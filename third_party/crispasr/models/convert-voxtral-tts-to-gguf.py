#!/usr/bin/env python3
"""
Convert mistralai/Voxtral-4B-TTS-2603 (HF safetensors) → GGUF F16.

Three-component TTS model:
  A. LLM backbone (Ministral 3B, 26L)  — autoregressive semantic token prediction
  B. Acoustic FM transformer (3L)       — flow-matching ODE for acoustic FSQ tokens
  C. Voxtral codec decoder             — transposed conv + transformer → 24 kHz PCM

Weight-norm convolutions in the codec are fused at conversion time:
  w = original0 * (original1 / ||original1||)

Voice embeddings (20 presets) are embedded as tensors.

Usage:
  python models/convert-voxtral-tts-to-gguf.py \\
      --input /path/to/Voxtral-4B-TTS-2603 \\
      --output voxtral-4b-tts-f16.gguf
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")
try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")


# ---------------------------------------------------------------------------
# Tensor name remapping (HF → GGUF)
# ---------------------------------------------------------------------------

# Component A: LLM backbone (layers.{0-25}, norm, mm_audio_embeddings)
DIRECT_LLM = {
    "norm.weight": "output_norm.weight",
    "mm_audio_embeddings.tok_embeddings.weight": "token_embd.weight",
    "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight": "audio_embd.weight",
}

LLM_LAYER_PATTERNS = [
    (r"layers\.(\d+)\.attention_norm\.weight", "blk.{}.attn_norm.weight"),
    (r"layers\.(\d+)\.attention\.wq\.weight", "blk.{}.attn_q.weight"),
    (r"layers\.(\d+)\.attention\.wk\.weight", "blk.{}.attn_k.weight"),
    (r"layers\.(\d+)\.attention\.wv\.weight", "blk.{}.attn_v.weight"),
    (r"layers\.(\d+)\.attention\.wo\.weight", "blk.{}.attn_output.weight"),
    (r"layers\.(\d+)\.ffn_norm\.weight", "blk.{}.ffn_norm.weight"),
    (r"layers\.(\d+)\.feed_forward\.w1\.weight", "blk.{}.ffn_gate.weight"),
    (r"layers\.(\d+)\.feed_forward\.w2\.weight", "blk.{}.ffn_down.weight"),
    (r"layers\.(\d+)\.feed_forward\.w3\.weight", "blk.{}.ffn_up.weight"),
]

# Component B: Acoustic FM transformer (acoustic_transformer.*)
DIRECT_FM = {
    "acoustic_transformer.input_projection.weight": "fm.input_proj.weight",
    "acoustic_transformer.llm_projection.weight": "fm.llm_proj.weight",
    "acoustic_transformer.time_projection.weight": "fm.time_proj.weight",
    "acoustic_transformer.acoustic_codebook_output.weight": "fm.acoustic_output.weight",
    "acoustic_transformer.semantic_codebook_output.weight": "fm.semantic_output.weight",
    # NOTE: the checkpoint has NO semantic_codebook_output.bias (verified from the
    # safetensors header); the reference runs the greedy argmax without one. Kept as
    # an optional mapping in case a future checkpoint adds it — harmless if absent.
    "acoustic_transformer.semantic_codebook_output.bias": "fm.semantic_output.bias",
    "acoustic_transformer.norm.weight": "fm.norm.weight",
}

FM_LAYER_PATTERNS = [
    (r"acoustic_transformer\.layers\.(\d+)\.attention_norm\.weight", "fm.blk.{}.attn_norm.weight"),
    (r"acoustic_transformer\.layers\.(\d+)\.attention\.wq\.weight", "fm.blk.{}.attn_q.weight"),
    (r"acoustic_transformer\.layers\.(\d+)\.attention\.wk\.weight", "fm.blk.{}.attn_k.weight"),
    (r"acoustic_transformer\.layers\.(\d+)\.attention\.wv\.weight", "fm.blk.{}.attn_v.weight"),
    (r"acoustic_transformer\.layers\.(\d+)\.attention\.wo\.weight", "fm.blk.{}.attn_output.weight"),
    (r"acoustic_transformer\.layers\.(\d+)\.ffn_norm\.weight", "fm.blk.{}.ffn_norm.weight"),
    (r"acoustic_transformer\.layers\.(\d+)\.feed_forward\.w1\.weight", "fm.blk.{}.ffn_gate.weight"),
    (r"acoustic_transformer\.layers\.(\d+)\.feed_forward\.w2\.weight", "fm.blk.{}.ffn_down.weight"),
    (r"acoustic_transformer\.layers\.(\d+)\.feed_forward\.w3\.weight", "fm.blk.{}.ffn_up.weight"),
]

# Component C: Codec decoder (audio_tokenizer.*)
# These are handled specially due to weight-norm fusing.
# Codec decoder block structure:
#   decoder_blocks.0 = Conv1d (k=3, s=1)     → codec.dec.conv.0
#   decoder_blocks.1 = Transformer (2 layers) → codec.dec.tfm.0
#   decoder_blocks.2 = ConvTranspose1d (k=4, s=2) → codec.dec.conv.1
#   decoder_blocks.3 = Transformer (2 layers) → codec.dec.tfm.1
#   decoder_blocks.4 = ConvTranspose1d (k=4, s=2) → codec.dec.conv.2
#   decoder_blocks.5 = Transformer (2 layers) → codec.dec.tfm.2
#   decoder_blocks.6 = ConvTranspose1d (k=4, s=2) → codec.dec.conv.3
#   decoder_blocks.7 = Transformer (2 layers) → codec.dec.tfm.3
#   output_proj = Conv1d (k=7) → codec.output
#   quantizer.semantic_codebook → codec.semantic_cb


def fuse_weight_norm(raw_sd: dict, base: str) -> np.ndarray:
    """Fuse PyTorch weight_norm parametrization: w = g * v / ||v||."""
    g_key = f"{base}.parametrizations.weight.original0"
    v_key = f"{base}.parametrizations.weight.original1"
    g = raw_sd[g_key].float().numpy()  # (out_channels,) or (out_channels, 1, ...)
    v = raw_sd[v_key].float().numpy()  # full weight shape
    # Normalize v per output channel
    axes = tuple(range(1, v.ndim))
    v_norm = np.sqrt(np.sum(v * v, axis=axes, keepdims=True))
    v_norm = np.maximum(v_norm, 1e-12)
    return (g * v / v_norm).astype(np.float32)


def has_weight_norm(raw_sd: dict, base: str) -> bool:
    return f"{base}.parametrizations.weight.original0" in raw_sd


def get_weight(raw_sd: dict, base: str) -> np.ndarray:
    """Get a weight tensor, handling both weight-normed and regular cases."""
    if has_weight_norm(raw_sd, base):
        return fuse_weight_norm(raw_sd, base)
    key = f"{base}.weight"
    t = raw_sd[key]
    if "bfloat" in str(t.dtype):
        t = t.float()
    return t.numpy().astype(np.float32)


def get_bias(raw_sd: dict, base: str) -> np.ndarray | None:
    key = f"{base}.bias"
    if key not in raw_sd:
        return None
    t = raw_sd[key]
    if "bfloat" in str(t.dtype):
        t = t.float()
    return t.numpy().astype(np.float32)


def remap_name(hf_name: str) -> str | None:
    """Map HF tensor name to GGUF name. Returns None for codec (handled separately)."""
    if hf_name in DIRECT_LLM:
        return DIRECT_LLM[hf_name]
    if hf_name in DIRECT_FM:
        return DIRECT_FM[hf_name]
    for pat, tmpl in LLM_LAYER_PATTERNS:
        m = re.match(pat, hf_name)
        if m:
            return tmpl.format(m.group(1))
    for pat, tmpl in FM_LAYER_PATTERNS:
        m = re.match(pat, hf_name)
        if m:
            return tmpl.format(m.group(1))
    return None


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    """Norms, biases, 1-D tensors, small projections stay F32."""
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name:
        return True
    if "scale" in gguf_name:
        return True
    if len(shape) <= 1:
        return True
    # FM projections from/to 36-dim acoustic space are tiny — keep F32
    if "fm.input_proj" in gguf_name or "fm.acoustic_output" in gguf_name:
        return True
    if "fm.time_proj" in gguf_name:
        return True
    # Codec tensors — keep small ones F32
    if gguf_name.startswith("codec.") and max(shape) < 4096:
        return True
    return False


# ---------------------------------------------------------------------------
# Tekken tokenizer serialization (same as voxtral4b converter)
# ---------------------------------------------------------------------------

def serialize_tekken_vocab(tekken: dict) -> bytes:
    vocab = tekken["vocab"]
    out = bytearray()
    for entry in vocab:
        b = base64.b64decode(entry["token_bytes"])
        if len(b) > 65535:
            raise ValueError(f"vocab entry too long: {len(b)} bytes")
        out += struct.pack("<H", len(b))
        out += b
    return bytes(out)


# ---------------------------------------------------------------------------
# Codec decoder tensor extraction
# ---------------------------------------------------------------------------

def extract_codec_tensors(raw_sd: dict) -> dict[str, np.ndarray]:
    """Extract and fuse all codec decoder tensors."""
    out: dict[str, np.ndarray] = {}

    # Debug: list all codec tensor keys to understand the naming
    codec_keys = sorted(k for k in raw_sd if k.startswith("audio_tokenizer."))
    print(f"  {len(codec_keys)} audio_tokenizer tensors")
    # Print unique prefixes up to 3rd dot
    prefixes = set()
    for k in codec_keys:
        parts = k.split(".")
        pfx = ".".join(parts[:4]) if len(parts) >= 4 else k
        prefixes.add(pfx)
    for p in sorted(prefixes):
        print(f"    {p}")
    print()

    # Block mapping: even indices are conv layers, odd are transformer layers
    # decoder_blocks.{0,2,4,6} → conv layers (indices 0..3)
    # decoder_blocks.{1,3,5,7} → transformer layers (indices 0..3)
    conv_block_map = {0: 0, 2: 1, 4: 2, 6: 3}
    tfm_block_map = {1: 0, 3: 1, 5: 2, 7: 3}

    for hf_blk_idx in [0, 2, 4, 6]:
        conv_idx = conv_block_map[hf_blk_idx]
        # Try multiple naming patterns for the conv weight
        # Pattern 1: direct (decoder_blocks.N.weight)
        # Pattern 2: nested conv (decoder_blocks.N.conv.weight)
        # Pattern 3: weight-normed direct (decoder_blocks.N.parametrizations.weight.original0)
        # Pattern 4: weight-normed nested (decoder_blocks.N.conv.parametrizations.weight.original0)
        base_candidates = [
            f"audio_tokenizer.decoder_blocks.{hf_blk_idx}",
            f"audio_tokenizer.decoder_blocks.{hf_blk_idx}.conv",
            f"audio_tokenizer.decoder_blocks.{hf_blk_idx}.conv_block",
        ]
        found = False
        for base in base_candidates:
            if has_weight_norm(raw_sd, base) or f"{base}.weight" in raw_sd:
                w = get_weight(raw_sd, base)
                out[f"codec.dec.conv.{conv_idx}.weight"] = w
                b = get_bias(raw_sd, base)
                if b is not None:
                    out[f"codec.dec.conv.{conv_idx}.bias"] = b
                print(f"  conv[{conv_idx}] from {base}: {w.shape}")
                found = True
                break
        if not found:
            print(f"  WARNING: conv block {hf_blk_idx} not found (tried {base_candidates})")

    for hf_blk_idx in [1, 3, 5, 7]:
        tfm_idx = tfm_block_map[hf_blk_idx]
        # Each transformer block has N layers (typically 2)
        # Auto-detect the number of layers from the keys
        n_layers_detected = 0
        for k in codec_keys:
            m = __import__('re').match(
                rf"audio_tokenizer\.decoder_blocks\.{hf_blk_idx}\.layers\.(\d+)\.", k)
            if m:
                n_layers_detected = max(n_layers_detected, int(m.group(1)) + 1)
        if n_layers_detected == 0:
            print(f"  WARNING: no transformer layers found in decoder_blocks.{hf_blk_idx}")
            continue
        print(f"  tfm[{tfm_idx}] has {n_layers_detected} layers")

        for li in range(n_layers_detected):
            base = f"audio_tokenizer.decoder_blocks.{hf_blk_idx}.layers.{li}"
            pfx = f"codec.dec.tfm.{tfm_idx}.blk.{li}"

            # Attention: q/k/v/o — try wq/wk/wv/wo and q_proj/k_proj/v_proj/o_proj
            for proj in ["q", "k", "v", "o"]:
                w_key = f"{base}.attention.w{proj}.weight"
                if w_key not in raw_sd:
                    w_key = f"{base}.attention.{proj}_proj.weight"
                if w_key in raw_sd:
                    t = raw_sd[w_key]
                    if "bfloat" in str(t.dtype):
                        t = t.float()
                    out[f"{pfx}.attn_{proj}.weight"] = t.numpy().astype(np.float32)

            # QK norm
            for norm_name in ["q_norm", "k_norm"]:
                w_key = f"{base}.attention.{norm_name}.weight"
                if w_key in raw_sd:
                    t = raw_sd[w_key]
                    if "bfloat" in str(t.dtype):
                        t = t.float()
                    out[f"{pfx}.{norm_name}.weight"] = t.numpy().astype(np.float32)

            # Attention norm + FFN norm
            for norm_pos, norm_suffix in [("attention_norm", "attn_norm"), ("ffn_norm", "ffn_norm")]:
                w_key = f"{base}.{norm_pos}.weight"
                if w_key in raw_sd:
                    t = raw_sd[w_key]
                    if "bfloat" in str(t.dtype):
                        t = t.float()
                    out[f"{pfx}.{norm_suffix}.weight"] = t.numpy().astype(np.float32)

            # FFN (SwiGLU: w1=gate, w2=down, w3=up)
            for ff_name, ff_suffix in [("w1", "ffn_gate"), ("w2", "ffn_down"), ("w3", "ffn_up")]:
                w_key = f"{base}.feed_forward.{ff_name}.weight"
                if w_key in raw_sd:
                    t = raw_sd[w_key]
                    if "bfloat" in str(t.dtype):
                        t = t.float()
                    out[f"{pfx}.{ff_suffix}.weight"] = t.numpy().astype(np.float32)

            # Layer scale
            for scale_name, scale_suffix in [("attention_scale", "attn_scale"), ("ffn_scale", "ffn_scale")]:
                s_key = f"{base}.{scale_name}"
                if s_key in raw_sd:
                    t = raw_sd[s_key]
                    if "bfloat" in str(t.dtype):
                        t = t.float()
                    out[f"{pfx}.{scale_suffix}"] = t.numpy().astype(np.float32)

    # Output projection (weight-normed Conv1d k=7) — try direct and nested
    for out_base in ["audio_tokenizer.output_proj", "audio_tokenizer.output_proj.conv"]:
        if has_weight_norm(raw_sd, out_base) or f"{out_base}.weight" in raw_sd:
            w = get_weight(raw_sd, out_base)
            out["codec.output.weight"] = w
            b = get_bias(raw_sd, out_base)
            if b is not None:
                out["codec.output.bias"] = b
            print(f"  output_proj from {out_base}: {w.shape}")
            break

    # Patch projection (input conv, weight-normed) — try direct and nested
    for patch_base in ["audio_tokenizer.patch_proj", "audio_tokenizer.patch_proj.conv"]:
        if has_weight_norm(raw_sd, patch_base) or f"{patch_base}.weight" in raw_sd:
            w = get_weight(raw_sd, patch_base)
            out["codec.patch_proj.weight"] = w
            b = get_bias(raw_sd, patch_base)
            if b is not None:
                out["codec.patch_proj.bias"] = b
            print(f"  patch_proj from {patch_base}: {w.shape}")
            break

    # Semantic codebook: EMA embedding_sum / cluster_usage. The checkpoint key
    # omits the `._codebook.` segment that older converters assumed (the reference
    # vLLM-Omni/voxtral-tts.c reads `...semantic_codebook.embedding_sum`); try both
    # so this survives either layout. This tensor is REQUIRED for codec decode.
    emb_sum = usage = None
    for base in ("audio_tokenizer.quantizer.semantic_codebook",
                 "audio_tokenizer.quantizer.semantic_codebook._codebook"):
        if f"{base}.embedding_sum" in raw_sd and f"{base}.cluster_usage" in raw_sd:
            emb_sum = raw_sd[f"{base}.embedding_sum"].float().numpy()
            usage = raw_sd[f"{base}.cluster_usage"].float().numpy()
            break
    if emb_sum is not None:
        usage = np.maximum(usage, 1e-5)
        embedding = emb_sum / usage[:, np.newaxis]
        out["codec.semantic_cb.weight"] = embedding.astype(np.float32)
        print(f"  semantic codebook: {embedding.shape}")
    else:
        print("  WARNING: semantic codebook (embedding_sum/cluster_usage) NOT FOUND — "
              "codec decode will be impossible without codec.semantic_cb.weight")

    return out


# ---------------------------------------------------------------------------
# Voice embedding extraction
# ---------------------------------------------------------------------------

def extract_voice_embeddings(model_dir: Path) -> dict[str, np.ndarray]:
    """Load pre-computed voice embeddings from voice_embedding/*.pt files."""
    voice_dir = model_dir / "voice_embedding"
    if not voice_dir.is_dir():
        print("  No voice_embedding/ directory found — skipping preset voices")
        return {}

    voices = {}
    for pt_file in sorted(voice_dir.glob("*.pt")):
        name = pt_file.stem
        data = torch.load(str(pt_file), map_location="cpu", weights_only=True)
        if isinstance(data, torch.Tensor):
            arr = data.float().numpy()
        elif isinstance(data, dict):
            # Some formats store as dict with 'codes' or 'tokens' key
            for k in ["codes", "tokens", "embedding"]:
                if k in data:
                    arr = data[k].float().numpy() if isinstance(data[k], torch.Tensor) else np.array(data[k], dtype=np.float32)
                    break
            else:
                print(f"  WARNING: voice {name} has unknown dict keys: {list(data.keys())}")
                continue
        else:
            print(f"  WARNING: voice {name} has unexpected type: {type(data)}")
            continue

        voices[f"voice.{name}"] = arr.astype(np.float32)
        print(f"  voice '{name}': {arr.shape}")

    return voices


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_dir: Path, out_path: Path) -> None:
    print(f"Loading: {input_dir}")

    # Load params.json (Voxtral-TTS uses params.json, not config.json)
    params_path = input_dir / "params.json"
    if not params_path.exists():
        sys.exit(f"missing params.json in {input_dir}")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    mm = params.get("multimodal", {})
    audio_args = mm.get("audio_model_args", {})
    fm_args = audio_args.get("acoustic_transformer_args", {})
    codec_args = mm.get("audio_tokenizer_args", {})
    encoding_args = audio_args.get("audio_encoding_args", {})

    print(f"  LLM: dim={params['dim']} layers={params['n_layers']} heads={params['n_heads']}/{params['n_kv_heads']}")
    print(f"  FM:  dim={fm_args['dim']} layers={fm_args['n_layers']} heads={fm_args['n_heads']}/{fm_args['n_kv_heads']}")
    print(f"  Codec: dim={codec_args['dim']} semantic_cb={codec_args['semantic_codebook_size']} "
          f"acoustic_fsq={codec_args['acoustic_codebook_size']}×{codec_args['acoustic_dim']}")

    # Load all safetensors into memory
    sf_path = input_dir / "consolidated.safetensors"
    if not sf_path.exists():
        # Try model.safetensors or sharded
        candidates = sorted(input_dir.glob("*.safetensors"))
        if not candidates:
            sys.exit(f"no safetensors files found in {input_dir}")
        sf_path = candidates[0]

    print(f"  loading {sf_path.name} ...")
    raw_sd: dict[str, torch.Tensor] = {}
    with safe_open(str(sf_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            raw_sd[k] = f.get_tensor(k)
    print(f"  {len(raw_sd)} tensors loaded")

    # Tekken tokenizer
    tekken_path = input_dir / "tekken.json"
    if not tekken_path.exists():
        sys.exit(f"missing tekken.json in {input_dir}")
    with open(tekken_path, "r", encoding="utf-8") as f:
        tekken = json.load(f)
    tekken_cfg = tekken.get("config", {})
    n_specials = len(tekken.get("special_tokens", []))
    n_vocab = len(tekken.get("vocab", []))
    print(f"  tekken: {n_specials} specials + {n_vocab} BPE = {n_specials + n_vocab} total")

    # ----- Write GGUF -----
    print(f"Writing: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch="voxtral_tts")

    # LLM params
    writer.add_uint32("voxtral_tts.llm.n_layers", params["n_layers"])
    writer.add_uint32("voxtral_tts.llm.dim", params["dim"])
    writer.add_uint32("voxtral_tts.llm.n_heads", params["n_heads"])
    writer.add_uint32("voxtral_tts.llm.n_kv_heads", params["n_kv_heads"])
    writer.add_uint32("voxtral_tts.llm.head_dim", params["head_dim"])
    writer.add_uint32("voxtral_tts.llm.hidden_dim", params["hidden_dim"])
    writer.add_float32("voxtral_tts.llm.rope_theta", float(params["rope_theta"]))
    writer.add_float32("voxtral_tts.llm.norm_eps", float(params["norm_eps"]))
    writer.add_uint32("voxtral_tts.llm.vocab_size", params["vocab_size"])
    writer.add_bool("voxtral_tts.llm.tied_embeddings", bool(params.get("tied_embeddings", True)))

    # FM transformer params
    writer.add_uint32("voxtral_tts.fm.n_layers", fm_args["n_layers"])
    writer.add_uint32("voxtral_tts.fm.dim", fm_args["dim"])
    writer.add_uint32("voxtral_tts.fm.n_heads", fm_args["n_heads"])
    writer.add_uint32("voxtral_tts.fm.n_kv_heads", fm_args["n_kv_heads"])
    writer.add_uint32("voxtral_tts.fm.head_dim", fm_args["head_dim"])
    writer.add_uint32("voxtral_tts.fm.hidden_dim", fm_args["hidden_dim"])
    writer.add_float32("voxtral_tts.fm.rope_theta", float(fm_args["rope_theta"]))
    writer.add_float32("voxtral_tts.fm.sigma", float(fm_args.get("sigma", 1e-5)))
    writer.add_float32("voxtral_tts.fm.sigma_max", float(fm_args.get("sigma_max", 1.0)))
    writer.add_uint32("voxtral_tts.fm.input_dim", fm_args.get("input_dim", fm_args["dim"]))

    # Audio encoding params
    writer.add_uint32("voxtral_tts.semantic_codebook_size", audio_args["semantic_codebook_size"])
    writer.add_uint32("voxtral_tts.acoustic_codebook_size", audio_args["acoustic_codebook_size"])
    writer.add_uint32("voxtral_tts.n_acoustic_codebook", audio_args["n_acoustic_codebook"])
    writer.add_uint32("voxtral_tts.n_codebooks", encoding_args.get("num_codebooks", 37))
    writer.add_uint32("voxtral_tts.sample_rate", encoding_args.get("sampling_rate", 24000))
    writer.add_float32("voxtral_tts.frame_rate", float(encoding_args.get("frame_rate", 12.5)))

    # Special token IDs
    writer.add_uint32("voxtral_tts.audio_token_id", audio_args.get("audio_token_id", 24))
    writer.add_uint32("voxtral_tts.begin_audio_token_id", audio_args.get("begin_audio_token_id", 25))
    writer.add_uint32("voxtral_tts.condition_dropped_token_id", audio_args.get("condition_dropped_token_id", 42))
    writer.add_uint32("voxtral_tts.bos_token_id", mm.get("bos_token_id", 1))

    # Codec decoder params
    writer.add_uint32("voxtral_tts.codec.dim", codec_args["dim"])
    writer.add_uint32("voxtral_tts.codec.hidden_dim", codec_args["hidden_dim"])
    writer.add_uint32("voxtral_tts.codec.n_heads", codec_args["n_heads"])
    writer.add_uint32("voxtral_tts.codec.n_kv_heads", codec_args["n_kv_heads"])
    writer.add_uint32("voxtral_tts.codec.head_dim", codec_args["head_dim"])
    writer.add_uint32("voxtral_tts.codec.semantic_dim", codec_args["semantic_dim"])
    writer.add_uint32("voxtral_tts.codec.acoustic_dim", codec_args["acoustic_dim"])
    writer.add_uint32("voxtral_tts.codec.patch_size", codec_args["pretransform_patch_size"])
    writer.add_uint32("voxtral_tts.codec.patch_proj_kernel", codec_args["patch_proj_kernel_size"])
    writer.add_uint32("voxtral_tts.codec.attn_window", codec_args.get("attn_sliding_window_size", 16))
    writer.add_float32("voxtral_tts.codec.norm_eps", float(codec_args.get("norm_eps", 0.01)))
    writer.add_float32("voxtral_tts.codec.qk_norm_eps", float(codec_args.get("qk_norm_eps", 1e-6)))
    writer.add_bool("voxtral_tts.codec.qk_norm", bool(codec_args.get("qk_norm", True)))
    writer.add_bool("voxtral_tts.codec.layer_scale", bool(codec_args.get("layer_scale", True)))
    writer.add_string("voxtral_tts.codec.conv_strides", codec_args.get("decoder_convs_strides_str", "1,2,2,2"))
    writer.add_string("voxtral_tts.codec.conv_kernels", codec_args.get("decoder_convs_kernels_str", "3,4,4,4"))
    writer.add_string("voxtral_tts.codec.tfm_lengths", codec_args.get("decoder_transformer_lengths_str", "2,2,2,2"))

    # Voice name → index mapping
    voice_map = codec_args.get("voice", {})
    if voice_map:
        writer.add_array("voxtral_tts.voice_names", list(voice_map.keys()))

    # Tekken tokenizer
    writer.add_string("tokenizer.tekken.pattern", tekken_cfg.get("pattern", ""))
    specials = [s["token_str"] for s in tekken.get("special_tokens", [])]
    writer.add_array("tokenizer.tekken.specials", specials)
    vocab_blob = serialize_tekken_vocab(tekken)
    print(f"  vocab blob: {len(vocab_blob) / 1024:.1f} KB")
    vocab_f32 = np.frombuffer(vocab_blob, dtype=np.uint8).astype(np.float32)
    writer.add_tensor("tokenizer.tekken.vocab_tensor", vocab_f32)
    writer.add_uint32("tokenizer.tekken.n_specials", len(specials))
    writer.add_uint32("tokenizer.tekken.n_vocab", n_vocab)

    # ----- Component A+B tensors (LLM + FM) -----
    n_written = 0
    n_f16 = 0
    n_f32 = 0
    n_skipped = 0
    skipped_names: list[str] = []

    for hf_name in sorted(raw_sd.keys()):
        # Skip codec tensors — handled separately
        if hf_name.startswith("audio_tokenizer."):
            continue

        gguf_name = remap_name(hf_name)
        if gguf_name is None:
            n_skipped += 1
            skipped_names.append(hf_name)
            continue

        t = raw_sd[hf_name]
        if "bfloat" in str(t.dtype):
            t = t.float()
        arr = t.numpy()
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)

        if is_f32_tensor(gguf_name, arr.shape):
            arr = arr.astype(np.float32)
            n_f32 += 1
        else:
            arr = arr.astype(np.float16)
            n_f16 += 1

        writer.add_tensor(gguf_name, arr)
        n_written += 1
        if n_written <= 20 or n_written % 50 == 0:
            print(f"    [{n_written:4d}] {gguf_name:50s} {str(arr.shape):26s} {arr.dtype}")

    print(f"  LLM+FM: {n_written} tensors (F16: {n_f16}, F32: {n_f32}), skipped: {n_skipped}")

    # ----- Component C: Codec decoder tensors -----
    codec_tensors = extract_codec_tensors(raw_sd)
    n_codec = 0
    for name, arr in sorted(codec_tensors.items()):
        if is_f32_tensor(name, arr.shape):
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float16)
        writer.add_tensor(name, arr)
        n_codec += 1
        if n_codec <= 20 or n_codec % 20 == 0:
            print(f"    [C{n_codec:3d}] {name:50s} {str(arr.shape):26s} {arr.dtype}")
    print(f"  Codec: {n_codec} tensors")

    # ----- Voice embeddings -----
    voice_tensors = extract_voice_embeddings(input_dir)
    for name, arr in sorted(voice_tensors.items()):
        writer.add_tensor(name, arr.astype(np.float32))
    print(f"  Voices: {len(voice_tensors)} preset embeddings")

    if skipped_names:
        print(f"\n  skipped {len(skipped_names)} tensors:")
        for n in skipped_names[:20]:
            print(f"    {n}")
        if len(skipped_names) > 20:
            print(f"    ... and {len(skipped_names) - 20} more")

    # ----- Finalize -----
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    total = n_written + n_codec + len(voice_tensors)
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB, {total} tensors)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Voxtral-4B-TTS-2603 HF safetensors → GGUF F16"
    )
    p.add_argument("--input", required=True, type=Path, help="HF model directory")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)

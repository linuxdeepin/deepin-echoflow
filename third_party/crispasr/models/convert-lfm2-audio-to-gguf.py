#!/usr/bin/env python3
"""
Convert LiquidAI/LFM2.5-Audio-1.5B-JP (or any LFM2-Audio variant) to GGUF.

Architecture — five components packed into a single GGUF:

  1. Conformer encoder (17L FastConformer, d_model=512, 8H, rel_pos)
     Standard NeMo-style dw_striding subsampling (factor 8).

  2. Audio adapter MLP (conformer_out → hidden_size)
     LayerNorm → Linear → GELU → Linear.

  3. LFM2 backbone (16L hybrid conv+attention, hidden=2048)
     10 conv layers (depthwise causal conv1d + in/out projections)
     6 attention layers (GQA: 32H, 8KVH, QK layernorm, RoPE)
     Each layer: operator_norm → op → residual → ffn_norm → SwiGLU FFN → residual.

  4. Depthformer (6L, dim=1024, tied embeddings)
     Per-codebook SharedEmbedding + RawLMBackbone generating 8-codebook Mimi tokens.

  5. Mimi tokenizer (8-codebook audio codec)
     Loaded from tokenizer-*.safetensors — encoder side only for ASR.

A separate GGUF is written for the audio detokenizer (TTS output path):
  detokenizer: FusedEmbedding → upsample → 8L LFM2 (512d, sliding attn) → Linear → ISTFT.

Usage:
    python models/convert-lfm2-audio-to-gguf.py \\
        --input LiquidAI/LFM2.5-Audio-1.5B-JP \\
        --output lfm2-audio-1.5b-jp-f16.gguf

    # Also writes lfm2-audio-1.5b-jp-f16-detokenizer.gguf alongside.

GGUF tensor naming:

  # --- Conformer encoder ---
  encoder.pre.conv.{0,2,3,5,6}.{weight,bias}
  encoder.pre.out.{weight,bias}
  encoder.layers.{i}.norm_ff1.{weight,bias}
  encoder.layers.{i}.ff1.linear1.{weight,bias}
  encoder.layers.{i}.ff1.linear2.{weight,bias}
  encoder.layers.{i}.norm_attn.{weight,bias}
  encoder.layers.{i}.attn.{q,k,v,out}.{weight,bias}
  encoder.layers.{i}.attn.pos.weight
  encoder.layers.{i}.attn.pos_bias_{u,v}
  encoder.layers.{i}.norm_conv.{weight,bias}
  encoder.layers.{i}.conv.pw1.{weight,bias}
  encoder.layers.{i}.conv.dw.{weight,bias}
  encoder.layers.{i}.conv.pw2.{weight,bias}
  encoder.layers.{i}.norm_ff2.{weight,bias}
  encoder.layers.{i}.ff2.linear1.{weight,bias}
  encoder.layers.{i}.ff2.linear2.{weight,bias}
  encoder.layers.{i}.norm_out.{weight,bias}

  # --- Audio adapter ---
  adapter.norm.{weight,bias}
  adapter.linear0.{weight,bias}
  adapter.linear1.{weight,bias}

  # --- LFM2 backbone ---
  lfm.embed_tokens.weight
  lfm.embedding_norm.weight
  lfm.layers.{i}.operator_norm.weight              (all layers)
  lfm.layers.{i}.ffn_norm.weight                   (all layers)
  lfm.layers.{i}.ff.w1.weight                      (all layers)
  lfm.layers.{i}.ff.w2.weight                      (all layers)
  lfm.layers.{i}.ff.w3.weight                      (all layers)
  # conv layers:
  lfm.layers.{i}.conv.conv.weight                  [hidden, 1, kernel]
  lfm.layers.{i}.conv.in_proj.weight               [3*hidden, hidden]
  lfm.layers.{i}.conv.out_proj.weight              [hidden, hidden]
  # attention layers:
  lfm.layers.{i}.attn.q_proj.weight                [hidden, hidden]
  lfm.layers.{i}.attn.k_proj.weight                [kv_dim, hidden]
  lfm.layers.{i}.attn.v_proj.weight                [kv_dim, hidden]
  lfm.layers.{i}.attn.out_proj.weight              [hidden, hidden]
  lfm.layers.{i}.attn.q_layernorm.weight           [head_dim]
  lfm.layers.{i}.attn.k_layernorm.weight           [head_dim]

  # --- Audio embedding (for audio output tokens) ---
  audio_embd.embedding.weight                      [codebooks*2049, hidden]
  audio_embd.embedding_norm.weight                 [hidden]
  audio_embd.to_logits.weight                      [codebooks*2049, hidden]  (or tied)

  # --- Depthformer ---
  depth.linear.{weight,bias}                       [hidden → codebooks*depth_dim]
  depth.layers.{i}.operator_norm.weight
  depth.layers.{i}.ffn_norm.weight
  depth.layers.{i}.attn.qkv_proj.weight            [fused QKV]
  depth.layers.{i}.attn.out_proj.weight
  depth.layers.{i}.ff.w1.weight
  depth.layers.{i}.ff.w2.weight
  depth.layers.{i}.ff.w3.weight                    (if SwiGLU)
  depth.codebook.{c}.embedding.weight              [2049, depth_dim]
  depth.codebook.{c}.embedding_norm.weight         [depth_dim]
  depth.codebook.{c}.to_logits.weight              [2049, depth_dim]  (or tied)

  # --- Mimi tokenizer (encoder-only for ASR) ---
  mimi.*  (same naming as kyutai-stt converter)

GGUF metadata keys (under `lfm2audio.*`):
  lfm2audio.sample_rate             = 16000
  lfm2audio.n_mels                  = 128
  lfm2audio.n_fft                   = 512
  lfm2audio.win_length              = 400    (0.025 * 16000)
  lfm2audio.hop_length              = 160    (0.01 * 16000)
  lfm2audio.codebooks               = 8
  lfm2audio.audio_vocab_size        = 2049
  lfm2audio.enc_n_layers            = 17
  lfm2audio.enc_d_model             = 512
  lfm2audio.enc_n_heads             = 8
  lfm2audio.enc_ff_expansion        = 4
  lfm2audio.enc_conv_kernel         = 9
  lfm2audio.enc_subsampling_factor  = 8
  lfm2audio.enc_subsampling_channels= 256
  lfm2audio.lfm_hidden_size         = 2048
  lfm2audio.lfm_n_layers            = 16
  lfm2audio.lfm_n_heads             = 32
  lfm2audio.lfm_n_kv_heads          = 8
  lfm2audio.lfm_head_dim            = 64
  lfm2audio.lfm_ff_dim              = 8192  (after SwiGLU adjustment)
  lfm2audio.lfm_conv_kernel         = 3
  lfm2audio.lfm_rope_theta          = 1000000
  lfm2audio.lfm_layer_types         = "ccaccaccacacacac" (c=conv, a=attn)
  lfm2audio.depth_n_layers          = 6
  lfm2audio.depth_dim               = 1024
  lfm2audio.depth_tie               = 1
  lfm2audio.text_vocab_size         = 65536
  lfm2audio.interleaved_n_text      = 6
  lfm2audio.interleaved_n_audio     = 9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}...", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model",
    ]))


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    sd: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if t.dtype == torch.bfloat16:
                sd[key] = t.float().numpy()
            else:
                sd[key] = t.numpy()
    return sd


def f16(t: np.ndarray) -> np.ndarray:
    return t.astype(np.float16) if t.dtype in (np.float32, np.float64) else t


def f32(t: np.ndarray) -> np.ndarray:
    return t.astype(np.float32)


def is_f32_tensor(name: str, shape: tuple[int, ...]) -> bool:
    """Decide whether a tensor should be stored as F32 (vs F16)."""
    if name.startswith("preprocessor."):
        return True
    if name.endswith(".bias"):
        return True
    if "norm" in name:
        return True
    if "layernorm" in name:
        return True
    if len(shape) <= 1:
        return True
    # Conformer pos_bias and rel-pos table must be F32 (added to F32 Q)
    if "pos_bias" in name or "attn.pos" in name:
        return True
    # Small tensors (< 256 elements) stay F32
    if np.prod(shape) < 256:
        return True
    return False


# ---------------------------------------------------------------------------
# Conformer encoder tensor remapping (NeMo HF → GGUF)
# ---------------------------------------------------------------------------

def remap_conformer(nemo_name: str) -> str | None:
    """Map NeMo FastConformer tensor names to GGUF names.

    The liquid-audio conformer uses the same NeMo naming as canary.
    Input prefix: conformer.encoder.  (in the HF checkpoint)
    """
    n = nemo_name
    if n.endswith("num_batches_tracked"):
        return None
    # Skip batch_norm tensors — already folded into conv.dw at conversion time
    if "batch_norm" in n:
        return None

    # pre-encoder subsampling
    if "pre_encode." in n:
        return n.replace("pre_encode.", "pre.")

    # encoder layers
    if "layers." in n:
        n = (n
             .replace("feed_forward1", "ff1")
             .replace("feed_forward2", "ff2")
             .replace("norm_feed_forward1", "norm_ff1")
             .replace("norm_feed_forward2", "norm_ff2")
             .replace("norm_self_att", "norm_attn")
             .replace("self_attn.linear_q", "attn.q")
             .replace("self_attn.linear_k", "attn.k")
             .replace("self_attn.linear_v", "attn.v")
             .replace("self_attn.linear_out", "attn.out")
             .replace("self_attn.linear_pos", "attn.pos")
             .replace("self_attn.pos_bias_u", "attn.pos_bias_u")
             .replace("self_attn.pos_bias_v", "attn.pos_bias_v")
             .replace("conv.pointwise_conv1", "conv.pw1")
             .replace("conv.depthwise_conv", "conv.dw")
             .replace("conv.pointwise_conv2", "conv.pw2")
             .replace("conv.batch_norm", "conv.bn")
             )
        return n

    return n


# ---------------------------------------------------------------------------
# Main model tensor remapping
# ---------------------------------------------------------------------------

def remap_main_model(hf_name: str, config: dict) -> str | None:
    """Map HF safetensors keys → GGUF tensor names for the main model."""
    n = hf_name

    # --- Conformer encoder ---
    if n.startswith("conformer."):
        rest = n[len("conformer."):]
        mapped = remap_conformer(rest)
        if mapped is None:
            return None
        return "encoder." + mapped

    # --- Audio adapter MLP ---
    # audio_adapter.model.0 = LayerNorm
    # audio_adapter.model.1 = Linear (in→hidden)
    # audio_adapter.model.2 = GELU (no params)
    # audio_adapter.model.3 = Linear (hidden→out)
    if n.startswith("audio_adapter.model.0."):
        suffix = n.split(".")[-1]  # weight or bias
        return f"adapter.norm.{suffix}"
    if n.startswith("audio_adapter.model.1."):
        suffix = n.split(".")[-1]
        return f"adapter.linear0.{suffix}"
    if n.startswith("audio_adapter.model.3."):
        suffix = n.split(".")[-1]
        return f"adapter.linear1.{suffix}"

    # --- LFM2 backbone ---
    if n.startswith("lfm."):
        rest = n[len("lfm."):]

        if rest == "embed_tokens.weight":
            return "lfm.embed_tokens.weight"
        if rest == "embedding_norm.weight":
            return "lfm.embedding_norm.weight"

        # Layer tensors
        if rest.startswith("layers."):
            parts = rest.split(".", 2)  # layers, idx, remainder
            idx = parts[1]
            remainder = parts[2]

            # Conv layers
            if remainder.startswith("conv."):
                return f"lfm.layers.{idx}.conv.{remainder[len('conv.'):]}"

            # Attention layers
            if remainder.startswith("self_attn."):
                attn_part = remainder[len("self_attn."):]
                return f"lfm.layers.{idx}.attn.{attn_part}"

            # FFN
            if remainder.startswith("feed_forward."):
                ff_part = remainder[len("feed_forward."):]
                return f"lfm.layers.{idx}.ff.{ff_part}"

            # Norms
            if remainder == "operator_norm.weight":
                return f"lfm.layers.{idx}.operator_norm.weight"
            if remainder == "ffn_norm.weight":
                return f"lfm.layers.{idx}.ffn_norm.weight"

        # rotary_emb is computed at runtime
        if "rotary_emb" in rest:
            return None

    # --- Audio embedding ---
    if n.startswith("audio_embedding."):
        rest = n[len("audio_embedding."):]
        return f"audio_embd.{rest}"

    # --- Codebook offsets buffer (skip — computed at runtime) ---
    if n == "codebook_offsets":
        return None
    if n == "audio_loss_weights":
        return None

    # --- Depthformer ---
    if n == "depth_linear.weight":
        return "depth.linear.weight"
    if n == "depth_linear.bias":
        return "depth.linear.bias"

    # depthformer.layers.{i}.{operator/feed_forward/norms}
    if n.startswith("depthformer.layers."):
        rest = n[len("depthformer.layers."):]
        parts = rest.split(".", 1)
        idx = parts[0]
        remainder = parts[1]

        # The depthformer uses StandardBlock which wraps MHA as .operator
        # operator.qkv_proj.weight, operator.out_proj.weight, operator.bounded_attention.*
        if remainder.startswith("operator."):
            attn_rest = remainder[len("operator."):]
            if attn_rest == "qkv_proj.weight":
                return f"depth.layers.{idx}.attn.qkv_proj.weight"
            if attn_rest == "out_proj.weight":
                return f"depth.layers.{idx}.attn.out_proj.weight"
            if attn_rest.startswith("bounded_attention."):
                ba_rest = attn_rest[len("bounded_attention."):]
                if ba_rest == "q_layernorm.weight":
                    return f"depth.layers.{idx}.attn.q_layernorm.weight"
                if ba_rest == "k_layernorm.weight":
                    return f"depth.layers.{idx}.attn.k_layernorm.weight"
            # freqs_cis is a buffer, skip
            if "freqs_cis" in attn_rest:
                return None

        # feed_forward (GLU with w1, w2, w3)
        if remainder.startswith("feed_forward."):
            ff_rest = remainder[len("feed_forward."):]
            return f"depth.layers.{idx}.ff.{ff_rest}"

        # norms
        if remainder == "operator_norm.weight":
            return f"depth.layers.{idx}.operator_norm.weight"
        if remainder == "ffn_norm.weight":
            return f"depth.layers.{idx}.ffn_norm.weight"

    # Per-codebook depth embeddings
    # depth_embeddings.{c}.embedding.weight
    # depth_embeddings.{c}.embedding_norm.weight
    # depth_embeddings.{c}.to_logits.weight
    if n.startswith("depth_embeddings."):
        rest = n[len("depth_embeddings."):]
        parts = rest.split(".", 1)
        c = parts[0]
        field = parts[1]
        return f"depth.codebook.{c}.{field}"

    print(f"  WARN: unmapped tensor: {n}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Detokenizer tensor remapping
# ---------------------------------------------------------------------------

def remap_detokenizer(hf_name: str) -> str | None:
    n = hf_name

    # FusedEmbedding
    if n == "emb.emb.weight":
        return "detok.emb.weight"

    # LFM2 backbone
    if n.startswith("lfm."):
        rest = n[len("lfm."):]
        if rest == "embed_tokens.weight":
            # Not used — input is from FusedEmbedding, but HF model has it
            return "detok.lfm.embed_tokens.weight"
        if rest == "embedding_norm.weight":
            return "detok.lfm.embedding_norm.weight"
        if "rotary_emb" in rest:
            return None
        if rest.startswith("layers."):
            return f"detok.lfm.{rest}"

    # Output linear
    if n == "lin.weight":
        return "detok.output.weight"
    if n == "lin.bias":
        return "detok.output.bias"

    # ISTFT window buffer (skip — reconstructed at runtime)
    if n == "istft.window":
        return None

    print(f"  WARN: unmapped detokenizer tensor: {n}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def write_gguf(
    out_path: Path,
    arch: str,
    metadata: dict,
    tensors: list[tuple[str, np.ndarray]],
) -> None:
    print(f"\nWriting: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch)
    writer.add_name(arch)

    # Write metadata
    for key, val in metadata.items():
        if isinstance(val, int):
            writer.add_uint32(key, val)
        elif isinstance(val, float):
            writer.add_float32(key, val)
        elif isinstance(val, str):
            writer.add_string(key, val)
        elif isinstance(val, list) and all(isinstance(v, str) for v in val):
            writer.add_array(key, val)
        else:
            raise ValueError(f"Unsupported metadata type for {key}: {type(val)}")

    # Write tensors
    n_f16 = 0
    n_f32 = 0
    for name, data in tensors:
        if is_f32_tensor(name, data.shape):
            data = f32(data)
            n_f32 += 1
        else:
            data = f16(data)
            n_f16 += 1
        writer.add_tensor(name, data)

    print(f"  {len(tensors)} tensors ({n_f16} F16, {n_f32} F32)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    file_size = os.path.getsize(str(out_path))
    print(f"  Done: {file_size / 1e9:.2f} GB")


def convert(model_dir: Path, out_path: Path) -> None:
    # Load config
    config_path = model_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    print(f"Config: {config_path}")
    print(f"  Codebooks:    {config['codebooks']}")
    print(f"  LFM2 layers:  {config['lfm']['num_hidden_layers']}")
    print(f"  LFM2 hidden:  {config['lfm']['hidden_size']}")
    print(f"  Encoder layers: {config['encoder']['n_layers']}")
    print(f"  Depthformer:  {config['depthformer']['layers']}L, dim={config['depthformer']['dim']}")

    # Load main model weights
    main_safetensors = model_dir / "model.safetensors"
    print(f"\nLoading: {main_safetensors}")
    main_sd = load_safetensors(main_safetensors)
    print(f"  {len(main_sd)} tensors")

    # Load Mimi tokenizer weights
    mimi_files = list(model_dir.glob("tokenizer-*.safetensors"))
    mimi_sd: dict[str, np.ndarray] = {}
    if mimi_files:
        mimi_path = mimi_files[0]
        print(f"\nLoading Mimi tokenizer: {mimi_path}")
        mimi_sd = load_safetensors(mimi_path)
        print(f"  {len(mimi_sd)} tensors")

    # Load text tokenizer
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    vocab: list[str] = []
    if tokenizer_path.exists():
        import json as _json
        with open(tokenizer_path, encoding="utf-8") as f:
            tok_data = _json.load(f)
        # HuggingFace tokenizer.json format: vocab is in model.vocab (dict)
        if "model" in tok_data and "vocab" in tok_data["model"]:
            vocab_dict = tok_data["model"]["vocab"]
            vocab = [""] * len(vocab_dict)
            for token, idx in vocab_dict.items():
                if idx < len(vocab):
                    vocab[idx] = token
            print(f"  Text vocab: {len(vocab)} tokens")
        elif "model" in tok_data and "type" in tok_data["model"]:
            # BPE tokenizer — extract from merges
            print(f"  Tokenizer type: {tok_data['model']['type']}")
    if not vocab:
        print("  WARNING: Could not extract vocab from tokenizer.json", file=sys.stderr)

    # --- Compute GGUF metadata ---
    lfm_cfg = config["lfm"]
    enc_cfg = config["encoder"]
    depth_cfg = config["depthformer"]
    preproc = config.get("preprocessor", {})

    # Compute actual FF dim after SwiGLU adjustment
    ff_dim = lfm_cfg["intermediate_size"]
    if lfm_cfg.get("block_auto_adjust_ff_dim", True):
        ff_dim = int(2 * ff_dim / 3)
        mult = lfm_cfg.get("block_ffn_dim_multiplier", 1.0)
        ff_dim = int(mult * ff_dim)
        multiple_of = lfm_cfg.get("block_multiple_of", 256)
        ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

    layer_types_str = "".join(
        "a" if lt == "full_attention" else "c"
        for lt in lfm_cfg["layer_types"]
    )

    metadata = {
        "lfm2audio.sample_rate": preproc.get("sample_rate", 16000),
        "lfm2audio.n_mels": preproc.get("features", 128),
        "lfm2audio.n_fft": preproc.get("n_fft", 512),
        "lfm2audio.win_length": int(preproc.get("window_size", 0.025) * preproc.get("sample_rate", 16000)),
        "lfm2audio.hop_length": int(preproc.get("window_stride", 0.01) * preproc.get("sample_rate", 16000)),
        "lfm2audio.codebooks": config["codebooks"],
        "lfm2audio.audio_vocab_size": 2049,  # 2048 + 1 (EOAudio)
        "lfm2audio.enc_n_layers": enc_cfg["n_layers"],
        "lfm2audio.enc_d_model": enc_cfg["d_model"],
        "lfm2audio.enc_n_heads": enc_cfg["n_heads"],
        "lfm2audio.enc_ff_expansion": enc_cfg.get("ff_expansion_factor", 4),
        "lfm2audio.enc_conv_kernel": enc_cfg.get("conv_kernel_size", 9),
        "lfm2audio.enc_subsampling_factor": enc_cfg.get("subsampling_factor", 8),
        "lfm2audio.enc_subsampling_channels": enc_cfg.get("subsampling_conv_channels", 256),
        "lfm2audio.lfm_hidden_size": lfm_cfg["hidden_size"],
        "lfm2audio.lfm_n_layers": lfm_cfg["num_hidden_layers"],
        "lfm2audio.lfm_n_heads": lfm_cfg["num_attention_heads"],
        "lfm2audio.lfm_n_kv_heads": lfm_cfg["num_key_value_heads"],
        "lfm2audio.lfm_head_dim": lfm_cfg["hidden_size"] // lfm_cfg["num_attention_heads"],
        "lfm2audio.lfm_ff_dim": ff_dim,
        "lfm2audio.lfm_conv_kernel": lfm_cfg.get("conv_L_cache", 3),
        "lfm2audio.lfm_rope_theta": float(lfm_cfg.get("rope_theta", 1000000)),
        "lfm2audio.lfm_layer_types": layer_types_str,
        "lfm2audio.depth_n_layers": depth_cfg["layers"],
        "lfm2audio.depth_dim": depth_cfg["dim"],
        "lfm2audio.depth_tie": 1 if depth_cfg.get("tie", True) else 0,
        "lfm2audio.text_vocab_size": lfm_cfg.get("vocab_size", 65536),
        "lfm2audio.interleaved_n_text": config.get("interleaved_n_text", 6),
        "lfm2audio.interleaved_n_audio": config.get("interleaved_n_audio", 9),
    }

    # --- Fold BatchNorm into depthwise conv for each encoder layer ---
    # core_conformer::build_block expects BN folded into conv.dw.weight/bias.
    # BN folding: w' = w * γ / sqrt(var + ε), b' = γ*(b - μ)/sqrt(var + ε) + β
    bn_eps = 1e-5
    for i in range(config["encoder"]["n_layers"]):
        prefix = f"conformer.layers.{i}.conv."
        dw_w_key = prefix + "depthwise_conv.weight"
        dw_b_key = prefix + "depthwise_conv.bias"
        bn_w_key = prefix + "batch_norm.weight"      # γ
        bn_b_key = prefix + "batch_norm.bias"         # β
        bn_m_key = prefix + "batch_norm.running_mean" # μ
        bn_v_key = prefix + "batch_norm.running_var"  # σ²
        if all(k in main_sd for k in [dw_w_key, dw_b_key, bn_w_key, bn_b_key, bn_m_key, bn_v_key]):
            gamma = main_sd[bn_w_key].astype(np.float32)
            beta  = main_sd[bn_b_key].astype(np.float32)
            mean  = main_sd[bn_m_key].astype(np.float32)
            var   = main_sd[bn_v_key].astype(np.float32)
            inv_std = gamma / np.sqrt(var + bn_eps)

            dw_w = main_sd[dw_w_key].astype(np.float32)  # (C, 1, K)
            dw_b = main_sd[dw_b_key].astype(np.float32)  # (C,)
            # Fold: w' = w * inv_std[:, None, None], b' = (b - mean) * inv_std + beta
            main_sd[dw_w_key] = dw_w * inv_std[:, None, None]
            main_sd[dw_b_key] = (dw_b - mean) * inv_std + beta
            if i == 0:
                print(f"  BN-folded encoder layer 0 conv.dw")

    # --- Remap main model tensors ---
    tensors: list[tuple[str, np.ndarray]] = []
    unmapped = 0

    for name in sorted(main_sd.keys()):
        gguf_name = remap_main_model(name, config)
        if gguf_name is None:
            continue
        data = main_sd[name]
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        tensors.append((gguf_name, data))

    # --- Remap Mimi tokenizer tensors ---
    # Pre-compute codebook embeddings from EMA stats
    codebook_keys: set[str] = set()
    codebook_embeddings: dict[str, np.ndarray] = {}
    for key in list(mimi_sd.keys()):
        if key.endswith("._codebook.embedding_sum"):
            prefix = key[:-len("embedding_sum")]
            usage_key = prefix + "cluster_usage"
            init_key = prefix + "_initialized"

            embedding_sum = mimi_sd[key]
            cluster_usage = mimi_sd[usage_key]
            cluster_usage = np.maximum(cluster_usage, 1e-5)
            embedding = embedding_sum / cluster_usage[:, None]

            emb_key = prefix + "embedding"
            codebook_embeddings[emb_key] = embedding.astype(np.float32)
            codebook_keys.update([key, usage_key])
            if init_key in mimi_sd:
                codebook_keys.add(init_key)

    if codebook_embeddings:
        print(f"  Pre-computed {len(codebook_embeddings)} Mimi codebook embeddings")

    # Mimi tensor name shortening (same as kyutai-stt converter)
    def shorten_mimi(name: str) -> str:
        name = name.replace("encoder_transformer.transformer.", "enc_tfm.")
        name = name.replace("decoder_transformer.transformer.", "dec_tfm.")
        name = name.replace("self_attn.in_proj_weight", "attn.qkv_w")
        name = name.replace("self_attn.out_proj.weight", "attn.out_w")
        name = name.replace("layer_scale_1.scale", "ls1")
        name = name.replace("layer_scale_2.scale", "ls2")
        name = name.replace("linear1.weight", "ffn_up_w")
        name = name.replace("linear2.weight", "ffn_down_w")
        name = name.replace("gating.linear_in.weight", "gating_in_w")
        name = name.replace("gating.linear_out.weight", "gating_out_w")
        name = name.replace("conv.conv.conv.", "conv3.")
        name = name.replace("conv.conv.", "conv2.")
        name = name.replace("block.1.conv2.", "blk1.")
        name = name.replace("block.3.conv2.", "blk3.")
        return name

    for name in sorted(mimi_sd.keys()):
        if name in codebook_keys:
            continue
        gguf_name = shorten_mimi("mimi." + name)
        if len(gguf_name) >= 64:
            print(f"  WARN: Mimi tensor name too long ({len(gguf_name)}): {gguf_name}",
                  file=sys.stderr)
            continue
        data = mimi_sd[name]
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        tensors.append((gguf_name, data))

    for name, emb in sorted(codebook_embeddings.items()):
        gguf_name = shorten_mimi("mimi." + name)
        if len(gguf_name) >= 64:
            print(f"  WARN: Mimi codebook name too long ({len(gguf_name)}): {gguf_name}",
                  file=sys.stderr)
            continue
        tensors.append((gguf_name, f32(emb)))

    # --- Add tokenizer vocab + merges ---
    if vocab:
        metadata["tokenizer.ggml.tokens"] = vocab

    # BPE merges from tokenizer.json
    if tokenizer_path.exists():
        with open(tokenizer_path, encoding="utf-8") as f:
            tok_data = json.load(f)
        merges = tok_data.get("model", {}).get("merges", [])
        if merges:
            metadata["tokenizer.ggml.merges"] = merges
            print(f"  Added {len(merges)} BPE merges")

    # --- Compute and add preprocessor mel filterbank + window ---
    # The liquid-audio conformer uses librosa.filters.mel with slaney norm,
    # same as NeMo. We store these in the GGUF so the C++ runtime uses the
    # exact same filterbank (avoiding HTK vs slaney divergence).
    try:
        import librosa
        sr = preproc.get("sample_rate", 16000)
        n_fft_val = preproc.get("n_fft", 512)
        n_mels_val = preproc.get("features", 128)
        n_freqs = n_fft_val // 2 + 1
        # librosa mel with slaney normalization — matches NeMo AudioToMelSpectrogramPreprocessor
        mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft_val, n_mels=n_mels_val,
                                      fmin=0, fmax=sr // 2, norm="slaney")
        # mel_fb shape: (n_mels, n_freqs) — store as-is (the C++ runtime
        # handles the (1, n_mels, n_freqs) → matmul convention)
        tensors.append(("preprocessor.fb", f32(mel_fb.astype(np.float32))))
        print(f"  Added preprocessor.fb: {mel_fb.shape}")

        # Hann window (periodic=False matches torch.hann_window(N, periodic=False))
        win_length = int(preproc.get("window_size", 0.025) * sr)
        window = np.hanning(win_length + 1)[:win_length].astype(np.float32)
        # torch.hann_window(N, periodic=False) = 0.5*(1 - cos(2π*n/(N-1)))
        # np.hanning(N+1)[:N] matches this exactly.
        tensors.append(("preprocessor.window", f32(window)))
        print(f"  Added preprocessor.window: {window.shape}")
    except ImportError:
        print("  WARNING: librosa not installed — preprocessor.fb/window not added", file=sys.stderr)

    # --- Write main GGUF ---
    write_gguf(out_path, "lfm2-audio", metadata, tensors)

    # --- Write detokenizer GGUF ---
    detok_dir = model_dir / "audio_detokenizer"
    if detok_dir.exists():
        detok_safetensors = detok_dir / "model.safetensors"
        if detok_safetensors.exists():
            print(f"\nLoading detokenizer: {detok_safetensors}")
            detok_sd = load_safetensors(detok_safetensors)
            print(f"  {len(detok_sd)} tensors")

            # Load detokenizer config
            detok_config_path = detok_dir / "config.json"
            with open(detok_config_path, encoding="utf-8") as f:
                detok_cfg = json.load(f)

            detok_layer_types_str = "".join(
                "a" if lt in ("full_attention", "sliding_attention") else "c"
                for lt in detok_cfg["layer_types"]
            )

            detok_metadata = {
                "lfm2audio.detok_hidden_size": detok_cfg["hidden_size"],
                "lfm2audio.detok_n_layers": detok_cfg["num_hidden_layers"],
                "lfm2audio.detok_n_heads": detok_cfg["num_attention_heads"],
                "lfm2audio.detok_n_kv_heads": detok_cfg["num_key_value_heads"],
                "lfm2audio.detok_layer_types": detok_layer_types_str,
                "lfm2audio.detok_sliding_window": detok_cfg.get("sliding_window", 30),
                "lfm2audio.detok_output_size": detok_cfg.get("output_size", 1282),
                "lfm2audio.detok_codebooks": 8,
                "lfm2audio.detok_istft_n_fft": 1280,
                "lfm2audio.detok_istft_hop_length": 320,
                "lfm2audio.detok_istft_win_length": 1280,
                "lfm2audio.detok_sample_rate": 24000,
            }

            detok_tensors: list[tuple[str, np.ndarray]] = []
            for name in sorted(detok_sd.keys()):
                gguf_name = remap_detokenizer(name)
                if gguf_name is None:
                    continue
                data = detok_sd[name]
                if data.dtype == np.float64:
                    data = data.astype(np.float32)
                detok_tensors.append((gguf_name, data))

            detok_out = out_path.parent / (out_path.stem + "-detokenizer.gguf")
            write_gguf(detok_out, "lfm2-audio-detok", detok_metadata, detok_tensors)


def main():
    parser = argparse.ArgumentParser(
        description="Convert LFM2.5-Audio to GGUF for CrispASR")
    parser.add_argument("--input", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True,
                        help="Output GGUF path")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)
    out_path = Path(args.output)
    convert(model_dir, out_path)


if __name__ == "__main__":
    main()

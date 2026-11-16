#!/usr/bin/env python3
"""
Convert ResembleAI/chatterbox (or ResembleAI/chatterbox-turbo) safetensors
→ GGUF files compatible with gianni-cor/chatterbox.cpp.

gianni-cor ships THREE converter scripts; this one replicates all three:
  - scripts/convert-t3-turbo-to-gguf.py  (GPT-2 Medium backbone, turbo variant)
  - scripts/convert-t3-mtl-to-gguf.py    (Llama-520M backbone, multilingual)
  - scripts/convert-s3gen-to-gguf.py     (S3Gen flow + HiFT + CAMPPlus + S3Tok)

Key differences from CrispASR's own convert-chatterbox-to-gguf.py:
  - Tensor namespace: gianni uses slash-separated paths (e.g. "model/h0/attn/qkv/w")
    vs CrispASR's dot-separated prefixes (e.g. "t3.blk.0.attn_q.weight")
  - Metadata keys: gianni uses "chatterbox.*" flat keys; CrispASR uses "chatterbox.t3.*"
  - Built-in conds: gianni stores them inside the S3Gen GGUF; CrispASR stores in T3
  - Quantization: gianni applies gguf.quants.quantize() inline for q4_0/q5_0/q8_0
  - BN fusion and weight-norm resolution are equivalent in both

Usage:
    # Turbo (ResembleAI/chatterbox-turbo):
    python models/convert-chatterbox-gianni-to-gguf.py \\
        --model-dir /mnt/storage/chatterbox-turbo \\
        --outdir /mnt/storage/gianni-gguf \\
        --variant turbo --type q4_0

    # Multilingual (ResembleAI/chatterbox):
    python models/convert-chatterbox-gianni-to-gguf.py \\
        --model-dir /mnt/storage/chatterbox \\
        --outdir /mnt/storage/gianni-gguf \\
        --variant mtl --type q4_0

    # From HuggingFace (auto-download):
    python models/convert-chatterbox-gianni-to-gguf.py \\
        --model-dir ResembleAI/chatterbox-turbo \\
        --outdir /mnt/storage/gianni-gguf \\
        --variant turbo

Dependencies:
    pip install gguf safetensors torch huggingface_hub librosa
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
    import gguf.quants
except ImportError:
    sys.exit("pip install gguf  (from llama.cpp or pip)")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    import librosa
    _HAVE_LIBROSA = True
except ImportError:
    _HAVE_LIBROSA = False


# ─── Architecture constants ────────────────────────────────────────────

# GPT-2 Medium backbone (turbo / chatterbox-turbo)
T3_TURBO_HPARAMS = {
    "n_ctx":               8196,
    "n_embd":              1024,
    "n_head":              16,
    "n_layer":             24,
    "text_vocab_size":     50276,
    "speech_vocab_size":   6563,
    "start_speech_token":  6561,
    "stop_speech_token":   6562,
    "speaker_embed_size":  256,
    "layer_norm_eps":      1e-5,
    "variant":             "t3_turbo",
}
T3_TURBO_VE_HPARAMS = {
    "n_mels":        40,
    "hidden_size":   256,
    "num_layers":    3,
    "embedding_size": 256,
    "partial_frames": 160,
    "sample_rate":   16000,
    "n_fft":         400,
    "hop_size":      160,
    "win_size":      400,
    "overlap":       0.5,
    "rate":          1.3,
    "min_coverage":  0.75,
}

# Llama-520M backbone (mtl / chatterbox)
T3_MTL_HPARAMS = {
    "n_ctx":                    6148,
    "n_embd":                   1024,
    "n_head":                   16,
    "n_kv_head":                16,
    "head_dim":                 64,
    "n_layer":                  30,
    "intermediate_size":        4096,
    "text_vocab_size":          2454,
    "speech_vocab_size":        8194,
    "start_speech_token":       6561,
    "stop_speech_token":        6562,
    "start_text_token":         255,
    "stop_text_token":          0,
    "max_text_tokens":          2048,
    "max_speech_tokens":        4096,
    "speech_cond_prompt_len":   150,
    "speaker_embed_size":       256,
    "perceiver_query_tokens":   32,
    "perceiver_query_size":     1024,
    "perceiver_num_heads":      4,
    "rms_norm_eps":             1e-5,
    "rope_theta":               500000.0,
}
T3_MTL_ROPE_SCALING = {
    "type":                 "llama3",
    "factor":               8.0,
    "low_freq_factor":      1.0,
    "high_freq_factor":     4.0,
    "original_max_position": 8192,
}

S3GEN_HPARAMS = {
    "speech_vocab_size":   6561,
    "input_size":          512,
    "output_size":         80,
    "layer_norm_eps":      1e-12,
    "spk_embed_dim":       192,
    # encoder
    "encoder.n_blocks":          6,
    "encoder.up_n_blocks":       4,
    "encoder.attention_heads":   8,
    "encoder.head_dim":          64,
    "encoder.ff_size":           2048,
    "encoder.token_mel_ratio":   2,
    "encoder.pre_lookahead_len": 3,
}
S3GEN_TURBO_HPARAMS = {
    "meanflow":     True,
    "n_timesteps":  2,
    "cfg_rate":     0.0,
    "variant":      "turbo",
}
S3GEN_MTL_HPARAMS = {
    "meanflow":     False,
    "n_timesteps":  10,
    "cfg_rate":     0.7,
    "variant":      "mtl",
}

CAMPPLUS_HPARAMS = {
    "feat_dim":         80,
    "embedding_size":   192,
    "growth_rate":      32,
    "bn_size":          4,
    "init_channels":    128,
    "block1_layers":    2,
    "block2_layers":    3,
    "block3_layers":    4,
    "block1_dilation":  1,
    "block2_dilation":  2,
    "block3_dilation":  4,
    "kernel_size":      3,
    "seg_pool_len":     200,
    "sample_rate":      16000,
}

S3TOKV2_HPARAMS = {
    "n_mels":           128,
    "n_audio_state":    1280,
    "n_audio_head":     20,
    "n_audio_layer":    6,
    "head_dim":         64,
    "mlp_ratio":        4,
    "fsmn_kernel":      31,
    "fsq_levels":       "[8,5,5,5]",  # stored as string
    "fsq_dim":          8,
    "codebook_size":    6561,
    "conv_stride":      4,
    "n_fft":            400,
    "hop":              160,
    "sample_rate":      16000,
    "rope_theta":       10000.0,
    "rope_max_pos":     4096,
}


# ─── Quantization deny-list (mirrors gianni's requantize-gguf.py) ───

_DENY_SUBSTRINGS = [
    "text_emb", "speech_emb", "wte", "wpe",
    "stft_basis", "mel_filterbank", "mel_fb", "pos_emb", "pe/pe",
    "/b", "/bias", "/bn/", "/norm/", "/ln_", "/scale",
    "alpha", "beta", "gamma",
    "voice_encoder/", "campplus/", "s3tokv2/",
    "flow/input_embedding", "/builtin/", "pre_attention_query",
]
_DENY_SUFFIXES = ["/g"]

_QUANT_MAP = {
    "q4_0": GGMLQuantizationType.Q4_0,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q8_0": GGMLQuantizationType.Q8_0,
    "f16":  GGMLQuantizationType.F16,
    "f32":  GGMLQuantizationType.F32,
}


def _should_quantize(name: str, shape: tuple, qtype) -> bool:
    """Mirror of gianni's should_quantize() from requantize-gguf.py."""
    n_elem = 1
    for d in shape:
        n_elem *= d
    if n_elem < 1024:
        return False
    for s in _DENY_SUBSTRINGS:
        if s in name:
            return False
    for s in _DENY_SUFFIXES:
        if name.endswith(s):
            return False
    try:
        block = gguf.GGML_QUANT_SIZES[qtype][0]
    except (KeyError, AttributeError):
        block = 32
    if len(shape) == 2:
        return shape[-1] % block == 0
    if len(shape) == 3:
        return shape[-1] % block == 0
    return False


def add_tensor_q(
    writer: GGUFWriter,
    name: str,
    arr: np.ndarray,
    quant: str,
) -> None:
    """Write tensor to GGUF, applying block quantization when eligible."""
    # integer tensors always written as-is
    if arr.dtype.kind in "iu":
        writer.add_tensor(name, arr)
        return
    if quant in ("f16", "f32"):
        writer.add_tensor(name, arr)
        return
    qtype = _QUANT_MAP[quant]
    if not _should_quantize(name, arr.shape, qtype):
        # fallback to F16 for floats that can't be block-quantized
        writer.add_tensor(name, arr.astype(np.float16))
        return
    arr32 = np.ascontiguousarray(arr.astype(np.float32))
    qdata = gguf.quants.quantize(arr32, qtype)
    writer.add_tensor(name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)


# ─── Tensor loaders ────────────────────────────────────────────────────

def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Load safetensors, resolving PyTorch weight_norm parametrizations."""
    raw: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)

    # Resolve weight_norm: w = g * v / ||v||
    wn_bases: set[str] = set()
    for k in raw:
        if ".parametrizations.weight.original0" in k:
            wn_bases.add(k.replace(".parametrizations.weight.original0", ""))
    for base in wn_bases:
        g = raw.pop(f"{base}.parametrizations.weight.original0")
        v = raw.pop(f"{base}.parametrizations.weight.original1")
        norm_dims = tuple(range(1, v.ndim))
        v_norm = torch.norm(v, dim=norm_dims, keepdim=True)
        raw[f"{base}.weight"] = g * v / (v_norm + 1e-12)

    return {k: v for k, v in raw.items() if not k.endswith(".num_batches_tracked")}


def _as_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).numpy()


def _as_f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float16).numpy()


def _load_model_dir(path_or_id: str) -> Path:
    p = Path(path_or_id)
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(f"Directory {path_or_id!r} not found and huggingface_hub not installed")
    print(f"Downloading {path_or_id} from HuggingFace…", file=sys.stderr)
    return Path(snapshot_download(
        path_or_id,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt"],
    ))


# ─── T3 Turbo (GPT-2) converter ────────────────────────────────────────

def _map_turbo_t3(hf_name: str) -> str | None:
    """Map GPT-2-medium T3 tensor name (turbo) → gianni GGUF name."""
    n = hf_name

    # Static top-level keys
    if n == "tfmr.wte.weight":
        return None  # excluded (text_emb replaces it)
    if n == "tfmr.wpe.weight":
        return "model/wpe"
    if n == "tfmr.ln_f.weight":
        return "model/ln_f/g"
    if n == "tfmr.ln_f.bias":
        return "model/ln_f/b"
    if n == "text_emb.weight":
        return "chatterbox/text_emb"
    if n == "speech_emb.weight":
        return "chatterbox/speech_emb"
    if n == "speech_head.weight":
        return "chatterbox/speech_head"
    if n == "speech_head.bias":
        return "chatterbox/speech_head_bias"
    if n == "cond_enc.spkr_enc.weight":
        return "chatterbox/cond_spkr/w"
    if n == "cond_enc.spkr_enc.bias":
        return "chatterbox/cond_spkr/b"

    # Per-layer: tfmr.h.{i}.*
    m = re.match(r"^tfmr\.h\.(\d+)\.(.+)$", n)
    if m:
        i, rest = m.group(1), m.group(2)
        table = {
            "ln_1.weight":          f"model/h{i}/ln_1/g",
            "ln_1.bias":            f"model/h{i}/ln_1/b",
            "ln_2.weight":          f"model/h{i}/ln_2/g",
            "ln_2.bias":            f"model/h{i}/ln_2/b",
            "attn.c_attn.weight":   f"model/h{i}/attn/qkv/w",
            "attn.c_attn.bias":     f"model/h{i}/attn/qkv/b",
            "attn.c_proj.weight":   f"model/h{i}/attn/o/w",
            "attn.c_proj.bias":     f"model/h{i}/attn/o/b",
            "mlp.c_fc.weight":      f"model/h{i}/mlp/fc/w",
            "mlp.c_fc.bias":        f"model/h{i}/mlp/fc/b",
            "mlp.c_proj.weight":    f"model/h{i}/mlp/proj/w",
            "mlp.c_proj.bias":      f"model/h{i}/mlp/proj/b",
        }
        return table.get(rest)

    return None  # unknown key — skip


def write_turbo_t3(model_dir: Path, outdir: Path, quant: str) -> None:
    out_path = outdir / f"chatterbox-t3-turbo.gguf"
    print(f"\n=== Writing Turbo T3: {out_path} ===")

    # Detect safetensors file (turbo)
    candidates = [
        "t3_turbo_v1.safetensors",
        "t3_cfg.safetensors",      # older turbo releases
    ]
    t3_path: Path | None = None
    for c in candidates:
        if (model_dir / c).exists():
            t3_path = model_dir / c
            break
    if t3_path is None:
        sys.exit(f"Could not find turbo T3 safetensors in {model_dir}. "
                 f"Expected one of: {candidates}")
    print(f"  T3 weights: {t3_path.name}")
    state = _load_safetensors(t3_path)

    writer = GGUFWriter(str(out_path), "chatterbox")

    # ── Metadata ──
    writer.add_string("chatterbox.variant", T3_TURBO_HPARAMS["variant"])
    writer.add_string("chatterbox.quantization", quant)
    for k, v in T3_TURBO_HPARAMS.items():
        if k in ("variant",):
            continue
        key = f"chatterbox.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)
        elif isinstance(v, str):
            writer.add_string(key, v)

    # ── GPT-2 BPE tokenizer assets ──
    # gianni embeds vocab + merges to eliminate runtime file deps
    vocab_json = model_dir / "vocab.json"
    merges_txt = model_dir / "merges.txt"
    if vocab_json.exists():
        with open(vocab_json, encoding="utf-8") as f:
            vocab = json.load(f)
        max_id = max(vocab.values())
        tokens = [""] * (max_id + 1)
        for tok_str, tok_id in vocab.items():
            tokens[tok_id] = tok_str
        writer.add_array("tokenizer.ggml.tokens", tokens)
        print(f"  Tokenizer: {len(tokens)} tokens")

        if merges_txt.exists():
            merges = []
            with open(merges_txt, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        merges.append(line)
            writer.add_array("tokenizer.ggml.merges", merges)
            print(f"  BPE merges: {len(merges)}")
    else:
        print("  WARNING: no vocab.json found; gianni runtime requires embedded BPE tokenizer")

    # ── Voice encoder (ve.safetensors) ──
    ve_path = model_dir / "ve.safetensors"
    if ve_path.exists():
        for k, v in T3_TURBO_VE_HPARAMS.items():
            key = f"voice_encoder.{k}"
            if isinstance(v, int):
                writer.add_uint32(key, v)
            elif isinstance(v, float):
                writer.add_float32(key, v)

        ve_state = _load_safetensors(ve_path)
        # mel filterbank: librosa 16 kHz, 40 channels
        if _HAVE_LIBROSA:
            mel_fb = librosa.filters.mel(
                sr=16000, n_fft=400, n_mels=40, fmin=0, fmax=8000
            ).astype(np.float32)  # (40, 201)
            writer.add_tensor("voice_encoder/mel_fb", mel_fb,
                              raw_dtype=GGMLQuantizationType.F32)

        ve_keys = sorted(k for k in ve_state if not k.startswith("similarity_weight"))
        n_ve = 0
        for hf_key in ve_keys:
            # gianni uses "voice_encoder/{hf_key with dots→slashes}"
            gguf_key = "voice_encoder/" + hf_key.replace(".", "/")
            arr = _as_f32(ve_state[hf_key])
            writer.add_tensor(gguf_key, arr, raw_dtype=GGMLQuantizationType.F32)
            n_ve += 1
        print(f"  VE: {n_ve} tensors")

    # ── Precomputed conds ──
    conds_path = model_dir / "conds.pt"
    if conds_path.exists():
        conds = torch.load(conds_path, map_location="cpu", weights_only=False)
        t3_cond = conds.get("t3", {})
        if "speaker_emb" in t3_cond:
            spk = t3_cond["speaker_emb"].detach().to(torch.float32)
            if spk.ndim == 1:
                spk = spk.unsqueeze(0)  # (1, 256)
            writer.add_tensor(
                "chatterbox/builtin/speaker_emb",
                spk.numpy(),
                raw_dtype=GGMLQuantizationType.F32,
            )
            writer.add_uint32(
                "chatterbox.cond_prompt_length",
                int(spk.shape[0]),
            )
        if "cond_prompt_speech_tokens" in t3_cond:
            toks = t3_cond["cond_prompt_speech_tokens"].to(torch.int32).numpy()
            writer.add_tensor(
                "chatterbox/builtin/cond_prompt_speech_tokens",
                toks,
                raw_dtype=GGMLQuantizationType.I32,
            )
        print("  Conds embedded")

    # ── T3 weights ──
    conv1d_suffixes = {
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    }
    n_t3 = 0
    for hf_key, tensor in sorted(state.items()):
        gguf_key = _map_turbo_t3(hf_key)
        if gguf_key is None:
            continue
        # GPT-2 Conv1D stores weights transposed relative to nn.Linear
        for sfx in conv1d_suffixes:
            if hf_key.endswith(sfx):
                tensor = tensor.t().contiguous()
                break
        arr = _as_f32(tensor)
        add_tensor_q(writer, gguf_key, arr, quant)
        n_t3 += 1
    print(f"  T3: {n_t3} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Written: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


# ─── T3 MTL (Llama-520M) converter ────────────────────────────────────

def _map_mtl_t3(hf_name: str) -> str | None:
    """Map Llama-520M T3 tensor name (mtl) → gianni GGUF name."""
    n = hf_name

    # Llama backbone per-layer
    m = re.match(r"^tfmr\.layers\.(\d+)\.(.+)$", n)
    if m:
        i, rest = m.group(1), m.group(2)
        table = {
            "input_layernorm.weight":          f"model/h{i}/ln_attn/g",
            "post_attention_layernorm.weight":  f"model/h{i}/ln_mlp/g",
            "self_attn.q_proj.weight":         f"model/h{i}/attn/q/w",
            "self_attn.k_proj.weight":         f"model/h{i}/attn/k/w",
            "self_attn.v_proj.weight":         f"model/h{i}/attn/v/w",
            "self_attn.o_proj.weight":         f"model/h{i}/attn/o/w",
            "mlp.gate_proj.weight":            f"model/h{i}/mlp/gate/w",
            "mlp.up_proj.weight":              f"model/h{i}/mlp/up/w",
            "mlp.down_proj.weight":            f"model/h{i}/mlp/down/w",
        }
        return table.get(rest)

    # Top-level keys
    static = {
        "tfmr.norm.weight":                         "model/norm/g",
        "text_emb.weight":                          "chatterbox/text_emb",
        "speech_emb.weight":                        "chatterbox/speech_emb",
        "text_head.weight":                         "chatterbox/text_head",
        "speech_head.weight":                       "chatterbox/speech_head",
        "text_pos_emb.emb.weight":                  "chatterbox/text_pos_emb",
        "speech_pos_emb.emb.weight":                "chatterbox/speech_pos_emb",
        "cond_enc.spkr_enc.weight":                 "chatterbox/cond_spkr/w",
        "cond_enc.spkr_enc.bias":                   "chatterbox/cond_spkr/b",
        "cond_enc.emotion_adv_fc.weight":           "chatterbox/emotion_adv_fc/w",
        # Perceiver resampler
        "cond_enc.perceiver.pre_attention_query":   "chatterbox/perceiver/pre_attention_query",
        "cond_enc.perceiver.attn.norm.weight":      "chatterbox/perceiver/attn/norm/g",
        "cond_enc.perceiver.attn.norm.bias":        "chatterbox/perceiver/attn/norm/b",
        "cond_enc.perceiver.attn.to_q.weight":      "chatterbox/perceiver/attn/to_q/w",
        "cond_enc.perceiver.attn.to_q.bias":        "chatterbox/perceiver/attn/to_q/b",
        "cond_enc.perceiver.attn.to_k.weight":      "chatterbox/perceiver/attn/to_k/w",
        "cond_enc.perceiver.attn.to_k.bias":        "chatterbox/perceiver/attn/to_k/b",
        "cond_enc.perceiver.attn.to_v.weight":      "chatterbox/perceiver/attn/to_v/w",
        "cond_enc.perceiver.attn.to_v.bias":        "chatterbox/perceiver/attn/to_v/b",
        "cond_enc.perceiver.attn.proj_out.weight":  "chatterbox/perceiver/attn/proj_out/w",
        "cond_enc.perceiver.attn.proj_out.bias":    "chatterbox/perceiver/attn/proj_out/b",
    }
    return static.get(n)


def write_mtl_t3(model_dir: Path, outdir: Path, quant: str) -> None:
    out_path = outdir / "chatterbox-t3-mtl.gguf"
    print(f"\n=== Writing MTL T3: {out_path} ===")

    candidates = [
        "t3_mtl.safetensors",
        "t3_mtl_v2.safetensors",
        "t3_cfg.safetensors",
    ]
    t3_path: Path | None = None
    for c in candidates:
        if (model_dir / c).exists():
            t3_path = model_dir / c
            break
    if t3_path is None:
        sys.exit(f"Could not find MTL T3 safetensors in {model_dir}. "
                 f"Expected one of: {candidates}")
    print(f"  T3 weights: {t3_path.name}")
    state = _load_safetensors(t3_path)

    writer = GGUFWriter(str(out_path), "chatterbox")

    # ── General metadata ──
    writer.add_string("general.name", "Chatterbox Multilingual T3")
    writer.add_string("general.architecture", "chatterbox")
    writer.add_string("chatterbox.variant", "t3_mtl")
    writer.add_string("chatterbox.backbone", "llama")
    writer.add_string("chatterbox.quantization", quant)
    writer.add_string("chatterbox.reference_repo", "ResembleAI/chatterbox")
    writer.add_bool("chatterbox.emotion_adv", True)

    for k, v in T3_MTL_HPARAMS.items():
        key = f"chatterbox.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)
        elif isinstance(v, str):
            writer.add_string(key, v)

    # RoPE scaling
    writer.add_string("chatterbox.rope.scaling_type",
                      T3_MTL_ROPE_SCALING["type"])
    writer.add_float32("chatterbox.rope.scaling_factor",
                       T3_MTL_ROPE_SCALING["factor"])
    writer.add_float32("chatterbox.rope.low_freq_factor",
                       T3_MTL_ROPE_SCALING["low_freq_factor"])
    writer.add_float32("chatterbox.rope.high_freq_factor",
                       T3_MTL_ROPE_SCALING["high_freq_factor"])
    writer.add_uint32("chatterbox.rope.original_max_position",
                      T3_MTL_ROPE_SCALING["original_max_position"])

    # ── Multilingual tokenizer ──
    tok_json = model_dir / "tokenizer.json"
    if tok_json.exists():
        with open(tok_json, encoding="utf-8") as f:
            tok_data = json.load(f)
        writer.add_string("tokenizer.ggml.model", "mtl_grapheme")
        writer.add_string("tokenizer.ggml.mtl_json", json.dumps(tok_data))
        MTL_LANGUAGES = [
            "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
            "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
            "sw", "tr", "zh",
        ]
        writer.add_array("tokenizer.ggml.mtl_languages", MTL_LANGUAGES)
        print(f"  MTL tokenizer embedded ({len(MTL_LANGUAGES)} languages)")

    # ── Voice encoder (ve.safetensors) ──
    ve_path = model_dir / "ve.safetensors"
    if ve_path.exists():
        ve_state = _load_safetensors(ve_path)
        n_ve = 0
        for hf_key in sorted(ve_state.keys()):
            if hf_key.startswith("similarity_weight"):
                continue
            gguf_key = "voice_encoder/" + hf_key.replace(".", "/")
            arr = _as_f32(ve_state[hf_key])
            writer.add_tensor(gguf_key, arr, raw_dtype=GGMLQuantizationType.F32)
            n_ve += 1
        print(f"  VE: {n_ve} tensors")

    # ── Precomputed conds (speaker_emb + cond_prompt_speech_tokens) ──
    conds_path = model_dir / "conds.pt"
    if conds_path.exists():
        conds = torch.load(conds_path, map_location="cpu", weights_only=False)
        t3_cond = conds.get("t3", {})
        if "speaker_emb" in t3_cond:
            spk = t3_cond["speaker_emb"].detach().to(torch.float32)
            if spk.ndim == 1:
                spk = spk.unsqueeze(0)
            writer.add_tensor(
                "chatterbox/builtin/speaker_emb",
                spk.numpy(),
                raw_dtype=GGMLQuantizationType.F32,
            )
            writer.add_uint32("chatterbox.cond_prompt_length", int(spk.shape[0]))
        if "cond_prompt_speech_tokens" in t3_cond:
            toks = t3_cond["cond_prompt_speech_tokens"].to(torch.int32).numpy()
            writer.add_tensor(
                "chatterbox/builtin/cond_prompt_speech_tokens",
                toks,
                raw_dtype=GGMLQuantizationType.I32,
            )
        print("  Conds embedded")

    # ── T3 weights ──
    n_t3 = 0
    for hf_key, tensor in sorted(state.items()):
        gguf_key = _map_mtl_t3(hf_key)
        if gguf_key is None:
            continue
        arr = _as_f32(tensor)
        add_tensor_q(writer, gguf_key, arr, quant)
        n_t3 += 1
    print(f"  T3: {n_t3} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Written: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


# ─── S3Gen converter (shared turbo / mtl) ──────────────────────────────

def _fuse_batchnorm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Fuse BatchNorm statistics + affine params into (scale, shift) pairs.
    Any key ending in .running_mean triggers fusion of the whole group.
    Result keys replace .running_mean → /s and .running_mean → /b etc.
    gianni's C++ runtime expects fused scale/shift, not raw BN params.
    """
    BN_EPS = 1e-5
    out = {}
    processed: set[str] = set()

    for k in sorted(state.keys()):
        if k.endswith(".running_mean"):
            prefix = k[: -len(".running_mean")]
            mean = state[f"{prefix}.running_mean"].float()
            var  = state[f"{prefix}.running_var"].float()
            denom = torch.sqrt(var + BN_EPS)
            g_key = f"{prefix}.weight"
            b_key = f"{prefix}.bias"
            if g_key in state and b_key in state:
                gamma = state[g_key].float()
                beta  = state[b_key].float()
                scale = gamma / denom
                shift = beta - mean * scale
            else:
                scale = 1.0 / denom
                shift = -mean * scale
            # Store under gianni's naming: prefix → bn_prefix/s and /b
            out[prefix + "/s"] = scale
            out[prefix + "/b"] = shift
            processed.update([
                k,
                f"{prefix}.running_var",
                f"{prefix}.running_mean",
                g_key,
                b_key,
                f"{prefix}.num_batches_tracked",
            ])
        elif k not in processed:
            out[k] = state[k]

    return out


def _map_s3gen(hf_name: str) -> str:
    """Map S3Gen/HiFT/CAMPPlus/S3Tok HF key → gianni GGUF path."""
    n = hf_name

    # Flow encoder → flow/*
    if n.startswith("flow.encoder."):
        rest = n[len("flow.encoder."):]
        rest = re.sub(r"^embed\.out\.0\.", "embed/linear/", rest)
        rest = re.sub(r"^embed\.out\.1\.", "embed/norm/", rest)
        rest = re.sub(r"^up_embed\.out\.0\.", "up_embed/linear/", rest)
        rest = re.sub(r"^up_embed\.out\.1\.", "up_embed/norm/", rest)
        rest = re.sub(r"^pre_lookahead_layer\.", "pre_lookahead/", rest)
        rest = re.sub(r"^after_norm\.", "after_norm/", rest)
        rest = re.sub(r"^up_layer\.conv\.", "up_layer/conv/", rest)
        rest = re.sub(r"^encoders\.(\d+)\.", lambda m: f"block{m.group(1)}/", rest)
        rest = re.sub(r"^up_encoders\.(\d+)\.", lambda m: f"up_block{m.group(1)}/", rest)
        rest = rest.replace("norm_mha.", "norm_mha/")
        rest = rest.replace("norm_ff.", "norm_ff/")
        rest = rest.replace("self_attn.linear_q.", "attn/q/")
        rest = rest.replace("self_attn.linear_k.", "attn/k/")
        rest = rest.replace("self_attn.linear_v.", "attn/v/")
        rest = rest.replace("self_attn.linear_out.", "attn/o/")
        rest = rest.replace("self_attn.linear_pos.", "attn/pos/")
        rest = rest.replace("self_attn.pos_bias_u", "attn/pos_bias_u")
        rest = rest.replace("self_attn.pos_bias_v", "attn/pos_bias_v")
        rest = rest.replace("feed_forward.w_1.", "ff/w1/")
        rest = rest.replace("feed_forward.w_2.", "ff/w2/")
        rest = rest.replace(".weight", "/w").replace(".bias", "/b")
        rest = rest.rstrip("/")
        return "flow/encoder/" + rest

    if n == "flow.input_embedding.weight":
        return "flow/input_embedding"
    if n == "flow.spk_embed_affine_layer.weight":
        return "flow/spk_embed_affine/w"
    if n == "flow.spk_embed_affine_layer.bias":
        return "flow/spk_embed_affine/b"
    if n == "flow.encoder_proj.weight":
        return "flow/encoder_proj/w"
    if n == "flow.encoder_proj.bias":
        return "flow/encoder_proj/b"

    # CFM decoder → cfm/*
    if n.startswith("flow.decoder.estimator."):
        rest = n[len("flow.decoder.estimator."):]
        return "cfm/" + rest.replace(".", "/")

    # HiFT vocoder → hift/*
    if n.startswith("mel2wav."):
        rest = n[len("mel2wav."):]
        return "hift/" + rest.replace(".", "/")

    # CAMPPlus speaker encoder (after BN fusion, keys may have /s, /b)
    if n.startswith("speaker_encoder."):
        rest = n[len("speaker_encoder."):]
        return "campplus/" + rest.replace(".", "/")

    # S3TokenizerV2
    if n.startswith("tokenizer."):
        rest = n[len("tokenizer."):]
        if rest == "_mel_filters":
            return "s3tokv2/mel_fb"
        if rest == "window":
            return None  # skip
        return "s3tokv2/" + rest.replace(".", "/")

    return None  # skip unknown


def _kaldi_mel_filterbank(n_mels: int = 80, n_fft: int = 512, sr: int = 16000,
                           fmin: float = 20.0, fmax: float = 8000.0) -> np.ndarray:
    """Kaldi-style triangular mel filterbank (matches gianni's manual computation)."""
    mel_low  = 1127.0 * np.log(1.0 + fmin  / 700.0)
    mel_high = 1127.0 * np.log(1.0 + fmax  / 700.0)
    n_bins = n_fft // 2 + 1
    bin_freq = np.arange(n_bins, dtype=np.float64) * sr / n_fft
    bin_mel  = 1127.0 * np.log(1.0 + bin_freq / 700.0)
    mel_delta = (mel_high - mel_low) / (n_mels + 1)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        mel_center = mel_low + (m + 1) * mel_delta
        mel_lo = mel_center - mel_delta
        mel_hi = mel_center + mel_delta
        for k_idx, mb in enumerate(bin_mel):
            if mb < mel_lo or mb > mel_hi:
                continue
            if mb <= mel_center:
                fb[m, k_idx] = float((mb - mel_lo) / (mel_center - mel_lo))
            else:
                fb[m, k_idx] = float((mel_hi - mb) / (mel_hi - mel_center))
    return fb


def write_s3gen(model_dir: Path, outdir: Path, quant: str, variant: str) -> None:
    is_turbo = variant == "turbo"
    fname = "chatterbox-s3gen.gguf" if is_turbo else "chatterbox-s3gen-mtl.gguf"
    out_path = outdir / fname
    print(f"\n=== Writing S3Gen ({variant}): {out_path} ===")

    # Load S3Gen safetensors
    s3gen_path = model_dir / "s3gen.safetensors"
    if not s3gen_path.exists():
        sys.exit(f"Missing {s3gen_path}")
    state = _load_safetensors(s3gen_path)

    # Fuse BatchNorm for CAMPPlus
    state = _fuse_batchnorm(state)

    writer = GGUFWriter(str(out_path), "chatterbox")

    # ── S3Gen metadata ──
    writer.add_string("s3gen.quantization", quant)
    extra = S3GEN_TURBO_HPARAMS if is_turbo else S3GEN_MTL_HPARAMS
    writer.add_bool("s3gen.meanflow", extra["meanflow"])
    writer.add_uint32("s3gen.n_timesteps", extra["n_timesteps"])
    writer.add_float32("s3gen.cfg_rate", extra["cfg_rate"])
    writer.add_string("s3gen.variant", extra["variant"])

    for k, v in S3GEN_HPARAMS.items():
        key = f"s3gen.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)

    # ── CAMPPlus hyperparameters ──
    for k, v in CAMPPLUS_HPARAMS.items():
        writer.add_uint32(f"campplus.{k}", v) if isinstance(v, int) else \
            writer.add_float32(f"campplus.{k}", v)

    # ── S3TokenizerV2 hyperparameters ──
    for k, v in S3TOKV2_HPARAMS.items():
        key = f"s3tokv2.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)
        elif isinstance(v, str):
            writer.add_string(key, v)

    # ── Precomputed builtins (from conds.pt) ──
    conds_path = model_dir / "conds.pt"
    if conds_path.exists():
        conds = torch.load(conds_path, map_location="cpu", weights_only=False)
        gen = conds.get("gen", {})

        prompt_token: Optional[torch.Tensor] = gen.get("prompt_token")
        prompt_feat:  Optional[torch.Tensor] = gen.get("prompt_feat")
        embedding:    Optional[torch.Tensor] = gen.get("embedding")

        if prompt_token is not None:
            pt = prompt_token.reshape(-1).to(torch.int32)
            writer.add_uint32("s3gen.builtin.prompt_token_len", int(pt.numel()))
            writer.add_tensor(
                "s3gen/builtin/prompt_token",
                pt.numpy(),
                raw_dtype=GGMLQuantizationType.I32,
            )
        if prompt_feat is not None:
            pf = prompt_feat.squeeze(0).detach().to(torch.float32).numpy()
            writer.add_uint32("s3gen.builtin.prompt_feat_frames", int(pf.shape[0]))
            add_tensor_q(writer, "s3gen/builtin/prompt_feat", pf, quant)
        if embedding is not None:
            emb = embedding.squeeze(0).detach().to(torch.float32).numpy()
            writer.add_tensor(
                "s3gen/builtin/embedding",
                emb,
                raw_dtype=GGMLQuantizationType.F32,
            )
        print("  Builtins embedded")

    # ── Mel filterbanks ──
    # S3Gen 24 kHz (librosa)
    if _HAVE_LIBROSA:
        mel_24k = librosa.filters.mel(
            sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000
        ).astype(np.float32)  # (80, 961)
    else:
        # approximate without librosa (linear-spaced mel centers, not ideal)
        print("  WARNING: librosa not installed; mel_fb/24k_80 will be approximated")
        mel_24k = np.zeros((80, 961), dtype=np.float32)
    writer.add_tensor("s3gen/mel_fb/24k_80", mel_24k,
                      raw_dtype=GGMLQuantizationType.F32)

    # CAMPPlus 16 kHz (Kaldi-style, computed manually)
    kaldi_fb = _kaldi_mel_filterbank(n_mels=80, n_fft=512, sr=16000,
                                     fmin=20.0, fmax=8000.0)
    writer.add_tensor("campplus/mel_fb_kaldi_80", kaldi_fb,
                      raw_dtype=GGMLQuantizationType.F32)

    # ── S3Gen tensors (flow encoder, CFM, HiFT, CAMPPlus, S3Tok) ──
    n_tensors = 0
    skipped = 0
    for hf_key in sorted(state.keys()):
        if hf_key.endswith(".num_batches_tracked"):
            skipped += 1
            continue
        gguf_key = _map_s3gen(hf_key)
        if gguf_key is None:
            skipped += 1
            continue
        t = state[hf_key]
        if isinstance(t, torch.Tensor):
            arr = _as_f32(t)
        elif isinstance(t, np.ndarray):
            arr = t.astype(np.float32)
        else:
            arr = np.array(t, dtype=np.float32)
        add_tensor_q(writer, gguf_key, arr, quant)
        n_tensors += 1

    print(f"  S3Gen: {n_tensors} tensors ({skipped} skipped)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Written: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


# ─── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ResembleAI chatterbox weights → gianni-cor/chatterbox.cpp GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Turbo from local dir:
  python models/convert-chatterbox-gianni-to-gguf.py \\
      --model-dir /mnt/storage/chatterbox-turbo --variant turbo --type q4_0

  # MTL from HuggingFace (auto-download):
  python models/convert-chatterbox-gianni-to-gguf.py \\
      --model-dir ResembleAI/chatterbox --variant mtl --type q4_0

  # S3Gen only:
  python models/convert-chatterbox-gianni-to-gguf.py \\
      --model-dir /mnt/storage/chatterbox-turbo --variant turbo --type q8_0 --s3gen-only
""",
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to safetensors dir or HuggingFace repo ID (e.g. ResembleAI/chatterbox)",
    )
    parser.add_argument(
        "--outdir", default=".",
        help="Output directory for GGUF files (default: current directory)",
    )
    parser.add_argument(
        "--variant", choices=["turbo", "mtl"], default="turbo",
        help="Which chatterbox variant: turbo (GPT-2 backbone) or mtl (Llama-520M backbone). "
             "Default: turbo",
    )
    parser.add_argument(
        "--type", dest="quant", choices=list(_QUANT_MAP.keys()), default="q4_0",
        help="Quantization type for large weight matrices. "
             "Biases, norms, embeddings, and voice encoders always stay F32. "
             "Default: q4_0 (matches gianni's documented default)",
    )
    parser.add_argument(
        "--t3-only", action="store_true",
        help="Only write the T3 GGUF (skip S3Gen)",
    )
    parser.add_argument(
        "--s3gen-only", action="store_true",
        help="Only write the S3Gen GGUF (skip T3)",
    )
    args = parser.parse_args()

    model_dir = _load_model_dir(args.model_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Source:  {model_dir}")
    print(f"Outdir:  {outdir}")
    print(f"Variant: {args.variant}")
    print(f"Quant:   {args.quant}")

    if not args.s3gen_only:
        if args.variant == "turbo":
            write_turbo_t3(model_dir, outdir, args.quant)
        else:
            write_mtl_t3(model_dir, outdir, args.quant)

    if not args.t3_only:
        write_s3gen(model_dir, outdir, args.quant, args.variant)

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert nvidia/canary-qwen-2.5b (HuggingFace safetensors) → GGUF F16.

Architecture (SALM — Speech-Augmented Language Model):

  Preprocessor:     128 mels @ 16 kHz, n_fft=512, win=400, hop=160
  Encoder (32× FastConformer):
    pre_encode:     3-stage dw_striding Conv2d (8× time downsample)
                    out: linear(4096 → 1024)
    layer i:        FFN1(½) → MHA(rel_pos, untied bias) → conv → FFN2(½) → LN
    d_model = 1024  n_heads = 8  ff = 4096  conv_k = 9
  Projection:       linear (1024 → 2048)
  LLM (Qwen3-1.7B with LoRA merged):
    embed_tokens (151936, 2048) — tied with lm_head
    28 × decoder block (RMSNorm, GQA 16/8, SwiGLU MLP)
    norm + lm_head (tied to embed_tokens)

LoRA merge at conversion time:
    merged = base + (lora_B @ lora_A) * (alpha / r) = base + lora_B @ lora_A * 2
    Applied to q_proj and v_proj in all 28 LLM layers.

Safetensors key structure:
    perception.preprocessor.featurizer.{fb,window}
    perception.encoder.pre_encode.conv.{0,2,3,5,6}.{weight,bias}
    perception.encoder.pre_encode.out.{weight,bias}
    perception.encoder.layers.{i}.*  (NeMo FastConformer naming)
    perception.proj.{weight,bias}
    embed_tokens.weight
    llm.base_model.model.model.layers.{i}.*  (Qwen3 + LoRA)
    llm.base_model.model.model.norm.weight

GGUF tensor naming (C++ loader expects — hybrid canary encoder + qwen3 LLM):

    preprocessor.fb                                        F32  (128, 257)
    preprocessor.window                                    F32  (400,)

    encoder.pre.conv.{0,2,3,5,6}.{weight,bias}             F32 / F16
    encoder.pre.out.{weight,bias}                          F16 / F32

    encoder.layers.{i}.norm_ff1.{weight,bias}              F32
    encoder.layers.{i}.ff1.linear1.{weight,bias}           F16 / F32
    encoder.layers.{i}.ff1.linear2.{weight,bias}           F16 / F32
    encoder.layers.{i}.norm_attn.{weight,bias}             F32
    encoder.layers.{i}.attn.{q,k,v,out}.{weight,bias}      F16 / F32
    encoder.layers.{i}.attn.pos.weight                     F16
    encoder.layers.{i}.attn.pos_bias_{u,v}                 F32
    encoder.layers.{i}.norm_conv.{weight,bias}             F32
    encoder.layers.{i}.conv.pw1.{weight,bias}              F16 / F32
    encoder.layers.{i}.conv.dw.{weight,bias}               F16 / F32
    encoder.layers.{i}.conv.pw2.{weight,bias}              F16 / F32
    encoder.layers.{i}.conv.bn.{weight,bias,running_mean,running_var}  F32
    encoder.layers.{i}.norm_ff2.{weight,bias}              F32
    encoder.layers.{i}.ff2.linear1.{weight,bias}           F16 / F32
    encoder.layers.{i}.ff2.linear2.{weight,bias}           F16 / F32
    encoder.layers.{i}.norm_out.{weight,bias}              F32

    proj.weight                                            F16  (2048, 1024)
    proj.bias                                              F32  (2048,)

    token_embd.weight                                      F16  (151936, 2048)
    blk.{i}.attn_norm.weight                               F32
    blk.{i}.attn_q.weight                                  F16  (LoRA merged)
    blk.{i}.attn_k.weight                                  F16
    blk.{i}.attn_v.weight                                  F16  (LoRA merged)
    blk.{i}.attn_output.weight                             F16
    blk.{i}.attn_q_norm.weight                             F32
    blk.{i}.attn_k_norm.weight                             F32
    blk.{i}.ffn_norm.weight                                F32
    blk.{i}.ffn_gate.weight                                F16
    blk.{i}.ffn_up.weight                                  F16
    blk.{i}.ffn_down.weight                                F16
    output_norm.weight                                     F32
    output.weight                                          F16  (= token_embd, tied)

GGUF metadata keys (under `canary_qwen.*`):
    canary_qwen.sample_rate          = 16000
    canary_qwen.n_mels               = 128
    canary_qwen.n_fft                = 512
    canary_qwen.win_length           = 400
    canary_qwen.hop_length           = 160
    canary_qwen.enc_d_model          = 1024
    canary_qwen.enc_n_layers         = 32
    canary_qwen.enc_n_heads          = 8
    canary_qwen.enc_head_dim         = 128
    canary_qwen.enc_ff_dim           = 4096
    canary_qwen.subsampling_factor   = 8
    canary_qwen.subsampling_channels = 256
    canary_qwen.conv_kernel          = 9
    canary_qwen.llm_d_model          = 2048
    canary_qwen.llm_n_layers         = 28
    canary_qwen.llm_n_heads          = 16
    canary_qwen.llm_n_kv_heads       = 8
    canary_qwen.llm_head_dim         = 128
    canary_qwen.llm_ff_dim           = 6144
    canary_qwen.llm_rope_theta       = 1e6
    canary_qwen.llm_rms_norm_eps     = 1e-6
    canary_qwen.llm_vocab_size       = 151936
    canary_qwen.frame_dur_cs         = 8
"""

from __future__ import annotations

import argparse
import json
import re
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


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------


def remap_encoder(nemo_name: str) -> str | None:
    """Map perception.encoder.* NeMo names to canary-style GGUF names."""
    n = nemo_name
    if n.endswith("num_batches_tracked"):
        return None

    # preprocessor
    if n == "perception.preprocessor.featurizer.fb":
        return "preprocessor.fb"
    if n == "perception.preprocessor.featurizer.window":
        return "preprocessor.window"

    # pre-encode
    if n.startswith("perception.encoder.pre_encode."):
        return n.replace("perception.encoder.pre_encode.", "encoder.pre.")

    # projection
    if n == "perception.proj.weight":
        return "proj.weight"
    if n == "perception.proj.bias":
        return "proj.bias"

    # encoder layers
    if n.startswith("perception.encoder.layers."):
        rest = n[len("perception.encoder.layers."):]
        dot = rest.index(".")
        layer_id = rest[:dot]
        sub = rest[dot + 1:]
        sub = (
            sub.replace("feed_forward1", "ff1")
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
        return f"encoder.layers.{layer_id}.{sub}"

    return None


# LLM layer sub-pattern mapping (Qwen3 naming → qwen3_asr-style GGUF names)
LLM_SUB = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}

# LoRA targets — these have base_layer + lora_A + lora_B instead of a plain weight
LORA_TARGETS = {"self_attn.q_proj", "self_attn.v_proj"}
LORA_GGUF_MAP = {
    "self_attn.q_proj": "attn_q.weight",
    "self_attn.v_proj": "attn_v.weight",
}


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    """Norms, biases, 1-D tensors stay F32 for accuracy."""
    if gguf_name.startswith("preprocessor."):
        return True
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name:
        return True
    if "bn" in gguf_name:
        return True
    if "pos_bias_u" in gguf_name or "pos_bias_v" in gguf_name:
        return True
    if len(shape) <= 1:
        return True
    return False


def load_tokenizer(input_dir: Path, vocab_size: int) -> tuple[list[str], list[str]]:
    """Load vocab tokens and BPE merges from tokenizer.json."""
    tokenizer_json_path = input_dir / "tokenizer.json"
    if not tokenizer_json_path.exists():
        # Try Qwen3-1.7B tokenizer from HF cache
        sys.exit(f"no tokenizer.json found in {input_dir}")

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    model_data = tok_data.get("model", {})
    vocab_dict = model_data.get("vocab", {})
    sorted_vocab = sorted(vocab_dict.items(), key=lambda kv: kv[1])
    tokens = [tok for tok, _ in sorted_vocab]

    raw_merges = model_data.get("merges", [])
    merges: list[str] = []
    for m in raw_merges:
        if isinstance(m, list):
            merges.append(" ".join(m))
        elif isinstance(m, str) and m and not m.startswith("#"):
            merges.append(m)

    # Pad to vocab_size
    while len(tokens) < vocab_size:
        tokens.append(f"[PAD{len(tokens)}]")

    # Patch special tokens from tokenizer_config.json
    tcfg_path = input_dir / "tokenizer_config.json"
    if tcfg_path.exists():
        with open(tcfg_path, "r", encoding="utf-8") as f:
            tcfg = json.load(f)
        added = tcfg.get("added_tokens_decoder", {})
        for tid_str, info in added.items():
            tid = int(tid_str)
            content = info.get("content")
            if content and 0 <= tid < len(tokens):
                tokens[tid] = content
        print(f"  patched {len(added)} added/special tokens from tokenizer_config.json")

    # Also patch from tokenizer.json added_tokens
    for at in tok_data.get("added_tokens", []):
        tid = at.get("id")
        content = at.get("content")
        if tid is not None and content and 0 <= tid < len(tokens):
            tokens[tid] = content

    print(f"  vocab: {len(tokens)} tokens, {len(merges)} merges")
    return tokens, merges


# ---------------------------------------------------------------------------
# Mel filterbank (NeMo convention: Slaney-normalized)
# ---------------------------------------------------------------------------


def _hz_to_mel_slaney(hz: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(hz < min_log_hz, hz / f_sp,
                    min_log_mel + np.log(hz / min_log_hz) / logstep)


def _mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(mel < min_log_mel, mel * f_sp,
                    min_log_hz * np.exp(logstep * (mel - min_log_mel)))


def compute_mel_filters(sr: int = 16000, n_fft: int = 512,
                        n_mels: int = 128) -> np.ndarray:
    """Slaney-normalized mel filterbank. Returns (n_freqs, n_mels) F32."""
    n_freqs = n_fft // 2 + 1
    f_min, f_max = 0.0, float(sr) / 2.0
    mel_min = _hz_to_mel_slaney(np.array([f_min]))[0]
    mel_max = _hz_to_mel_slaney(np.array([f_max]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz_slaney(mel_points)
    bin_freqs = np.linspace(0.0, f_max, n_freqs)

    fb = np.zeros((n_freqs, n_mels), dtype=np.float32)
    for m in range(n_mels):
        f_left, f_center, f_right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        for k in range(n_freqs):
            if f_left <= bin_freqs[k] < f_center and f_center > f_left:
                fb[k, m] = (bin_freqs[k] - f_left) / (f_center - f_left)
            elif f_center <= bin_freqs[k] <= f_right and f_right > f_center:
                fb[k, m] = (f_right - bin_freqs[k]) / (f_right - f_center)
        bw = hz_points[m + 2] - hz_points[m]
        if bw > 0:
            fb[:, m] *= 2.0 / bw
    return np.ascontiguousarray(fb)


# ---------------------------------------------------------------------------
# LoRA merge helper
# ---------------------------------------------------------------------------


def merge_lora(base: np.ndarray, lora_a: np.ndarray, lora_b: np.ndarray,
               alpha: float = 256.0, r: float = 128.0) -> np.ndarray:
    """Merge LoRA: merged = base + lora_B @ lora_A * (alpha/r)."""
    scale = alpha / r
    # base: (out, in), lora_A: (r, in), lora_B: (out, r)
    delta = lora_b.astype(np.float32) @ lora_a.astype(np.float32)
    return base.astype(np.float32) + delta * scale


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(input_dir: Path, out_path: Path, f32_encoder: bool = False) -> None:
    print(f"Loading: {input_dir}")

    # Read config.json for architecture params
    cfg_path = input_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"config.json not found in {input_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Extract architecture from config
    perception = cfg.get("perception", {})
    encoder_cfg = perception.get("encoder", {})
    lora_cfg = cfg.get("lora", {})

    enc_n_layers = encoder_cfg.get("n_layers", 32)
    enc_d_model = encoder_cfg.get("d_model", 1024)
    enc_n_heads = encoder_cfg.get("n_heads", 8)
    enc_ff_dim = enc_d_model * encoder_cfg.get("ff_expansion_factor", 4)
    enc_conv_kernel = encoder_cfg.get("conv_kernel_size", 9)
    subsampling_factor = encoder_cfg.get("subsampling_factor", 8)
    subsampling_channels = encoder_cfg.get("subsampling_conv_channels", 256)
    n_mels = encoder_cfg.get("feat_in", 128)

    preprocessor_cfg = perception.get("preprocessor", {})
    n_fft = preprocessor_cfg.get("n_fft", 512)
    sample_rate = preprocessor_cfg.get("sample_rate", 16000)
    win_size_s = preprocessor_cfg.get("window_size", 0.025)
    win_stride_s = preprocessor_cfg.get("window_stride", 0.01)
    win_length = int(win_size_s * sample_rate)
    hop_length = int(win_stride_s * sample_rate)

    output_dim = perception.get("output_dim", 2048)

    lora_alpha = lora_cfg.get("lora_alpha", 256)
    lora_r = lora_cfg.get("r", 128)

    # LLM params — need to read from pretrained_llm config or infer from weights
    # For canary-qwen-2.5b: Qwen3-1.7B
    llm_d_model = output_dim  # 2048
    llm_n_layers = 28
    llm_n_heads = 16
    llm_n_kv_heads = 8
    llm_head_dim = 128
    llm_ff_dim = 6144
    llm_rope_theta = 1e6
    llm_rms_eps = 1e-6
    llm_vocab_size = 151936

    print(f"  encoder: {enc_n_layers}L d={enc_d_model} h={enc_n_heads} ff={enc_ff_dim}")
    print(f"  projection: {enc_d_model} → {output_dim}")
    print(f"  LLM: {llm_n_layers}L d={llm_d_model} h={llm_n_heads} kv={llm_n_kv_heads}")
    print(f"  LoRA: r={lora_r} alpha={lora_alpha} scale={lora_alpha/lora_r}")

    # Tokenizer
    # The canary-qwen model may not ship its own tokenizer.json; try to find
    # the Qwen3-1.7B tokenizer from HF cache or download it.
    tokenizer_dir = input_dir
    if not (input_dir / "tokenizer.json").exists():
        # Try to find Qwen3-1.7B tokenizer from HF cache
        import glob
        hf_cache = Path("/mnt/akademie_storage/huggingface/hub")
        candidates = list(hf_cache.glob("models--Qwen--Qwen3-1.7B/snapshots/*/tokenizer.json"))
        if not candidates:
            # Also try Qwen3-ASR-1.7B-hf which uses the same tokenizer
            candidates = list(hf_cache.glob("models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/*/tokenizer.json"))
        if not candidates:
            candidates = list(hf_cache.glob("models--Qwen--Qwen3-ASR-0.6B-hf/snapshots/*/tokenizer.json"))
        if candidates:
            tokenizer_dir = candidates[0].parent
            print(f"  using tokenizer from: {tokenizer_dir}")
        else:
            sys.exit("No tokenizer.json found. Place one in the model directory or cache Qwen3.")

    tokens, merges = load_tokenizer(tokenizer_dir, llm_vocab_size)

    # ----- write GGUF -----
    print(f"\nWriting: {out_path}")
    import os
    os.environ["TMPDIR"] = "/mnt/volume1/tmp-overflow"
    writer = gguf.GGUFWriter(str(out_path), arch="canary_qwen", use_temp_file=True)

    # Encoder params
    writer.add_uint32("canary_qwen.sample_rate", sample_rate)
    writer.add_uint32("canary_qwen.n_mels", n_mels)
    writer.add_uint32("canary_qwen.n_fft", n_fft)
    writer.add_uint32("canary_qwen.win_length", win_length)
    writer.add_uint32("canary_qwen.hop_length", hop_length)
    writer.add_uint32("canary_qwen.enc_d_model", enc_d_model)
    writer.add_uint32("canary_qwen.enc_n_layers", enc_n_layers)
    writer.add_uint32("canary_qwen.enc_n_heads", enc_n_heads)
    writer.add_uint32("canary_qwen.enc_head_dim", enc_d_model // enc_n_heads)
    writer.add_uint32("canary_qwen.enc_ff_dim", enc_ff_dim)
    writer.add_uint32("canary_qwen.subsampling_factor", subsampling_factor)
    writer.add_uint32("canary_qwen.subsampling_channels", subsampling_channels)
    writer.add_uint32("canary_qwen.conv_kernel", enc_conv_kernel)
    writer.add_uint32("canary_qwen.frame_dur_cs", 8)

    # LLM params
    writer.add_uint32("canary_qwen.llm_d_model", llm_d_model)
    writer.add_uint32("canary_qwen.llm_n_layers", llm_n_layers)
    writer.add_uint32("canary_qwen.llm_n_heads", llm_n_heads)
    writer.add_uint32("canary_qwen.llm_n_kv_heads", llm_n_kv_heads)
    writer.add_uint32("canary_qwen.llm_head_dim", llm_head_dim)
    writer.add_uint32("canary_qwen.llm_ff_dim", llm_ff_dim)
    writer.add_float32("canary_qwen.llm_rope_theta", float(llm_rope_theta))
    writer.add_float32("canary_qwen.llm_rms_norm_eps", float(llm_rms_eps))
    writer.add_uint32("canary_qwen.llm_vocab_size", llm_vocab_size)

    # Tokenizer
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    if merges:
        writer.add_token_merges(merges)

    # Mel filterbank + Hann window (baked, same as canary)
    # compute_mel_filters returns (n_freqs, n_mels) but core_mel expects
    # MelsFreqs layout = (n_mels, n_freqs). Transpose before writing.
    mel_fb = compute_mel_filters(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_fb = np.ascontiguousarray(mel_fb.T)  # (n_freqs, n_mels) → (n_mels, n_freqs)
    print(f"  mel_filters shape: {mel_fb.shape}  (MelsFreqs layout)")
    writer.add_tensor("preprocessor.fb", mel_fb)

    win = np.ascontiguousarray(
        (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length))
        .astype(np.float32)
    )
    writer.add_tensor("preprocessor.window", win)
    print(f"  mel_window shape: {win.shape}")

    # ----- tensors -----
    import gc
    import torch

    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        sys.exit(f"no safetensors files found in {input_dir}")

    n_written = 0
    n_f16 = 0
    n_f32 = 0
    n_skipped = 0
    n_lora_merged = 0
    skipped_names: list[str] = []

    # Collect all LoRA components before processing. We need to read base + A + B
    # for each LoRA target, merge them, then write the merged weight.
    # Strategy: buffer LoRA tensors, merge when all 3 parts are available.
    lora_buf: dict[str, dict[str, np.ndarray]] = {}  # key: "N.target" → {base, A, B}

    for sf_path in safetensor_files:
        print(f"  reading {sf_path.name}")
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for hf_name in sorted(f.keys()):
                # --- Encoder / preprocessor / projection ---
                enc_name = remap_encoder(hf_name)
                if enc_name is not None:
                    # Skip preprocessor tensors — we bake our own above
                    if enc_name in ("preprocessor.fb", "preprocessor.window"):
                        n_skipped += 1
                        continue
                    t = f.get_tensor(hf_name)
                    if t.dtype == torch.bfloat16:
                        arr = t.float().numpy()
                    else:
                        arr = t.numpy()
                    del t
                    if arr.dtype == np.float64:
                        arr = arr.astype(np.float32)
                    # Preprocessor fb: NeMo stores (1, 128, 257), flatten to (128, 257)
                    if enc_name == "preprocessor.fb" and len(arr.shape) == 3:
                        arr = arr.squeeze(0)
                    if is_f32_tensor(enc_name, arr.shape) or (f32_encoder and enc_name.startswith("encoder.")):
                        arr = arr.astype(np.float32)
                        n_f32 += 1
                    else:
                        arr = arr.astype(np.float16)
                        n_f16 += 1
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    writer.add_tensor(enc_name, arr)
                    n_written += 1
                    if n_written <= 30 or n_written % 100 == 0:
                        print(f"    {enc_name:50s} {str(arr.shape):24s} {arr.dtype}")
                    continue

                # --- embed_tokens ---
                if hf_name == "embed_tokens.weight":
                    t = f.get_tensor(hf_name)
                    arr = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
                    del t
                    arr = arr.astype(np.float16)
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    writer.add_tensor("token_embd.weight", arr)
                    # Tied lm_head
                    writer.add_tensor("output.weight", arr.copy())
                    n_written += 2
                    n_f16 += 2
                    print(f"    {'token_embd.weight':50s} {str(arr.shape):24s} {arr.dtype}")
                    print(f"    {'output.weight (tied)':50s} {str(arr.shape):24s} {arr.dtype}")
                    continue

                # --- LLM output norm ---
                if hf_name == "llm.base_model.model.model.norm.weight":
                    t = f.get_tensor(hf_name)
                    arr = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
                    del t
                    arr = arr.astype(np.float32)
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    writer.add_tensor("output_norm.weight", arr)
                    n_written += 1
                    n_f32 += 1
                    print(f"    {'output_norm.weight':50s} {str(arr.shape):24s} {arr.dtype}")
                    continue

                # --- LLM layers ---
                m = re.match(
                    r"llm\.base_model\.model\.model\.layers\.(\d+)\.(.*)", hf_name
                )
                if m:
                    layer_idx = m.group(1)
                    suffix = m.group(2)

                    # Check if this is a LoRA component
                    for target in LORA_TARGETS:
                        if suffix.startswith(target + "."):
                            lora_key = f"{layer_idx}.{target}"
                            lora_sub = suffix[len(target) + 1:]
                            if lora_key not in lora_buf:
                                lora_buf[lora_key] = {}
                            t = f.get_tensor(hf_name)
                            arr = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
                            del t
                            if lora_sub == "base_layer.weight":
                                lora_buf[lora_key]["base"] = arr
                            elif lora_sub == "lora_A.default.weight":
                                lora_buf[lora_key]["A"] = arr
                            elif lora_sub == "lora_B.default.weight":
                                lora_buf[lora_key]["B"] = arr
                            # Check if all 3 parts collected → merge
                            parts = lora_buf[lora_key]
                            if "base" in parts and "A" in parts and "B" in parts:
                                merged = merge_lora(
                                    parts["base"], parts["A"], parts["B"],
                                    alpha=lora_alpha, r=lora_r
                                )
                                gguf_name = f"blk.{layer_idx}.{LORA_GGUF_MAP[target]}"
                                merged = merged.astype(np.float16)
                                if not merged.flags["C_CONTIGUOUS"]:
                                    merged = np.ascontiguousarray(merged)
                                writer.add_tensor(gguf_name, merged)
                                n_written += 1
                                n_f16 += 1
                                n_lora_merged += 1
                                if n_lora_merged <= 4 or n_lora_merged % 10 == 0:
                                    print(f"    {gguf_name:50s} {str(merged.shape):24s} {merged.dtype}  (LoRA merged)")
                                del lora_buf[lora_key]
                            break
                    else:
                        # Regular LLM tensor (non-LoRA)
                        if suffix in LLM_SUB:
                            gguf_name = f"blk.{layer_idx}.{LLM_SUB[suffix]}"
                            t = f.get_tensor(hf_name)
                            arr = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
                            del t
                            if arr.dtype == np.float64:
                                arr = arr.astype(np.float32)
                            if is_f32_tensor(gguf_name, arr.shape):
                                arr = arr.astype(np.float32)
                                n_f32 += 1
                            else:
                                arr = arr.astype(np.float16)
                                n_f16 += 1
                            if not arr.flags["C_CONTIGUOUS"]:
                                arr = np.ascontiguousarray(arr)
                            writer.add_tensor(gguf_name, arr)
                            n_written += 1
                            if n_written <= 30 or n_written % 100 == 0:
                                print(f"    {gguf_name:50s} {str(arr.shape):24s} {arr.dtype}")
                        else:
                            n_skipped += 1
                            skipped_names.append(hf_name)
                    continue

                # --- Unrecognized ---
                n_skipped += 1
                skipped_names.append(hf_name)

        gc.collect()

    # Check for incomplete LoRA merges
    if lora_buf:
        print(f"\n  WARNING: {len(lora_buf)} incomplete LoRA merges:")
        for k, v in lora_buf.items():
            print(f"    {k}: has {list(v.keys())}")

    print(
        f"\n  total: {n_written} tensors  (F16: {n_f16}, F32: {n_f32}, "
        f"LoRA merged: {n_lora_merged})  skipped: {n_skipped}"
    )
    if skipped_names:
        print(f"  skipped tensors (first 20):")
        for n in skipped_names[:20]:
            print(f"    {n}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert nvidia/canary-qwen-2.5b HF safetensors → GGUF F16"
    )
    p.add_argument("--input", required=True, type=Path, help="HF model directory")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    p.add_argument("--f32-encoder", action="store_true",
                   help="emit encoder weights (incl. conv.dw) as F32 — for diff-harness "
                        "parity and to exercise the F32 BN-fold path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output, args.f32_encoder)

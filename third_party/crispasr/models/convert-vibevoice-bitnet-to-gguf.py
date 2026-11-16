#!/usr/bin/env python3
"""Convert VibeVoice-ASR-BitNet (microsoft/VibeVoice-ASR-BitNet) to GGUF.

The BitNet variant stores ternary-trained weights ({-1, 0, 1} after scaling)
in full-precision safetensors.  This script quantizes the LM projection
weights to ggml's native TQ2_0 (ternary 2-bit, 2.06 bpw) and keeps the
VAE encoder / norms / biases at F32.

Result: ~700 MB GGUF loadable by the existing vibevoice backend (no code
changes needed — ggml mul_mat handles TQ2_0 transparently).

Usage:
  python models/convert-vibevoice-bitnet-to-gguf.py \\
      --input microsoft/VibeVoice-ASR-BitNet \\
      --output /mnt/storage/gguf-models/vibevoice-asr-bitnet-tq2.gguf
"""

import argparse
import gc
import json
import os
import struct
import sys

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# gguf-py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
import gguf  # noqa: E402

# Keywords identifying BitNet projection weights (ternary-quantizable).
BITNET_WEIGHT_KEYWORDS = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
]


def ternary_quantize(weight_f32: np.ndarray) -> np.ndarray:
    """Quantize to ternary {-1, 0, +1} scaled by mean(|w|).

    ⚠ This is genuinely lossy, not a repack. The checkpoint ships the
    full-precision LATENT QAT weights — 12.6 M distinct values per tensor,
    range +-2.7 against mean|w| ~= 0.034 — so the rounding rule below is doing
    real work and has to match BitNet's. It does: verified against upstream's
    I2_S codes at 2 differing values in 13.76 M, ties at the rounding boundary.

    ⚠ It is also only HALF of BitNet inference. b1.58 quantizes ACTIVATIONS to
    int8 per token (127/max|x|) at every BitLinear, and the model was trained
    that way; the runtime does not. See the model card. An A/B that varies only
    --lm-quant cannot see either of these, because both arms come through here.

    Mirrors quant_weight_fp16() from Microsoft's convert_lm_to_gguf.py:
        s = 1 / mean(|w|)
        w_ternary = round(w * s).clamp(-1, 1) / s

    After this, the values are exactly {-mean, 0, +mean}.
    """
    abs_mean = np.abs(weight_f32).mean()
    if abs_mean < 1e-5:
        abs_mean = 1e-5
    s = 1.0 / abs_mean
    ternary = np.round(weight_f32 * s).clip(-1, 1) / s
    return ternary.astype(np.float32)


# ---------- name shortening (same as convert-vibevoice-large-to-gguf.py) -----
def shorten(name: str) -> str:
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("tts_eos_classifier.", "tts_eos.")
    name = name.replace("acoustic_tokenizer.encoder.", "at_enc.")
    name = name.replace("acoustic_tokenizer.decoder.", "at_dec.")
    name = name.replace("semantic_tokenizer.encoder.", "st_enc.")
    name = name.replace("semantic_tokenizer.decoder.", "st_dec.")
    name = name.replace("upsample_layers.", "us.")
    name = name.replace("convtr.convtr.", "convtr.")
    name = name.replace("head.conv.conv.", "head.")
    name = name.replace("acoustic_connector.", "at_conn.")
    name = name.replace("semantic_connector.", "se_conn.")
    name = name.replace("language_model.", "lm.")
    name = name.replace("prediction_head.", "pred.")
    name = name.replace("downsample_layers.", "ds.")
    name = name.replace("stages.", "s.")
    name = name.replace("mixer.conv.conv.conv.", "dw_conv.")
    name = name.replace("ffn.linear1.", "ffn.up.")
    name = name.replace("ffn.linear2.", "ffn.down.")
    name = name.replace("ffn_norm.", "ffn_ln.")
    name = name.replace("input_layernorm.", "attn_ln.")
    name = name.replace("post_attention_layernorm.", "ffn_ln.")
    name = name.replace("self_attn.", "attn.")
    name = name.replace("mlp.gate_proj.", "ffn.gate.")
    name = name.replace("mlp.up_proj.", "ffn.up.")
    name = name.replace("mlp.down_proj.", "ffn.down.")
    name = name.replace("embed_tokens.", "tok_emb.")
    name = name.replace("conv.conv.", "conv.")
    name = name.replace("adaLN_modulation.1.", "adaln.")
    name = name.replace("t_embedder.mlp.", "t_emb.")
    name = name.replace("noisy_images_proj.", "noisy_proj.")
    name = name.replace("final_layer.", "final.")
    name = name.replace("cond_proj.", "cond.")
    name = name.replace("tts_language_model.", "tts_lm.")
    name = name.replace("tts_input_types.", "tts_types.")
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Convert VibeVoice-ASR-BitNet to GGUF (TQ2_0 LM weights)")
    parser.add_argument("--input", required=True,
                        help="HF model ID or local path")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--vae-quant", default="q8_0",
                        choices=["f16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1"],
                        help="Quantization for VAE encoder weights (default: q8_0)")
    parser.add_argument("--lm-quant", default="tq2_0",
                        choices=["tq2_0", "f16", "f32", "q8_0"],
                        help="storage for the ternary LM projections. The VALUES are "
                             "identical in every case -- ternary_quantize() has already "
                             "collapsed them to {-mean, 0, +mean} -- so this changes only "
                             "the ggml matmul path, and with it how ACTIVATIONS are "
                             "handled: TQ2_0 multiplies against block-quantized int8 "
                             "activations, f16/f32 against unquantized ones. That is the "
                             "one axis on which we differ from Microsoft's I2_S kernel, "
                             "which quantizes activations per token. f16 is the control "
                             "for #369's BitNet-vs-demo gap; it costs ~3x the file size "
                             "and is not meant for release.")
    parser.add_argument("--embed-quant", default="f16",
                        choices=["f16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1"],
                        help="Quantization for LM embedding (default: f16)")
    args = parser.parse_args()

    # Resolve model directory
    if os.path.isdir(args.input):
        model_dir = args.input
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.input} …")
        model_dir = snapshot_download(args.input)

    # Read config
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    dec_cfg = cfg.get("decoder_config", {})
    at_cfg = cfg.get("acoustic_tokenizer_config", {})
    st_cfg = cfg.get("semantic_tokenizer_config", {})

    d_lm = dec_cfg["hidden_size"]
    n_lm_layers = dec_cfg["num_hidden_layers"]
    n_heads = dec_cfg["num_attention_heads"]
    n_kv_heads = dec_cfg["num_key_value_heads"]
    d_ffn = dec_cfg["intermediate_size"]
    vocab_size = dec_cfg["vocab_size"]
    rope_theta = dec_cfg.get("rope_theta", 1000000.0)
    head_dim = d_lm // n_heads
    vae_dim_acoustic = at_cfg.get("vae_dim", 64)
    vae_dim_semantic = st_cfg.get("vae_dim", 128) if st_cfg else 128
    encoder_ratios = at_cfg.get("encoder_ratios", [8, 5, 5, 4, 2, 2])
    encoder_depths_str = at_cfg.get("encoder_depths", "3-3-3-3-3-3-8")
    encoder_depths = ([int(x) for x in encoder_depths_str.split("-")]
                      if isinstance(encoder_depths_str, str)
                      else encoder_depths_str)
    n_filters = at_cfg.get("encoder_n_filters", 32)
    n_stages = len(encoder_depths)
    total_downsample = 1
    for r in encoder_ratios:
        total_downsample *= r

    print(f"VibeVoice-ASR-BitNet")
    print(f"  LM: d={d_lm}, layers={n_lm_layers}, heads={n_heads}/{n_kv_heads}, "
          f"ffn={d_ffn}, vocab={vocab_size}")
    print(f"  VAE: acoustic_dim={vae_dim_acoustic}, semantic_dim={vae_dim_semantic}")
    print(f"  Encoder: {n_stages} stages, downsample={total_downsample}×")

    # ----- Detect actual base LM layer count -----
    from safetensors import safe_open
    shard_files = sorted([f for f in os.listdir(model_dir)
                          if f.endswith(".safetensors")])
    if not shard_files:
        print("ERROR: no .safetensors files found")
        sys.exit(1)

    detected_layers = set()
    for shard in shard_files:
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for name in f.keys():
                if "language_model.layers." in name and "tts_" not in name:
                    try:
                        layer = int(name.split("language_model.layers.")[1].split(".")[0])
                        detected_layers.add(layer)
                    except (ValueError, IndexError):
                        pass
    if detected_layers:
        actual = max(detected_layers) + 1
        if actual != n_lm_layers:
            print(f"  NOTE: actual LM layers = {actual} (config says {n_lm_layers})")
            n_lm_layers = actual

    # ----- Create GGUF writer -----
    writer = gguf.GGUFWriter(args.output, "vibevoice-asr")
    writer.add_name("VibeVoice-ASR-BitNet")

    # Hyperparameters (same layout as the existing converter)
    writer.add_uint32("vibevoice.d_lm", d_lm)
    writer.add_uint32("vibevoice.n_lm_layers", n_lm_layers)
    writer.add_uint32("vibevoice.n_heads", n_heads)
    writer.add_uint32("vibevoice.n_kv_heads", n_kv_heads)
    writer.add_uint32("vibevoice.d_ffn", d_ffn)
    writer.add_uint32("vibevoice.vocab_size", vocab_size)
    writer.add_uint32("vibevoice.head_dim", head_dim)
    writer.add_float32("vibevoice.rope_theta", rope_theta)
    writer.add_uint32("vibevoice.vae_dim_acoustic", vae_dim_acoustic)
    writer.add_uint32("vibevoice.vae_dim_semantic", vae_dim_semantic)
    writer.add_uint32("vibevoice.n_encoder_stages", n_stages)
    writer.add_uint32("vibevoice.n_filters", n_filters)
    writer.add_uint32("vibevoice.total_downsample", total_downsample)
    writer.add_array("vibevoice.encoder_ratios", encoder_ratios)
    writer.add_uint32("vibevoice.has_decoder", 0)  # ASR-only
    writer.add_array("vibevoice.encoder_depths", encoder_depths)
    writer.add_uint32("vibevoice.tts_n_layers", 0)

    # ----- Tokenizer -----
    try:
        from transformers import AutoTokenizer
        tok = None
        for tok_src in (model_dir, "Qwen/Qwen2.5-1.5B"):
            try:
                tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
                print(f"  loaded tokenizer from: {tok_src}")
                break
            except Exception:
                pass
        if tok:
            vocab_map = tok.get_vocab()
            inv = {v: k for k, v in vocab_map.items()}
            max_id = max(inv.keys())
            vocab_list = [inv.get(i, f"<unk_{i}>") for i in range(max_id + 1)]
            writer.add_array("tokenizer.ggml.tokens", vocab_list)
            writer.add_uint32("vibevoice.has_tokenizer", 1)
            print(f"  tokenizer: {len(vocab_list)} tokens embedded")
    except Exception as e:
        writer.add_uint32("vibevoice.has_tokenizer", 0)
        print(f"  tokenizer not embedded: {e}")

    # ----- Process tensors shard by shard (low memory) -----
    import torch

    tensor_count = 0
    tq2_count = 0
    q8_count = 0
    skipped = 0

    for shard_idx, shard in enumerate(shard_files):
        path = os.path.join(model_dir, shard)
        print(f"\n  shard {shard_idx + 1}/{len(shard_files)}: {shard}")

        with safe_open(path, framework="pt") as f:
            for name in sorted(f.keys()):
                # Skip decoder weights (ASR-only)
                if "acoustic_tokenizer.decoder" in name:
                    skipped += 1
                    continue
                if "semantic_tokenizer.decoder" in name:
                    skipped += 1
                    continue

                gguf_name = shorten(name)
                if len(gguf_name) >= 64:
                    gguf_name = gguf_name.replace("layers.", "l.")
                    if len(gguf_name) >= 64:
                        print(f"    SKIP (name too long): {name}")
                        skipped += 1
                        continue

                raw = f.get_tensor(name)

                # Skip lm_head — tied to tok_emb (config tie_word_embeddings=true).
                # The vibevoice backend falls back to tok_emb when lm_head is absent.
                if name == "lm_head.weight":
                    skipped += 1
                    continue

                # Resolve quant type enum from CLI flag
                QTYPE_MAP = {
                    "q4_0": gguf.GGMLQuantizationType.Q4_0,
                    "q4_1": gguf.GGMLQuantizationType.Q4_1,
                    "q5_0": gguf.GGMLQuantizationType.Q5_0,
                    "q5_1": gguf.GGMLQuantizationType.Q5_1,
                    "q8_0": gguf.GGMLQuantizationType.Q8_0,
                }
                # Block sizes: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 all use blocks of 32
                QBLOCK = 32

                is_keep_f32 = (
                    "norm" in name or "gamma" in name or
                    name.endswith(".bias") or raw.ndim <= 1 or
                    "scaling_factor" in name or "bias_factor" in name
                )
                is_bitnet_proj = any(kw in name for kw in BITNET_WEIGHT_KEYWORDS)
                is_vae_weight = (
                    "at_enc." in gguf_name or "st_enc." in gguf_name or
                    "at_conn." in gguf_name or "se_conn." in gguf_name
                )
                is_embed = ("tok_emb" in gguf_name)

                # Pick target quant for this tensor
                target_q = None
                if is_keep_f32:
                    pass  # F32
                elif is_bitnet_proj:
                    pass  # TQ2_0 (handled separately)
                elif is_vae_weight and args.vae_quant != "f16":
                    target_q = QTYPE_MAP.get(args.vae_quant)
                elif is_embed and args.embed_quant != "f16":
                    target_q = QTYPE_MAP.get(args.embed_quant)

                tag = ""
                if is_keep_f32:
                    data = raw.to(torch.float32).numpy()
                    writer.add_tensor(gguf_name, data)
                elif is_bitnet_proj:
                    w_f32 = raw.to(torch.float32).numpy()
                    w_ternary = ternary_quantize(w_f32)
                    shape = w_ternary.shape
                    if args.lm_quant == "tq2_0":
                        packed = gguf.quants.quantize(w_ternary.flatten(),
                                                      gguf.GGMLQuantizationType.TQ2_0)
                        nrow = shape[0]
                        bytes_per_row = len(packed) // nrow
                        packed = packed.reshape(nrow, bytes_per_row)
                        writer.add_tensor(gguf_name, packed,
                                          raw_dtype=gguf.GGMLQuantizationType.TQ2_0)
                        tag = " [TQ2_0]"
                    elif args.lm_quant == "f16":
                        # Same numbers, unquantized activations. ⚠ add_tensor's
                        # raw_dtype LABELS bytes and never converts, so the cast
                        # has to happen here.
                        writer.add_tensor(gguf_name, w_ternary.astype(np.float16))
                        tag = " [F16 ternary]"
                    elif args.lm_quant == "f32":
                        writer.add_tensor(gguf_name, w_ternary)
                        tag = " [F32 ternary]"
                    else:  # q8_0
                        packed = gguf.quants.quantize(w_ternary.flatten(),
                                                      gguf.GGMLQuantizationType.Q8_0)
                        nrow = shape[0]
                        bytes_per_row = len(packed) // nrow
                        packed = packed.reshape(nrow, bytes_per_row)
                        writer.add_tensor(gguf_name, packed,
                                          raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                        tag = " [Q8_0 ternary]"
                    tq2_count += 1
                elif target_q is not None and raw.ndim >= 2:
                    w_f32 = raw.to(torch.float32).numpy()
                    shape = w_f32.shape
                    ncol = shape[-1] if len(shape) > 1 else shape[0]
                    if ncol % QBLOCK == 0:
                        packed = gguf.quants.quantize(w_f32.flatten(), target_q)
                        nrow = shape[0]
                        bytes_per_row = len(packed) // nrow
                        packed = packed.reshape(nrow, bytes_per_row)
                        writer.add_tensor(gguf_name, packed, raw_dtype=target_q)
                        q8_count += 1
                        tag = f" [{target_q.name}]"
                    else:
                        data = w_f32.astype(np.float16)
                        writer.add_tensor(gguf_name, data)
                else:
                    data = raw.to(torch.float16).numpy()
                    writer.add_tensor(gguf_name, data)

                tensor_count += 1
                del raw
                if tensor_count <= 5 or tensor_count % 50 == 0:
                    print(f"    [{tensor_count}] {gguf_name}{tag}")

        gc.collect()

    print(f"\n  total: {tensor_count} tensors ({tq2_count} TQ2_0, {q8_count} Q8_0, {skipped} skipped)")

    # ----- Write GGUF -----
    print("  writing GGUF file …")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e9:.2f} GB, {tensor_count} tensors, "
          f"{tq2_count} TQ2_0, {q8_count} Q8_0)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert KugelAudio-0-Open to GGUF.

Architecture: Qwen2.5-7B LLM + 4-layer DiT diffusion head +
  acoustic VAE decoder (+ optional encoders for voice cloning).

Produces a single GGUF with all components needed for TTS inference:
  - Qwen2.5-7B language model (28L, 3584d, GQA 28/4)
  - Diffusion prediction head (4L DiT, AdaLN)
  - Acoustic connector (MLP bridge)
  - Acoustic tokenizer decoder (ConvNeXt, 6 upsample stages)
  - speech_scaling_factor, speech_bias_factor
  - Qwen2.5 vocabulary (for text tokenization)

Optionally includes encoders (acoustic + semantic) and semantic connector
for future voice cloning support.

Usage (run on Kaggle with 16+ GB RAM):
  python models/convert-kugelaudio-to-gguf.py \\
      --input kugelaudio/kugelaudio-0-open \\
      --output kugelaudio-0-open-f16.gguf

  # Inference-only (no encoders, ~14 GB smaller):
  python models/convert-kugelaudio-to-gguf.py \\
      --input kugelaudio/kugelaudio-0-open \\
      --output kugelaudio-0-open-f16.gguf \\
      --no-encoders
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser(description="Convert KugelAudio to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--no-encoders", action="store_true",
                        help="Skip encoder tensors (inference-only, saves ~4 GB)")
    parser.add_argument("--type", choices=["f32", "f16", "bf16"], default="f16",
                        help="Output weight type (default: f16)")
    args = parser.parse_args()

    from safetensors import safe_open

    # ── Resolve model directory ──────────────────────────────────────────
    if os.path.isdir(args.input):
        model_dir = args.input
    else:
        from huggingface_hub import snapshot_download
        print(f"downloading {args.input}...")
        model_dir = snapshot_download(args.input)

    # ── Read config ──────────────────────────────────────────────────────
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    dec_cfg = cfg.get("decoder_config", {})
    at_cfg = cfg.get("acoustic_tokenizer_config", {})
    st_cfg = cfg.get("semantic_tokenizer_config", {})
    diff_cfg = cfg.get("diffusion_head_config", {})

    # Architecture params
    d_lm = dec_cfg["hidden_size"]              # 3584
    n_lm_layers = dec_cfg["num_hidden_layers"]  # 28
    n_heads = dec_cfg["num_attention_heads"]     # 28
    n_kv_heads = dec_cfg["num_key_value_heads"]  # 4
    d_ffn = dec_cfg["intermediate_size"]         # 18944
    vocab_size = dec_cfg["vocab_size"]            # 152064
    rope_theta = dec_cfg.get("rope_theta", 1000000.0)
    rms_norm_eps = dec_cfg.get("rms_norm_eps", 1e-6)
    head_dim = d_lm // n_heads                   # 128

    # VAE dims
    vae_dim_acoustic = at_cfg.get("vae_dim", 64)
    vae_dim_semantic = st_cfg.get("vae_dim", 128) if st_cfg else 128

    # Tokenizer encoder/decoder params
    encoder_ratios = at_cfg.get("encoder_ratios", [8, 5, 5, 4, 2, 2])
    decoder_ratios = at_cfg.get("decoder_ratios", encoder_ratios)
    encoder_depths_str = at_cfg.get("encoder_depths", "3-3-3-3-3-3-8")
    encoder_depths = [int(x) for x in encoder_depths_str.split("-")]
    # Decoder depths: reversed encoder depths (per tokenizer.py logic)
    decoder_depths = list(reversed(encoder_depths))
    n_filters = at_cfg.get("encoder_n_filters", 32)
    dec_n_filters = at_cfg.get("decoder_n_filters", 32)
    n_stages = len(encoder_depths)
    at_norm_eps = at_cfg.get("layernorm_eps", 1e-5)

    total_upsample = 1
    for r in decoder_ratios:
        total_upsample *= r

    # Diffusion head params
    head_layers = diff_cfg.get("head_layers", 4)
    head_ffn_ratio = diff_cfg.get("head_ffn_ratio", 3.0)
    latent_size = diff_cfg.get("latent_size", 64)
    head_hidden = diff_cfg.get("hidden_size", d_lm)  # Should be same as d_lm
    ddpm_num_steps = diff_cfg.get("ddpm_num_steps", 1000)
    ddpm_inference_steps = diff_cfg.get("ddpm_num_inference_steps", 20)
    prediction_type = diff_cfg.get("prediction_type", "v_prediction")
    algorithm_type = diff_cfg.get("ddpm_algorithm_type", "sde-dpmsolver++")
    beta_schedule = diff_cfg.get("ddpm_beta_schedule", "cosine")
    diff_norm_eps = diff_cfg.get("rms_norm_eps", 1e-5)

    print(f"KugelAudio: d_lm={d_lm}, n_layers={n_lm_layers}, heads={n_heads}/{n_kv_heads}")
    print(f"  d_ffn={d_ffn}, vocab={vocab_size}, rope_theta={rope_theta}")
    print(f"  acoustic vae_dim={vae_dim_acoustic}, semantic vae_dim={vae_dim_semantic}")
    print(f"  decoder: {n_stages} stages, depths={decoder_depths}, ratios={decoder_ratios}")
    print(f"  total upsample: {total_upsample}x, base_filters={dec_n_filters}")
    print(f"  diffusion: {head_layers}L, latent={latent_size}, steps={ddpm_inference_steps}")
    print(f"  prediction={prediction_type}, algorithm={algorithm_type}, schedule={beta_schedule}")

    # ── Create GGUF writer ───────────────────────────────────────────────
    model_name = os.path.basename(args.input.rstrip("/"))
    arch = "kugelaudio"
    writer = gguf.GGUFWriter(args.output, arch)
    writer.add_name(model_name)

    # ── Write hyperparameters ────────────────────────────────────────────
    writer.add_uint32("kugelaudio.d_lm", d_lm)
    writer.add_uint32("kugelaudio.n_lm_layers", n_lm_layers)
    writer.add_uint32("kugelaudio.n_heads", n_heads)
    writer.add_uint32("kugelaudio.n_kv_heads", n_kv_heads)
    writer.add_uint32("kugelaudio.d_ffn", d_ffn)
    writer.add_uint32("kugelaudio.vocab_size", vocab_size)
    writer.add_uint32("kugelaudio.head_dim", head_dim)
    writer.add_float32("kugelaudio.rope_theta", rope_theta)
    writer.add_float32("kugelaudio.rms_norm_eps", rms_norm_eps)
    writer.add_uint32("kugelaudio.vae_dim_acoustic", vae_dim_acoustic)
    writer.add_uint32("kugelaudio.vae_dim_semantic", vae_dim_semantic)

    # Tokenizer/decoder params
    writer.add_uint32("kugelaudio.n_decoder_stages", n_stages)
    writer.add_uint32("kugelaudio.dec_n_filters", dec_n_filters)
    writer.add_uint32("kugelaudio.total_upsample", total_upsample)
    writer.add_array("kugelaudio.decoder_ratios", decoder_ratios)
    writer.add_array("kugelaudio.decoder_depths", decoder_depths)
    writer.add_float32("kugelaudio.at_norm_eps", at_norm_eps)
    writer.add_uint32("kugelaudio.has_encoders", 0 if args.no_encoders else 1)
    writer.add_array("kugelaudio.encoder_ratios", encoder_ratios)
    writer.add_array("kugelaudio.encoder_depths", encoder_depths)

    # Diffusion head params
    writer.add_uint32("kugelaudio.head_layers", head_layers)
    writer.add_float32("kugelaudio.head_ffn_ratio", head_ffn_ratio)
    writer.add_uint32("kugelaudio.latent_size", latent_size)
    writer.add_uint32("kugelaudio.ddpm_num_steps", ddpm_num_steps)
    writer.add_uint32("kugelaudio.ddpm_inference_steps", ddpm_inference_steps)
    writer.add_string("kugelaudio.prediction_type", prediction_type)
    writer.add_string("kugelaudio.algorithm_type", algorithm_type)
    writer.add_string("kugelaudio.beta_schedule", beta_schedule)
    writer.add_float32("kugelaudio.diff_norm_eps", diff_norm_eps)

    # Special token IDs
    writer.add_uint32("kugelaudio.speech_start_id", 151652)
    writer.add_uint32("kugelaudio.speech_end_id", 151653)
    writer.add_uint32("kugelaudio.speech_diffusion_id", 151654)
    writer.add_uint32("kugelaudio.eos_token_id", 151643)

    # ── Embed tokenizer vocabulary ───────────────────────────────────────
    print("\nloading tokenizer...")
    try:
        from transformers import AutoTokenizer

        # Try model dir first, fall back to Qwen2.5 base tokenizer
        tok = None
        for tok_src in [model_dir, "Qwen/Qwen2.5-7B"]:
            try:
                tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
                print(f"  tokenizer from: {tok_src}")
                break
            except Exception:
                continue

        if tok is not None:
            # Build vocab arrays
            n_vocab = tok.vocab_size
            if hasattr(tok, "get_vocab"):
                vmap = tok.get_vocab()
                n_vocab = max(n_vocab, max(vmap.values()) + 1)
            else:
                vmap = {}

            # Reverse map: id -> token string
            id2tok = {}
            for text, tid in vmap.items():
                id2tok[tid] = text

            tokens = []
            scores = []
            toktypes = []
            for i in range(n_vocab):
                piece = id2tok.get(i, f"<unk_{i}>")
                tokens.append(piece.encode("utf-8", errors="replace"))
                scores.append(0.0)
                toktypes.append(1)  # NORMAL

            writer.add_token_list(tokens)
            writer.add_token_scores(scores)
            writer.add_token_types(toktypes)
            writer.add_uint32("kugelaudio.n_vocab", n_vocab)
            print(f"  embedded {n_vocab} tokens")
    except ImportError:
        print("  WARNING: transformers not available, skipping tokenizer")

    # ── Load and convert tensors ─────────────────────────────────────────
    print("\nloading tensors...")

    shard_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    if not shard_files:
        raise RuntimeError(f"No .safetensors files found in {model_dir}")

    # Determine output data type
    dtype_map = {
        "f32": (np.float32, gguf.GGMLQuantizationType.F32),
        "f16": (np.float16, gguf.GGMLQuantizationType.F16),
        "bf16": (None, gguf.GGMLQuantizationType.BF16),
    }
    out_np_dtype, out_ggml_type = dtype_map[args.type]

    tensor_count = 0
    skipped_encoder = 0

    # Name mapping: PyTorch -> GGUF
    # We preserve the PyTorch naming convention with "model." prefix stripped
    # and "." replaced by "." (no change needed, GGUF supports dots).

    for shard in shard_files:
        path = os.path.join(model_dir, shard)
        print(f"  processing {shard}...")

        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                # Skip encoder tensors if requested
                if args.no_encoders:
                    if ".encoder." in name or "semantic_tokenizer" in name or "semantic_connector" in name:
                        skipped_encoder += 1
                        continue

                tensor = f.get_tensor(name)
                shape = list(tensor.shape)

                # Convert to target dtype
                if tensor.dtype == torch.bfloat16:
                    if args.type == "bf16":
                        # Keep as bf16 view
                        data = tensor.view(torch.uint16).numpy()
                        ggml_type = gguf.GGMLQuantizationType.BF16
                    elif args.type == "f16":
                        data = tensor.float().half().numpy()
                        ggml_type = gguf.GGMLQuantizationType.F16
                    else:
                        data = tensor.float().numpy()
                        ggml_type = gguf.GGMLQuantizationType.F32
                elif tensor.dtype == torch.float16:
                    if args.type == "f32":
                        data = tensor.float().numpy()
                        ggml_type = gguf.GGMLQuantizationType.F32
                    else:
                        data = tensor.numpy()
                        ggml_type = gguf.GGMLQuantizationType.F16
                else:  # float32
                    if args.type == "f16":
                        data = tensor.half().numpy()
                        ggml_type = gguf.GGMLQuantizationType.F16
                    else:
                        data = tensor.numpy()
                        ggml_type = gguf.GGMLQuantizationType.F32

                # For 1D tensors (norms, biases, scalars), always use F32
                if len(shape) <= 1:
                    data = tensor.float().numpy()
                    ggml_type = gguf.GGMLQuantizationType.F32

                writer.add_tensor(name, data, raw_dtype=ggml_type)
                tensor_count += 1

                if tensor_count % 100 == 0:
                    print(f"    {tensor_count} tensors...")

    print(f"\n  total: {tensor_count} tensors written")
    if skipped_encoder > 0:
        print(f"  skipped: {skipped_encoder} encoder tensors (--no-encoders)")

    # ── Write file ───────────────────────────────────────────────────────
    print(f"\nwriting {args.output}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"done: {file_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()

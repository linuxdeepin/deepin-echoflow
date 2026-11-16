#!/usr/bin/env python3
"""Convert Mini-Omni2 (gpt-omni/mini-omni2) to GGUF.

Architecture:
  - Audio encoder: Whisper-small (80 mel, 12 layers, 768d, 12 heads)
  - Adapter: whisperMLP (SwiGLU: fc_1 768→4864, fc_2 768→4864, proj 4864→896)
  - LLM decoder: Qwen2-0.5B (896d, 24L, 14 heads, 2 KV groups, RMSNorm, SwiGLU)
  - Vocab: text 152000 + 7×audio 4160 = 181120 padded

Weight sources:
  - lit_model.pth: LLM + adapter weights (litgpt format)
  - small.pt: Whisper-small encoder weights (OpenAI format)
  - model_config.yaml: litgpt config
  - tokenizer.json: Qwen2 tokenizer

Usage:
  python models/convert-mini-omni2-to-gguf.py \\
      --input /path/to/mini-omni2 \\
      --output mini-omni2.gguf
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser(description="Convert Mini-Omni2 to GGUF")
    parser.add_argument("--input", required=True, help="Path to mini-omni2 directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    import torch
    import whisper

    model_dir = args.input
    print(f"Loading from: {model_dir}")

    # --- Load LitGPT weights ---
    lit_path = os.path.join(model_dir, "lit_model.pth")
    print(f"  Loading lit_model.pth ...")
    lit_sd = torch.load(lit_path, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(lit_sd)} tensors from lit_model.pth")

    # --- Load Whisper-small encoder ---
    whisper_path = os.path.join(model_dir, "small.pt")
    if os.path.exists(whisper_path):
        print(f"  Loading Whisper-small from {whisper_path}")
        whisper_model = whisper.load_model(whisper_path, device="cpu")
    else:
        print("  Loading Whisper-small from OpenAI")
        whisper_model = whisper.load_model("small", device="cpu")
    whisper_sd = whisper_model.state_dict()
    print(f"  Loaded {len(whisper_sd)} tensors from Whisper-small")

    # --- Read config ---
    # Qwen2-0.5B via model_config.yaml
    # Hardcoded from the known model_config.yaml values
    enc_hidden = 768
    enc_n_layers = 12
    enc_n_heads = 12
    enc_ff = 3072      # whisper-small intermediate
    n_mels = 80
    enc_max_pos = 1500
    adapter_dim = 768   # whisper output dim

    llm_hidden = 896
    llm_n_layers = 24
    llm_n_heads = 14
    llm_n_kv_heads = 2
    llm_ff = 4864
    llm_vocab = 181120  # padded_vocab_size (152000 text + 29120 audio)
    llm_max_pos = 2048
    rms_eps = 1e-6
    rope_base = 1000000

    text_vocab_size = 152000
    audio_vocab_size = 4160
    n_audio_streams = 7

    # --- Create GGUF writer ---
    writer = gguf.GGUFWriter(args.output, "mini-omni2")
    writer.add_name("Mini-Omni2")

    # Audio encoder hparams
    writer.add_uint32("mini_omni2.audio.hidden_size", enc_hidden)
    writer.add_uint32("mini_omni2.audio.num_layers", enc_n_layers)
    writer.add_uint32("mini_omni2.audio.num_heads", enc_n_heads)
    writer.add_uint32("mini_omni2.audio.intermediate_size", enc_ff)
    writer.add_uint32("mini_omni2.audio.num_mel_bins", n_mels)
    writer.add_uint32("mini_omni2.audio.max_position_embeddings", enc_max_pos)

    # Adapter hparams
    writer.add_uint32("mini_omni2.adapter.input_dim", adapter_dim)
    writer.add_uint32("mini_omni2.adapter.intermediate_size", llm_ff)
    writer.add_uint32("mini_omni2.adapter.output_dim", llm_hidden)

    # LLM hparams
    writer.add_uint32("mini_omni2.llm.hidden_size", llm_hidden)
    writer.add_uint32("mini_omni2.llm.num_layers", llm_n_layers)
    writer.add_uint32("mini_omni2.llm.num_heads", llm_n_heads)
    writer.add_uint32("mini_omni2.llm.num_kv_heads", llm_n_kv_heads)
    writer.add_uint32("mini_omni2.llm.intermediate_size", llm_ff)
    writer.add_uint32("mini_omni2.llm.vocab_size", llm_vocab)
    writer.add_uint32("mini_omni2.llm.max_position_embeddings", llm_max_pos)
    writer.add_float32("mini_omni2.llm.rms_norm_eps", rms_eps)
    writer.add_uint32("mini_omni2.llm.rope_base", rope_base)

    # Vocab split
    writer.add_uint32("mini_omni2.text_vocab_size", text_vocab_size)
    writer.add_uint32("mini_omni2.audio_vocab_size", audio_vocab_size)
    writer.add_uint32("mini_omni2.n_audio_streams", n_audio_streams)

    # --- Tokenizer ---
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        if vocab:
            vocab_size = max(vocab.values()) + 1
            vocab_list = [""] * vocab_size
            for token, idx in vocab.items():
                if idx < vocab_size:
                    vocab_list[idx] = token
            writer.add_array("tokenizer.ggml.tokens", vocab_list)
            print(f"  Tokenizer: {vocab_size} tokens")
            # BPE merges (needed for TTS text encoding)
            merges = tok_data.get("model", {}).get("merges", [])
            if merges:
                writer.add_array("tokenizer.ggml.merges", merges)
                print(f"  Merges: {len(merges)} merge rules")
    else:
        print("  Warning: tokenizer.json not found")

    # --- Mel filterbank ---
    # Whisper-small uses 80 mel bins, n_fft=400 → 201 frequency bins
    mel_filters = whisper_model.encoder.conv1.weight.new_zeros(0)  # placeholder
    # Get mel filters from whisper library
    mel_80 = whisper.audio.mel_filters(whisper_model.device, n_mels=80)  # (80, 201)
    mel_np = mel_80.cpu().numpy().astype(np.float32)
    # Periodic Hann window matching torch.hann_window(400) used by Whisper.
    # NOT np.hanning(401)[:-1] which is the symmetric variant.
    win = torch.hann_window(400).numpy().astype(np.float32)

    writer.add_tensor("audio.mel_filters", mel_np)
    writer.add_tensor("audio.mel_window", win)
    tensor_count = 2
    print(f"  audio.mel_filters  {mel_np.shape}  F32")
    print(f"  audio.mel_window   {win.shape}  F32")

    # --- Helper functions ---
    def to_np(t):
        if isinstance(t, torch.Tensor):
            return t.float().numpy()
        return np.asarray(t, dtype=np.float32)

    def is_f32_tensor(name, shape):
        if "norm" in name:
            return True
        if name.endswith(".bias"):
            return True
        if "positional_embedding" in name:
            return True
        if len(shape) <= 1:
            return True
        return False

    def add_tensor(gguf_name, data_np):
        nonlocal tensor_count
        if is_f32_tensor(gguf_name, data_np.shape):
            writer.add_tensor(gguf_name, data_np.astype(np.float32))
            dtype_str = "F32"
        else:
            writer.add_tensor(gguf_name, data_np.astype(np.float16))
            dtype_str = "F16"
        print(f"  {gguf_name:60s} {str(data_np.shape):20s} {dtype_str}")
        tensor_count += 1

    # --- Write Whisper encoder tensors ---
    print("\nWhisper encoder tensors:")
    whisper_name_map = {
        "encoder.conv1.weight": "audio.conv1.weight",
        "encoder.conv1.bias": "audio.conv1.bias",
        "encoder.conv2.weight": "audio.conv2.weight",
        "encoder.conv2.bias": "audio.conv2.bias",
        "encoder.positional_embedding": "audio.positional_embedding",
        "encoder.ln_post.weight": "audio.norm.weight",
        "encoder.ln_post.bias": "audio.norm.bias",
    }

    for hf_name, gguf_name in whisper_name_map.items():
        if hf_name in whisper_sd:
            add_tensor(gguf_name, to_np(whisper_sd[hf_name]))

    # Encoder blocks
    for il in range(enc_n_layers):
        hf_pfx = f"encoder.blocks.{il}"
        gg_pfx = f"audio.blk.{il}"
        layer_map = {
            f"{hf_pfx}.attn_ln.weight": f"{gg_pfx}.attn_norm.weight",
            f"{hf_pfx}.attn_ln.bias": f"{gg_pfx}.attn_norm.bias",
            f"{hf_pfx}.attn.query.weight": f"{gg_pfx}.attn_q.weight",
            f"{hf_pfx}.attn.query.bias": f"{gg_pfx}.attn_q.bias",
            f"{hf_pfx}.attn.key.weight": f"{gg_pfx}.attn_k.weight",
            f"{hf_pfx}.attn.value.weight": f"{gg_pfx}.attn_v.weight",
            f"{hf_pfx}.attn.value.bias": f"{gg_pfx}.attn_v.bias",
            f"{hf_pfx}.attn.out.weight": f"{gg_pfx}.attn_out.weight",
            f"{hf_pfx}.attn.out.bias": f"{gg_pfx}.attn_out.bias",
            f"{hf_pfx}.mlp_ln.weight": f"{gg_pfx}.ffn_norm.weight",
            f"{hf_pfx}.mlp_ln.bias": f"{gg_pfx}.ffn_norm.bias",
            f"{hf_pfx}.mlp.0.weight": f"{gg_pfx}.ffn.up.weight",
            f"{hf_pfx}.mlp.0.bias": f"{gg_pfx}.ffn.up.bias",
            f"{hf_pfx}.mlp.2.weight": f"{gg_pfx}.ffn.down.weight",
            f"{hf_pfx}.mlp.2.bias": f"{gg_pfx}.ffn.down.bias",
        }
        for hf_name, gguf_name in layer_map.items():
            if hf_name in whisper_sd:
                add_tensor(gguf_name, to_np(whisper_sd[hf_name]))

    # --- Write adapter tensors ---
    print("\nAdapter (whisperMLP) tensors:")
    # Note: config.bias=false → adapter Linears have no bias
    adapter_map = {
        "whisper_adapter.fc_1.weight": "adapter.fc_1.weight",
        "whisper_adapter.fc_2.weight": "adapter.fc_2.weight",
        "whisper_adapter.proj.weight": "adapter.proj.weight",
    }
    for lit_name, gguf_name in adapter_map.items():
        if lit_name in lit_sd:
            add_tensor(gguf_name, to_np(lit_sd[lit_name]))
        else:
            print(f"  WARNING: {lit_name} not found in lit_model.pth")

    # --- Write LLM tensors ---
    print("\nLLM (Qwen2-0.5B) tensors:")

    # Token embedding
    if "transformer.wte.weight" in lit_sd:
        add_tensor("llm.token_embd.weight", to_np(lit_sd["transformer.wte.weight"]))

    # LM head — tie_word_embeddings=true means lm_head.weight is the same
    # tensor as transformer.wte.weight and won't appear separately in the
    # state dict. Store a flag so the C++ runtime knows to reuse wte.
    if "lm_head.weight" in lit_sd:
        add_tensor("lm_head.weight", to_np(lit_sd["lm_head.weight"]))
    else:
        writer.add_bool("mini_omni2.tie_word_embeddings", True)
        print("  lm_head tied to token_embd (tie_word_embeddings=true)")

    # Final norm
    if "transformer.ln_f.weight" in lit_sd:
        add_tensor("llm.output_norm.weight", to_np(lit_sd["transformer.ln_f.weight"]))

    # Transformer blocks
    # litgpt naming: transformer.h.{il}.{component}
    for il in range(llm_n_layers):
        lit_pfx = f"transformer.h.{il}"
        gg_pfx = f"llm.blk.{il}"

        llm_layer_map = {
            # Attention norm
            f"{lit_pfx}.norm_1.weight": f"{gg_pfx}.attn_norm.weight",
            # FFN norm
            f"{lit_pfx}.norm_2.weight": f"{gg_pfx}.ffn_norm.weight",
            # Attention Q/K/V/O
            f"{lit_pfx}.attn.attn.weight": None,  # fused QKV — split below
            f"{lit_pfx}.attn.attn.bias": None,     # fused QKV bias — split below
            f"{lit_pfx}.attn.proj.weight": f"{gg_pfx}.attn_out.weight",
            # FFN (SwiGLU: separate gate fc_1 + up fc_2, down proj)
            # Note: bias=false in config for attn.proj, mlp.fc_1/fc_2/proj
            f"{lit_pfx}.mlp.fc_1.weight": f"{gg_pfx}.ffn.gate.weight",
            f"{lit_pfx}.mlp.fc_2.weight": f"{gg_pfx}.ffn.up.weight",
            f"{lit_pfx}.mlp.proj.weight": f"{gg_pfx}.ffn.down.weight",
        }

        # Direct mappings
        for lit_name, gguf_name in llm_layer_map.items():
            if gguf_name is None:
                continue
            if lit_name in lit_sd:
                add_tensor(gguf_name, to_np(lit_sd[lit_name]))

        # Split fused QKV: litgpt stores attn.attn.weight interleaved per
        # query group: [group0_Q(q_per_kv*hd), group0_K(hd), group0_V(hd),
        #               group1_Q(q_per_kv*hd), group1_K(hd), group1_V(hd), ...]
        # Qwen2-0.5B: n_heads=14, n_kv=2, q_per_kv=7, hd=64
        #   group size = (7+1+1)*64 = 576, total = 2*576 = 1152
        qkv_key = f"{lit_pfx}.attn.attn.weight"
        if qkv_key in lit_sd:
            qkv = to_np(lit_sd[qkv_key])
            head_dim = llm_hidden // llm_n_heads  # 64
            q_per_kv = llm_n_heads // llm_n_kv_heads  # 7
            group_size = (q_per_kv + 2) * head_dim  # 576

            q_parts, k_parts, v_parts = [], [], []
            for g in range(llm_n_kv_heads):
                offset = g * group_size
                q_parts.append(qkv[offset : offset + q_per_kv * head_dim])
                k_parts.append(qkv[offset + q_per_kv * head_dim : offset + (q_per_kv + 1) * head_dim])
                v_parts.append(qkv[offset + (q_per_kv + 1) * head_dim : offset + group_size])
            add_tensor(f"{gg_pfx}.attn_q.weight", np.concatenate(q_parts, axis=0))
            add_tensor(f"{gg_pfx}.attn_k.weight", np.concatenate(k_parts, axis=0))
            add_tensor(f"{gg_pfx}.attn_v.weight", np.concatenate(v_parts, axis=0))

        qkv_bias_key = f"{lit_pfx}.attn.attn.bias"
        if qkv_bias_key in lit_sd:
            qkv_b = to_np(lit_sd[qkv_bias_key])
            head_dim = llm_hidden // llm_n_heads
            q_per_kv = llm_n_heads // llm_n_kv_heads
            group_size = (q_per_kv + 2) * head_dim

            q_b_parts, k_b_parts, v_b_parts = [], [], []
            for g in range(llm_n_kv_heads):
                offset = g * group_size
                q_b_parts.append(qkv_b[offset : offset + q_per_kv * head_dim])
                k_b_parts.append(qkv_b[offset + q_per_kv * head_dim : offset + (q_per_kv + 1) * head_dim])
                v_b_parts.append(qkv_b[offset + (q_per_kv + 1) * head_dim : offset + group_size])
            add_tensor(f"{gg_pfx}.attn_q.bias", np.concatenate(q_b_parts, axis=0))
            add_tensor(f"{gg_pfx}.attn_k.bias", np.concatenate(k_b_parts, axis=0))
            add_tensor(f"{gg_pfx}.attn_v.bias", np.concatenate(v_b_parts, axis=0))

    # --- Finalize ---
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({file_size / 1e9:.2f} GB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()

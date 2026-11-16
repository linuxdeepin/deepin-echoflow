#!/usr/bin/env python3
"""Convert OpenMOSS-Team/MOSS-Transcribe-Diarize-0.9B to GGUF format for CrispASR.

Architecture:
  Whisper encoder: 80-mel → Conv1d stem (conv1+conv2, stride 1/2) →
    +learned pos embed → 24 pre-LN Whisper-style layers (1024d, 16 heads,
    FFN 4096, global attention) → ln_post.
  VQAdaptor: 4x temporal merge (reshape T/4 × 4096) → Linear(4096→1024) +
    SiLU + Linear(1024→1024) + LayerNorm(1024).
  LM: Qwen3-0.6B (28L, 1024d, 16Q/8KV, head_dim 128, SwiGLU 3072,
    QK-norm, RoPE θ=1e6, vocab 151936, TIED embeddings → no lm_head).

Streams tensors one-at-a-time via safe_open (BF16→F16). ~4 GB peak RAM.

Usage:
    python models/convert-moss-transcribe-diarize-to-gguf.py \\
        --input OpenMOSS-Team/MOSS-Transcribe-Diarize-0.9B \\
        --output moss-transcribe-diarize-0.9b-f16.gguf

    # Then quantize:
    crispasr-quantize moss-transcribe-diarize-0.9b-f16.gguf \\
                      moss-transcribe-diarize-0.9b-q4_k.gguf q4_k
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors package not found. Install with: pip install safetensors")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub package not found. Install with: pip install huggingface_hub")
    sys.exit(1)


def load_model_dir(model_id: str) -> Path:
    model_path = Path(model_id)
    if model_path.is_dir():
        return model_path
    print(f"Downloading model from HuggingFace: {model_id}")
    path = snapshot_download(model_id,
                             allow_patterns=["*.safetensors", "*.json",
                                             "merges.txt", "vocab.json"])
    return Path(path)


def map_tensor_name(hf_name: str):
    """Map HF tensor name → GGUF name for MOSS-Transcribe-Diarize.

    Returns None for tensors that should be skipped (e.g. a separate lm_head,
    which is tied to embed_tokens and absent in this checkpoint anyway).
    """
    name = hf_name

    # Strip the top-level MossModel prefix.
    if name.startswith("model."):
        name = name[len("model."):]

    # ---- Whisper encoder (stock Conv1d stem + 24L transformer) ----
    if name.startswith("whisper_encoder."):
        name = name[len("whisper_encoder."):]
        # Top-level layer_norm (whisper_encoder.layer_norm.* → enc.ln_post.*)
        if name.startswith("layer_norm."):
            return "enc.ln_post." + name[len("layer_norm."):]
        # Conv1d stem
        name = name.replace("conv1.", "enc.conv1.")
        name = name.replace("conv2.", "enc.conv2.")
        # Learned position embeddings
        name = name.replace("embed_positions.", "enc.pos_embed.")
        # Transformer layers — do longer patterns first to avoid substring clobber
        name = name.replace("layers.", "enc.blk.")
        name = name.replace(".self_attn_layer_norm.", ".attn_norm.")
        name = name.replace(".final_layer_norm.", ".ffn_norm.")
        name = name.replace(".self_attn.q_proj.", ".attn_q.")
        name = name.replace(".self_attn.k_proj.", ".attn_k.")
        name = name.replace(".self_attn.v_proj.", ".attn_v.")
        name = name.replace(".self_attn.out_proj.", ".attn_out.")
        name = name.replace(".fc1.", ".ffn_up.")
        name = name.replace(".fc2.", ".ffn_down.")
        return name

    # ---- VQAdaptor (Linear + SiLU + Linear + LayerNorm) ----
    if name.startswith("vq_adaptor."):
        name = name[len("vq_adaptor."):]
        # layers.0 = Linear(4096, 1024)
        # layers.1 = SiLU (no params)
        # layers.2 = Linear(1024, 1024)
        # layers.3 = LayerNorm(1024)
        name = name.replace("layers.0.", "adaptor.fc1.")
        name = name.replace("layers.2.", "adaptor.fc2.")
        name = name.replace("layers.3.", "adaptor.norm.")
        return name

    # ---- Language model (Qwen3-0.6B) ----
    if name.startswith("language_model."):
        name = name[len("language_model."):]
        # Top-level norm (language_model.norm.weight → output_norm.weight)
        if name.startswith("norm."):
            return "output_norm." + name[len("norm."):]
        name = name.replace("embed_tokens.", "token_embd.")
        name = name.replace("layers.", "blk.")
        # Order matters: do longer/more-specific patterns first
        name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
        name = name.replace(".input_layernorm.", ".attn_norm.")
        name = name.replace(".self_attn.q_norm.", ".q_norm.")
        name = name.replace(".self_attn.k_norm.", ".k_norm.")
        name = name.replace(".self_attn.q_proj.", ".attn_q.")
        name = name.replace(".self_attn.k_proj.", ".attn_k.")
        name = name.replace(".self_attn.v_proj.", ".attn_v.")
        name = name.replace(".self_attn.o_proj.", ".attn_o.")
        name = name.replace(".mlp.gate_proj.", ".ffn_gate.")
        name = name.replace(".mlp.up_proj.", ".ffn_up.")
        name = name.replace(".mlp.down_proj.", ".ffn_down.")
        return name

    # Separate lm_head (tied → ignore if present)
    if name.startswith("lm_head."):
        return None

    return name


def main():
    parser = argparse.ArgumentParser(description="Convert MOSS-Transcribe-Diarize-0.9B to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16"],
                        help="Output data type for 2D+ tensors (default: f16)")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    # The diarize model uses a stock Whisper encoder: 80 mel, n_fft=400, hop=160
    ac = config.get("audio_config", config.get("whisper_config", {}))
    lc = config.get("language_config", config.get("text_config", {}))

    n_mels = ac.get("num_mel_bins", 80)
    enc_layers = ac.get("encoder_layers", 24)
    enc_d_model = ac.get("d_model", 1024)
    enc_n_heads = ac.get("encoder_attention_heads", 16)
    enc_ffn_dim = ac.get("encoder_ffn_dim", 4096)
    enc_max_pos = ac.get("max_source_positions", 1500)

    llm_hidden = lc.get("hidden_size", 1024)
    llm_layers = lc.get("num_hidden_layers", 28)
    llm_n_heads = lc.get("num_attention_heads", 16)
    llm_n_kv_heads = lc.get("num_key_value_heads", 8)
    llm_head_dim = lc.get("head_dim", 128)
    llm_ff_dim = lc.get("intermediate_size", 3072)
    llm_vocab_size = lc.get("vocab_size", 151936)
    llm_rope_theta = lc.get("rope_theta", 1000000.0)
    llm_rms_eps = lc.get("rms_norm_eps", 1e-6)

    audio_merge_size = config.get("audio_merge_size", 4)
    adaptor_in_dim = config.get("adaptor_input_dim", enc_d_model * audio_merge_size)
    adaptor_out_dim = llm_hidden  # 1024

    # Special tokens
    audio_token_id = config.get("audio_token_id", 151671)
    # Read from processor_config.json if available, else defaults
    proc_cfg_path = model_dir / "processor_config.json"
    if proc_cfg_path.exists():
        with open(proc_cfg_path, encoding="utf-8") as f:
            proc_cfg = json.load(f)
        audio_tokens_per_second = proc_cfg.get("audio_tokens_per_second", 12.5)
        time_marker_every_seconds = proc_cfg.get("time_marker_every_seconds", 5)
    else:
        audio_tokens_per_second = 12.5
        time_marker_every_seconds = 5

    print(f"\nMOSS-Transcribe-Diarize-0.9B")
    print(f"  Whisper encoder: {enc_layers}L, d={enc_d_model}, "
          f"heads={enc_n_heads}, FFN={enc_ffn_dim}, mels={n_mels}")
    print(f"  VQAdaptor: merge={audio_merge_size}x, {adaptor_in_dim}→{adaptor_out_dim}")
    print(f"  LM: {llm_layers}L, hidden={llm_hidden}, "
          f"heads={llm_n_heads}, kv_heads={llm_n_kv_heads}, "
          f"head_dim={llm_head_dim}, ffn={llm_ff_dim}")
    print(f"  Vocab: {llm_vocab_size} (tied embeddings)")

    if args.outtype == "f16":
        out_dtype = np.float16
        ggml_type = GGMLQuantizationType.F16
    else:
        out_dtype = np.float32
        ggml_type = GGMLQuantizationType.F32

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        st_files = sorted(model_dir.glob("model-*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    tensor_names = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            tensor_names[name] = idx
    print(f"  Safetensors: {len(tensor_names)} tensors in {len(st_files)} file(s)")

    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "moss_transcribe_diarize", use_temp_file=True)
    writer.add_name("MOSS-Transcribe-Diarize-0.9B")

    # Encoder params
    writer.add_uint32("moss_diarize.enc.n_layers", enc_layers)
    writer.add_uint32("moss_diarize.enc.d_model", enc_d_model)
    writer.add_uint32("moss_diarize.enc.n_heads", enc_n_heads)
    writer.add_uint32("moss_diarize.enc.n_mels", n_mels)
    writer.add_uint32("moss_diarize.enc.ffn_dim", enc_ffn_dim)
    writer.add_uint32("moss_diarize.enc.max_pos", enc_max_pos)

    # Adaptor params
    writer.add_uint32("moss_diarize.adaptor.in_dim", adaptor_in_dim)
    writer.add_uint32("moss_diarize.adaptor.out_dim", adaptor_out_dim)

    # LLM params (Qwen3-0.6B)
    writer.add_uint32("moss_diarize.llm.n_layers", llm_layers)
    writer.add_uint32("moss_diarize.llm.dim", llm_hidden)
    writer.add_uint32("moss_diarize.llm.n_heads", llm_n_heads)
    writer.add_uint32("moss_diarize.llm.n_kv_heads", llm_n_kv_heads)
    writer.add_uint32("moss_diarize.llm.head_dim", llm_head_dim)
    writer.add_uint32("moss_diarize.llm.ff_dim", llm_ff_dim)
    writer.add_uint32("moss_diarize.llm.vocab_size", llm_vocab_size)
    writer.add_bool("moss_diarize.llm.tied_embeddings", lc.get("tie_word_embeddings", True))
    writer.add_float32("moss_diarize.llm.rope_theta", llm_rope_theta)
    writer.add_float32("moss_diarize.llm.rms_eps", llm_rms_eps)

    # Audio/diarize params
    writer.add_uint32("moss_diarize.audio_token_id", audio_token_id)
    writer.add_uint32("moss_diarize.time_marker_every_seconds", time_marker_every_seconds)
    writer.add_float32("moss_diarize.audio_tokens_per_second", audio_tokens_per_second)
    writer.add_uint32("moss_diarize.audio_merge_size", audio_merge_size)

    # Bake mel filterbank + Hann window from WhisperFeatureExtractor
    try:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor(
            feature_size=n_mels, sampling_rate=16000,
            hop_length=160, n_fft=400)
        mel_filters = np.ascontiguousarray(np.asarray(fe.mel_filters, dtype=np.float32))
        writer.add_tensor("audio.mel_filters", mel_filters)
        print(f"  mel_filters shape: {mel_filters.shape}")
        win = np.asarray(
            [0.5 * (1.0 - np.cos(2.0 * np.pi * i / 400)) for i in range(400)],
            dtype=np.float32)
        writer.add_tensor("audio.mel_window", win)
        print(f"  mel_window shape: {win.shape}")
    except ImportError:
        print("  WARNING: transformers not available, skipping mel filter bake")

    # Tokenizer: GPT-2 byte-level BPE vocab + merges (Qwen3)
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path, encoding="utf-8") as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        added = tok_data.get("added_tokens", [])
        tokens = [f"[PAD{i}]" for i in range(llm_vocab_size)]
        for token, idx in vocab.items():
            if 0 <= idx < llm_vocab_size:
                tokens[idx] = token
        for entry in added:
            tid = entry.get("id")
            content = entry.get("content")
            if content and tid is not None and 0 <= tid < llm_vocab_size:
                tokens[tid] = content
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        print(f"  Tokenizer: {len(tokens)} tokens ({len(vocab)} BPE + {len(added)} added)")
        raw_merges = tok_data.get("model", {}).get("merges", [])
        merges = [" ".join(m) if isinstance(m, list) else m for m in raw_merges]
    else:
        merges = []

    merges_path = model_dir / "merges.txt"
    if merges_path.exists():
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line and not line.startswith("#"):
                    merges.append(line)
    if merges:
        writer.add_token_merges(merges)
        print(f"  Merges: {len(merges)}")

    # Map and write tensors — stream one-at-a-time to minimize RAM.
    mapped = 0
    skipped = []
    for hf_name in sorted(tensor_names.keys()):
        gguf_name = map_tensor_name(hf_name)
        if gguf_name is None:
            skipped.append(hf_name)
            continue
        h = handles[tensor_names[hf_name]]
        tensor = h.get_tensor(hf_name)
        if tensor.dtype == torch.bfloat16:
            arr = tensor.to(torch.float32).numpy()
        else:
            arr = tensor.numpy()
        if arr.ndim >= 2:
            arr = arr.astype(out_dtype)
            dtype = ggml_type
        else:
            arr = arr.astype(np.float32)
            dtype = GGMLQuantizationType.F32
        writer.add_tensor(gguf_name, arr, raw_dtype=dtype)
        mapped += 1
        del tensor, arr

    print(f"\n  Mapped {mapped} tensors, skipped {len(skipped)}")
    for s in skipped:
        print(f"    SKIP: {s}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\n  Written: {outfile} ({outfile.stat().st_size / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert OpenMOSS-Team/MOSS-Audio-4B-Instruct to GGUF format for CrispASR.

Architecture: 32-layer Whisper-style audio encoder (3× Conv2d stem,
1280d, 20 heads, FFN 5120) + DeepStack 3-tap adapter +
GatedMLP audio adapter + 36-layer Qwen3 LM (2560d, 32Q/8KV heads,
SwiGLU 9728, QK-norm, RoPE θ=1M, vocab=151936).

Streams tensors one-at-a-time via safe_open (BF16→F16). ~12 GB peak RAM.

Usage:
    python models/convert-moss-audio-to-gguf.py \\
        --input OpenMOSS-Team/MOSS-Audio-4B-Instruct \\
        --output moss-audio-4b-instruct-f16.gguf

    # Then quantize:
    crispasr-quantize moss-audio-4b-instruct-f16.gguf \\
                      moss-audio-4b-instruct-q4_k.gguf q4_k
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


def map_tensor_name(hf_name: str) -> str:
    """Map HF tensor name to GGUF name for MOSS-Audio."""
    name = hf_name

    # Audio encoder convolutions
    name = name.replace("audio_encoder.conv1.", "enc.conv1.")
    name = name.replace("audio_encoder.conv2.", "enc.conv2.")
    name = name.replace("audio_encoder.conv3.", "enc.conv3.")

    # Audio encoder stem projection + layer norm
    name = name.replace("audio_encoder.stem_proj.", "enc.stem_proj.")
    name = name.replace("audio_encoder.layer_norm.", "enc.norm.")

    # Audio encoder transformer layers
    name = name.replace("audio_encoder.layers.", "enc.blk.")

    # Whisper encoder layer internals
    name = name.replace(".self_attn_layer_norm.", ".attn_norm.")
    name = name.replace(".final_layer_norm.", ".ffn_norm.")
    name = name.replace(".self_attn.q_proj.", ".attn.q.")
    name = name.replace(".self_attn.k_proj.", ".attn.k.")
    name = name.replace(".self_attn.v_proj.", ".attn.v.")
    name = name.replace(".self_attn.out_proj.", ".attn.o.")
    name = name.replace(".fc1.", ".ffn.fc1.")
    name = name.replace(".fc2.", ".ffn.fc2.")

    # Audio adapter (final encoder → LM space)
    name = name.replace("audio_adapter.gate_proj.", "adapter.gate.")
    name = name.replace("audio_adapter.up_proj.", "adapter.up.")
    name = name.replace("audio_adapter.down_proj.", "adapter.down.")

    # DeepStack mergers (3 independent GatedMLPs)
    name = name.replace("deepstack_audio_merger_list.", "deepstack.")
    # deepstack.{i}.gate_proj. → deepstack.{i}.gate. etc
    for prefix in ["deepstack.0.", "deepstack.1.", "deepstack.2."]:
        name = name.replace(f"{prefix}gate_proj.", f"{prefix}gate.")
        name = name.replace(f"{prefix}up_proj.", f"{prefix}up.")
        name = name.replace(f"{prefix}down_proj.", f"{prefix}down.")

    # Language model embedding + final norm
    name = name.replace("language_model.embed_tokens.", "llm.embed.")
    name = name.replace("language_model.norm.", "llm.final_norm.")

    # Language model transformer layers
    name = name.replace("language_model.layers.", "llm.blk.")
    # Qwen3 LM MLP (must come before generic replacements)
    name = name.replace(".mlp.gate_proj.", ".ffn.gate.")
    name = name.replace(".mlp.up_proj.", ".ffn.up.")
    name = name.replace(".mlp.down_proj.", ".ffn.down.")
    # Qwen3 attention: q_proj, k_proj, v_proj, o_proj + q_norm, k_norm
    name = name.replace(".self_attn.q_proj.", ".attn.q.")
    name = name.replace(".self_attn.k_proj.", ".attn.k.")
    name = name.replace(".self_attn.v_proj.", ".attn.v.")
    name = name.replace(".self_attn.o_proj.", ".attn.o.")
    name = name.replace(".self_attn.q_norm.", ".attn.q_norm.")
    name = name.replace(".self_attn.k_norm.", ".attn.k_norm.")
    name = name.replace(".input_layernorm.", ".attn_norm.")
    name = name.replace(".post_attention_layernorm.", ".ffn_norm.")

    # LM head
    name = name.replace("lm_head.", "llm.lm_head.")

    return name


def main():
    parser = argparse.ArgumentParser(description="Convert MOSS-Audio-4B-Instruct to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16"],
                        help="Output data type for 2D+ tensors (default: f16)")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    ac = config.get("audio_config", {})
    lc = config.get("language_config", {})

    print(f"\nMOSS-Audio-4B-Instruct")
    print(f"  Audio encoder: {ac.get('encoder_layers', 32)}L, d={ac.get('d_model', 1280)}, "
          f"heads={ac.get('encoder_attention_heads', 20)}, FFN={ac.get('encoder_ffn_dim', 5120)}")
    print(f"  DeepStack taps: {ac.get('deepstack_encoder_layer_indexes', [8,16,24])}, "
          f"inject_layers={config.get('deepstack_num_inject_layers', 3)}")
    print(f"  Adapter hidden: {config.get('adapter_hidden_size', 8192)}")
    print(f"  LM: {lc.get('num_hidden_layers', 36)}L, hidden={lc.get('hidden_size', 2560)}, "
          f"heads={lc.get('num_attention_heads', 32)}, kv_heads={lc.get('num_key_value_heads', 8)}")
    print(f"  Vocab: {lc.get('vocab_size', 151936)}")

    if args.outtype == "f16":
        out_dtype = np.float16
        ggml_type = GGMLQuantizationType.F16
    else:
        out_dtype = np.float32
        ggml_type = GGMLQuantizationType.F32

    # Open safetensors (sharded)
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        st_files = sorted(model_dir.glob("model-*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    tensor_names = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            tensor_names[name] = idx
    print(f"  Safetensors: {len(tensor_names)} tensors in {len(st_files)} file(s)")

    # Create GGUF
    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "moss_audio", use_temp_file=True)
    writer.add_name("MOSS-Audio-4B-Instruct")

    # Audio encoder params
    writer.add_uint32("moss_audio.enc.num_mel_bins", ac.get("num_mel_bins", 128))
    writer.add_uint32("moss_audio.enc.encoder_layers", ac.get("encoder_layers", 32))
    writer.add_uint32("moss_audio.enc.d_model", ac.get("d_model", 1280))
    writer.add_uint32("moss_audio.enc.encoder_attention_heads", ac.get("encoder_attention_heads", 20))
    writer.add_uint32("moss_audio.enc.encoder_ffn_dim", ac.get("encoder_ffn_dim", 5120))
    writer.add_uint32("moss_audio.enc.downsample_rate", ac.get("downsample_rate", 8))
    writer.add_uint32("moss_audio.enc.downsample_hidden_size", ac.get("downsample_hidden_size", 480))
    writer.add_uint32("moss_audio.enc.max_source_positions", ac.get("max_source_positions", 1500))
    writer.add_uint32("moss_audio.enc.encoder_attention_window_size",
                       ac.get("encoder_attention_window_size", 100))
    writer.add_uint32("moss_audio.enc.output_dim", ac.get("output_dim", 1280))
    writer.add_float32("moss_audio.enc.layer_norm_eps", ac.get("layer_norm_eps", 1e-5))

    # DeepStack params
    ds_taps = ac.get("deepstack_encoder_layer_indexes", [8, 16, 24])
    writer.add_uint32("moss_audio.deepstack.num_taps", len(ds_taps))
    for i, tap in enumerate(ds_taps):
        writer.add_uint32(f"moss_audio.deepstack.tap.{i}", tap)
    writer.add_uint32("moss_audio.deepstack.num_inject_layers",
                       config.get("deepstack_num_inject_layers", len(ds_taps)))

    # Adapter params
    writer.add_uint32("moss_audio.adapter.hidden_size", config.get("adapter_hidden_size", 8192))

    # LM params (Qwen3)
    writer.add_uint32("moss_audio.llm.hidden_size", lc.get("hidden_size", 2560))
    writer.add_uint32("moss_audio.llm.num_layers", lc.get("num_hidden_layers", 36))
    writer.add_uint32("moss_audio.llm.num_heads", lc.get("num_attention_heads", 32))
    writer.add_uint32("moss_audio.llm.num_kv_heads", lc.get("num_key_value_heads", 8))
    writer.add_uint32("moss_audio.llm.head_dim", lc.get("head_dim", 128))
    writer.add_uint32("moss_audio.llm.intermediate_size", lc.get("intermediate_size", 9728))
    writer.add_uint32("moss_audio.llm.vocab_size", lc.get("vocab_size", 151936))
    writer.add_uint32("moss_audio.llm.max_position_embeddings",
                       lc.get("max_position_embeddings", 40960))
    writer.add_float32("moss_audio.llm.rope_theta", lc.get("rope_theta", 1000000.0))
    writer.add_float32("moss_audio.llm.rms_norm_eps", lc.get("rms_norm_eps", 1e-6))

    # Special tokens
    writer.add_uint32("moss_audio.bos_token_id", config.get("bos_token_id", 151643))
    writer.add_uint32("moss_audio.eos_token_id", config.get("eos_token_id", 151645))
    writer.add_uint32("moss_audio.audio_token_id", 151654)
    writer.add_uint32("moss_audio.audio_start_id", 151669)
    writer.add_uint32("moss_audio.audio_end_id", 151670)

    # Bake mel filterbank + Hann window from WhisperFeatureExtractor
    # so the C++ runtime uses the exact same filters as the Python reference.
    try:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor(
            feature_size=ac.get("num_mel_bins", 128),
            sampling_rate=16000,
            hop_length=ac.get("mel_hop_length", 160) if "mel_hop_length" in ac else 160,
            n_fft=ac.get("mel_n_fft", 400) if "mel_n_fft" in ac else 400,
        )
        mel_filters = np.ascontiguousarray(np.asarray(fe.mel_filters, dtype=np.float32))
        writer.add_tensor("audio.mel_filters", mel_filters)
        print(f"  mel_filters shape: {mel_filters.shape}")

        n_fft = fe.n_fft if hasattr(fe, 'n_fft') else 400
        win = np.asarray(
            [0.5 * (1.0 - np.cos(2.0 * np.pi * i / n_fft)) for i in range(n_fft)],
            dtype=np.float32)
        writer.add_tensor("audio.mel_window", win)
        print(f"  mel_window shape: {win.shape}")
    except ImportError:
        print("  WARNING: transformers not available, skipping mel filter bake")

    # Tokenizer: BPE vocab + merges
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path, encoding="utf-8") as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        added = tok_data.get("added_tokens", [])
        vocab_size = lc.get("vocab_size", 151936)
        tokens = [f"[PAD{i}]" for i in range(vocab_size)]
        for token, idx in vocab.items():
            if 0 <= idx < vocab_size:
                tokens[idx] = token
        for entry in added:
            tid = entry.get("id")
            content = entry.get("content")
            if content and tid is not None and 0 <= tid < vocab_size:
                tokens[tid] = content
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        print(f"  Tokenizer: {len(tokens)} tokens "
              f"({len(vocab)} BPE + {len(added)} added)")
    elif (model_dir / "vocab.json").exists():
        with open(model_dir / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab_size = lc.get("vocab_size", 151936)
        tokens = [f"[PAD{i}]" for i in range(vocab_size)]
        for token, idx in vocab.items():
            if 0 <= idx < vocab_size:
                tokens[idx] = token
        # Patch added_tokens
        at_path = model_dir / "added_tokens.json"
        if at_path.exists():
            with open(at_path, encoding="utf-8") as f:
                added = json.load(f)
            for token, idx in added.items():
                if 0 <= idx < vocab_size:
                    tokens[idx] = token
            print(f"  Added tokens: {len(added)}")
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        print(f"  Tokenizer: {len(tokens)} tokens from vocab.json")

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
    elif tok_path.exists():
        raw_merges = tok_data.get("model", {}).get("merges", [])
        merges = []
        for m in raw_merges:
            if isinstance(m, list):
                merges.append(" ".join(m))
            else:
                merges.append(m)
        if merges:
            writer.add_token_merges(merges)
            print(f"  Merges: {len(merges)} (from tokenizer.json)")

    # Map and write tensors — stream one-at-a-time to minimize RAM.
    # BF16→F16 via torch (numpy can't handle BF16); delete immediately.
    mapped = 0
    skipped = []
    for hf_name in sorted(tensor_names.keys()):
        h = handles[tensor_names[hf_name]]
        tensor = h.get_tensor(hf_name)

        gguf_name = map_tensor_name(hf_name)

        # Convert to numpy
        if tensor.dtype == torch.bfloat16:
            arr = tensor.to(torch.float32).numpy()
        else:
            arr = tensor.numpy()

        # For 2D+ tensors, convert to target dtype
        if arr.ndim >= 2:
            arr = arr.astype(out_dtype)
            dtype = ggml_type
        else:
            # 1D tensors (biases, norms) stay F32
            arr = arr.astype(np.float32)
            dtype = GGMLQuantizationType.F32

        writer.add_tensor(gguf_name, arr, raw_dtype=dtype)
        mapped += 1
        del tensor, arr

    for h in handles:
        h.__exit__(None, None, None) if hasattr(h, '__exit__') else None

    print(f"\n  Mapped {mapped} tensors, skipped {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    SKIP: {s}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\n  Written: {outfile} ({outfile.stat().st_size / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()

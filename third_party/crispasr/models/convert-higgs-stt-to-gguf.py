#!/usr/bin/env python3
"""
Convert bosonai/higgs-audio-v3-stt (HuggingFace safetensors) -> GGUF F16.

Architecture (verified against config.json + model.safetensors.index.json +
modeling_higgs_audio.py):

  Audio encoder (audio_tower.*) -- Whisper-large-v3 encoder verbatim:
    conv1                                 (1280, 128, 3)   stride 1
    conv2                                 (1280, 1280, 3)  stride 2
    embed_positions                       (1500, 1280)     learned absolute pos
    32 x encoder block (Whisper-style pre-LN, biased, GELU FFN):
      self_attn_layer_norm.{weight,bias}  (1280,)
      self_attn.q_proj.{weight,bias}      (1280, 1280)
      self_attn.k_proj.weight             (1280, 1280)   (NO bias -- Whisper quirk)
      self_attn.v_proj.{weight,bias}      (1280, 1280)
      self_attn.out_proj.{weight,bias}    (1280, 1280)
      final_layer_norm.{weight,bias}      (1280,)
      fc1.{weight,bias}                   (5120, 1280)
      fc2.{weight,bias}                   (1280, 5120)
    layer_norm.{weight,bias}              (1280,)        post-encoder norm
    [AvgPool1d(2, stride=2) -- parameterless, applied in forward -> 25 fps]

  Audio encoder projector (audio_encoder_proj.*) -- depthwise temporal conv + MLP:
    temporal.{weight,bias}                (1280, 1, 3)   depthwise Conv1d stride 2
    linear1.{weight,bias}                 (2048, 1280)   Linear(1280 -> 2048)
    [ReLU]
    linear2.{weight,bias}                 (2048, 2048)   Linear(2048 -> 2048)
    -> 12.5 tps, dim 2048 (= LLM hidden)

  Text decoder (Qwen3-1.7B-Base, 28 layers, bare top-level names):
    embed_tokens.weight                   (151936, 2048)
    per layer:
      input_layernorm.weight              (2048,)        RMSNorm
      self_attn.q_proj.weight             (2048, 2048)   16 heads x 128
      self_attn.k_proj.weight             (1024, 2048)   8 KV heads x 128 (GQA)
      self_attn.v_proj.weight             (1024, 2048)
      self_attn.o_proj.weight             (2048, 2048)
      self_attn.q_norm.weight             (128,)         Qwen3 per-head RMSNorm
      self_attn.k_norm.weight             (128,)
      post_attention_layernorm.weight     (2048,)
      mlp.gate_proj.weight                (6144, 2048)   SwiGLU
      mlp.up_proj.weight                  (6144, 2048)
      mlp.down_proj.weight                (2048, 6144)
    norm.weight                           (2048,)
    audio_decoder_proj.text_lm_head.weight (151936, 2048) text logits head

  Skipped (TTS-only generation path, not needed for STT):
    audio_codebook_embeddings.*, audio_decoder_proj.audio_lm_head.*

GGUF tensor naming (mirrors what src/higgs_stt.cpp expects):

  audio.conv.{1,2}.{weight,bias}              F16/F32
  audio.embed_positions                       F32
  audio.blk.{i}.attn_norm.{weight,bias}       F32
  audio.blk.{i}.attn_{q,k,v,out}.{weight,bias?}  F16/F32
  audio.blk.{i}.ffn_norm.{weight,bias}        F32
  audio.blk.{i}.ffn_{up,down}.{weight,bias}   F16/F32
  audio.ln_post.{weight,bias}                 F32

  proj.temporal.{weight,bias}                 F16/F32
  proj.linear1.{weight,bias}                  F16/F32
  proj.linear2.{weight,bias}                  F16/F32

  token_embd.weight                           F16
  blk.{i}.attn_norm.weight                    F32
  blk.{i}.attn_{q,k,v,output}.weight          F16
  blk.{i}.attn_{q,k}_norm.weight              F32
  blk.{i}.ffn_norm.weight                     F32
  blk.{i}.ffn_{gate,up,down}.weight           F16
  output_norm.weight                          F32
  output.weight                               F16  (= audio_decoder_proj.text_lm_head)

GGUF metadata keys (under `higgs.*`) + GPT-2 BPE tokenizer (tokenizer.ggml.*).
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

DIRECT = {
    # Audio encoder front-end
    "audio_tower.conv1.weight": "audio.conv.1.weight",
    "audio_tower.conv1.bias": "audio.conv.1.bias",
    "audio_tower.conv2.weight": "audio.conv.2.weight",
    "audio_tower.conv2.bias": "audio.conv.2.bias",
    "audio_tower.embed_positions.weight": "audio.embed_positions",
    "audio_tower.layer_norm.weight": "audio.ln_post.weight",
    "audio_tower.layer_norm.bias": "audio.ln_post.bias",
    # Projector (depthwise temporal conv + 2x Linear)
    "audio_encoder_proj.temporal.weight": "proj.temporal.weight",
    "audio_encoder_proj.temporal.bias": "proj.temporal.bias",
    "audio_encoder_proj.linear1.weight": "proj.linear1.weight",
    "audio_encoder_proj.linear1.bias": "proj.linear1.bias",
    "audio_encoder_proj.linear2.weight": "proj.linear2.weight",
    "audio_encoder_proj.linear2.bias": "proj.linear2.bias",
    # Text decoder -- bare top-level names (no model./language_model. prefix)
    "embed_tokens.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
    # Text logits head (tie_word_embeddings=true, but the explicit head is used)
    "audio_decoder_proj.text_lm_head.weight": "output.weight",
}

# Audio encoder body -- Whisper-block style
AUDIO_LAYER_PATTERNS = [
    (r"audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.weight", "audio.blk.{}.attn_norm.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.bias", "audio.blk.{}.attn_norm.bias"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.weight", "audio.blk.{}.attn_q.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.bias", "audio.blk.{}.attn_q.bias"),
    # K has weight only (Whisper quirk)
    (r"audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.weight", "audio.blk.{}.attn_k.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.weight", "audio.blk.{}.attn_v.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.bias", "audio.blk.{}.attn_v.bias"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.weight", "audio.blk.{}.attn_out.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.bias", "audio.blk.{}.attn_out.bias"),
    (r"audio_tower\.layers\.(\d+)\.final_layer_norm\.weight", "audio.blk.{}.ffn_norm.weight"),
    (r"audio_tower\.layers\.(\d+)\.final_layer_norm\.bias", "audio.blk.{}.ffn_norm.bias"),
    (r"audio_tower\.layers\.(\d+)\.fc1\.weight", "audio.blk.{}.ffn_up.weight"),
    (r"audio_tower\.layers\.(\d+)\.fc1\.bias", "audio.blk.{}.ffn_up.bias"),
    (r"audio_tower\.layers\.(\d+)\.fc2\.weight", "audio.blk.{}.ffn_down.weight"),
    (r"audio_tower\.layers\.(\d+)\.fc2\.bias", "audio.blk.{}.ffn_down.bias"),
]

# Text decoder body -- Qwen3
TEXT_LAYER_PATTERNS = [
    (r"layers\.(\d+)\.input_layernorm\.weight", "blk.{}.attn_norm.weight"),
    (r"layers\.(\d+)\.self_attn\.q_proj\.weight", "blk.{}.attn_q.weight"),
    (r"layers\.(\d+)\.self_attn\.k_proj\.weight", "blk.{}.attn_k.weight"),
    (r"layers\.(\d+)\.self_attn\.v_proj\.weight", "blk.{}.attn_v.weight"),
    (r"layers\.(\d+)\.self_attn\.o_proj\.weight", "blk.{}.attn_output.weight"),
    (r"layers\.(\d+)\.self_attn\.q_norm\.weight", "blk.{}.attn_q_norm.weight"),
    (r"layers\.(\d+)\.self_attn\.k_norm\.weight", "blk.{}.attn_k_norm.weight"),
    (r"layers\.(\d+)\.post_attention_layernorm\.weight", "blk.{}.ffn_norm.weight"),
    (r"layers\.(\d+)\.mlp\.gate_proj\.weight", "blk.{}.ffn_gate.weight"),
    (r"layers\.(\d+)\.mlp\.up_proj\.weight", "blk.{}.ffn_up.weight"),
    (r"layers\.(\d+)\.mlp\.down_proj\.weight", "blk.{}.ffn_down.weight"),
]

# Explicitly-dropped tensors (TTS-only generation path)
SKIP_PREFIXES = (
    "audio_codebook_embeddings",
    "audio_decoder_proj.audio_lm_head",
)


def remap_name(hf_name: str) -> str | None:
    if any(hf_name.startswith(p) for p in SKIP_PREFIXES):
        return None
    if hf_name in DIRECT:
        return DIRECT[hf_name]
    for pat, tmpl in AUDIO_LAYER_PATTERNS:
        m = re.match(pat + r"$", hf_name)
        if m:
            return tmpl.format(m.group(1))
    for pat, tmpl in TEXT_LAYER_PATTERNS:
        m = re.match(pat + r"$", hf_name)
        if m:
            return tmpl.format(m.group(1))
    return None


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    """Norms, biases, 1-D tensors, embed_positions stay F32 for accuracy."""
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name or "ln_post" in gguf_name:
        return True
    if "embed_positions" in gguf_name:
        return True
    if len(shape) <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(input_dir: Path, out_path: Path) -> None:
    print(f"Loading: {input_dir}")
    with open(input_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    audio = cfg["audio_encoder_config"]
    text = cfg["text_config"]

    safetensor_files = sorted(input_dir.glob("model-*.safetensors"))
    if not safetensor_files:
        safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        sys.exit(f"no safetensors files found in {input_dir}")
    print(f"  shards: {[p.name for p in safetensor_files]}")

    # ----- Tokenizer (GPT-2 byte-level BPE, like Qwen3-ASR) -----
    with open(input_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    sorted_vocab = sorted(vocab_dict.items(), key=lambda kv: kv[1])
    vocab_size = text.get("vocab_size", 151936)
    tokens = [tok for tok, _ in sorted_vocab]
    while len(tokens) < vocab_size:
        tokens.append(f"[PAD{len(tokens)}]")
    # Patch special tokens (e.g. <|im_start|>, <|AUDIO|>) from
    # tokenizer_config.json's added_tokens_decoder at their proper IDs.
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
    print(f"  vocab: {len(tokens)} tokens")

    merges = []
    merges_path = input_dir / "merges.txt"
    if merges_path.exists():
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line and not line.startswith("#"):
                    merges.append(line)
        print(f"  merges: {len(merges)}")

    # ----- write GGUF -----
    print(f"Writing: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch="higgs-stt")

    # Audio params
    writer.add_uint32("higgs.sample_rate", 16000)
    writer.add_uint32("higgs.n_mels", audio.get("num_mel_bins", 128))
    writer.add_uint32("higgs.n_fft", 400)
    writer.add_uint32("higgs.win_length", 400)
    writer.add_uint32("higgs.hop_length", 160)
    writer.add_uint32("higgs.audio.n_layers", audio.get("encoder_layers", 32))
    writer.add_uint32("higgs.audio.d_model", audio.get("d_model", 1280))
    writer.add_uint32("higgs.audio.n_heads", audio.get("encoder_attention_heads", 20))
    writer.add_uint32(
        "higgs.audio.head_dim",
        audio.get("d_model", 1280) // audio.get("encoder_attention_heads", 20),
    )
    writer.add_uint32("higgs.audio.ff_dim", audio.get("encoder_ffn_dim", 5120))
    writer.add_uint32("higgs.audio.max_pos", audio.get("max_source_positions", 1500))
    # Encoder applies AvgPool1d(2,2) after the final layer_norm (frame_rate 25).
    writer.add_uint32("higgs.audio.avgpool", 2)

    # Projector params (depthwise temporal conv stride 2 + Linear(1280->2048) + ReLU + Linear(2048->2048))
    writer.add_uint32("higgs.proj.in_dim", audio.get("d_model", 1280))
    writer.add_uint32("higgs.proj.hidden_dim", text.get("hidden_size", 2048))
    writer.add_uint32("higgs.proj.out_dim", text.get("hidden_size", 2048))
    writer.add_uint32(
        "higgs.proj.temporal_downsample", cfg.get("projector_temporal_downsample", 2)
    )

    # LLM params (Qwen3-1.7B)
    writer.add_uint32("higgs.llm.n_layers", text.get("num_hidden_layers", 28))
    writer.add_uint32("higgs.llm.d_model", text.get("hidden_size", 2048))
    writer.add_uint32("higgs.llm.n_heads", text.get("num_attention_heads", 16))
    writer.add_uint32("higgs.llm.n_kv_heads", text.get("num_key_value_heads", 8))
    writer.add_uint32("higgs.llm.head_dim", text.get("head_dim", 128))
    writer.add_uint32("higgs.llm.ff_dim", text.get("intermediate_size", 6144))
    writer.add_float32("higgs.llm.rope_theta", float(text.get("rope_theta", 1000000)))
    writer.add_float32("higgs.llm.rms_norm_eps", float(text.get("rms_norm_eps", 1e-6)))
    writer.add_uint32("higgs.llm.vocab_size", vocab_size)
    writer.add_uint32("higgs.llm.max_pos", text.get("max_position_embeddings", 32768))

    # Special tokens
    writer.add_uint32("higgs.audio_bos_token_id", 151669)  # <|audio_bos|>
    writer.add_uint32(
        "higgs.audio_in_token_id", cfg.get("audio_in_token_idx", 151672)
    )  # <|AUDIO|>
    writer.add_uint32(
        "higgs.audio_eos_token_id", cfg.get("audio_eos_token_id", 151670)
    )  # <|audio_eos|>
    writer.add_uint32("higgs.im_start_token_id", 151644)
    writer.add_uint32("higgs.im_end_token_id", 151645)
    writer.add_uint32("higgs.eos_token_id", text.get("eos_token_id", 151643))
    writer.add_uint32("higgs.pad_token_id", cfg.get("pad_token_id", 151643))

    # Tokenizer
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    if merges:
        writer.add_token_merges(merges)

    # ----- Mel filterbank + Hann window (Whisper-large-v3) -----
    try:
        from transformers import WhisperFeatureExtractor

        try:
            fe = WhisperFeatureExtractor.from_pretrained(str(input_dir))
        except Exception:
            fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    except Exception as e:
        sys.exit(f"need transformers for WhisperFeatureExtractor mel filters: {e}")
    mel_filters = np.ascontiguousarray(
        np.asarray(fe.mel_filters, dtype=np.float32)
    )  # (n_freqs, n_mels)
    print(f"  mel_filters shape: {mel_filters.shape}")
    writer.add_tensor("audio.mel_filters", mel_filters)
    n_fft_w = 400
    win = np.ascontiguousarray(
        (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft_w) / n_fft_w)).astype(np.float32)
    )
    writer.add_tensor("audio.mel_window", win)
    print(f"  mel_window shape: {win.shape}")

    # ----- tensors -----
    n_written = n_f16 = n_f32 = n_skipped = 0
    skipped_names: list[str] = []

    for sf_path in safetensor_files:
        print(f"  reading {sf_path.name}")
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for hf_name in sorted(f.keys()):
                gguf_name = remap_name(hf_name)
                if gguf_name is None:
                    n_skipped += 1
                    skipped_names.append(hf_name)
                    continue
                t = f.get_tensor(hf_name)
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
                if not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)
                writer.add_tensor(gguf_name, arr)
                n_written += 1
                if n_written <= 30 or n_written % 100 == 0:
                    print(f"    {gguf_name:46s} {str(arr.shape):24s} {arr.dtype}")

    print(
        f"\n  total: {n_written} tensors  (F16: {n_f16}, F32: {n_f32})  skipped: {n_skipped}"
    )
    if skipped_names:
        print("  skipped tensors (first 12):")
        for n in skipped_names[:12]:
            print(f"    {n}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert bosonai/higgs-audio-v3-stt HF safetensors -> GGUF F16"
    )
    p.add_argument("--input", required=True, type=Path, help="HF model directory")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)

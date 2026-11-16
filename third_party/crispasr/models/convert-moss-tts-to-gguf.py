#!/usr/bin/env python3
"""Convert OpenMOSS-Team/MOSS-TTS-v1.5 (MossTTSDelay) to GGUF for CrispASR.

Architecture (verified against config.json + pwilkin/openmoss, 2026-07-12):

  Backbone  : Qwen3-8B — 36 layers, hidden 4096, 32 Q-heads / 8 KV-heads,
              head_dim 128, SwiGLU intermediate 12288, QK-norm, RoPE (NEOX)
              theta 1e6, rms_eps 1e-6, vocab 155648.
  Audio     : 32 RVQ codebooks. Input embedding = text_emb + Sum_i audio_emb_i;
              33 LM heads (1 text = lm_heads.0, 32 audio = lm_heads.1..32).
              emb_ext.{0..31}.weight  (audio_vocab+1 = 1025, hidden 4096).
  Codec     : separate repo OpenMOSS-Team/MOSS-Audio-Tokenizer — a 1.6B
              pure-transformer RVQ codec (32 codebooks, dim 8, rvq 512, out 768,
              4 ProjectedTransformer decoder stages, hop 1920 -> 24 kHz).

Output layout (CrispASR convention, mirrors qwen3-tts talker + companion codec):

  <output>.gguf            backbone GGUF, arch "moss-tts":
      llm.*                Qwen3 backbone (quantizable by crispasr-quantize)
      moss.audio_embed.{i} 32 audio embedding tables      (kept F16)
      moss.audio_head.{i}  32 audio LM heads              (kept F16)
      moss_tts.llm.* KV    backbone hparams
      moss.* KV            n_vq, audio vocab, token ids, sampling/frame rate
      gpt2 BPE tokenizer   (Qwen3 vocab + merges)

  <output>-codec.gguf      companion codec GGUF, arch "moss-tts-codec" (F16):
      moss.codec.*         encoder/decoder/quantizer tensors (openmoss names)
      moss.codec.* KV      dims, frame rate, sliding-window context

Streams tensors one-at-a-time via safe_open (BF16->F16); peak RAM stays modest.

Usage:
    # Backbone only (start the runtime before the codec is ported):
    python models/convert-moss-tts-to-gguf.py \\
        --input OpenMOSS-Team/MOSS-TTS-v1.5 \\
        --output moss-tts-v1.5-f16.gguf

    # With the codec companion:
    python models/convert-moss-tts-to-gguf.py \\
        --input OpenMOSS-Team/MOSS-TTS-v1.5 \\
        --codec OpenMOSS-Team/MOSS-Audio-Tokenizer \\
        --output moss-tts-v1.5-f16.gguf

    # Then quantize the backbone (codec stays F16):
    crispasr-quantize moss-tts-v1.5-f16.gguf moss-tts-v1.5-q4_k.gguf q4_k
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("Error: torch not found. Install torch (BF16->F16 needs it).")
    sys.exit(1)

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


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading model from HuggingFace: {model_id}")
    return Path(snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json",
                        "tokenizer.json", "added_tokens.json"]))


# ---------------------------------------------------------------------------
# Backbone tensor-name mapping: MossTTSDelay language_model.* -> llm.*
# (identical Qwen3 layout to MOSS-Audio, so we reuse that convention verbatim.)
# ---------------------------------------------------------------------------

def map_backbone_name(hf_name: str) -> str:
    name = hf_name
    name = name.replace("language_model.embed_tokens.", "llm.embed.")
    name = name.replace("language_model.norm.", "llm.final_norm.")
    name = name.replace("language_model.layers.", "llm.blk.")
    # Qwen3 MLP (before generic attn replacements)
    name = name.replace(".mlp.gate_proj.", ".ffn.gate.")
    name = name.replace(".mlp.up_proj.", ".ffn.up.")
    name = name.replace(".mlp.down_proj.", ".ffn.down.")
    # Qwen3 attention + QK-norm
    name = name.replace(".self_attn.q_proj.", ".attn.q.")
    name = name.replace(".self_attn.k_proj.", ".attn.k.")
    name = name.replace(".self_attn.v_proj.", ".attn.v.")
    name = name.replace(".self_attn.o_proj.", ".attn.o.")
    name = name.replace(".self_attn.q_norm.", ".attn.q_norm.")
    name = name.replace(".self_attn.k_norm.", ".attn.k_norm.")
    name = name.replace(".input_layernorm.", ".attn_norm.")
    name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
    return name


def classify_moss_tensor(hf_name: str):
    """Return (kind, gguf_name) for a MOSS-TTS safetensors tensor, or (None, None)
    to skip. kind in {"backbone", "audio_embed", "audio_head", "text_head"}."""
    if hf_name.startswith("language_model."):
        return "backbone", map_backbone_name(hf_name)
    if hf_name.startswith("emb_ext.") and hf_name.endswith(".weight"):
        idx = int(hf_name.split(".")[1])
        return "audio_embed", f"moss.audio_embed.{idx}.weight"
    if hf_name.startswith("lm_heads.") and hf_name.endswith(".weight"):
        idx = int(hf_name.split(".")[1])
        if idx == 0:
            # Text head -> backbone lm_head (Qwen3-8B is not tied).
            return "text_head", "llm.lm_head.weight"
        return "audio_head", f"moss.audio_head.{idx - 1}.weight"
    return None, None


# ---------------------------------------------------------------------------
# Codec tensor-name shortening (fit the 64-byte GGUF name limit).
# Ported verbatim from pwilkin/openmoss scripts/convert_hf_to_gguf.py so the
# codec runtime can follow openmoss codec.cpp's tensor names exactly.
# Order matters: longer / more-specific patterns first.
# ---------------------------------------------------------------------------

_CODEC_RENAMES = (
    ("parametrizations.weight.original0", "wp0"),
    ("parametrizations.weight.original1", "wp1"),
    ("transformer.layers", "tr.l"),
    ("encoder", "enc"),
    ("decoder", "dec"),
    ("self_attn", "attn"),
    ("in_projs", "inp"),
    ("out_projs", "outp"),
    ("quantizers", "q"),
    ("input_proj", "iproj"),
    ("output_proj", "oproj"),
    ("in_proj", "iproj"),
    ("out_proj", "oproj"),
)


def shorten_codec_name(name: str) -> str:
    s = name
    for pat, rep in _CODEC_RENAMES:
        s = s.replace(pat, rep)
    return "moss.codec." + s


# ---------------------------------------------------------------------------
# Tokenizer bake (Qwen3 gpt2-style BPE) — mirrors convert-moss-audio.
# ---------------------------------------------------------------------------

def bake_tokenizer(writer: GGUFWriter, model_dir: Path, vocab_size: int):
    tok_path = model_dir / "tokenizer.json"
    added = []
    if tok_path.exists():
        with open(tok_path, encoding="utf-8") as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        added = tok_data.get("added_tokens", [])
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
        raw_merges = tok_data.get("model", {}).get("merges", [])
        merges = [" ".join(m) if isinstance(m, list) else m for m in raw_merges]
        if merges:
            writer.add_token_merges(merges)
        print(f"  Tokenizer: {len(tokens)} tokens "
              f"({len(vocab)} BPE + {len(added)} added), {len(merges)} merges")
        return

    if (model_dir / "vocab.json").exists():
        with open(model_dir / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        tokens = [f"[PAD{i}]" for i in range(vocab_size)]
        for token, idx in vocab.items():
            if 0 <= idx < vocab_size:
                tokens[idx] = token
        at_path = model_dir / "added_tokens.json"
        if at_path.exists():
            with open(at_path, encoding="utf-8") as f:
                for token, idx in json.load(f).items():
                    if 0 <= idx < vocab_size:
                        tokens[idx] = token
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        merges_path = model_dir / "merges.txt"
        if merges_path.exists():
            with open(merges_path, encoding="utf-8") as f:
                merges = [ln.rstrip("\n") for ln in f
                          if ln.strip() and not ln.startswith("#")]
            if merges:
                writer.add_token_merges(merges)
            print(f"  Tokenizer: vocab.json + {len(merges)} merges")
        return

    print("  WARNING: no tokenizer.json / vocab.json found — tokenizer not baked")


# ---------------------------------------------------------------------------
# Backbone + audio GGUF
# ---------------------------------------------------------------------------

def to_gguf_array(tensor, out_dtype, ggml_type, keep_f16=False):
    """torch tensor -> (numpy array, GGMLQuantizationType). 1D stays F32
    (norms/biases); 2D+ becomes out_dtype unless keep_f16 forces F16."""
    if tensor.dtype == torch.bfloat16:
        arr = tensor.to(torch.float32).numpy()
    else:
        arr = tensor.numpy()
    if arr.ndim >= 2:
        if keep_f16:
            return arr.astype(np.float16), GGMLQuantizationType.F16
        return arr.astype(out_dtype), ggml_type
    return arr.astype(np.float32), GGMLQuantizationType.F32


def write_backbone_gguf(model_dir: Path, config: dict, out_path: Path,
                        out_dtype, ggml_type):
    lc = config.get("language_config", {})
    n_vq = int(config.get("n_vq", 32))
    audio_vocab_size = int(config.get("audio_vocab_size", 1024))
    audio_pad_code = int(config.get("audio_pad_code", audio_vocab_size))
    sampling_rate = int(config.get("sampling_rate", 24000))
    downsample_rate = 1920  # codec hop, fixed by upstream
    vocab_size = int(lc.get("vocab_size", 155648))

    print("\nMOSS-TTS-v1.5 (MossTTSDelay)")
    print(f"  Backbone: {lc.get('num_hidden_layers', 36)}L, hidden={lc.get('hidden_size', 4096)}, "
          f"heads={lc.get('num_attention_heads', 32)}, kv={lc.get('num_key_value_heads', 8)}, "
          f"head_dim={lc.get('head_dim', 128)}, ffn={lc.get('intermediate_size', 12288)}")
    print(f"  Audio: n_vq={n_vq}, audio_vocab={audio_vocab_size}(+1 pad), vocab={vocab_size}")

    writer = GGUFWriter(str(out_path), "moss-tts", use_temp_file=True)
    writer.add_name("MOSS-TTS-v1.5")

    # Backbone (Qwen3) hparams
    writer.add_uint32("moss_tts.llm.hidden_size", int(lc.get("hidden_size", 4096)))
    writer.add_uint32("moss_tts.llm.num_layers", int(lc.get("num_hidden_layers", 36)))
    writer.add_uint32("moss_tts.llm.num_heads", int(lc.get("num_attention_heads", 32)))
    writer.add_uint32("moss_tts.llm.num_kv_heads", int(lc.get("num_key_value_heads", 8)))
    writer.add_uint32("moss_tts.llm.head_dim", int(lc.get("head_dim", 128)))
    writer.add_uint32("moss_tts.llm.intermediate_size", int(lc.get("intermediate_size", 12288)))
    writer.add_uint32("moss_tts.llm.vocab_size", vocab_size)
    writer.add_uint32("moss_tts.llm.max_position_embeddings",
                      int(lc.get("max_position_embeddings", 40960)))
    writer.add_float32("moss_tts.llm.rope_theta", float(lc.get("rope_theta", 1000000.0)))
    writer.add_float32("moss_tts.llm.rms_norm_eps", float(lc.get("rms_norm_eps", 1e-6)))

    # Audio / delay hparams (moss.* namespace, matching openmoss KV)
    writer.add_uint32("moss.n_vq", n_vq)
    writer.add_uint32("moss.audio_vocab_size", audio_vocab_size)
    writer.add_uint32("moss.audio_pad_code", audio_pad_code)
    writer.add_uint32("moss.sampling_rate", sampling_rate)
    writer.add_uint32("moss.downsample_rate", downsample_rate)
    writer.add_float32("moss.frame_rate", float(sampling_rate) / float(downsample_rate))

    # Special token ids (defaults from MOSS-TTS-v1.5 config.json)
    writer.add_uint32("moss.token.audio_start", int(config.get("audio_start_token_id", 151652)))
    writer.add_uint32("moss.token.audio_end", int(config.get("audio_end_token_id", 151653)))
    writer.add_uint32("moss.token.audio_user_slot",
                      int(config.get("audio_user_slot_token_id", 151654)))
    writer.add_uint32("moss.token.audio_gen_slot",
                      int(config.get("audio_assistant_gen_slot_token_id", 151656)))
    writer.add_uint32("moss.token.audio_delay_slot",
                      int(config.get("audio_assistant_delay_slot_token_id", 151662)))
    writer.add_uint32("moss.token.im_start", int(config.get("im_start_token_id", 151644)))
    writer.add_uint32("moss.token.im_end",
                      int(lc.get("eos_token_id", config.get("im_end_token_id", 151645))))
    writer.add_uint32("moss.token.pad",
                      int(lc.get("pad_token_id", config.get("pad_token_id", 151643))))

    bake_tokenizer(writer, model_dir, vocab_size)

    # Stream tensors. Audio embeds/heads are precision-sensitive -> keep F16.
    st_files = sorted(model_dir.glob("*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_handle = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            name_to_handle[name] = idx
    print(f"  Safetensors: {len(name_to_handle)} tensors in {len(st_files)} file(s)")

    counts = {"backbone": 0, "audio_embed": 0, "audio_head": 0, "text_head": 0}
    skipped = []
    for hf_name in sorted(name_to_handle):
        kind, gguf_name = classify_moss_tensor(hf_name)
        if kind is None:
            skipped.append(hf_name)
            continue
        tensor = handles[name_to_handle[hf_name]].get_tensor(hf_name)
        keep_f16 = kind in ("audio_embed", "audio_head")
        arr, dtype = to_gguf_array(tensor, out_dtype, ggml_type, keep_f16=keep_f16)
        writer.add_tensor(gguf_name, arr, raw_dtype=dtype)
        counts[kind] += 1
        del tensor, arr

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Backbone tensors: {counts}")
    if skipped:
        print(f"  Skipped {len(skipped)} non-backbone tensors (first few): {skipped[:6]}")
    print(f"  Written backbone: {out_path} ({out_path.stat().st_size / 1024**3:.2f} GB)")


# ---------------------------------------------------------------------------
# Codec companion GGUF
# ---------------------------------------------------------------------------

def write_codec_gguf(codec_dir: Path, config: dict, out_path: Path):
    print(f"\nCodec (MOSS-Audio-Tokenizer) -> {out_path}")
    codec_cfg = {}
    ccfg_path = codec_dir / "config.json"
    if ccfg_path.exists():
        with open(ccfg_path, encoding="utf-8") as f:
            codec_cfg = json.load(f)

    sampling_rate = int(config.get("sampling_rate", 24000))
    downsample_rate = 1920

    writer = GGUFWriter(str(out_path), "moss-tts-codec", use_temp_file=True)
    writer.add_name("MOSS-Audio-Tokenizer")

    # Codec dims (fixed by upstream; also emitted for the runtime to assert).
    writer.add_uint32("moss.codec.num_quantizers", int(codec_cfg.get("num_quantizers", 32)))
    writer.add_uint32("moss.codec.codebook_size", int(codec_cfg.get("codebook_size", 1024)))
    writer.add_uint32("moss.codec.codebook_dim", int(codec_cfg.get("codebook_dim", 8)))
    writer.add_uint32("moss.codec.rvq_dim", int(codec_cfg.get("rvq_dim", 512)))
    writer.add_uint32("moss.codec.input_dim", int(codec_cfg.get("input_dim", 768)))
    writer.add_uint32("moss.codec.output_dim", int(codec_cfg.get("output_dim", 768)))
    writer.add_uint32("moss.codec.sampling_rate", sampling_rate)
    writer.add_uint32("moss.codec.downsample_rate", downsample_rate)
    writer.add_float32("moss.codec.frame_rate", float(sampling_rate) / float(downsample_rate))
    writer.add_uint32("moss.codec.context_duration_s",
                      int(codec_cfg.get("causal_transformer_context_duration", 10)))
    writer.add_float32("moss.codec.rope_max_period", 10000.0)
    writer.add_bool("moss.codec.present", True)

    st_files = sorted(codec_dir.glob("*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_handle = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            name_to_handle[name] = idx
    print(f"  Codec safetensors: {len(name_to_handle)} tensors in {len(st_files)} file(s)")

    seen = {}
    count = 0
    for hf_name in sorted(name_to_handle):
        gguf_name = shorten_codec_name(hf_name)
        if len(gguf_name) > 63:
            raise RuntimeError(
                f"codec tensor name over 64-byte GGUF limit ({len(gguf_name)}): {gguf_name}\n"
                f"  original: {hf_name}\n  add a rule to _CODEC_RENAMES")
        if gguf_name in seen:
            raise RuntimeError(f"codec name collision after rename: {gguf_name}\n"
                               f"  {seen[gguf_name]}\n  {hf_name}")
        seen[gguf_name] = hf_name
        tensor = handles[name_to_handle[hf_name]].get_tensor(hf_name)
        # Codec stored all-F16 to match openmoss codec.cpp's f16 weight-norm
        # reconstruction path (validated end-to-end there). Integer tensors, if
        # any, pass through their native dtype.
        if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            arr = (tensor.to(torch.float32).numpy().astype(np.float16))
            writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F16)
        else:
            writer.add_tensor(gguf_name, tensor.numpy())
        count += 1
        del tensor

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Codec tensors: {count}")
    print(f"  Written codec: {out_path} ({out_path.stat().st_size / 1024**3:.2f} GB)")


def main():
    ap = argparse.ArgumentParser(description="Convert MOSS-TTS-v1.5 to GGUF for CrispASR")
    ap.add_argument("--input", required=True,
                    help="HF id or local dir of MOSS-TTS-v1.5 (OpenMOSS-Team/MOSS-TTS-v1.5)")
    ap.add_argument("--codec", default=None,
                    help="HF id or local dir of MOSS-Audio-Tokenizer; omit to skip the codec")
    ap.add_argument("--output", required=True, help="Output backbone GGUF path")
    ap.add_argument("--codec-output", default=None,
                    help="Codec GGUF path (default: <output stem>-codec.gguf)")
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32"],
                    help="Backbone dtype for 2D+ tensors (default: f16)")
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)
    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    if args.outtype == "f16":
        out_dtype, ggml_type = np.float16, GGMLQuantizationType.F16
    else:
        out_dtype, ggml_type = np.float32, GGMLQuantizationType.F32

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_backbone_gguf(model_dir, config, out_path, out_dtype, ggml_type)

    if args.codec:
        codec_dir = load_model_dir(args.codec)
        codec_out = (Path(args.codec_output) if args.codec_output
                     else out_path.with_name(out_path.stem + "-codec.gguf"))
        write_codec_gguf(codec_dir, config, codec_out)

    print("\nDone.")


if __name__ == "__main__":
    main()

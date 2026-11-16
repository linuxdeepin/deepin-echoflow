#!/usr/bin/env python3
"""
Convert a MarianMT / Opus-MT checkpoint (Helsinki-NLP/opus-mt-de-en,
opus-mt-en-de, …) from HuggingFace PyTorch format to GGUF F16.

Architecture: MarianMTModel
  - Encoder-decoder transformer, POST-norm (normalize_before = false), no
    final encoder/decoder LayerNorm, activation from config.json (swish for
    the Opus-MT checkpoints)
  - One embedding table shared by encoder, decoder and lm_head, plus a
    `final_logits_bias` added to the logits
  - Static sinusoidal positions in Marian's layout: sin in the first half of
    the vector, cos in the second, position 0 first (no padding offset)
  - Decoder starts from <pad>; <pad> is a bad word (never generated)
  - Tokenizer: a SOURCE SentencePiece model for segmentation and a separate
    vocab.json for the ids — two tables, both embedded here (see
    src/core/marian_tokenizer.h)

Runs through the m2m100 runtime (src/m2m100.cpp), which branches on the
metadata written here.

Usage:
  python models/convert-marian-to-gguf.py \\
      --input /path/to/opus-mt-de-en \\
      --output /path/to/opus-mt-de-en-f16.gguf

The input directory needs config.json, vocab.json, source.spm, target.spm and
pytorch_model.bin (or model.safetensors); generation_config.json and
tokenizer_config.json are read when present.

GGUF tensor naming (same as convert-m2m100-to-gguf.py, minus the output
LayerNorms, plus the logits bias):

  shared.embed.weight                            F16  (vocab_size, d_model)
  final_logits_bias                              F32  (vocab_size,)
  enc.pos_emb / dec.pos_emb                      F32  (max_pos, d_model)
  enc.blk.N.{attn_q,attn_k,attn_v,attn_o,ffn_up,ffn_down}.{weight,bias}
  enc.blk.N.{attn_ln,ffn_ln}.{weight,bias}
  dec.blk.N.{attn_*,cross_*,ffn_*}.{weight,bias}
  dec.blk.N.{attn_ln,cross_ln,ffn_ln}.{weight,bias}

GGUF metadata:
  general.architecture              = "marian"
  marian.vocab_size, d_model, encoder.*, decoder.*, max_position_embeddings
  marian.scale_embedding, normalize_before, activation_function
  marian.eos_token_id, pad_token_id, unk_token_id, decoder_start_token_id
  marian.suppress_token_ids         = single-token bad_words_ids
  marian.gen.num_beams, gen.max_length, gen.early_stopping
  marian.source_lang, marian.target_lang
  tokenizer.ggml.model              = "marian"
  tokenizer.ggml.tokens             = vocab.json, by id
  tokenizer.marian.{source,target}.pieces / scores / types / charsmap
  tokenizer.marian.{source,target}.add_dummy_prefix / remove_extra_whitespaces
                                    / escape_whitespaces
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
    import torch
except ImportError:
    sys.exit("pip install torch")
try:
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2
except ImportError:
    sys.exit("pip install sentencepiece protobuf")


# ---------------------------------------------------------------------------
# Sinusoidal positional embeddings (Marian layout)
# ---------------------------------------------------------------------------


def make_marian_positions(num_positions: int, d_model: int) -> np.ndarray:
    """MarianSinusoidalPositionalEmbedding.create_weight: the XLM table with
    the features not interleaved — sin in [:d/2], cos in [d/2:]."""
    position_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]
         for pos in range(num_positions)]
    )
    out = np.zeros((num_positions, d_model), dtype=np.float32)
    sentinel = d_model // 2 if d_model % 2 == 0 else (d_model // 2) + 1
    out[:, :sentinel] = np.sin(position_enc[:, 0::2]).astype(np.float32)
    out[:, sentinel:] = np.cos(position_enc[:, 1::2]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Tensor name remapping: PyTorch → GGUF
# ---------------------------------------------------------------------------

_ATTN = [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("out_proj", "o")]


def _remap_layer_sub(sub: str, decoder: bool) -> str | None:
    for proj, out in _ATTN:
        for wb in ("weight", "bias"):
            if sub == f"self_attn.{proj}.{wb}":
                return f"attn_{out}.{wb}"
            if decoder and sub == f"encoder_attn.{proj}.{wb}":
                return f"cross_{out}.{wb}"
    for wb in ("weight", "bias"):
        if sub == f"self_attn_layer_norm.{wb}":
            return f"attn_ln.{wb}"
        if decoder and sub == f"encoder_attn_layer_norm.{wb}":
            return f"cross_ln.{wb}"
        if sub == f"fc1.{wb}":
            return f"ffn_up.{wb}"
        if sub == f"fc2.{wb}":
            return f"ffn_down.{wb}"
        if sub == f"final_layer_norm.{wb}":
            return f"ffn_ln.{wb}"
    return None


# Tied to model.shared.weight (checked, not assumed) or regenerated.
_SKIP = {
    "lm_head.weight",
    "model.encoder.embed_tokens.weight",
    "model.decoder.embed_tokens.weight",
    "model.encoder.embed_positions.weight",
    "model.decoder.embed_positions.weight",
}


def remap_name(pt_name: str) -> str | None:
    """HuggingFace Marian state-dict key → GGUF tensor name; None = skipped on
    purpose. Anything unrecognised is an error, never a silent drop."""
    n = pt_name
    if n == "model.shared.weight":
        return "shared.embed.weight"
    if n == "final_logits_bias":
        return "final_logits_bias"
    if n in _SKIP:
        return None
    for side, prefix in (("encoder", "enc"), ("decoder", "dec")):
        head = f"model.{side}.layers."
        if n.startswith(head):
            layer_id, sub = n[len(head):].split(".", 1)
            mapped = _remap_layer_sub(sub, decoder=(side == "decoder"))
            if mapped is None:
                sys.exit(f"unmapped {side} tensor: {n}")
            return f"{prefix}.blk.{layer_id}.{mapped}"
    sys.exit(f"unmapped tensor: {n} — this converter covers plain MarianMTModel "
             f"(no layernorm_embedding, no final layer norm)")


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    if "pos_emb" in gguf_name or gguf_name.endswith(".bias") or "_ln." in gguf_name:
        return True
    return len(shape) <= 1


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def load_spm(path: Path) -> dict:
    """Everything the runtime needs from one SentencePiece model. Exits on any
    feature the runtime does not implement rather than converting a tokenizer
    that would then segment differently."""
    if not path.exists():
        sys.exit(f"Missing {path}")
    m = sp_pb2.ModelProto()
    m.ParseFromString(path.read_bytes())
    ts, ns = m.trainer_spec, m.normalizer_spec
    if ts.model_type != sp_pb2.TrainerSpec.UNIGRAM:
        sys.exit(f"{path.name}: SentencePiece model_type {ts.model_type} is not UNIGRAM; "
                 f"the Marian runtime implements unigram only")
    if ts.byte_fallback:
        sys.exit(f"{path.name}: byte_fallback is not implemented")
    if ts.treat_whitespace_as_suffix:
        sys.exit(f"{path.name}: treat_whitespace_as_suffix is not implemented")
    if m.HasField("denormalizer_spec") and m.denormalizer_spec.precompiled_charsmap:
        sys.exit(f"{path.name}: a denormalizer is not implemented")
    pieces = [p.piece for p in m.pieces]
    if len(set(pieces)) != len(pieces):
        sys.exit(f"{path.name}: duplicate pieces")
    return {
        "pieces": pieces,
        "scores": [float(p.score) for p in m.pieces],
        "types": [int(p.type) for p in m.pieces],
        "charsmap": bytes(ns.precompiled_charsmap),
        "norm_name": ns.name,
        "add_dummy_prefix": bool(ns.add_dummy_prefix),
        "remove_extra_whitespaces": bool(ns.remove_extra_whitespaces),
        "escape_whitespaces": bool(ns.escape_whitespaces),
    }


def load_vocab(path: Path, vocab_size: int) -> list[str]:
    if not path.exists():
        sys.exit(f"Missing {path}")
    with open(path, encoding="utf-8") as f:
        vj = json.load(f)
    inv = {int(v): k for k, v in vj.items()}
    if len(inv) != len(vj):
        sys.exit("vocab.json: two tokens share an id")
    if sorted(inv) != list(range(vocab_size)):
        sys.exit(f"vocab.json: ids are not exactly 0..{vocab_size - 1} "
                 f"({len(inv)} entries, max id {max(inv)})")
    return [inv[i] for i in range(vocab_size)]


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(input_dir: Path, out_path: Path) -> None:
    print(f"Loading: {input_dir}")

    cfg_path = input_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"Missing config.json at {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("model_type") != "marian":
        sys.exit(f"config.json model_type is {cfg.get('model_type')!r}, not 'marian'")

    gen_path = input_dir / "generation_config.json"
    gen = json.loads(gen_path.read_text(encoding="utf-8")) if gen_path.exists() else {}

    def gen_or_cfg(key, default):
        # generation_config.json wins; old checkpoints carry these in config.json
        if key in gen:
            return gen[key]
        return cfg.get(key, default)

    vocab_size = cfg["vocab_size"]
    d_model = cfg["d_model"]
    enc_n_layers = cfg["encoder_layers"]
    dec_n_layers = cfg["decoder_layers"]
    enc_n_heads = cfg["encoder_attention_heads"]
    dec_n_heads = cfg["decoder_attention_heads"]
    enc_ffn_dim = cfg["encoder_ffn_dim"]
    dec_ffn_dim = cfg["decoder_ffn_dim"]
    max_pos = cfg["max_position_embeddings"]
    scale_embed = int(bool(cfg.get("scale_embedding", False)))
    normalize_before = int(bool(cfg.get("normalize_before", False)))
    activation = cfg.get("activation_function", "gelu")
    eos_id = gen_or_cfg("eos_token_id", 0)
    pad_id = gen_or_cfg("pad_token_id", None)
    dec_start_id = gen_or_cfg("decoder_start_token_id", None)
    if pad_id is None or dec_start_id is None:
        sys.exit("config.json: pad_token_id / decoder_start_token_id missing")

    if activation not in ("swish", "silu", "relu", "gelu"):
        sys.exit(f"activation_function {activation!r} is not implemented in the runtime")
    if cfg.get("normalize_embedding"):
        sys.exit("normalize_embedding (layernorm_embedding) is not implemented")
    if cfg.get("add_final_layer_norm"):
        sys.exit("add_final_layer_norm is not implemented")
    if cfg.get("decoder_vocab_size", vocab_size) != vocab_size or cfg.get("share_encoder_decoder_embeddings") is False:
        sys.exit("separate encoder/decoder vocabularies are not implemented")
    if enc_n_heads != dec_n_heads:
        sys.exit("encoder and decoder head counts differ; the runtime assumes one head_dim")

    suppress: list[int] = []
    for bad in gen_or_cfg("bad_words_ids", None) or []:
        if len(bad) != 1:
            sys.exit(f"bad_words_ids entry {bad} is a multi-token sequence; only single tokens are implemented")
        suppress.append(int(bad[0]))

    print(f"  arch:    Marian  d_model={d_model}  enc={enc_n_layers}L×{enc_n_heads}H  "
          f"dec={dec_n_layers}L×{dec_n_heads}H  vocab={vocab_size}  act={activation}  "
          f"{'pre' if normalize_before else 'post'}-norm")

    # ---- weights ----
    st_path = input_dir / "model.safetensors"
    pt_path = input_dir / "pytorch_model.bin"
    if st_path.exists():
        from safetensors.torch import load_file
        sd = load_file(str(st_path))
        print(f"  weights: {st_path}")
    elif pt_path.exists():
        sd = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        print(f"  weights: {pt_path}")
    else:
        sys.exit(f"Missing model.safetensors / pytorch_model.bin in {input_dir}")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    print(f"  tensors: {len(sd)} in state dict")

    # The runtime keeps ONE table for encoder, decoder and lm_head. Prove the
    # checkpoint really ties them instead of trusting the config flag.
    shared = sd["model.shared.weight"]
    if tuple(shared.shape) != (vocab_size, d_model):
        sys.exit(f"model.shared.weight is {tuple(shared.shape)}, expected {(vocab_size, d_model)}")
    for tied in ("model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"):
        if tied in sd and not torch.equal(sd[tied], shared):
            sys.exit(f"{tied} differs from model.shared.weight — untied embeddings are not implemented")
    if "final_logits_bias" not in sd:
        sys.exit("final_logits_bias missing from the checkpoint")

    # Positions: static, regenerated from the formula. If the checkpoint
    # carries a copy, it must be this table.
    pos_emb = make_marian_positions(max_pos, d_model)
    for key in ("model.encoder.embed_positions.weight", "model.decoder.embed_positions.weight"):
        if key in sd:
            ck = sd[key].float().numpy()
            err = float(np.abs(ck - pos_emb).max()) if ck.shape == pos_emb.shape else float("inf")
            if err > 1e-5:
                sys.exit(f"{key} is not Marian's sinusoidal table (max abs diff {err:g}); "
                         f"learned positions are not implemented")
            print(f"  pos_emb: {key} matches the formula (max abs diff {err:.2e})")

    # ---- tokenizer ----
    tokens = load_vocab(input_dir / "vocab.json", vocab_size)
    for name, tid in (("</s>", eos_id), ("<pad>", pad_id)):
        if tokens[tid] != name:
            sys.exit(f"vocab.json id {tid} is {tokens[tid]!r}, config says it is {name}")
    if "<unk>" not in tokens:
        sys.exit("vocab.json has no <unk>")
    unk_id = tokens.index("<unk>")
    src_spm = load_spm(input_dir / "source.spm")
    tgt_spm = load_spm(input_dir / "target.spm")
    token_set = set(tokens)
    for side, sp in (("source", src_spm), ("target", tgt_spm)):
        absent = [p for p, t in zip(sp["pieces"], sp["types"]) if t == 1 and p not in token_set]
        print(f"  {side}.spm: {len(sp['pieces'])} pieces, normalizer {sp['norm_name']!r} "
              f"({len(sp['charsmap'])} B charsmap), {len(absent)} pieces absent from vocab.json "
              f"(→ <unk>, as in HF): {absent[:8]}")

    tok_cfg_path = input_dir / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8")) if tok_cfg_path.exists() else {}
    source_lang = tok_cfg.get("source_lang") or ""
    target_lang = tok_cfg.get("target_lang") or ""
    print(f"  langs:   {source_lang or '?'} → {target_lang or '?'}")

    # ---- write GGUF ----
    print(f"\nWriting: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="marian")
    writer.add_name(input_dir.name)

    writer.add_uint32("marian.vocab_size",               vocab_size)
    writer.add_uint32("marian.d_model",                  d_model)
    writer.add_uint32("marian.encoder.n_layers",         enc_n_layers)
    writer.add_uint32("marian.encoder.n_heads",          enc_n_heads)
    writer.add_uint32("marian.encoder.ffn_dim",          enc_ffn_dim)
    writer.add_uint32("marian.decoder.n_layers",         dec_n_layers)
    writer.add_uint32("marian.decoder.n_heads",          dec_n_heads)
    writer.add_uint32("marian.decoder.ffn_dim",          dec_ffn_dim)
    writer.add_uint32("marian.max_position_embeddings",  max_pos)
    writer.add_uint32("marian.scale_embedding",          scale_embed)
    writer.add_uint32("marian.normalize_before",         normalize_before)
    writer.add_string("marian.activation_function",      activation)
    writer.add_uint32("marian.eos_token_id",             eos_id)
    writer.add_uint32("marian.pad_token_id",             pad_id)
    writer.add_uint32("marian.unk_token_id",             unk_id)
    writer.add_uint32("marian.decoder_start_token_id",   dec_start_id)
    if suppress:
        writer.add_array("marian.suppress_token_ids",    suppress)
    writer.add_uint32("marian.gen.num_beams",            int(gen_or_cfg("num_beams", 1)))
    writer.add_uint32("marian.gen.max_length",           int(gen_or_cfg("max_length", 512)))
    writer.add_uint32("marian.gen.early_stopping",       1 if gen_or_cfg("early_stopping", False) is True else 0)
    writer.add_string("marian.source_lang",              source_lang)
    writer.add_string("marian.target_lang",              target_lang)

    writer.add_string("tokenizer.ggml.model", "marian")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    for side, sp in (("source", src_spm), ("target", tgt_spm)):
        pre = f"tokenizer.marian.{side}."
        writer.add_array(pre + "pieces", sp["pieces"])
        writer.add_array(pre + "scores", sp["scores"])
        writer.add_array(pre + "types", sp["types"])
        # bytes → a GGUF UINT8 array (a list of ints would be written as INT32)
        if sp["charsmap"]:
            writer.add_array(pre + "charsmap", sp["charsmap"])
        writer.add_string(pre + "normalizer", sp["norm_name"])
        writer.add_bool(pre + "add_dummy_prefix", sp["add_dummy_prefix"])
        writer.add_bool(pre + "remove_extra_whitespaces", sp["remove_extra_whitespaces"])
        writer.add_bool(pre + "escape_whitespaces", sp["escape_whitespaces"])

    n_written = n_f16 = n_f32 = 0

    def write_tensor(gguf_name: str, arr: np.ndarray) -> None:
        nonlocal n_written, n_f16, n_f32
        if is_f32_tensor(gguf_name, arr.shape):
            arr = arr.astype(np.float32)
            n_f32 += 1
        else:
            arr = arr.astype(np.float16)
            n_f16 += 1
        writer.add_tensor(gguf_name, arr)
        n_written += 1
        if n_written <= 12 or n_written % 50 == 0:
            print(f"  {gguf_name:44s}  {str(arr.shape):16s}  {arr.dtype}")

    write_tensor("enc.pos_emb", pos_emb)
    write_tensor("dec.pos_emb", pos_emb)
    for pt_name in sorted(sd.keys()):
        gguf_name = remap_name(pt_name)
        if gguf_name is None:
            continue
        arr = sd[pt_name].cpu().float().numpy()
        if gguf_name == "final_logits_bias":
            arr = arr.reshape(-1)
            if arr.shape[0] != vocab_size:
                sys.exit(f"final_logits_bias has {arr.shape[0]} entries, expected {vocab_size}")
        write_tensor(gguf_name, arr)

    # enc: 16 per layer; dec: 26 per layer; shared.embed + final_logits_bias + 2 pos_emb
    expected = enc_n_layers * 16 + dec_n_layers * 26 + 4
    if n_written != expected:
        sys.exit(f"tensor count: wrote {n_written}, expected {expected}")
    print(f"\n  total tensors written: {n_written}  (F16: {n_f16}, F32: {n_f32})")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a MarianMT / Opus-MT checkpoint (HuggingFace) → GGUF F16",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="HuggingFace model directory (config.json, vocab.json, source.spm, "
                        "target.spm, pytorch_model.bin or model.safetensors)")
    p.add_argument("--output", required=True, type=Path, help="output GGUF file path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)

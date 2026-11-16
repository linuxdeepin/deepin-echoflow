#!/usr/bin/env python3
"""
Convert k2-fsa/OmniVoice (or finetunes like ModelsLab/omnivoice-singing)
from HuggingFace safetensors → GGUF F16 for the CrispASR `omnivoice` backend.

Architecture (from config.json, confirmed against the safetensors tensor keys):

  model_type            = "omnivoice"
  architectures         = ["OmniVoice"]

  llm_config — standard Qwen3 0.6B backbone:
    hidden_size          = 1024
    num_hidden_layers    = 28
    num_attention_heads  = 16
    num_key_value_heads  = 8
    head_dim             = 128
    intermediate_size    = 3072
    vocab_size           = 151676
    max_position_embeddings = 40960
    rope_theta           = 1000000
    rms_norm_eps         = 1e-06
    hidden_act           = "silu"
    use_sliding_window   = false
    tie_word_embeddings  = true

  Audio:
    audio_vocab_size     = 1025   (1024 codes + 1 mask)
    audio_mask_id        = 1024
    num_audio_codebook   = 8
    audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]

  Generation: masked iterative (NOT autoregressive). The LLM runs
  multiple forward passes, each time predicting scores for all masked
  positions, then unmasking the top-k highest-confidence tokens.

  Audio tokenizer: separate HiggsAudioV2TokenizerModel in audio_tokenizer/
  subfolder — gets its own converter (convert-omnivoice-tokenizer-to-gguf.py).

Usage:

    python models/convert-omnivoice-to-gguf.py \\
        --input k2-fsa/OmniVoice \\
        --output omnivoice.gguf

    python models/convert-omnivoice-to-gguf.py \\
        --input ModelsLab/omnivoice-singing \\
        --output omnivoice-singing.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub")


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}…", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "*.jinja",
    ]))


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    """Map a HuggingFace tensor name to the GGUF name the C++ runtime
    expects.  Returns None to skip."""

    if hf_name.endswith("num_batches_tracked"):
        return None

    n = hf_name

    # ── Audio-specific weights (top-level, outside the LLM) ──────────
    n = n.replace("audio_embeddings.weight", "audio_embd.weight")
    n = n.replace("audio_heads.weight", "audio_output.weight")

    # ── LLM backbone (Qwen3) ─────────────────────────────────────────
    n = n.replace("llm.embed_tokens.", "llm.token_embd.")
    n = n.replace("llm.norm.", "llm.output_norm.")
    n = n.replace("llm.layers.", "llm.blk.")

    # ── Per-layer renames ─────────────────────────────────────────────
    n = n.replace(".self_attn.q_proj.", ".attn_q.")
    n = n.replace(".self_attn.k_proj.", ".attn_k.")
    n = n.replace(".self_attn.v_proj.", ".attn_v.")
    n = n.replace(".self_attn.o_proj.", ".attn_output.")
    n = n.replace(".self_attn.q_norm.", ".attn_q_norm.")
    n = n.replace(".self_attn.k_norm.", ".attn_k_norm.")
    n = n.replace(".input_layernorm.", ".attn_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".ffn_gate.")
    n = n.replace(".mlp.up_proj.", ".ffn_up.")
    n = n.replace(".mlp.down_proj.", ".ffn_down.")

    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert OmniVoice to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. k2-fsa/OmniVoice) or local dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    llm = cfg.get("llm_config", {})

    print(f"\nOmniVoice — {cfg.get('model_type', '?')}")
    print(f"  LLM backbone:  {llm.get('num_hidden_layers')}L  "
          f"hidden={llm.get('hidden_size')}  "
          f"heads={llm.get('num_attention_heads')}/{llm.get('num_key_value_heads')}  "
          f"head_dim={llm.get('head_dim')}  "
          f"ff={llm.get('intermediate_size')}  vocab={llm.get('vocab_size')}")
    print(f"  Audio:         {cfg.get('num_audio_codebook')} codebooks  "
          f"vocab={cfg.get('audio_vocab_size')}  "
          f"mask_id={cfg.get('audio_mask_id')}")
    print(f"  Codebook wts:  {cfg.get('audio_codebook_weights')}")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"no safetensors in {model_dir}")
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_idx = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            name_to_idx[k] = i
    print(f"  Safetensors:   {len(name_to_idx)} tensors in {len(st_files)} file(s)")

    out_path = Path(args.output)
    w = GGUFWriter(str(out_path), arch="omnivoice", use_temp_file=False)

    # ----- metadata -----------------------------------------------------
    w.add_name(f"omnivoice")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))
    def boolv(k, v): w.add_bool(k, bool(v))

    # LLM backbone
    u32("omnivoice.llm.n_layers",      llm.get("num_hidden_layers", 28))
    u32("omnivoice.llm.d_model",       llm.get("hidden_size", 1024))
    u32("omnivoice.llm.n_heads",       llm.get("num_attention_heads", 16))
    u32("omnivoice.llm.n_kv_heads",    llm.get("num_key_value_heads", 8))
    u32("omnivoice.llm.head_dim",      llm.get("head_dim", 128))
    u32("omnivoice.llm.ff_dim",        llm.get("intermediate_size", 3072))
    u32("omnivoice.llm.vocab_size",    llm.get("vocab_size", 151676))
    u32("omnivoice.llm.max_pos",       llm.get("max_position_embeddings", 40960))
    f32("omnivoice.llm.rms_norm_eps",  llm.get("rms_norm_eps", 1e-6))
    boolv("omnivoice.llm.tie_word_embeddings", llm.get("tie_word_embeddings", True))

    rope_params = llm.get("rope_parameters", {}) or {}
    f32("omnivoice.llm.rope_theta",    rope_params.get("rope_theta",
                                       llm.get("rope_theta", 1_000_000)))

    # Audio
    u32("omnivoice.audio.vocab_size",      cfg.get("audio_vocab_size", 1025))
    u32("omnivoice.audio.mask_id",         cfg.get("audio_mask_id", 1024))
    u32("omnivoice.audio.n_codebooks",     cfg.get("num_audio_codebook", 8))
    cb_weights = cfg.get("audio_codebook_weights", [8, 8, 6, 6, 4, 4, 2, 2])
    w.add_array("omnivoice.audio.codebook_weights", cb_weights)

    # Token sentinels
    u32("omnivoice.eos_token_id",      cfg.get("eos_token_id", 151645))
    u32("omnivoice.pad_token_id",      cfg.get("pad_token_id", 151643))

    # Tokenizer (tokenizer.json + tokenizer_config.json)
    tok_cfg_path = model_dir / "tokenizer_config.json"
    tok_json_path = model_dir / "tokenizer.json"
    if tok_cfg_path.exists():
        with open(tok_cfg_path, encoding="utf-8") as f:
            tok_cfg = json.load(f)
        w.add_string("omnivoice.tokenizer_class",
                     tok_cfg.get("tokenizer_class", "Qwen2Tokenizer"))

    # Embed the tokenizer.json so the runtime can decode it
    if tok_json_path.exists():
        with open(tok_json_path, encoding="utf-8") as f:
            tok_data = json.load(f)
        # Extract vocab from tokenizer.json's model.vocab
        vocab_dict = tok_data.get("model", {}).get("vocab", {})
        if vocab_dict:
            toks = [""] * len(vocab_dict)
            for tok, idx in vocab_dict.items():
                if idx < len(toks):
                    toks[idx] = tok
            w.add_token_list(toks)
            print(f"  Tokens:        {len(toks)} entries from tokenizer.json")

        # Extract merges — tokenizer.json stores merges as [["a","b"], ...]
        # but the GGUF C reader expects flat strings "a b". Flatten them.
        raw_merges = tok_data.get("model", {}).get("merges", [])
        if raw_merges:
            merges = []
            for m in raw_merges:
                if isinstance(m, list):
                    merges.append(" ".join(m))
                else:
                    merges.append(str(m))
            w.add_token_merges(merges)
            print(f"  Merges:        {len(merges)} entries from tokenizer.json")

        # Extract added_tokens for special tokens
        added = tok_data.get("added_tokens", [])
        if added:
            special_names = []
            special_ids = []
            for t in added:
                if t.get("special", False):
                    special_names.append(t["content"])
                    special_ids.append(int(t["id"]))
            if special_names:
                w.add_array("omnivoice.special_token_names", special_names)
                w.add_array("omnivoice.special_token_ids", special_ids)
                print(f"  Special toks:  {len(special_names)}")

    # Chat template
    chat_tpl_path = model_dir / "chat_template.jinja"
    if chat_tpl_path.exists():
        with open(chat_tpl_path, encoding="utf-8") as f:
            w.add_string("omnivoice.chat_template", f.read())

    # ----- tensors ------------------------------------------------------
    n_mapped = 0
    n_skipped = 0
    skipped_examples = []
    for hf_name in sorted(name_to_idx.keys()):
        gn = map_tensor_name(hf_name)
        if gn is None:
            n_skipped += 1
            continue

        # Verify all tensors land under known prefixes
        if not gn.startswith(("llm.", "audio_")):
            if len(skipped_examples) < 20:
                skipped_examples.append(f"  [WARN unmapped] {hf_name} → {gn}")
            n_skipped += 1
            continue

        t = handles[name_to_idx[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()
        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        if n_mapped <= 30 or n_mapped % 100 == 0:
            print(f"  [{n_mapped}] {gn:55s} {t.shape}  {t.dtype}")

    if skipped_examples:
        print("\n".join(skipped_examples), file=sys.stderr)
        print(f"\n  WARNING: {len(skipped_examples)} unmapped tensor(s)",
              file=sys.stderr)

    print(f"\nMapped: {n_mapped}, skipped: {n_skipped}")
    print(f"Writing {out_path}…")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / 1e9
    print(f"Done: {out_path}  ({sz:.2f} GB, {n_mapped} tensors)")


if __name__ == "__main__":
    main()

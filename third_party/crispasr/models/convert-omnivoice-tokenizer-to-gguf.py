#!/usr/bin/env python3
"""
Convert the OmniVoice audio tokenizer (HiggsAudioV2TokenizerModel)
from HuggingFace safetensors → GGUF F16 for the CrispASR `omnivoice` backend.

The audio tokenizer lives in k2-fsa/OmniVoice/audio_tokenizer/ (or as a
separate model at the same path in finetunes). It consists of:

  1. Semantic encoder: HuBERT (12L, 768 hidden, 16 kHz input)
     - 7-layer 1D conv feature extractor (strides 5,2,2,2,2,2,2)
     - Feature projection (512→768) + layer norm
     - 12 transformer layers with conv-position embeddings
     - Produces semantic features for the vector quantizer

  2. Acoustic encoder/decoder: DAC-based (16 kHz input, 9 codebooks)
     - Encoder: downsampling ratios [8,5,4,2,3] → hop_length 960
     - Decoder: upsampling ratios [3,2,4,5,8] (reversed)
     - 9 codebooks × 1024 entries × 8-dim per codebook

  3. Quantizer bridge:
     - Semantic projection (768→codebook_dim 64) + VQ
     - Acoustic projection + residual VQ (9 codebooks)
     - Combined output: 1 semantic + 8 acoustic = 9 code streams
       (OmniVoice uses 8 of these for its codebooks)

  4. Decoder (waveform synthesis from codes):
     - VQ lookup → upsample → DAC decoder → 24 kHz waveform

Usage:

    python models/convert-omnivoice-tokenizer-to-gguf.py \\
        --input k2-fsa/OmniVoice \\
        --output omnivoice-tokenizer.gguf

    # Or if the audio_tokenizer is a separate directory:
    python models/convert-omnivoice-tokenizer-to-gguf.py \\
        --input /path/to/audio_tokenizer \\
        --output omnivoice-tokenizer.gguf
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
    """Resolve to a local directory containing the audio tokenizer."""
    p = Path(model_id)

    # Direct path to audio_tokenizer subfolder
    if p.is_dir() and (p / "config.json").exists():
        return p

    # Parent repo with audio_tokenizer/ subfolder
    if p.is_dir() and (p / "audio_tokenizer" / "config.json").exists():
        return p / "audio_tokenizer"

    # HuggingFace download
    print(f"Downloading {model_id}…", file=sys.stderr)
    downloaded = Path(snapshot_download(model_id, allow_patterns=[
        "audio_tokenizer/*.safetensors",
        "audio_tokenizer/*.json",
        "*.safetensors",
        "*.json",
    ]))

    if (downloaded / "audio_tokenizer" / "config.json").exists():
        return downloaded / "audio_tokenizer"
    if (downloaded / "config.json").exists():
        return downloaded

    sys.exit(f"Cannot find audio_tokenizer config in {downloaded}")


def map_tensor_name(hf_name: str) -> str | None:
    """Map HuggingFace tensor name to GGUF name."""

    if hf_name.endswith("num_batches_tracked"):
        return None

    n = hf_name

    # ── Semantic model (HuBERT) → sem.* ──────────────────────────────
    n = n.replace("semantic_model.feature_extractor.conv_layers.", "sem.conv.")
    n = n.replace("semantic_model.feature_projection.layer_norm.", "sem.feat_proj_ln.")
    n = n.replace("semantic_model.feature_projection.projection.", "sem.feat_proj.")
    n = n.replace("semantic_model.encoder.pos_conv_embed.conv.", "sem.pos_conv.")
    n = n.replace("semantic_model.encoder.layer_norm.", "sem.enc_ln.")
    n = n.replace("semantic_model.encoder.layers.", "sem.blk.")

    # HuBERT layer renames
    n = n.replace(".attention.q_proj.", ".attn_q.")
    n = n.replace(".attention.k_proj.", ".attn_k.")
    n = n.replace(".attention.v_proj.", ".attn_v.")
    n = n.replace(".attention.out_proj.", ".attn_output.")
    n = n.replace(".layer_norm.", ".ln.")
    n = n.replace(".feed_forward.intermediate_dense.", ".fc1.")
    n = n.replace(".feed_forward.output_dense.", ".fc2.")
    n = n.replace(".final_layer_norm.", ".ffn_ln.")

    # ── Acoustic model (DAC) → ac.* ──────────────────────────────────
    n = n.replace("acoustic_model.encoder.", "ac.enc.")
    n = n.replace("acoustic_model.decoder.", "ac.dec.")
    n = n.replace("acoustic_model.quantizer.", "ac.quant.")

    # ── Quantizer / projections → quant.* ────────────────────────────
    n = n.replace("semantic_quantizer.", "sem_quant.")
    n = n.replace("acoustic_quantizer.", "ac_quant.")
    n = n.replace("semantic_project_in.", "sem_proj_in.")
    n = n.replace("semantic_project_out.", "sem_proj_out.")
    n = n.replace("acoustic_project_in.", "ac_proj_in.")
    n = n.replace("acoustic_project_out.", "ac_proj_out.")

    # ── Unit decoder → unit_dec.* ────────────────────────────────────
    n = n.replace("unit_decoder.", "unit_dec.")

    return n


def main():
    ap = argparse.ArgumentParser(
        description="Convert OmniVoice audio tokenizer to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. k2-fsa/OmniVoice) or local dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    sem = cfg.get("semantic_model_config", {})
    ac = cfg.get("acoustic_model_config", {})

    print(f"\nOmniVoice Audio Tokenizer — {cfg.get('model_type', '?')}")
    print(f"  Sample rate:   {cfg.get('sample_rate', 24000)}")
    print(f"  Downsample:    {cfg.get('downsample_factor', 320)}")
    print(f"  Codebook size: {cfg.get('codebook_size', 1024)}")
    print(f"  Codebook dim:  {cfg.get('codebook_dim', 64)}")
    print(f"  Semantic:      HuBERT {sem.get('num_hidden_layers', 12)}L  "
          f"hidden={sem.get('hidden_size', 768)}  "
          f"heads={sem.get('num_attention_heads', 12)}")
    print(f"  Acoustic:      DAC {ac.get('n_codebooks', 9)} codebooks  "
          f"hidden={ac.get('hidden_size', 256)}  "
          f"hop={ac.get('hop_length', 960)}")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = (GGMLQuantizationType.F16 if args.outtype == "f16"
              else GGMLQuantizationType.F32)

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
    w = GGUFWriter(str(out_path), arch="omnivoice_tok", use_temp_file=False)

    # ----- metadata -----------------------------------------------------
    w.add_name("omnivoice-tokenizer")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))

    u32("omnivoice_tok.sample_rate",       cfg.get("sample_rate", 24000))
    u32("omnivoice_tok.downsample_factor", cfg.get("downsample_factor", 320))
    u32("omnivoice_tok.codebook_size",     cfg.get("codebook_size", 1024))
    u32("omnivoice_tok.codebook_dim",      cfg.get("codebook_dim", 64))
    u32("omnivoice_tok.kernel_size",       cfg.get("kernel_size", 3))

    # Semantic model
    u32("omnivoice_tok.sem.n_layers",      sem.get("num_hidden_layers", 12))
    u32("omnivoice_tok.sem.d_model",       sem.get("hidden_size", 768))
    u32("omnivoice_tok.sem.n_heads",       sem.get("num_attention_heads", 12))
    u32("omnivoice_tok.sem.ff_dim",        sem.get("intermediate_size", 3072))
    u32("omnivoice_tok.sem.sample_rate",   cfg.get("semantic_sample_rate", 16000))
    f32("omnivoice_tok.sem.ln_eps",        sem.get("layer_norm_eps", 1e-5))
    n_conv_layers = sem.get("num_feat_extract_layers", 7)
    u32("omnivoice_tok.sem.n_conv_layers", n_conv_layers)
    conv_dims = sem.get("conv_dim", [512]*7)
    conv_kernels = sem.get("conv_kernel", [10, 3, 3, 3, 3, 2, 2])
    conv_strides = sem.get("conv_stride", [5, 2, 2, 2, 2, 2, 2])
    w.add_array("omnivoice_tok.sem.conv_dims", conv_dims)
    w.add_array("omnivoice_tok.sem.conv_kernels", conv_kernels)
    w.add_array("omnivoice_tok.sem.conv_strides", conv_strides)

    # Acoustic model
    u32("omnivoice_tok.ac.n_codebooks",     ac.get("n_codebooks", 9))
    u32("omnivoice_tok.ac.codebook_size",   ac.get("codebook_size", 1024))
    u32("omnivoice_tok.ac.codebook_dim",    ac.get("codebook_dim", 8))
    u32("omnivoice_tok.ac.hidden_size",     ac.get("hidden_size", 256))
    u32("omnivoice_tok.ac.decoder_hidden",  ac.get("decoder_hidden_size", 1024))
    u32("omnivoice_tok.ac.hop_length",      ac.get("hop_length", 960))
    u32("omnivoice_tok.ac.sample_rate",     ac.get("sampling_rate", 16000))
    ds_ratios = ac.get("downsampling_ratios", [8, 5, 4, 2, 3])
    w.add_array("omnivoice_tok.ac.downsampling_ratios", ds_ratios)

    # ----- tensors ------------------------------------------------------
    n_mapped = 0
    n_skipped = 0
    skipped_examples = []
    for hf_name in sorted(name_to_idx.keys()):
        gn = map_tensor_name(hf_name)
        if gn is None:
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

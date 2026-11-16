#!/usr/bin/env python3
"""Convert HTDemucs (Meta Demucs v4) weights to CrispASR GGUF format.

Usage:
    python models/convert-htdemucs-to-gguf.py --model htdemucs --output htdemucs.gguf [--dtype f16]

The --model argument is a pretrained model name (htdemucs, htdemucs_ft,
htdemucs_6s) or a path to a .th / .pt checkpoint.

Requires: pip install demucs (for model loading only — weights are
streamed to GGUF tensor-by-tensor to stay under 8 GB RAM).
"""

import argparse
import os
import sys
import json
import numpy as np

# Throttle BLAS threads (torch + OMP deadlocks on bf16→f16 casts)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

TMPDIR = os.environ.get("TMPDIR", "/mnt/volume1/tmp-overflow")
os.environ["TMPDIR"] = TMPDIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default="htdemucs",
                   help="Pretrained model name (htdemucs, htdemucs_ft, htdemucs_6s) or path")
    p.add_argument("--output", type=str, required=True, help="Output GGUF path")
    p.add_argument("--dtype", choices=["f32", "f16"], default="f16",
                   help="Weight storage precision (default: f16)")
    return p.parse_args()


def load_model(name_or_path):
    """Load HTDemucs model, return (module, config_dict)."""
    from demucs.pretrained import get_model
    from demucs.htdemucs import HTDemucs

    bag = get_model(name_or_path)
    if hasattr(bag, "models"):
        assert len(bag.models) == 1, f"Expected 1 model in bag, got {len(bag.models)}"
        model = bag.models[0]
    else:
        model = bag
    assert isinstance(model, HTDemucs), f"Expected HTDemucs, got {type(model).__name__}"
    model.eval()

    cfg = {}
    cfg["sources"] = model.sources
    cfg["audio_channels"] = model.audio_channels
    cfg["channels"] = model.channels
    cfg["nfft"] = model.nfft
    cfg["depth"] = model.depth
    cfg["bottom_channels"] = model.bottom_channels
    cfg["samplerate"] = model.samplerate
    cfg["segment"] = float(model.segment)
    cfg["cac"] = model.cac
    cfg["kernel_size"] = model.kernel_size
    cfg["stride"] = model.stride
    cfg["context"] = model.context
    cfg["use_train_segment"] = model.use_train_segment

    ct = model.crosstransformer
    if ct is not None:
        cfg["t_layers"] = ct.num_layers
        cfg["t_emb"] = ct.emb
        cfg["t_max_period"] = ct.max_period
        cfg["t_weight_pos_embed"] = ct.weight_pos_embed
        cfg["t_classic_parity"] = ct.classic_parity

    # Derive norm/dconv params from encoder[0]
    enc0 = model.encoder[0]
    cfg["dconv_depth"] = enc0.dconv.depth if enc0.dconv else 0
    cfg["dconv_compress"] = int(enc0.dconv.compress) if enc0.dconv else 4
    cfg["has_rewrite"] = enc0.rewrite is not None

    # Freq embedding
    cfg["has_freq_emb"] = model.freq_emb is not None
    if model.freq_emb is not None:
        cfg["freq_emb_scale"] = model.freq_emb_scale

    # Transformer head count (from first layer's self_attn)
    if ct and len(ct.layers) > 0:
        layer0 = ct.layers[0]
        if hasattr(layer0, "self_attn"):
            cfg["t_heads"] = layer0.self_attn.num_heads
        elif hasattr(layer0, "cross_attn"):
            cfg["t_heads"] = layer0.cross_attn.num_heads

    return model, cfg


def should_keep_f32(name):
    """Biases, norms, LayerScale, small embeddings → always F32."""
    if name.endswith(".bias") or name.endswith("_bias"):
        return True
    if ".norm" in name and ".weight" in name:
        return True
    if "scale" in name:
        return True
    if "freq_emb" in name:
        return True
    return False


def convert(args):
    import torch
    from gguf import GGUFWriter, GGMLQuantizationType

    model, cfg = load_model(args.model)
    sd = model.state_dict()
    n_tensors = len(sd)
    print(f"Loaded {type(model).__name__}: {n_tensors} tensors, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    print(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    w = GGUFWriter(args.output, "htdemucs", use_temp_file=True)
    w.add_name(f"HTDemucs ({args.model})")
    w.add_description("HTDemucs — hybrid transformer source separation (Meta)")

    # Write hyperparameters
    w.add_uint32("htdemucs.audio_channels", cfg["audio_channels"])
    w.add_uint32("htdemucs.channels", cfg["channels"])
    w.add_uint32("htdemucs.nfft", cfg["nfft"])
    w.add_uint32("htdemucs.depth", cfg["depth"])
    w.add_uint32("htdemucs.bottom_channels", cfg["bottom_channels"])
    w.add_uint32("htdemucs.samplerate", cfg["samplerate"])
    w.add_float32("htdemucs.segment", cfg["segment"])
    w.add_uint32("htdemucs.cac", 1 if cfg["cac"] else 0)
    w.add_uint32("htdemucs.kernel_size", cfg["kernel_size"])
    w.add_uint32("htdemucs.stride", cfg["stride"])
    w.add_uint32("htdemucs.context", cfg["context"])
    w.add_uint32("htdemucs.n_sources", len(cfg["sources"]))
    w.add_uint32("htdemucs.dconv_depth", cfg["dconv_depth"])
    w.add_uint32("htdemucs.dconv_compress", cfg["dconv_compress"])
    w.add_uint32("htdemucs.has_rewrite", 1 if cfg["has_rewrite"] else 0)
    w.add_uint32("htdemucs.has_freq_emb", 1 if cfg["has_freq_emb"] else 0)
    if cfg["has_freq_emb"]:
        w.add_float32("htdemucs.freq_emb_scale", cfg["freq_emb_scale"])

    if "t_layers" in cfg:
        w.add_uint32("htdemucs.t_layers", cfg["t_layers"])
        w.add_string("htdemucs.t_emb", cfg["t_emb"])
        w.add_float32("htdemucs.t_max_period", cfg["t_max_period"])
        w.add_float32("htdemucs.t_weight_pos_embed", cfg["t_weight_pos_embed"])
        w.add_uint32("htdemucs.t_classic_parity", cfg["t_classic_parity"])
        if "t_heads" in cfg:
            w.add_uint32("htdemucs.t_heads", cfg["t_heads"])

    # Source names as a JSON array in a string KV
    w.add_string("htdemucs.sources_json", json.dumps(cfg["sources"]))

    # Write tensors
    f32_type = GGMLQuantizationType.F32
    f16_type = GGMLQuantizationType.F16
    written = 0
    for name, tensor in sd.items():
        data = tensor.float().numpy()
        # ndim == 1: every 1-D tensor is a bias / norm affine / LayerScale that
        # ends up as src1 of a broadcast add/mul in the ggml graph, and CUDA's
        # binbcast kernels reject non-F32 src1 there (issue #398). The name
        # rules alone missed the DConv GroupNorm affines (`dconv.layers.N.4.
        # weight` — a bare nn.Sequential index, no ".norm" in the name).
        keep_f32 = should_keep_f32(name) or data.size < 256 or data.ndim == 1
        if keep_f32 or args.dtype == "f32":
            w.add_tensor(name, data.astype(np.float32), raw_dtype=f32_type)
        else:
            w.add_tensor(name, data.astype(np.float16), raw_dtype=f16_type)
        written += 1
        if written % 100 == 0:
            print(f"  [{written}/{n_tensors}] {name} {list(data.shape)}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {args.output}: {written} tensors, {size_mb:.1f} MB")


if __name__ == "__main__":
    convert(parse_args())

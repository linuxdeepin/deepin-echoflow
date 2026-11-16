#!/usr/bin/env python3
"""
Convert Meta AudioSeal PyTorch checkpoints → GGUF for the CrispASR
C++ audioseal watermark generator + detector.

The AudioSeal library handles weight_norm fusion internally, so we
load via AudioSeal.load_generator / load_detector and dump the
state_dict directly.

PyTorch state_dict layout (generator, verified live):

  Encoder (encoder.model.{idx}):
    0:  Conv1d(1, 32, k=7)      — input projection
    1:  ResBlock(32)             — .block.{1,3}.conv.conv.inner_conv
    (2: ELU — no params)
    3:  Conv1d(32, 64, k=4, s=2) — downsample (ratio=2)
    4:  ResBlock(64)
    (5: ELU)
    6:  Conv1d(64, 128, k=8, s=4) — downsample (ratio=4)
    7:  ResBlock(128)
    (8: ELU)
    9:  Conv1d(128, 256, k=10, s=5) — downsample (ratio=5)
    10: ResBlock(256)
    (11: ELU)
    12: Conv1d(256, 512, k=16, s=8) — downsample (ratio=8)
    13: StreamableLSTM(512, 2 layers)
    (14: ELU)
    15: Conv1d(512, 128, k=7) — output projection

  Decoder (decoder.model.{idx}):
    0:  Conv1d(128, 512, k=7) — input projection
    1:  StreamableLSTM(512, 2 layers)
    (2: ELU)
    3:  ConvTranspose1d(512, 256, k=16, s=8)
    4:  ResBlock(256)
    (5: ELU)
    6:  ConvTranspose1d(256, 128, k=10, s=5)
    7:  ResBlock(128)
    (8: ELU)
    9:  ConvTranspose1d(128, 64, k=8, s=4)
    10: ResBlock(64)
    (11: ELU)
    12: ConvTranspose1d(64, 32, k=4, s=2)
    13: ResBlock(32)
    (14: ELU)
    15: Conv1d(32, 1, k=7) — output projection

  Message: msg_processor.msg_processor.weight — Embedding(32, 128)

  Detector (detector.{0|1}):
    0.model.*: same encoder as generator
    0.reverse_convolution: ConvTranspose1d(128, 32, k=320, s=320)
    1: Conv1d(32, 18, k=1) — detection+message head

GGUF tensor naming convention:
  audioseal.gen.enc.{idx}.{subpath}   — encoder layer
  audioseal.gen.dec.{idx}.{subpath}   — decoder layer
  audioseal.gen.msg.weight            — message embedding
  audioseal.det.enc.{idx}.{subpath}   — detector encoder
  audioseal.det.reverse.weight/.bias  — reverse convolution
  audioseal.det.head.weight/.bias     — detection head

Usage:
    pip install audioseal gguf
    python models/convert-audioseal-to-gguf.py --output audioseal.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)


def remap_generator(sd: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Remap generator state_dict to GGUF names."""
    out = {}
    for key, val in sd.items():
        # Shorten the deeply nested conv paths:
        # encoder.model.0.conv.conv.inner_conv.weight → audioseal.gen.enc.0.weight
        # encoder.model.1.block.1.conv.conv.inner_conv.weight → audioseal.gen.enc.1.block.1.weight
        k = key
        k = k.replace(".conv.conv.inner_conv.", ".")
        k = k.replace(".convtr.convtr.inner_conv.", ".")
        k = k.replace("encoder.model.", "audioseal.gen.enc.")
        k = k.replace("decoder.model.", "audioseal.gen.dec.")
        k = k.replace("msg_processor.msg_processor.", "audioseal.gen.msg.")
        # LSTM: encoder.model.13.lstm.weight_ih_l0 → audioseal.gen.enc.13.lstm.weight_ih_l0
        out[k] = val.float().numpy()
    return out


def remap_detector(sd: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Remap detector state_dict to GGUF names."""
    out = {}
    for key, val in sd.items():
        k = key
        k = k.replace(".conv.conv.inner_conv.", ".")
        k = k.replace("detector.0.model.", "audioseal.det.enc.")
        k = k.replace("detector.0.reverse_convolution.", "audioseal.det.reverse.")
        k = k.replace("detector.1.", "audioseal.det.head.")
        out[k] = val.float().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert AudioSeal to GGUF")
    ap.add_argument("--output", "-o", required=True, help="Output GGUF path")
    args = ap.parse_args()

    from audioseal import AudioSeal

    print("Loading AudioSeal generator...")
    gen = AudioSeal.load_generator("audioseal_wm_16bits")
    gen_tensors = remap_generator(gen.state_dict())
    print(f"  {len(gen_tensors)} tensors, {sum(v.size for v in gen_tensors.values())} params")

    print("Loading AudioSeal detector...")
    det = AudioSeal.load_detector("audioseal_detector_16bits")
    det_tensors = remap_detector(det.state_dict())
    print(f"  {len(det_tensors)} tensors, {sum(v.size for v in det_tensors.values())} params")

    all_tensors = {}
    all_tensors.update(gen_tensors)
    all_tensors.update(det_tensors)

    # Verify no key collisions
    assert len(all_tensors) == len(gen_tensors) + len(det_tensors), \
        "Key collision between generator and detector!"

    # Write GGUF
    writer = GGUFWriter(args.output, "audioseal")

    # Metadata
    writer.add_uint32("audioseal.sample_rate", 16000)
    writer.add_uint32("audioseal.dimension", 128)
    writer.add_uint32("audioseal.n_filters", 32)
    writer.add_uint32("audioseal.n_residual_layers", 1)
    writer.add_uint32("audioseal.nbits", 16)
    writer.add_uint32("audioseal.lstm_layers", 2)
    # Encoder ratios (used for downsampling strides)
    writer.add_array("audioseal.ratios", [2, 4, 5, 8])
    # Channel progression: 32 → 64 → 128 → 256 → 512
    writer.add_array("audioseal.channels", [32, 64, 128, 256, 512])

    # Print key mapping for verification
    print(f"\nGGUF tensor names ({len(all_tensors)}):")
    for name in sorted(all_tensors.keys()):
        arr = all_tensors[name]
        print(f"  {name:55s}  {str(list(arr.shape)):20s}  {arr.dtype}")

    # Write tensors. Default to F32 for maximum parity; pass --f16 for
    # smaller files (at the cost of watermark-only cosine).
    use_f16 = "--f16" in sys.argv
    n_written = 0
    for name, arr in sorted(all_tensors.items()):
        if use_f16 and arr.size > 256 and "bias" not in name:
            writer.add_tensor(name, arr.astype(np.float16),
                              raw_dtype=GGMLQuantizationType.F16)
        else:
            writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    total_bytes = Path(args.output).stat().st_size
    print(f"\nWrote {n_written} tensors to {args.output} ({total_bytes / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()

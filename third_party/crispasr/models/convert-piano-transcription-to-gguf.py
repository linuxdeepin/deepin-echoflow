#!/usr/bin/env python3
"""
Convert ByteDance/Kong piano_transcription_inference checkpoint → GGUF.

The checkpoint (model.pth) contains two sub-models:
  - note_model: Regress_onset_offset_frame_velocity_CRNN
  - pedal_model: Regress_pedal_CRNN

Both share the same front-end (spectrogram + logmel + BN0) but have separate
AcousticModelCRnn8Dropout sub-networks.

GGUF tensor naming:
  piano.note.bn0.{weight,bias,running_mean,running_var}
  piano.note.{frame,onset,offset,velocity}.conv{1..4}.conv{1,2}.weight
  piano.note.{frame,onset,offset,velocity}.conv{1..4}.bn{1,2}.{w,b,rm,rv}
  piano.note.{frame,onset,offset,velocity}.fc5.weight
  piano.note.{frame,onset,offset,velocity}.bn5.{w,b,rm,rv}
  piano.note.{frame,onset,offset,velocity}.gru.{weight_ih,weight_hh,bias_ih,bias_hh}_l{0..1}[_reverse]
  piano.note.{frame,onset,offset,velocity}.fc.{weight,bias}
  piano.note.onset_refine_gru.{...}
  piano.note.onset_refine_fc.{weight,bias}
  piano.note.frame_refine_gru.{...}
  piano.note.frame_refine_fc.{weight,bias}
  piano.note.logmel.melW
  piano.note.stft.conv_real.weight
  piano.note.stft.conv_imag.weight

  piano.pedal.bn0.{...}
  piano.pedal.{onset,offset,frame}.conv{1..4}...
  piano.pedal.{onset,offset,frame}.gru...
  piano.pedal.{onset,offset,frame}.fc.{weight,bias}

Usage:
    python models/convert-piano-transcription-to-gguf.py \
        --input /mnt/storage/models/piano-transcription/model.pth \
        --output /mnt/storage/gguf-models/piano-transcription-f16.gguf \
        [--f16]
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


# Map from PyTorch state_dict key prefixes → GGUF prefixes
NOTE_MODEL_MAP = {
    "frame_model":    "piano.note.frame",
    "reg_onset_model":  "piano.note.onset",
    "reg_offset_model": "piano.note.offset",
    "velocity_model":   "piano.note.velocity",
}

PEDAL_MODEL_MAP = {
    "reg_pedal_onset_model":  "piano.pedal.onset",
    "reg_pedal_offset_model": "piano.pedal.offset",
    "reg_pedal_frame_model":  "piano.pedal.frame",
}


def remap_acoustic_model(key: str, prefix_map: dict[str, str]) -> str | None:
    """Map a single key from the PyTorch state_dict to GGUF name."""
    for pt_prefix, gguf_prefix in prefix_map.items():
        if key.startswith(pt_prefix + "."):
            suffix = key[len(pt_prefix) + 1:]
            return gguf_prefix + "." + suffix
    return None


def remap_note_model(sd: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Remap note_model state_dict to GGUF names."""
    out = {}
    for key, val in sd.items():
        # Skip num_batches_tracked — not needed for inference
        if "num_batches_tracked" in key:
            continue

        arr = val.float().numpy()

        # Top-level BN0
        if key.startswith("bn0."):
            out["piano.note." + key] = arr
            continue

        # Mel filterbank and STFT (frozen params)
        if key.startswith("logmel_extractor."):
            suffix = key[len("logmel_extractor."):]
            out["piano.note.logmel." + suffix] = arr
            continue
        if key.startswith("spectrogram_extractor."):
            suffix = key[len("spectrogram_extractor."):]
            # stft.conv_real.weight, stft.conv_imag.weight
            out["piano.note." + suffix] = arr
            continue

        # Refinement GRUs and FCs
        if key.startswith("reg_onset_gru."):
            suffix = key[len("reg_onset_gru."):]
            out["piano.note.onset_refine_gru." + suffix] = arr
            continue
        if key.startswith("reg_onset_fc."):
            suffix = key[len("reg_onset_fc."):]
            out["piano.note.onset_refine_fc." + suffix] = arr
            continue
        if key.startswith("frame_gru."):
            suffix = key[len("frame_gru."):]
            out["piano.note.frame_refine_gru." + suffix] = arr
            continue
        if key.startswith("frame_fc."):
            suffix = key[len("frame_fc."):]
            out["piano.note.frame_refine_fc." + suffix] = arr
            continue

        # Acoustic sub-models
        name = remap_acoustic_model(key, NOTE_MODEL_MAP)
        if name:
            out[name] = arr
            continue

        print(f"WARNING: unmapped note_model key: {key}")

    return out


def remap_pedal_model(sd: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Remap pedal_model state_dict to GGUF names."""
    out = {}
    for key, val in sd.items():
        if "num_batches_tracked" in key:
            continue

        arr = val.float().numpy()

        # Top-level BN0
        if key.startswith("bn0."):
            out["piano.pedal." + key] = arr
            continue

        # Pedal model also has its own STFT/logmel, but shares the same weights
        # as the note model. We skip them here — the C++ backend will use the
        # note model's STFT/logmel for both.
        if key.startswith("logmel_extractor.") or key.startswith("spectrogram_extractor."):
            continue

        name = remap_acoustic_model(key, PEDAL_MODEL_MAP)
        if name:
            out[name] = arr
            continue

        print(f"WARNING: unmapped pedal_model key: {key}")

    return out


def main():
    ap = argparse.ArgumentParser(description="Convert piano_transcription_inference to GGUF")
    ap.add_argument("--input", "-i", required=True, help="Path to model.pth")
    ap.add_argument("--output", "-o", required=True, help="Output GGUF path")
    ap.add_argument("--f16", action="store_true", help="Store weights as F16")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.input}")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    model = checkpoint["model"]

    note_sd = model["note_model"]
    pedal_sd = model["pedal_model"]
    print(f"  note_model: {len(note_sd)} tensors")
    print(f"  pedal_model: {len(pedal_sd)} tensors")

    note_tensors = remap_note_model(note_sd)
    pedal_tensors = remap_pedal_model(pedal_sd)

    all_tensors = {}
    all_tensors.update(note_tensors)
    all_tensors.update(pedal_tensors)
    assert len(all_tensors) == len(note_tensors) + len(pedal_tensors), "Key collision!"

    # Write GGUF
    writer = GGUFWriter(args.output, "piano-transcription")

    # Metadata
    writer.add_uint32("piano.sample_rate", 16000)
    writer.add_uint32("piano.n_fft", 2048)
    writer.add_uint32("piano.hop_size", 160)
    writer.add_uint32("piano.n_mels", 229)
    writer.add_uint32("piano.fmin", 30)
    writer.add_uint32("piano.fmax", 8000)
    writer.add_uint32("piano.frames_per_second", 100)
    writer.add_uint32("piano.classes_num", 88)
    writer.add_uint32("piano.begin_note", 21)
    # Conv block channel progression
    writer.add_array("piano.conv_channels", [48, 64, 96, 128])
    # GRU hidden size
    writer.add_uint32("piano.gru_hidden", 256)
    # Fully-connected intermediate
    writer.add_uint32("piano.fc5_out", 768)
    # Midfeat = 128 * (229 / 2^4) = 128 * 14 = 1792
    writer.add_uint32("piano.midfeat", 1792)

    print(f"\nGGUF tensor names ({len(all_tensors)}):")
    for name in sorted(all_tensors.keys()):
        arr = all_tensors[name]
        print(f"  {name:65s}  {str(list(arr.shape)):20s}  {arr.dtype}")

    n_written = 0
    for name, arr in sorted(all_tensors.items()):
        if args.f16 and arr.size > 256 and "bias" not in name and "running_" not in name:
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

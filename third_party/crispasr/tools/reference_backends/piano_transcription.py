#!/usr/bin/env python3
"""
Piano transcription reference dumper for crispasr-diff parity testing.

Runs the upstream PyTorch model, dumps intermediate activations at each
boundary for comparison with the C++ GGUF implementation:

  1. mel_spectrogram    — (1, 1, T, 229) after logmel + BN0
  2. conv_block_output  — (1, 128, T, 14) after 4 ConvBlocks (for onset model)
  3. gru_output         — (1, T, 512) after BiGRU (for onset model)
  4. note_outputs       — (T, 88) × 4 heads (frame, onset, offset, velocity)
  5. midi_events        — detected note events as JSON

Usage:
    python tools/reference_backends/piano_transcription.py \\
        --model /mnt/storage/models/piano-transcription/model.pth \\
        --audio samples/jfk.wav \\
        --output-dir /mnt/volume1/tmp-overflow/piano-ref

With crispasr-diff:
    python tools/dump_reference.py piano_transcription \\
        --model /mnt/storage/models/piano-transcription/model.pth \\
        --audio samples/jfk.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load the Note_pedal model from checkpoint."""
    from piano_transcription_inference.models import Note_pedal

    model = Note_pedal(frames_per_second=100, classes_num=88)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model


def dump_intermediates(model, audio: np.ndarray, output_dir: str):
    """Run inference and dump per-stage intermediates."""
    os.makedirs(output_dir, exist_ok=True)

    device = "cpu"
    model = model.to(device)

    # Input: (1, audio_samples) — mono 16kHz
    x = torch.from_numpy(audio).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # Stage 1: Spectrogram → LogMel
        note_model = model.note_model
        spec = note_model.spectrogram_extractor(x)  # (1, 1, T, 1025)
        mel = note_model.logmel_extractor(spec)       # (1, 1, T, 229)

        # Stage 2: BN0 (applied on transposed mel: BN over mel_bins axis)
        mel_bn = mel.transpose(1, 3)        # (1, 229, T, 1)
        mel_bn = note_model.bn0(mel_bn)
        mel_bn = mel_bn.transpose(1, 3)     # (1, 1, T, 229)

        save_tensor(mel_bn, "mel_spectrogram", output_dir)

        # Stage 3: Each acoustic model (run through one for intermediate dumps)
        # We'll dump the onset model's intermediates as representative
        onset_m = note_model.reg_onset_model

        # ConvBlocks
        h = mel_bn  # (1, 1, T, 229)
        for i, cb in enumerate([onset_m.conv_block1, onset_m.conv_block2,
                                onset_m.conv_block3, onset_m.conv_block4]):
            h = torch.relu_(cb.bn1(cb.conv1(h)))
            h = torch.relu_(cb.bn2(cb.conv2(h)))
            h = torch.nn.functional.avg_pool2d(h, kernel_size=(1, 2))

        save_tensor(h, "conv_block_output", output_dir)
        # h shape: (1, 128, T, 14)  (229 / 2^4 ≈ 14)

        # Flatten and FC5+BN5
        h = h.transpose(1, 2).flatten(2)      # (1, T, 128*14=1792)
        h = torch.relu(onset_m.bn5(onset_m.fc5(h).transpose(1, 2)).transpose(1, 2))
        save_tensor(h, "fc5_output", output_dir)  # (1, T, 768)

        # BiGRU
        h_gru, _ = onset_m.gru(h)  # (1, T, 512)
        save_tensor(h_gru, "gru_output", output_dir)

        # FC output
        onset_raw = torch.sigmoid(onset_m.fc(h_gru))  # (1, T, 88)

        # Now run all 4 models fully for the final outputs
        frame_output = note_model.frame_model(mel_bn)      # (1, T, 88)
        reg_onset_output = note_model.reg_onset_model(mel_bn)
        reg_offset_output = note_model.reg_offset_model(mel_bn)
        velocity_output = note_model.velocity_model(mel_bn)

        # Onset refinement
        x_onset = torch.cat(
            (reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()),
            dim=2)
        x_onset, _ = note_model.reg_onset_gru(x_onset)
        reg_onset_output = torch.sigmoid(note_model.reg_onset_fc(x_onset))

        # Frame refinement
        x_frame = torch.cat(
            (frame_output, reg_onset_output.detach(), reg_offset_output.detach()),
            dim=2)
        x_frame, _ = note_model.frame_gru(x_frame)
        frame_output = torch.sigmoid(note_model.frame_fc(x_frame))

        # Save all 4 output heads
        outputs = {
            "frame_output": frame_output[0].numpy(),
            "onset_output": reg_onset_output[0].numpy(),
            "offset_output": reg_offset_output[0].numpy(),
            "velocity_output": velocity_output[0].numpy(),
        }

        for name, arr in outputs.items():
            save_tensor(torch.from_numpy(arr), name, output_dir)

        # Also run pedal model
        pedal_frame = note_model.frame_model(mel_bn)  # placeholder — actual pedal uses pedal_model
        pedal_model = model.pedal_model
        spec_p = pedal_model.spectrogram_extractor(x)
        mel_p = pedal_model.logmel_extractor(spec_p)
        mel_p_bn = mel_p.transpose(1, 3)
        mel_p_bn = pedal_model.bn0(mel_p_bn)
        mel_p_bn = mel_p_bn.transpose(1, 3)

        pedal_onset = pedal_model.reg_pedal_onset_model(mel_p_bn)
        pedal_offset = pedal_model.reg_pedal_offset_model(mel_p_bn)
        pedal_frame_out = pedal_model.reg_pedal_frame_model(mel_p_bn)

        save_tensor(pedal_onset[0], "pedal_onset_output", output_dir)
        save_tensor(pedal_offset[0], "pedal_offset_output", output_dir)
        save_tensor(pedal_frame_out[0], "pedal_frame_output", output_dir)

        # Post-process to MIDI events
        from piano_transcription_inference.utilities import RegressionPostProcessor
        from piano_transcription_inference import config

        output_dict = {
            "reg_onset_output": reg_onset_output[0].numpy(),
            "reg_offset_output": reg_offset_output[0].numpy(),
            "frame_output": frame_output[0].numpy(),
            "velocity_output": velocity_output[0].numpy(),
            "reg_pedal_onset_output": pedal_onset[0].numpy(),
            "reg_pedal_offset_output": pedal_offset[0].numpy(),
            "pedal_frame_output": pedal_frame_out[0].numpy(),
        }

        post_processor = RegressionPostProcessor(
            frames_per_second=config.frames_per_second,
            classes_num=config.classes_num,
            onset_threshold=0.3,
            offset_threshold=0.3,
            frame_threshold=0.1,
            pedal_offset_threshold=0.2,
        )

        est_note_events, est_pedal_events = post_processor.output_dict_to_midi_events(output_dict)

        events_json = {
            "note_events": est_note_events[:50] if est_note_events else [],
            "pedal_events": est_pedal_events[:20] if est_pedal_events else [],
            "total_notes": len(est_note_events) if est_note_events else 0,
            "total_pedals": len(est_pedal_events) if est_pedal_events else 0,
        }

        with open(os.path.join(output_dir, "midi_events.json"), "w") as f:
            json.dump(events_json, f, indent=2, default=float)

        print(f"Detected {events_json['total_notes']} notes, {events_json['total_pedals']} pedal events")

    # Write all intermediates to a single GGUF for crispasr-diff
    write_ref_gguf(output_dir)


def save_tensor(tensor: torch.Tensor, name: str, output_dir: str):
    """Save tensor as .npy for quick inspection."""
    arr = tensor.detach().cpu().numpy()
    path = os.path.join(output_dir, f"{name}.npy")
    np.save(path, arr)
    print(f"  {name}: {list(arr.shape)} [{arr.min():.6f}, {arr.max():.6f}]")


def write_ref_gguf(output_dir: str):
    """Write all .npy intermediates into a single reference GGUF for crispasr-diff."""
    gguf_path = os.path.join(output_dir, "ref.gguf")
    writer = GGUFWriter(gguf_path, "piano-transcription-ref")

    npy_files = sorted(Path(output_dir).glob("*.npy"))
    for npy_file in npy_files:
        name = npy_file.stem
        arr = np.load(npy_file).astype(np.float32)
        # Flatten any batch dimension
        if arr.ndim >= 3 and arr.shape[0] == 1:
            arr = arr[0]
        writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
        print(f"  ref: {name} {list(arr.shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Reference GGUF: {gguf_path}")


def main():
    ap = argparse.ArgumentParser(description="Piano transcription reference dumper")
    ap.add_argument("--model", "-m", required=True, help="Path to model.pth")
    ap.add_argument("--audio", "-a", required=True, help="Path to audio file (16kHz mono WAV)")
    ap.add_argument("--output-dir", "-o", default="/mnt/volume1/tmp-overflow/piano-ref",
                    help="Output directory")
    args = ap.parse_args()

    import librosa
    audio, sr = librosa.load(args.audio, sr=16000, mono=True)
    print(f"Audio: {args.audio} ({len(audio)} samples, {len(audio)/16000:.2f}s at 16kHz)")

    model = load_model(args.model)
    dump_intermediates(model, audio, args.output_dir)


if __name__ == "__main__":
    main()

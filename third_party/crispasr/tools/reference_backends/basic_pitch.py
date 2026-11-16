#!/usr/bin/env python3
"""
Basic Pitch (Spotify, ICASSP 2022) reference dumper for crispasr-diff parity.

Runs the UPSTREAM nmp.onnx through onnxruntime and dumps every stage boundary
the C++ port has to reproduce, so a front-end difference can never masquerade
as a model failure:

  1. audio_window0     — (43844,)   the exact float samples fed to the model
  2. cqt_magnitude     — (172, 309) nnAudio CQT2010v2 magnitude
  3. normalized_log    — (172, 309) after signal.NormalizedLog
  4. harmonic_stack    — (172, 264, 8) after BN + HarmonicStacking
  5. head_contour      — (172, 264) sigmoid
  6. head_note         — (172, 88)  sigmoid
  7. head_onset        — (172, 88)  sigmoid
  8. unwrapped_{contour,note,onset} — full-file stitched posteriorgrams
  9. note_events.json  — decoded note events (start_s, end_s, midi, amplitude)

Stages 1-7 are for WINDOW 0 only; the C++ diff compares that window so the
comparison is exact rather than averaged over a file.

AUDIO CONDITIONING (must match the C++ byte for byte — see the "Ref dumper
conditioning" discipline in docs/contributing.md):
    audio = librosa.load(path, sr=22050, mono=True)      # float32
    audio = concat([zeros(3840), audio])                 # LEADING pad only
    window i = audio[i*36164 : i*36164 + 43844], right-zero-padded if short
No trailing silence, no normalisation, no dithering.

The ONNX is the one that ships inside the pip package / git repo:
    basic_pitch/saved_models/icassp_2022/nmp.onnx
    sha256 2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec

Usage:
    python tools/reference_backends/basic_pitch.py \\
        --model /mnt/storage/gguf-models/basic-pitch-src/nmp.onnx \\
        --audio samples/jfk.wav \\
        --output-dir /mnt/volume1/tmp-overflow/basic-pitch-ref

With crispasr-diff:
    python tools/dump_reference.py basic_pitch \\
        --model /mnt/storage/gguf-models/basic-pitch-src/nmp.onnx \\
        --audio samples/jfk.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)


# ── constants, verbatim from basic_pitch/constants.py + inference.py ─────────

AUDIO_SAMPLE_RATE = 22050
FFT_HOP = 256
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP          # 86
AUDIO_WINDOW_LENGTH = 2
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH  # 172
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP  # 43844
N_FREQ_BINS_CONTOURS = 264
CONTOURS_BINS_PER_SEMITONE = 3
ANNOTATIONS_BASE_FREQUENCY = 27.5

DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MINIMUM_NOTE_LENGTH_MS = 127.7
DEFAULT_OVERLAPPING_FRAMES = 30

MIDI_OFFSET = 21
MAX_FREQ_IDX = 87
MAGIC_ALIGNMENT_OFFSET = 0.0018

# ONNX output names, verbatim from inference.py::Model.predict.
ONNX_OUTPUTS = {
    "note": "StatefulPartitionedCall:1",
    "onset": "StatefulPartitionedCall:2",
    "contour": "StatefulPartitionedCall:0",
}
ONNX_INPUT = "serving_default_input_2:0"


# ── locating the intermediate tensors inside the ONNX graph ─────────────────
#
# The graph is a tf2onnx export, so node names are long semicolon-joined
# TensorFlow scope strings. Matching on substrings of those scopes is stable
# across re-exports in a way that matching on `const_fold_opt__NNN` is not.

def find_intermediates(model):
    """Return {logical_name: onnx_tensor_name} for the stages we dump."""
    g = model.graph
    out = {}

    def pick(pred, what):
        hits = [n for n in g.node if pred(n)]
        if len(hits) != 1:
            raise RuntimeError(f"expected exactly 1 node for {what}, got {len(hits)}")
        return hits[0].output[0]

    # CQT magnitude: the last Transpose inside the cq_t2010v2 scope, whose
    # input is the Sqrt of the summed squares.
    out["cqt_magnitude"] = pick(
        lambda n: n.op_type == "Transpose" and "cq_t2010v2" in n.name and "transpose_70" in n.name,
        "cqt magnitude",
    )
    # NormalizedLog output: the final Reshape in the normalized_log scope.
    out["normalized_log"] = pick(
        lambda n: n.op_type == "Reshape" and "normalized_log" in n.name and "Reshape_2" in n.name,
        "normalized log",
    )
    # HarmonicStacking output: the trailing Slice that keeps the first 264 bins.
    out["harmonic_stack"] = pick(
        lambda n: n.op_type == "Slice" and "harmonic-stacking" in n.name and "strided_slice_7" in n.name,
        "harmonic stack",
    )
    return out


def make_session(model_path: str, want_intermediates: bool):
    import onnx
    import onnxruntime as ort

    model = onnx.load(model_path)
    extra = {}
    if want_intermediates:
        extra = find_intermediates(model)
        for tname in extra.values():
            model.graph.output.append(onnx.ValueInfoProto(name=tname))
    so = ort.SessionOptions()
    # Constant folding would delete the intermediate tensors we just exposed.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return sess, extra


# ── windowing, verbatim from inference.py ───────────────────────────────────

def window_audio(audio: np.ndarray, hop_size: int, overlap_len: int):
    """Yield (window (43844,), window_index). Mirrors get_audio_input()."""
    padded = np.concatenate([np.zeros((overlap_len // 2,), dtype=np.float32), audio])
    idx = 0
    for i in range(0, padded.shape[0], hop_size):
        w = padded[i:i + AUDIO_N_SAMPLES]
        if len(w) < AUDIO_N_SAMPLES:
            w = np.pad(w, pad_width=[[0, AUDIO_N_SAMPLES - len(w)]])
        yield w.astype(np.float32), idx
        idx += 1


def unwrap_output(output: np.ndarray, audio_original_length: int, n_overlapping_frames: int, hop_size: int):
    """Verbatim from inference.py::unwrap_output."""
    if len(output.shape) != 3:
        return None
    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        output = output[:, n_olap:-n_olap, :]
    s = output.shape
    unwrapped = output.reshape(s[0] * s[1], s[2])
    n_expected_windows = audio_original_length / hop_size
    n_frames_per_window = (AUDIO_WINDOW_LENGTH * ANNOTATIONS_FPS) - n_overlapping_frames
    return unwrapped[: int(n_expected_windows * n_frames_per_window), :]


# ── note creation, transcribed from basic_pitch/note_creation.py ────────────
#
# Copied rather than imported because note_creation.py imports pretty_midi,
# mir_eval and resampy at module scope, none of which are needed to produce
# note events and none of which are installed here. The logic below is
# line-for-line equivalent to output_to_notes_polyphonic / get_pitch_bends /
# model_frames_to_time.

def get_infered_onsets(onsets, frames, n_diff=2):
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    frame_diff = np.max(onsets) * frame_diff / np.max(frame_diff)
    return np.max([onsets, frame_diff], axis=0)


def model_frames_to_time(n_frames: int) -> np.ndarray:
    original_times = np.arange(n_frames) * FFT_HOP / AUDIO_SAMPLE_RATE
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + MAGIC_ALIGNMENT_OFFSET
    return original_times - (window_offset * window_numbers)


def output_to_notes_polyphonic(frames, onsets, onset_thresh, frame_thresh, min_note_len,
                               infer_onsets=True, melodia_trick=True, energy_tol=11):
    import scipy.signal

    n_frames = frames.shape[0]
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]
    onset_freq_idx = onset_idx[1][::-1]

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]

    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        if note_start_idx >= n_frames - 1:
            continue
        i = note_start_idx + 1
        k = 0
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1
        i -= k
        if i - note_start_idx <= min_note_len:
            continue
        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0
        amplitude = np.mean(frames[note_start_idx:i, freq_idx])
        note_events.append((note_start_idx, i, freq_idx + MIDI_OFFSET, amplitude))

    if melodia_trick:
        energy_shape = remaining_energy.shape
        while np.max(remaining_energy) > frame_thresh:
            i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), energy_shape)
            remaining_energy[i_mid, freq_idx] = 0

            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i += 1
            i_end = i - 1 - k

            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i -= 1
            i_start = i + 1 + k

            if i_end - i_start <= min_note_len:
                continue
            amplitude = np.mean(frames[i_start:i_end, freq_idx])
            note_events.append((i_start, i_end, freq_idx + MIDI_OFFSET, amplitude))

    return note_events


# ── main ────────────────────────────────────────────────────────────────────

def save_tensor(arr: np.ndarray, name: str, output_dir: str):
    arr = np.asarray(arr, dtype=np.float32)
    np.save(os.path.join(output_dir, name + ".npy"), arr)
    print(f"  {name}: {list(arr.shape)}")


def write_ref_gguf(output_dir: str):
    gguf_path = os.path.join(output_dir, "ref.gguf")
    writer = GGUFWriter(gguf_path, "basic-pitch-ref")
    for npy_file in sorted(Path(output_dir).glob("*.npy")):
        arr = np.load(npy_file).astype(np.float32)
        if arr.ndim >= 3 and arr.shape[0] == 1:
            arr = arr[0]
        writer.add_tensor(npy_file.stem, arr, raw_dtype=GGMLQuantizationType.F32)
        print(f"  ref: {npy_file.stem} {list(arr.shape)}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Reference GGUF: {gguf_path}")


def main():
    ap = argparse.ArgumentParser(description="Basic Pitch reference dumper")
    ap.add_argument("--model", "-m", default="/mnt/storage/gguf-models/basic-pitch-src/nmp.onnx")
    ap.add_argument("--audio", "-a", required=True)
    ap.add_argument("--output-dir", "-o", default="/mnt/volume1/tmp-overflow/basic-pitch-ref")
    ap.add_argument("--onset-threshold", type=float, default=DEFAULT_ONSET_THRESHOLD)
    ap.add_argument("--frame-threshold", type=float, default=DEFAULT_FRAME_THRESHOLD)
    ap.add_argument("--minimum-note-length", type=float, default=DEFAULT_MINIMUM_NOTE_LENGTH_MS)
    ap.add_argument("--no-gguf", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import librosa

    audio, _ = librosa.load(args.audio, sr=AUDIO_SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)
    original_length = audio.shape[0]
    print(f"Audio: {args.audio} ({original_length} samples, "
          f"{original_length / AUDIO_SAMPLE_RATE:.2f}s at {AUDIO_SAMPLE_RATE} Hz)")

    n_overlapping_frames = DEFAULT_OVERLAPPING_FRAMES
    overlap_len = n_overlapping_frames * FFT_HOP        # 7680
    hop_size = AUDIO_N_SAMPLES - overlap_len            # 36164

    sess, extra = make_session(args.model, want_intermediates=True)
    out_names = [ONNX_OUTPUTS["note"], ONNX_OUTPUTS["onset"], ONNX_OUTPUTS["contour"]] + list(extra.values())
    extra_keys = list(extra.keys())

    acc = {"note": [], "onset": [], "contour": []}
    for window, widx in window_audio(audio, hop_size, overlap_len):
        x = window[None, :, None]                       # (1, 43844, 1)
        res = sess.run(out_names, {ONNX_INPUT: x})
        note, onset, contour = res[0], res[1], res[2]
        acc["note"].append(note)
        acc["onset"].append(onset)
        acc["contour"].append(contour)
        if widx == 0:
            save_tensor(window, "audio_window0", args.output_dir)
            for i, key in enumerate(extra_keys):
                save_tensor(res[3 + i][0], key, args.output_dir)
            save_tensor(contour[0], "head_contour", args.output_dir)
            save_tensor(note[0], "head_note", args.output_dir)
            save_tensor(onset[0], "head_onset", args.output_dir)

    unwrapped = {
        k: unwrap_output(np.concatenate(acc[k]), original_length, n_overlapping_frames, hop_size)
        for k in acc
    }
    for k, v in unwrapped.items():
        save_tensor(v, "unwrapped_" + k, args.output_dir)

    min_note_len = int(np.round(args.minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    events = output_to_notes_polyphonic(
        unwrapped["note"], unwrapped["onset"],
        onset_thresh=args.onset_threshold,
        frame_thresh=args.frame_threshold,
        min_note_len=min_note_len,
    )
    times_s = model_frames_to_time(unwrapped["contour"].shape[0])
    events_s = [
        {
            "start_s": float(times_s[a]),
            "end_s": float(times_s[b]),
            "midi": int(p),
            "amplitude": float(amp),
            "velocity": int(np.round(127 * amp)),
        }
        for a, b, p, amp in events
    ]
    events_s.sort(key=lambda e: (e["start_s"], e["midi"]))
    meta = {
        "audio": os.path.abspath(args.audio),
        "n_samples": int(original_length),
        "n_windows": len(acc["note"]),
        "min_note_len": int(min_note_len),
        "onset_thresh": args.onset_threshold,
        "frame_thresh": args.frame_threshold,
        "n_notes": len(events_s),
        "notes": events_s,
    }
    with open(os.path.join(args.output_dir, "note_events.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"  note_events.json: {len(events_s)} notes")

    if not args.no_gguf:
        write_ref_gguf(args.output_dir)


if __name__ == "__main__":
    main()

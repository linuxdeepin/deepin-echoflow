#!/usr/bin/env python3
"""Reference for beat-this WINDOWING + POSTPROCESSING (§251b-1).

    PYTHONPATH=<beat_this-src> python tools/gen_beat_this_track_ref.py \
        --ckpt final0.ckpt --outdir <dir> [--seconds 45]

Writes, into <dir>:
    track_input.bin   float32 mono 22050 Hz PCM  (synthetic click track)
    track_logmel.bin  float32 (frames,128) log-mel
    track_ref.npz     beat_logits, downbeat_logits, beat_time, downbeat_time

WHY A LONG SIGNAL. The 101-frame front-end fixture is a SINGLE chunk, so it
exercises none of the windowing: no overlap, no border discard, no keep_first
precedence, no avoid_short_end shift. At 45 s the piece is 2 chunks with a
~975-frame overlap region, which is where a reversed()-order mistake shows up.
Raise --seconds to 65 for a 3-chunk case.

WHY A CLICK TRACK rather than noise. The postprocessing only does anything if
some logits actually clear 0, and a real beat grid also produces plateaus and
adjacent peaks — the cases deduplicate_peaks exists for. Noise would exercise
the code with an empty peak list and prove nothing.

Both sides consume the SAME track_logmel.bin, so a mismatch here can never be
blamed on the front end (which is separately at cos 1.00000000).
"""
import argparse
import os
import sys
from pathlib import Path

# Same self-shadowing guard as tools/reference_backends/beat_this.py: this
# directory must not sit ahead of the upstream package on sys.path.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]

import numpy as np
import torch
import torchaudio


def click_track(sr, seconds, bpm=120.0, beats_per_bar=4):
    """Exponentially-decaying clicks, louder and lower on the downbeat."""
    n = int(sr * seconds)
    x = np.zeros(n, dtype=np.float64)
    period = 60.0 / bpm
    idx = 0
    while True:
        t0 = int(idx * period * sr)
        if t0 >= n:
            break
        downbeat = (idx % beats_per_bar) == 0
        freq, amp, dur = (180.0, 1.0, 0.12) if downbeat else (330.0, 0.55, 0.06)
        m = min(int(dur * sr), n - t0)
        tt = np.arange(m) / sr
        x[t0 : t0 + m] += amp * np.sin(2 * np.pi * freq * tt) * np.exp(-tt / (dur / 4))
        idx += 1
    rng = np.random.default_rng(0)
    x += 0.005 * rng.standard_normal(n)
    return (x / np.abs(x).max() * 0.9).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=45.0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    from beat_this.inference import split_predict_aggregate
    from beat_this.model.beat_tracker import BeatThis
    from beat_this.model.postprocessor import Postprocessor

    sr = 22050
    pcm = click_track(sr, args.seconds)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=441, f_min=30, f_max=11000,
        n_mels=128, mel_scale="slaney", normalized="frame_length", power=1)
    spect = torch.log1p(1000.0 * mel(torch.tensor(pcm)).T)

    m = BeatThis().eval()
    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARNING: missing={len(missing)} unexpected={len(unexpected)}")

    with torch.no_grad():
        pred = split_predict_aggregate(
            spect=spect, chunk_size=1500, border_size=6,
            overlap_mode="keep_first", model=m)
    beat_time, downbeat_time = Postprocessor(type="minimal", fps=50)(
        pred["beat"], pred["downbeat"])

    pcm.tofile(args.outdir / "track_input.bin")
    spect.numpy().astype(np.float32).tofile(args.outdir / "track_logmel.bin")
    np.savez(
        args.outdir / "track_ref.npz",
        beat_logits=pred["beat"].numpy().astype(np.float32),
        downbeat_logits=pred["downbeat"].numpy().astype(np.float32),
        beat_time=np.asarray(beat_time, dtype=np.float64),
        downbeat_time=np.asarray(downbeat_time, dtype=np.float64),
    )
    frames = spect.shape[0]
    print(f"frames={frames} chunks={1 + max(0, (frames - 6 - 1) // 1488)} "
          f"beats={len(beat_time)} downbeats={len(downbeat_time)}")
    print(f"wrote {args.outdir}/track_{{input.bin,logmel.bin,ref.npz}}")


if __name__ == "__main__":
    main()

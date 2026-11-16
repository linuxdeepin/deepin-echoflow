#!/usr/bin/env python3
"""Generate the torchaudio fixtures that tests/test_beat_this_frontend.cpp checks.

    python tools/gen_beat_this_frontend_ref.py <outdir>

Writes fe_input.bin (float32 mono 22050 Hz PCM) and fe_ref.bin (float32
(frames,128) log-mel), then:

    ./build/bin/test-beat-this-frontend beat-this-f16.gguf \
        <outdir>/fe_input.bin <outdir>/fe_ref.bin

Mirrors beat_this/preprocessing.py LogMelSpect exactly. Two parameters are
load-bearing and easy to get wrong:
  * power=1  -> MAGNITUDE, not power. Squaring yields a plausible spectrogram
                and wrong beats.
  * normalized="frame_length" -> divides by SQRT(n_fft) = 32, NOT n_fft,
                despite the name. Verified: the ratio between normalized=False
                and normalized="frame_length" is exactly 32.0 at n_fft=1024.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    sr, n = 22050, 22050 * 2
    torch.manual_seed(0)
    t = torch.arange(n) / sr
    x = (0.5 * torch.sin(2 * np.pi * 220 * t)
         + 0.3 * torch.sin(2 * np.pi * 1000 * t)
         + 0.05 * torch.randn(n)).float()
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=441, f_min=30, f_max=11000,
        n_mels=128, mel_scale="slaney", normalized="frame_length", power=1)
    ref = torch.log1p(1000.0 * mel(x).T)
    x.numpy().astype(np.float32).tofile(out / "fe_input.bin")
    ref.numpy().astype(np.float32).tofile(out / "fe_ref.bin")
    print(f"wrote {out}/fe_input.bin ({n} samples) and {out}/fe_ref.bin {tuple(ref.shape)}")


if __name__ == "__main__":
    main()

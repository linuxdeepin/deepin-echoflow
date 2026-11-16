#!/usr/bin/env python
"""Phase-invariant log-magnitude-spectrogram correlation between two wavs (§207 parity metric)."""
import sys
import numpy as np
import soundfile as sf


def logmag(path, n_fft=1024, hop=256):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    win = np.hanning(n_fft)
    frames = []
    for i in range(0, len(x) - n_fft, hop):
        frames.append(np.abs(np.fft.rfft(x[i:i + n_fft] * win)))
    S = np.array(frames)
    return np.log(S + 1e-6), sr


def main():
    a, sra = logmag(sys.argv[1])
    b, srb = logmag(sys.argv[2])
    n = min(len(a), len(b))
    a, b = a[:n].ravel(), b[:n].ravel()
    corr = np.corrcoef(a, b)[0, 1]
    # also raw sample-domain stats
    xa, _ = sf.read(sys.argv[1])
    xb, _ = sf.read(sys.argv[2])
    if xa.ndim > 1:
        xa = xa.mean(1)
    if xb.ndim > 1:
        xb = xb.mean(1)
    m = min(len(xa), len(xb))
    max_abs = float(np.max(np.abs(xa[:m] - xb[:m]))) if m else float("nan")
    print(f"frames a={len(a)//(logmag.__defaults__ and 1 or 1)} sr={sra}/{srb}")
    print(f"len_a={len(xa)} len_b={len(xb)}")
    print(f"logmag-spectral-corr = {corr:.6f}")
    print(f"sample-domain max|Δ| = {max_abs:.6e}")
    print("PASS" if corr >= 0.999 else "FAIL (need >=0.999)")


if __name__ == "__main__":
    main()

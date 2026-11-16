#!/usr/bin/env python3
"""Score src/core/cqt.h against librosa's CQT.

Usage:
    python tools/cqt_librosa_parity.py <path-to-test-core-cqt-dump-binary>

The C++ side (tests/test-core-cqt.cpp --dump) writes a raw float32 magnitude
matrix for a deterministic test signal; this script computes librosa's CQT on
the identical signal and reports per-bin correlation AND per-bin magnitude
ratio. The magnitude check is not optional: correlation and peak-bin match are
scale-invariant and once hid a sqrt(N_k) per-bin scale error here.

Why correlation and not cosine-to-1.0: librosa's CQT uses recursive downsampling
with a different normalisation lineage, so an exact match is not the target. The
question BTC actually cares about is whether the two front ends rank bins the
same way and put energy in the same places. Report the numbers, do not paper
over them.
"""
import sys
import struct
import numpy as np

SR = 22050
FMIN = 32.703195662574829  # C1
N_BINS = 144
BPO = 24
HOP = 2048


def test_signal(n):
    """Deterministic: three sustained tones an octave apart + a chirp tail.
    Must match tests/test-core-cqt.cpp exactly."""
    t = np.arange(n) / SR
    x = np.zeros(n, dtype=np.float64)
    third = n // 3
    for i, f in enumerate((130.8127826502993, 261.6255653005986, 523.2511306011972)):
        lo, hi = i * third, (i + 1) * third if i < 2 else n
        x[lo:hi] = 0.5 * np.sin(2 * np.pi * f * t[lo:hi])
    return x.astype(np.float32)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: cqt_librosa_parity.py <dump.bin>")
    import librosa

    with open(sys.argv[1], "rb") as fh:
        T, K = struct.unpack("<ii", fh.read(8))
        mine = np.frombuffer(fh.read(T * K * 4), dtype=np.float32).reshape(T, K)
    print(f"c++  : {T} frames x {K} bins")

    x = test_signal(T * HOP)
    ref = np.abs(librosa.cqt(y=x.astype(np.float64), sr=SR, fmin=FMIN, n_bins=N_BINS,
                             bins_per_octave=BPO, hop_length=HOP)).T  # (frames, bins)
    print(f"librosa: {ref.shape[0]} frames x {ref.shape[1]} bins")

    n = min(T, ref.shape[0])
    a, b = mine[:n], ref[:n]

    # Per-frame correlation over the bin axis: does the spectral SHAPE agree?
    cors = []
    for i in range(n):
        va, vb = a[i], b[i]
        if va.std() < 1e-12 or vb.std() < 1e-12:
            continue
        cors.append(float(np.corrcoef(va, vb)[0, 1]))
    cors = np.array(cors)

    # Does the peak bin agree? That is what a chord/pitch model keys on.
    peak_match = float(np.mean(a.argmax(1) == b.argmax(1)))
    # ...and within one bin (half a semitone at bpo=24)?
    peak_close = float(np.mean(np.abs(a.argmax(1).astype(int) - b.argmax(1).astype(int)) <= 1))

    print(f"\nper-frame shape correlation: mean {cors.mean():.4f}  min {cors.min():.4f}  "
          f"median {np.median(cors):.4f}  (n={len(cors)})")
    print(f"peak-bin exact match : {peak_match:.1%}")
    print(f"peak-bin within +/-1 : {peak_close:.1%}")

    # SCALE-SENSITIVE check. Correlation and peak-bin match are both invariant
    # to magnitude, so on their own they cannot see a per-bin scale error --
    # and one lived here undetected: core/cqt.h was missing librosa's
    # scale=True normalisation, leaving every bin off by sqrt(N_k) (bin 0 was
    # 152x). BTC then fed out-of-distribution features to the model and
    # predicted N everywhere while the reference predicted real chords.
    #
    # Same class of miss as the htdemucs iSTFT 1/sqrt(nfft) bug, which cosine
    # also could not see. Any parity tool that only scores shape needs a
    # magnitude assertion beside it.
    # Scale can only be verified where there IS signal. This reference tone is
    # sparse (three sustained pitches), so most bins sit on the epsilon floor
    # and their ratio is meaningless noise -- judging those would fail the
    # check for no reason. Test a bin only when its own peak carries real
    # energy, and inside such a bin use only the frames that do.
    testable, ratios = [], []
    for k in range(a.shape[1]):
        col_ref = b[:, k]
        if col_ref.max() < b.max() * 0.05:
            continue
        m = col_ref > col_ref.max() * 0.5
        if not m.any():
            continue
        testable.append(k)
        ratios.append(float(np.median(a[m, k] / np.maximum(col_ref[m], 1e-12))))
    if ratios:
        ratios = np.array(ratios)
        # Assert on the MEDIAN, report the worst. A normalisation error is
        # SYSTEMATIC -- the sqrt(N_k) one moved this median to 152 at the
        # bottom octave -- so the median catches it with enormous margin.
        # Individual bins on a tone's skirt can differ by tens of percent for a
        # legitimate reason: librosa uses recursive per-octave downsampling
        # while these are direct kernels, so the two disagree in group delay
        # (the same effect that drags the shape-correlation MIN down while the
        # median stays 0.9999). Asserting the worst bin would fail on that.
        scale_err = float(np.abs(np.median(ratios) - 1.0))
        print(f"per-bin magnitude ratio: {len(testable)} bins with real energy, "
              f"median {np.median(ratios):.4f} (asserted)  "
              f"worst |ratio-1| {float(np.max(np.abs(ratios - 1.0))):.4f} (informational)")
    else:
        scale_err = 0.0
        print("per-bin magnitude ratio: no bin carried enough energy to test")

    # Where the expected tones land, as an absolute sanity check.
    for f in (130.8127826502993, 261.6255653005986, 523.2511306011972):
        k = int(round(BPO * np.log2(f / FMIN)))
        print(f"  {f:8.2f} Hz -> bin {k:3d}")

    ok = cors.mean() > 0.9 and peak_close > 0.9 and scale_err < 0.05
    print("\n" + ("PASS" if ok else "FAIL") + " (shape corr > 0.9 and peak within 1 bin > 90%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

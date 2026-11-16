#!/usr/bin/env python3
"""Independent within-arm boundary check for mel-band-roformer segmentation (#422).

A SECOND IMPLEMENTATION, on purpose. tools/kaggle/melband-seg-evidence-v2 computes
this in-kernel; this script recomputes it offline from the stems alone. The point
is that it can DISAGREE — if the two implementations differ, one has a bug, and
finding that is worth more than the verdict itself. (Run 2026-09-03: they agreed
to 0.01 dB on both arms, from separately written window models.)

WHAT IT MEASURES. Within a single segmented arm, SDR inside the overlap-add
windows vs SDR over the segment interiors, both against arm A (C++ with
CRISPASR_MELBAND_NO_SEGMENT=1) as the reference. Arm A is the right reference,
not the torch oracle: same binary, same weights, same graph path, so the ONLY
difference is segmentation — the variable under test. Measuring against torch
would fold the port's own numerical differences into the answer.

WHY NOT A CROSS-ARM COMPARISON. Comparing arm B against arm C cannot separate
"the metric cannot see boundary effects" from "there were no boundary effects to
see" — opposite states, same null. This is a localised positive measurement
instead, and it never compares across segment lengths.

    THE EXPECTED RESULT IS POSITIVE, NOT ZERO.
An overlap-add crossfade averages TWO independent estimates of the same samples.
Averaging two independent errors of equal variance halves the noise power, which
is 10*log10(2) = 3.01 dB. So a HEALTHY overlap-add makes the boundary regions
score BETTER than the interiors by roughly 3 dB. The first version of this script
treated any positive delta as suspicious and reported the healthy signature of
arm B (+3.75 dB) as "unexpected; suspect the window model" — a readout that
mislabels a correct state, which is the same disease it was written to detect.
A delta near ZERO is the suspicious outcome: it means the averaging gain is not
appearing, i.e. overlap-add may not be doing what it claims.

    A NEGATIVE DELTA IS NOT AUTOMATICALLY AN ARTEFACT.
If an arm's segment length is shorter than the checkpoint's TRAINED chunk, that
arm feeds the model out-of-distribution buffers, and a boundary-vs-interior loss
then conflates two causes: crossfade artefacts and OOD segment length. This
script cannot separate them and says so rather than announcing a finding.
(Observed: arm C at 3 s against an 8 s trained chunk scored -2.62 dB.)

Usage:
    python tools/melband-within-arm-check.py <out_dir> [--trained-chunk 352800]
where <out_dir> holds A_noseg/, B_default/, C_seg3/ from the evidence kernel.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
OVERLAP = 0.25
TRAINED_CHUNK_DEFAULT = 352800  # Kim vocals, 8.0 s @ 44100 (GGUF mel-band-roformer.chunk_size)

# Segment length in SAMPLES per arm, matching the kernel's arm definitions.
ARM_SEG = {"A_noseg": None, "B_default": TRAINED_CHUNK_DEFAULT, "C_seg3": 3 * SR}

# Averaging two independent estimates halves noise power.
EXPECTED_GAIN_DB = 10.0 * np.log10(2.0)  # 3.01
GAIN_TOL_DB = 2.0  # how far from EXPECTED_GAIN_DB still reads as "consistent with averaging"

# Above this the residual is numerically indistinguishable from zero and the
# reported SDR is an artefact of the epsilon, not a measurement. Bit-identical
# arms would otherwise have SDR pinned to each region's SIGNAL energy, and the
# differing energy of boundary vs interior regions manufactures a delta at
# exactly the moment segmentation is a perfect no-op.
FLOOR_DB = 120.0


def load_stem(d: Path):
    w = sorted(d.rglob("*vocal*.wav")) or sorted(d.rglob("*.wav"))
    if not w:
        return None, None
    a, _sr = sf.read(str(w[0]), dtype="float64", always_2d=True)
    return a, w[0].name


def segment_windows(n_samples: int, seg_len: int):
    """Replicate the C++ schedule: off += stride, stride = 0.75*seg_len.

    Returns (overlap_mask, n_internal_boundaries). A sample is 'overlap' when
    covered by more than one segment — precisely where overlap-add arithmetic
    is exercised.
    """
    stride = int((1.0 - OVERLAP) * seg_len) or seg_len
    cover = np.zeros(n_samples, dtype=np.int32)
    n_seg = 0
    off = 0
    while off < n_samples:
        n_seg += 1
        valid = min(seg_len, n_samples - off)
        cover[off:off + valid] += 1
        off += stride
    return cover > 1, max(n_seg - 1, 0)


def sdr(est, ref, mask=None):
    """Returns (dB, n_samples). inf = provably identical; nan = below the floor."""
    e, r = est.reshape(-1, est.shape[-1]), ref.reshape(-1, ref.shape[-1])
    if mask is not None:
        e, r = e[mask], r[mask]
    if e.size == 0:
        return float("nan"), 0
    ef, rf = e.reshape(-1), r.reshape(-1)
    noise = ef - rf
    ne, se = float(noise @ noise), float(rf @ rf)
    if ne <= 0.0:
        return float("inf"), e.shape[0]
    val = 10.0 * np.log10(se / ne)
    return (float("nan") if val > FLOOR_DB else val), e.shape[0]


def main(out_dir: str, trained_chunk: int) -> int:
    root = Path(out_dir)
    ref, ref_name = load_stem(root / "A_noseg")
    if ref is None:
        print("A_noseg missing — cannot localise without the unsegmented arm.")
        return 2
    print(f"reference (unsegmented C++ arm A): {ref_name}  {ref.shape}")
    print(f"trained chunk: {trained_chunk} samples ({trained_chunk / SR:.2f} s)")
    print(f"expected boundary gain from 2-estimate averaging: +{EXPECTED_GAIN_DB:.2f} dB\n")
    print(f"{'arm':<12}{'bounds':>7}{'seg_s':>7}{'whole dB':>10}{'interior':>10}"
          f"{'boundary':>10}{'delta':>8}{'n_int':>10}{'n_bnd':>9}")

    rows = []
    for arm, seg_len in ARM_SEG.items():
        if arm == "A_noseg":
            continue
        est, _ = load_stem(root / arm)
        if est is None:
            print(f"{arm:<12}{'VOID (no output)':>45}")
            continue
        n = min(len(est), len(ref))
        e, r = est[:n], ref[:n]
        bmask, nb = segment_windows(n, seg_len)
        whole, _ = sdr(e, r)
        inter, n_i = sdr(e, r, ~bmask)
        bound, n_b = sdr(e, r, bmask)
        delta = bound - inter
        print(f"{arm:<12}{nb:>7}{seg_len / SR:>7.1f}{whole:10.2f}{inter:10.2f}"
              f"{bound:10.2f}{delta:+8.2f}{n_i:10d}{n_b:9d}")
        rows.append((arm, seg_len, nb, inter, bound, delta, n_b))

    print("\n--- within-arm verdict ---")
    for arm, seg_len, nb, inter, bound, delta, n_b in rows:
        ood = seg_len < trained_chunk
        if n_b == 0 or nb == 0:
            print(f"{arm}: no overlap region / 0 internal boundaries — uninformative.")
        elif np.isinf(inter) or np.isinf(bound):
            print(f"{arm}: BIT-IDENTICAL to unsegmented in at least one region. No artefact.")
        elif np.isnan(inter) or np.isnan(bound):
            print(f"{arm}: NEAR FLOOR (>{FLOOR_DB:.0f} dB) — differs, but too little to measure.")
        elif abs(delta - EXPECTED_GAIN_DB) <= GAIN_TOL_DB:
            print(f"{arm}: delta {delta:+.2f} dB is CONSISTENT WITH HEALTHY OVERLAP-ADD "
                  f"(expected ~+{EXPECTED_GAIN_DB:.2f} from averaging two estimates). "
                  f"Positive evidence of NO boundary artefact over {nb} boundaries.")
        elif delta > EXPECTED_GAIN_DB + GAIN_TOL_DB:
            print(f"{arm}: delta {delta:+.2f} dB EXCEEDS what averaging explains "
                  f"(~+{EXPECTED_GAIN_DB:.2f}). Not a defect signature, but unexplained — "
                  f"check whether the boundary regions coincide with quieter audio.")
        elif delta > -1.0:
            print(f"{arm}: delta {delta:+.2f} dB — the ~+{EXPECTED_GAIN_DB:.2f} dB averaging "
                  f"gain is NOT APPEARING. Suspicious: is overlap-add actually running, "
                  f"or is one segment dominating the crossfade?")
        elif ood:
            print(f"{arm}: delta {delta:+.2f} dB (boundary WORSE). NOT ATTRIBUTABLE — this arm's "
                  f"{seg_len / SR:.1f} s segments are shorter than the {trained_chunk / SR:.1f} s "
                  f"trained chunk, so boundary artefact and out-of-distribution segment length "
                  f"are confounded. This script cannot separate them.")
        else:
            print(f"{arm}: delta {delta:+.2f} dB (boundary WORSE) at or above the trained chunk, "
                  f"so OOD length does not explain it — localised crossfade artefact. REAL FINDING.")
        if nb <= 1:
            print(f"    CAVEAT: {nb} internal boundary — underpowered, not a verdict on the default.")

    print("\nNOTE: arm A is the reference, so this measures SEGMENTATION ONLY. It says nothing "
          "about whether the unsegmented path is itself correct — arm A and any unsegmented "
          "oracle both extrapolate past the trained chunk on a long clip and can agree with "
          "each other precisely because they do the same unusual thing.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="directory holding A_noseg/, B_default/, C_seg3/")
    ap.add_argument("--trained-chunk", type=int, default=TRAINED_CHUNK_DEFAULT,
                    help="checkpoint's trained chunk in SAMPLES (GGUF mel-band-roformer.chunk_size)")
    a = ap.parse_args()
    sys.exit(main(a.out_dir, a.trained_chunk))

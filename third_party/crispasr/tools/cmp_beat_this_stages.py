#!/usr/bin/env python3
"""Score beat-this C++ stage dumps against the torch reference (§251b-1).

    python tools/cmp_beat_this_stages.py <bt_stages.npz> <dump.bin> [<dump.bin> ...]
    python tools/cmp_beat_this_stages.py <bt_stages.npz> --prefix <dir/> <stage> ...

LAYOUT. The dump carries its ggml `ne` in the header, and ggml's ne is the
exact reverse of torch's shape — so reshaping to `reversed(ne)` lands on the
reference's layout with no transpose at all. Leading singleton axes are then
dropped to meet 3-D references (the sub-block hooks have no batch dim, because
einops folded the batch into the sequence count). Doing it this way rather than
with a hand-written per-stage transpose is deliberate: a layout slip would
otherwise show up as a plausible numerical error.

Prints |mine| and |ref| as well as cosine and RELATIVE max error. Magnitudes
matter: a 10-30x outlier on either side says "same name, wrong data" instantly,
whereas cosine alone reads as plausible drift. And relative error rather than
absolute, because these layers carry magnitudes around 200 where pure f16
weight rounding already costs ~2.5e-3 absolute — an absolute threshold would
either pass everything or fail everything depending on the layer's scale.
"""
import struct
import sys
from pathlib import Path

import numpy as np

COS_MIN = 0.9999
REL_MAX = 5e-3


def load_dump(path):
    with open(path, "rb") as f:
        ne = struct.unpack("<iiii", f.read(16))
        n = int(np.prod(ne))
        a = np.frombuffer(f.read(n * 4), dtype=np.float32)
    return a.reshape(tuple(reversed(ne)))


def score(name, mine, ref):
    while mine.ndim > ref.ndim and mine.shape[0] == 1:
        mine = mine[0]
    print(f"{name:16s} cpp={mine.shape} ref={ref.shape}")
    if mine.shape != ref.shape:
        print(f"{'':16s}   SHAPE MISMATCH")
        return False
    a, b = mine.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    scale = np.abs(b).max()
    rel = float(np.abs(a - b).max() / (scale + 1e-30))
    ok = cos > COS_MIN and rel < REL_MAX
    print(
        f"{'':16s}   cos={cos:.8f}  rel_max={rel:.3e}  "
        f"|mine|={np.linalg.norm(a):.4f} |ref|={np.linalg.norm(b):.4f}  "
        + ("PASS" if ok else "FAIL")
    )
    return ok


def main():
    argv = sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        return 2
    npz = np.load(argv[0])
    items = []
    if argv[1] == "--prefix":
        prefix = argv[2]
        for stage in argv[3:]:
            items.append((stage, f"{prefix}{stage}.bin"))
    else:
        for p in argv[1:]:
            items.append((Path(p).stem, p))

    all_ok = True
    for stage, path in items:
        if stage not in npz.files:
            print(f"{stage:16s} not in reference npz")
            all_ok = False
            continue
        all_ok &= score(stage, load_dump(path), npz[stage])
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

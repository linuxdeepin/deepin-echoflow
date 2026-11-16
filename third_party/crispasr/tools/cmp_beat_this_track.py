#!/usr/bin/env python3
"""Score beat-this windowing + postprocessing against the reference (§251b-1).

    python tools/cmp_beat_this_track.py <track_ref.npz> <out_prefix>

Checks three things separately, because they fail differently:

  logits    our 1500-frame chunking / border discard / keep_first aggregation
            vs the reference's. Reported as cosine AND max |diff| over frames
            that the reference actually covered.
  postp     our peak-picker fed the REFERENCE's own logits vs the reference's
            events. This must be EXACT to ~1e-6 — same input, same algorithm,
            no numerics in between. Any drift here is an algorithm difference,
            most likely the fractional mean in deduplicate_peaks.
  events    our logits through our peak-picker, end to end. Allowed to differ
            from the reference by one frame (20 ms) on individual beats.

A frame left at the -1000 sentinel means the chunking failed to cover it, which
is checked explicitly: it is silent at the event level (a -1000 logit simply
never peaks) but would mean a hole in the middle of the piece.
"""
import struct
import sys

import numpy as np

FPS = 50.0


def read_events(path):
    """int32 count, then (float32 time_s, int32 is_downbeat) pairs."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<i", f.read(4))
        rec = [struct.unpack("<fi", f.read(8)) for _ in range(n)]
    t = np.asarray([r[0] for r in rec], dtype=np.float64)
    d = np.asarray([r[1] for r in rec], dtype=np.int32)
    return t, d


def match(name, mine, ref, tol):
    ok = len(mine) == len(ref)
    if not ok:
        print(f"  {name}: COUNT {len(mine)} vs ref {len(ref)}  FAIL")
        n = min(len(mine), len(ref))
        if n:
            print(f"    first {min(5, n)} mine={np.round(mine[:5], 4)} ref={np.round(ref[:5], 4)}")
        return False
    if len(mine) == 0:
        print(f"  {name}: both empty  PASS")
        return True
    d = np.abs(mine - ref)
    ok = d.max() <= tol
    print(f"  {name}: n={len(mine)} max_dt={d.max():.6f}s (tol {tol}s)  " + ("PASS" if ok else "FAIL"))
    if not ok:
        i = int(d.argmax())
        print(f"    worst at {i}: mine={mine[i]:.6f} ref={ref[i]:.6f}")
    return ok


def main():
    ref = np.load(sys.argv[1])
    p = sys.argv[2]
    all_ok = True

    rb, rd = ref["beat_logits"].astype(np.float64), ref["downbeat_logits"].astype(np.float64)
    T = len(rb)
    raw = np.fromfile(p + "logits.bin", dtype=np.float32).astype(np.float64)
    mb, md = raw[:T], raw[T:]

    n_uncovered = int((mb <= -999.0).sum())
    print(f"logits: T={T}  uncovered frames (still -1000): {n_uncovered}")
    if n_uncovered:
        print("  CHUNKING GAP — some frames were never written  FAIL")
        all_ok = False
    for nm, a, b in (("beat", mb, rb), ("downbeat", md, rd)):
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        mx = float(np.abs(a - b).max())
        ok = cos > 0.9999 and mx < 0.05
        print(f"  {nm}: cos={cos:.8f} max_abs={mx:.4e} |mine|={np.linalg.norm(a):.4f} "
              f"|ref|={np.linalg.norm(b):.4f}  " + ("PASS" if ok else "FAIL"))
        all_ok &= ok

    rbt = np.asarray(ref["beat_time"], dtype=np.float64)
    rdt = np.asarray(ref["downbeat_time"], dtype=np.float64)

    print("postp (our peak-pick on the REFERENCE's logits — must be exact):")
    t, d = read_events(p + "postp.bin")
    all_ok &= match("beats", t, rbt, 1e-6)
    all_ok &= match("downbeats", t[d == 1], rdt, 1e-6)

    print("events (end to end):")
    t, d = read_events(p + "events.bin")
    all_ok &= match("beats", t, rbt, 1.0 / FPS)
    all_ok &= match("downbeats", t[d == 1], rdt, 1.0 / FPS)

    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

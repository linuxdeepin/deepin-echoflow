#!/usr/bin/env python3
"""Diarization Error Rate against reference labels (NIST / dscore conventions).

DER = (missed speech + false alarm + speaker confusion) / total reference
speech, after mapping hypothesis speakers onto reference speakers to maximise
agreement. A collar is excluded around every reference boundary, because human
boundary annotation is not millisecond-accurate and scoring those edges
measures the annotator rather than the system.

The mapping is EXACT by enumeration for <= 8 hypothesis speakers and greedy
above that. Overlapping speech is not modelled: each instant has at most one
reference and one hypothesis speaker.

This is the Python twin of src/core/der.h, kept for benchmarking pipelines that
are not CrispASR (e.g. comparing against an upstream reference implementation).
See docs/foxnose-diarize/PLAN.md for the VoxConverse recipe that uses it.

    from der_score import score
    s = score(ref_turns, hyp_turns, collar=0.25)   # [(start, end, speaker), ...]
    print(s["der"])
"""

import itertools, sys

def score(ref, hyp, collar=0.25):
    if not ref: return None
    T = max([t[1] for t in ref] + [t[1] for t in hyp] + [0])
    holes = []
    for a, b, _ in ref: holes += [(a-collar, a+collar), (b-collar, b+collar)]
    holes.sort(); merged = []
    for h in holes:
        if merged and h[0] <= merged[-1][1]: merged[-1] = (merged[-1][0], max(merged[-1][1], h[1]))
        else: merged.append(list(h)); merged[-1] = tuple(merged[-1])
    keep = []; cur = 0.0
    for h in merged:
        if h[0] > cur: keep.append((cur, min(h[0], T)))
        cur = max(cur, h[1])
    if cur < T: keep.append((cur, T))

    R = sorted({t[2] for t in ref}); H = sorted({t[2] for t in hyp})
    def at(ts, t):
        for a, b, s in ts:
            if a <= t < b: return s
        return None
    pts = sorted({x for a, b, _ in ref+hyp for x in (a, b)} | {x for k in keep for x in k})
    def in_scored(a, b):
        return any(a >= k[0] and b <= k[1] for k in keep)

    # optimal 1:1 mapping by total overlap (exact for <= 8 hyp speakers)
    ov = {}
    for i in range(len(pts)-1):
        a, b = pts[i], pts[i+1]
        if b <= a or not in_scored(a, b): continue
        r, h = at(ref, (a+b)/2), at(hyp, (a+b)/2)
        if r is not None and h is not None: ov[(r, h)] = ov.get((r, h), 0) + (b-a)
    best_map, best_tot = {}, -1
    slots = list(R) + [None]*max(0, len(H)-len(R))
    if len(H) <= 8 and len(slots) <= 8:
        for perm in set(itertools.permutations(slots)):
            m = {H[i]: perm[i] for i in range(len(H)) if perm[i] is not None}
            tot = sum(ov.get((m[h], h), 0) for h in m)
            if tot > best_tot: best_tot, best_map = tot, m
    else:
        for (r, h), v in sorted(ov.items(), key=lambda kv: -kv[1]):
            if h in best_map or r in best_map.values(): continue
            best_map[h] = r

    miss = fa = conf = tot = 0.0
    for i in range(len(pts)-1):
        a, b = pts[i], pts[i+1]
        if b <= a or not in_scored(a, b): continue
        d = b-a; mid = (a+b)/2
        r, h = at(ref, mid), at(hyp, mid)
        if r is not None: tot += d
        if r is not None and h is None: miss += d
        elif r is None and h is not None: fa += d
        elif r is not None and h is not None and best_map.get(h) != r: conf += d
    return dict(miss=miss, fa=fa, conf=conf, total=tot,
                der=(miss+fa+conf)/tot if tot else 0.0,
                n_ref_spk=len(R), n_hyp_spk=len(H))

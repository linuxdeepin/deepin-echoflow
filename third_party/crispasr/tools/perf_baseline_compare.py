#!/usr/bin/env python3
"""Compare two ``benchmark_asr_engines.py --json`` snapshots for perf regressions.

docs/perf-sweep/PLAN.md TODO-4 (perf-regression gate). Two gates, deliberately
asymmetric because GitHub-runner RTF is noisy (the regression manifest itself notes
"GH runner decode diverges significantly"):

  HARD (crash / fake-win detection — LEARNINGS 4a): an (engine, quant, mode, audio)
    present in the BASELINE is MISSING in CURRENT, or ran with realtime_factor <= 0
    (a wrong-backend load-failure that exits ~0.5 s mints a bogus "100x RT"), or has
    an empty transcript where the baseline had one. These are deterministic signals —
    a dropped/crashed engine is a real regression regardless of runner noise.

  SOFT (algorithmic blow-up — runner-noise-tolerant): current realtime_factor is
    below baseline / ``--factor`` (default 2.0), i.e. a coarse >=2x slowdown. Logged,
    never failed on its own, so runner jitter (measured ~+-20% on a loaded box)
    cannot red the build; only an order-of-magnitude algorithmic regression trips it.

Exit code: 0 always, UNLESS ``--strict`` is passed AND at least one HARD issue is
found (then 1). SOFT warnings never change the exit code. This lets the nightly job
run it informationally first (no ``--strict``) and tighten to ``--strict`` once a few
runs establish the noise floor.

Usage:
    perf_baseline_compare.py <baseline.json> <current.json> [--factor 2.0] [--strict]
    perf_baseline_compare.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _key(r: dict) -> tuple:
    """Stable identity of a benchmark row across snapshots."""
    return (r.get("engine", ""), r.get("quant", ""), r.get("mode", ""), r.get("audio", ""))


def _index(results: list[dict]) -> dict[tuple, dict]:
    return {_key(r): r for r in results}


def _rtf(r: dict) -> float:
    try:
        return float(r.get("realtime_factor", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def compare(baseline: dict, current: dict, factor: float = 2.0) -> tuple[list[str], list[str]]:
    """Return (hard_issues, soft_warnings) comparing two benchmark payloads.

    ``baseline`` / ``current`` are the parsed JSON objects (each has a ``results``
    list). Pure function — no I/O, no globals — so it is directly unit-testable.
    """
    hard: list[str] = []
    soft: list[str] = []
    base_idx = _index(baseline.get("results", []))
    cur_idx = _index(current.get("results", []))

    for key, b in sorted(base_idx.items()):
        label = "/".join(str(x) for x in key)
        c = cur_idx.get(key)
        if c is None:
            hard.append(f"[MISSING] {label}: in baseline, absent in current (dropped or crashed engine)")
            continue
        b_rtf, c_rtf = _rtf(b), _rtf(c)
        # A wrong-backend load-failure exits fast and reports a fake huge RTF or a
        # zero/absent one; either way an empty transcript with a non-empty baseline
        # transcript is a crash. Guard both.
        b_txt = (b.get("transcript_sample") or "").strip()
        c_txt = (c.get("transcript_sample") or "").strip()
        if c_rtf <= 0.0 and b_rtf > 0.0:
            hard.append(f"[NO-WORK] {label}: current realtime_factor={c_rtf} (baseline {b_rtf:.2f}) — likely crash/no-op")
        elif b_txt and not c_txt:
            hard.append(f"[EMPTY] {label}: current transcript empty (baseline non-empty) — likely crash")
        elif b_rtf > 0.0 and c_rtf > 0.0 and c_rtf < b_rtf / factor:
            soft.append(
                f"[SLOWER] {label}: realtime_factor {c_rtf:.2f} < {b_rtf:.2f}/{factor:g} "
                f"= {b_rtf / factor:.2f} ({b_rtf / c_rtf:.1f}x slower than baseline)"
            )

    # New rows in current with no baseline are informational, not a regression.
    for key in sorted(set(cur_idx) - set(base_idx)):
        soft.append(f"[NEW] {'/'.join(str(x) for x in key)}: present in current, no baseline entry (informational)")

    return hard, soft


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _report(hard: list[str], soft: list[str]) -> None:
    if hard:
        print("HARD issues (deterministic — a dropped/crashed engine):")
        for m in hard:
            print("  " + m)
    if soft:
        print("SOFT warnings (coarse RTF regression — informational, runner-noise-tolerant):")
        for m in soft:
            print("  " + m)
    if not hard and not soft:
        print("OK: no regressions detected.")


def _selftest() -> int:
    """Minimal in-process checks — mirrors tests/test_perf_baseline_compare.py so the
    logic can be smoke-checked without pytest. Returns 0 on success."""
    base = {
        "results": [
            {"engine": "whisper", "quant": "q8_0", "mode": "whole", "audio": "short",
             "realtime_factor": 10.0, "transcript_sample": "hello world"},
            {"engine": "parakeet", "quant": "q4_k", "mode": "whole", "audio": "long",
             "realtime_factor": 50.0, "transcript_sample": "the quick brown fox"},
        ]
    }
    # identical -> clean
    h, s = compare(base, base)
    assert not h and not s, (h, s)
    # dropped engine -> HARD
    cur = {"results": base["results"][:1]}
    h, s = compare(base, cur)
    assert any("MISSING" in m and "parakeet" in m for m in h), h
    # zero RTF (fake-win/crash) -> HARD
    cur = {"results": [dict(base["results"][0], realtime_factor=0.0)] + base["results"][1:]}
    h, s = compare(base, cur)
    assert any("NO-WORK" in m for m in h), h
    # empty transcript -> HARD
    cur = {"results": [dict(base["results"][0], transcript_sample="")] + base["results"][1:]}
    h, s = compare(base, cur)
    assert any("EMPTY" in m for m in h), h
    # 3x slower -> SOFT (not hard)
    cur = {"results": [dict(base["results"][0], realtime_factor=3.0)] + base["results"][1:]}
    h, s = compare(base, cur)
    assert not h and any("SLOWER" in m for m in s), (h, s)
    # 1.5x slower with factor=2.0 -> tolerated (no warning)
    cur = {"results": [dict(base["results"][0], realtime_factor=6.6)] + base["results"][1:]}
    h, s = compare(base, cur)
    assert not h and not any("SLOWER" in m for m in s), (h, s)
    print("selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", nargs="?", help="baseline benchmark JSON")
    ap.add_argument("current", nargs="?", help="current benchmark JSON")
    ap.add_argument("--factor", type=float, default=2.0, help="soft-warn when RTF drops below baseline/factor (default 2.0)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any HARD issue is found (default: always exit 0)")
    ap.add_argument("--selftest", action="store_true", help="run in-process logic checks and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.baseline or not args.current:
        ap.error("baseline and current JSON paths are required (or pass --selftest)")

    hard, soft = compare(_load(args.baseline), _load(args.current), factor=args.factor)
    _report(hard, soft)
    return 1 if (args.strict and hard) else 0


if __name__ == "__main__":
    raise SystemExit(main())

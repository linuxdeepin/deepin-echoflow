#!/usr/bin/env python
"""Parse the longform_results/*.json outputs into a coverage table.

Each crispasr -of with --output-json-full produces <stem>.json with a
`transcription` array of segments. We compute:
  - n_segments
  - first_ts, last_ts (seconds)
  - char_count (non-space)
  - coverage_pct (sum of segment durations / clip duration * 100)
  - max_gap_s (largest silence between consecutive segments)
"""
from __future__ import annotations
import json
import sys
import re
import csv
from pathlib import Path

RESULTS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/christianstrobele/code/issue89-stash/longform_results")

def parse_ts(s):
    """'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' → seconds"""
    s = s.replace(",", ".")
    m = re.match(r"(\d+):(\d+):(\d+\.\d+)", s)
    if not m:
        return 0.0
    h, mi, se = m.groups()
    return int(h)*3600 + int(mi)*60 + float(se)

def metrics(path: Path, clip_duration_s: float):
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception as e:
        return {"err": str(e)}
    segs = data.get("transcription", [])
    if not segs:
        return {"err": "empty_transcription"}
    n = len(segs)
    text = "".join(s.get("text","").strip() for s in segs)
    char_count = sum(1 for c in text if not c.isspace())
    # parse timestamps
    intervals = []
    for s in segs:
        ts = s.get("timestamps", {})
        f, t = ts.get("from",""), ts.get("to","")
        if f and t:
            intervals.append((parse_ts(f), parse_ts(t)))
    intervals.sort()
    if not intervals:
        return {"err": "no_timestamps", "n": n, "chars": char_count}
    first_ts = intervals[0][0]
    last_ts = intervals[-1][1]
    # coverage: merged interval length / clip duration
    merged = []
    for a, b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    covered = sum(b - a for a, b in merged)
    coverage_pct = covered / clip_duration_s * 100 if clip_duration_s else 0
    # max gap
    max_gap = 0.0
    for i in range(1, len(intervals)):
        g = intervals[i][0] - intervals[i-1][1]
        max_gap = max(max_gap, g)
    return {
        "n": n,
        "chars": char_count,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "coverage_pct": round(coverage_pct, 1),
        "max_gap_s": round(max_gap, 1),
        "head": text[:60],
    }

def main():
    summary_csv = RESULTS_DIR / "_summary.csv"
    runs = []
    if summary_csv.exists():
        with summary_csv.open() as f:
            rows = list(csv.DictReader(f))
        runs = rows

    rows = []
    for json_path in sorted(RESULTS_DIR.glob("*.json")):
        if json_path.name.startswith("_"):
            continue
        stem = json_path.stem
        m = re.match(r"(.+?)_(\d+)s$", stem)
        if not m:
            continue
        label, length = m.group(1), int(m.group(2))
        wall_s = next((int(r["wall_s"]) for r in runs if r["label"] == stem), None)
        rc = next((r["rc"] for r in runs if r["label"] == stem), None)
        met = metrics(json_path, length)
        rows.append({"label": label, "length_s": length, "wall_s": wall_s, "rc": rc, **met})

    # Pretty-print
    print(f"{'label':<28}{'len':>4}  {'n':>4}  {'first':>6}  {'last':>7}  {'cov%':>5}  {'gap':>5}  {'chars':>5}  {'wall':>5}  rc  head")
    for r in sorted(rows, key=lambda r: (r["length_s"], r["label"])):
        if "err" in r:
            print(f"{r['label']:<28}{r['length_s']:>4}  ERR: {r['err']}")
            continue
        print(f"{r['label']:<28}{r['length_s']:>4}  {r['n']:>4}  {r['first_ts']:>6.1f}  {r['last_ts']:>7.1f}  {r['coverage_pct']:>5.1f}  {r['max_gap_s']:>5.1f}  {r['chars']:>5}  {r['wall_s'] if r['wall_s'] is not None else '?':>5}  {r['rc'] or '?':>2}  {r.get('head','')[:60]}")

if __name__ == "__main__":
    main()

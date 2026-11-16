#!/usr/bin/env python3
"""Harvest a Kaggle kernel's results from its LOG, not its output files.

`kaggle kernels output` pages at 500 files with no way to pass the next-page
token back in (the API returns one but accepts none). Any kernel that writes a
loose tree into /kaggle/working therefore buries everything sorting after it —
and `.ccache/` sorts before basically anything, so `results.json` and
`outputs/*.wav` become unreachable. Verified 2026-07-20: the returned token
decoded to `.../output/.ccache/6/`.

The kernel LOG has no such limit. It is a single artifact containing every
stdout/stderr line as JSON, so anything the kernel PRINTED is recoverable even
when its files are not. That makes "print your results, don't only write them"
the durable pattern for Kaggle kernels.

Usage:
    python tools/kaggle/harvest_kernel_log.py <owner/kernel-slug> [--out DIR]
"""

import argparse
import json
import re
import sys
from pathlib import Path


def fetch_log(slug: str, out: Path) -> Path | None:
    from kaggle import KaggleApi
    api = KaggleApi()
    api.authenticate()
    out.mkdir(parents=True, exist_ok=True)
    # file_pattern is a REGEX here, not a glob — "*.log" raises "nothing to
    # repeat". The log is returned on page 1 regardless of how many output
    # files a kernel wrote, which is exactly why this route survives the cap.
    for pattern in (r".*\.log", None):
        try:
            api.kernels_output(slug, path=str(out), file_pattern=pattern)
        except Exception as e:
            print(f"kernels_output(file_pattern={pattern!r}) failed: {e}", file=sys.stderr)
        if list(out.glob("*.log")):
            break
    logs = sorted(out.glob("*.log"))
    return logs[0] if logs else None


def lines_from_log(log: Path):
    """Yield plain text lines from Kaggle's JSON stream log.

    The file is a JSON array of {stream_name,time,data} records, but a running
    or truncated kernel can leave it unparseable as a whole — fall back to
    per-record regex so a partial log still yields data.
    """
    raw = log.read_text(errors="replace")
    try:
        for rec in json.loads(raw):
            for ln in str(rec.get("data", "")).splitlines():
                yield ln
        return
    except json.JSONDecodeError:
        pass
    for m in re.finditer(r'"data"\s*:\s*"((?:[^"\\]|\\.)*)"', raw):
        try:
            text = json.loads('"' + m.group(1) + '"')
        except json.JSONDecodeError:
            continue
        for ln in text.splitlines():
            yield ln


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--out", default="./kernel-harvest")
    args = ap.parse_args()
    out = Path(args.out)

    log = fetch_log(args.slug, out)
    if not log:
        print("no log retrieved", file=sys.stderr)
        return 1
    print(f"log: {log}  ({log.stat().st_size/1e3:.0f} KB)\n")

    text_lines = list(lines_from_log(log))
    (out / "kernel-stdout.txt").write_text("\n".join(text_lines))

    tests, steps, summary = [], [], None
    for ln in text_lines:
        s = ln.strip()
        m = re.match(r"^(PASS|FAIL)\s+(\S+)\s*(\{.*\})?$", s)
        if m:
            try:
                kw = json.loads(m.group(3)) if m.group(3) else {}
            except json.JSONDecodeError:
                kw = {"raw": m.group(3)}
            tests.append({"test": m.group(2), "pass": m.group(1) == "PASS", **kw})
            continue
        if s.startswith("[step"):
            steps.append(s)
        if '"passed"' in s and '"total"' in s:
            try:
                summary = json.loads(s)
            except json.JSONDecodeError:
                pass

    (out / "harvested-results.json").write_text(
        json.dumps({"slug": args.slug, "tests": tests, "summary": summary,
                    "steps": steps}, indent=2))

    for t in tests:
        print(("PASS  " if t["pass"] else "FAIL  ") + t["test"])
    n_pass = sum(1 for t in tests if t["pass"])
    print(f"\n{n_pass}/{len(tests)} assertions passed")
    if summary:
        print(json.dumps(summary, indent=2))
    print(f"\nwrote {out/'harvested-results.json'} and {out/'kernel-stdout.txt'}")
    return 0 if tests and n_pass == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())

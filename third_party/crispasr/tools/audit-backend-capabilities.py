#!/usr/bin/env python3
"""
audit-backend-capabilities.py — drift report for declared vs tracked
capability bits across the CrispASR backend matrix.

Reads three sources:
  1. The compiled binary's declared capability bits via
     `crispasr --list-backends-json` (the source of truth).
  2. The hand-maintained `Backend(... capabilities=(...))` tuples in
     `tools/test-all-backends.py` (what the regression script tests).
  3. (Optional) The README feature matrix table — left as a TODO; the
     binary JSON is the canonical source, the README is downstream.

Reports two kinds of drift:

  * Backend in binary but not in test-all-backends.py
    → recently added backend that lacks regression coverage.

  * Cap declared by binary but not tested in test-all-backends.py
    (and a test function exists for the cap)
    → the test script's per-backend tuple is under-claiming what the
    binary supports. Either widen the tuple or remove the cap from
    the backend.

Usage:
  python3 tools/audit-backend-capabilities.py [--crispasr PATH]

Default path: ./build/bin/crispasr (or build-ninja-compile/bin/crispasr,
or build-test/bin/crispasr — first one found wins).

Exit code: 0 on no drift, 1 if drift is detected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Cap-slug → test-script-cap-name mapping
# ---------------------------------------------------------------------------
#
# Slugs come from the C++ side (kCapSlugs in crispasr_backend.cpp).
# Test script names come from the Python side (test_* functions in
# test-all-backends.py + the capabilities=(...) tuples that cite them).
# The mapping is one-to-one for most caps, with a few binary slugs that
# have no corresponding python test (token-confidence is mostly an
# output property; flash-attn is a config flag; etc.).

CAP_SLUG_TO_TEST_NAME = {
    # binary slug         → test-script cap name (or None if no test)
    "timestamps-native":   None,                # captured implicitly by transcribe
    "timestamps-ctc":      "word-timestamps",   # -am post-step
    "word-timestamps":     "word-timestamps",
    "token-confidence":    None,                # output property
    "language-detect":     "lid",
    "translate":           None,                # no dedicated test yet
    "diarize":             None,                # framework post-step, ubiquitous
    "grammar":             None,                # whisper-only, no dedicated test
    "temperature":         "temperature",
    "beam-search":         "beam",
    "flash-attn":          None,                # config-only, no dedicated test
    "punctuation-toggle":  "punctuation",
    "src-tgt-language":    None,                # subset of translate
    "auto-download":       None,                # implicit in fetch_model
    "parallel-processors": None,                # whisper-only n_processors
    "vad-internal":        None,                # whisper internal-vad path
    "tts":                 "tts-roundtrip",
}

# ---------------------------------------------------------------------------
# Locate the crispasr binary
# ---------------------------------------------------------------------------

def locate_binary(override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.is_file():
            sys.exit(f"crispasr binary not found at: {override}")
        return p
    candidates = [
        Path("build/bin/crispasr"),
        Path("build-ninja-compile/bin/crispasr"),
        Path("build-test/bin/crispasr"),
        Path("build-libs/bin/crispasr"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    on_path = shutil.which("crispasr")
    if on_path:
        return Path(on_path)
    sys.exit(
        "Could not locate crispasr binary. Pass --crispasr PATH or build first.\n"
        "Tried: " + ", ".join(str(c) for c in candidates)
    )

# ---------------------------------------------------------------------------
# Read declared caps from the binary
# ---------------------------------------------------------------------------

def read_declared(binary: Path) -> dict[str, set[str]]:
    """name → set of cap slugs"""
    out = subprocess.check_output([str(binary), "--list-backends-json"])
    data = json.loads(out)
    return {b["name"]: set(b["caps"]) for b in data["backends"]}

# ---------------------------------------------------------------------------
# Parse test-all-backends.py's hand-maintained capability tuples
# ---------------------------------------------------------------------------

# Backend("name", "label", "model.gguf", ..., capabilities=("a", "b", ...))
# We only need name + capabilities tuple.
BACKEND_RE = re.compile(
    # Allow `.` in backend names so granite-4.1, qwen3-tts-1.7b, etc.
    # are captured. Earlier version used [a-z0-9_-] which silently
    # skipped any dotted name.
    r'Backend\(\s*"(?P<name>[a-z0-9_.-]+)".*?capabilities\s*=\s*\((?P<caps>[^)]*)\)',
    re.DOTALL,
)

def read_tracked(test_script: Path) -> dict[str, set[str]]:
    text = test_script.read_text()
    result: dict[str, set[str]] = {}
    for m in BACKEND_RE.finditer(text):
        name = m["name"]
        caps_raw = m["caps"]
        caps = {
            s.strip().strip('"').strip("'")
            for s in caps_raw.split(",")
            if s.strip().strip('"').strip("'")
        }
        result[name] = caps
    return result

# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crispasr", help="path to crispasr binary (default: auto-detect)")
    p.add_argument("--test-script", default="tools/test-all-backends.py",
                   help="path to test-all-backends.py (default: tools/test-all-backends.py)")
    args = p.parse_args()

    binary = locate_binary(args.crispasr)
    print(f"# binary:      {binary}")
    print(f"# test-script: {args.test_script}")
    print()

    declared = read_declared(binary)
    tracked = read_tracked(Path(args.test_script))

    drift_found = False

    # ---------------------------------------------------------------- check 1
    # Backend in binary but not in test-all-backends.py
    only_in_binary = sorted(set(declared) - set(tracked))
    if only_in_binary:
        drift_found = True
        print("## Backends in binary but not tested by test-all-backends.py")
        print()
        for name in only_in_binary:
            caps = sorted(declared[name])
            print(f"  - {name:24s}  caps: {', '.join(caps) if caps else '(none)'}")
        print()
        print("  → Add a Backend(...) entry to tools/test-all-backends.py.")
        print()

    only_in_tracked = sorted(set(tracked) - set(declared))
    if only_in_tracked:
        drift_found = True
        print("## Backends tested but not in binary (compiled out or renamed?)")
        print()
        for name in only_in_tracked:
            print(f"  - {name}")
        print()

    # ---------------------------------------------------------------- check 2
    # Per-backend cap drift: declared by binary but the test-script tuple
    # doesn't list the matching test. Only counts caps that have a test.
    print("## Per-backend cap drift (declared but not tested)")
    print()
    drift_per_backend = []
    for name in sorted(set(declared) & set(tracked)):
        bin_caps = declared[name]
        track_caps = tracked[name]
        missing_tests = []
        for slug in bin_caps:
            test_name = CAP_SLUG_TO_TEST_NAME.get(slug)
            if test_name and test_name not in track_caps:
                missing_tests.append((slug, test_name))
        if missing_tests:
            drift_per_backend.append((name, missing_tests))

    if not drift_per_backend:
        print("  (none — every declared+testable cap is in the test tuple)")
    else:
        drift_found = True
        for name, missing in drift_per_backend:
            print(f"  - {name}:")
            for slug, test_name in missing:
                print(f"      declares {slug:24s} but test tuple lacks {test_name!r}")
        print()
        print("  → Widen the capabilities=(...) tuple in tools/test-all-backends.py.")
        print()

    # ---------------------------------------------------------------- summary
    print("## Summary")
    print()
    print(f"  binary backends:        {len(declared)}")
    print(f"  tested by test script:  {len(tracked)}")
    print(f"  caps declared (total):  {sum(len(v) for v in declared.values())}")
    print(f"  caps tested (total):    {sum(len(v) for v in tracked.values())}")
    if drift_found:
        print()
        print("  → drift detected. exit code 1.")
        return 1
    print("  → no drift.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

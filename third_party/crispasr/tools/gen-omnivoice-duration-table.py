#!/usr/bin/env python
"""Generate src/core/omnivoice_duration_table.h from OmniVoice's duration.py.

OmniVoice decides how long an utterance should be from a per-character
"phonetic weight" table (``RuleDurationEstimator``). The model is a masked
*iterative* generator — it fills exactly the number of frames it is handed —
so this weight table is not cosmetic: get it wrong and the utterance is
compressed or padded, and the tail suffers first (#363).

The C++ port used to reimplement the classifier by hand, as a chain of Unicode
range checks. That drifted. Upstream dispatches on the Unicode CATEGORY first
(``P*``/``S*`` -> 0.5, ``Z*`` -> 0.2, ``M*`` -> 0.0, ``N*`` -> 3.5) and only
then consults its script ranges, so everything punctuation- or mark-like
outside the ranges the port happened to know fell through to a 1.0
"letter-ish" default. A codepoint sweep found 1110 disagreements — including
Devanagari matras and Arabic harakat, which upstream costs 0 and the port
charged as full spoken characters.

Hand-fixing ranges is how that bug was introduced in the first place, and
fixing it that way introduced two more (a ternary that changed the precedence
of a 20-term ``||`` chain; a punctuation range that swallowed ``ª``, ``µ`` and
``º``). So this script does not reimplement anything: it RUNS upstream's own
``_get_char_weight`` over the entire Unicode space and emits the result as a
run-length table. Whatever upstream does, including its quirks, is what the
table says.

That compresses to ~1400 runs, because the weights are piecewise-constant.

The source is VENDORED at ``third_party/omnivoice/duration.py`` (byte-identical
to upstream) so ``--check`` is hermetic: a gate that needs the network is a gate
that goes flaky and then gets disabled. Upstream drift is a separate,
best-effort question — ``--check-upstream`` answers it and only that step needs
a network.

Usage:
    python tools/gen-omnivoice-duration-table.py             # regenerate
    python tools/gen-omnivoice-duration-table.py --check     # hermetic; CI gates on this
    python tools/gen-omnivoice-duration-table.py --check-upstream  # advisory drift check
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED = os.path.join(REPO, "third_party", "omnivoice", "duration.py")
HEADER = os.path.join(REPO, "src", "core", "omnivoice_duration_table.h")
UPSTREAM_RAW = "https://raw.githubusercontent.com/k2-fsa/OmniVoice/master/omnivoice/utils/duration.py"

# Surrogates are not valid scalar values; chr() yields a lone surrogate that
# cannot be encoded, and no real text contains them. Skipped, not weighted.
SURROGATES = range(0xD800, 0xE000)


def load_estimator(path: str):
    spec = importlib.util.spec_from_file_location("omnivoice_duration_vendored", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RuleDurationEstimator()


def build_runs(est) -> list[tuple[int, int, float]]:
    """Run-length encode upstream's weight over the whole Unicode space."""
    runs: list[tuple[int, int, float]] = []
    prev: float | None = None
    start = 0
    for cp in range(0x110000):
        w = None if cp in SURROGATES else est._get_char_weight(chr(cp))
        if w != prev:
            if prev is not None:
                runs.append((start, cp - 1, prev))
            prev, start = w, cp
    if prev is not None:
        runs.append((start, 0x10FFFF, prev))
    return [r for r in runs if r[2] is not None]


def render(runs, src_sha: str) -> str:
    weights = sorted({w for _, _, w in runs})
    lines = [
        "// src/core/omnivoice_duration_table.h — GENERATED, do not edit by hand.",
        "//",
        "// Regenerate with:  python tools/gen-omnivoice-duration-table.py",
        "// Verify with:      python tools/gen-omnivoice-duration-table.py --check",
        "//",
        "// Every entry is produced by RUNNING OmniVoice's own",
        "// RuleDurationEstimator._get_char_weight over the whole Unicode space",
        "// (third_party/omnivoice/duration.py, vendored byte-identical to",
        f"// k2-fsa/OmniVoice; sha256 {src_sha}).",
        "//",
        "// Nothing here is reimplemented, which is the point: the previous",
        "// hand-written range chain disagreed with upstream on 1110 codepoints,",
        "// and two of the attempts to fix it by hand introduced fresh bugs.",
        "//",
        f"// {len(runs)} runs; weights used: {', '.join(f'{w:g}' for w in weights)}.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace core_omnivoice_duration {",
        "",
        "struct DurationWeightRun {",
        "    uint32_t lo;",
        "    uint32_t hi;",
        "    double weight;",
        "};",
        "",
        "// Sorted by `lo`, non-overlapping — binary-searchable.",
        "// `double`, not `float`: Python floats ARE doubles, so this reproduces",
        "// upstream bit-exactly. As float32, 0.2 becomes 0.20000000298 and an exact",
        "// comparison against upstream fails on 3358 codepoints for no real reason.",
        "// One run per line stays diffable; clang-format would repack these onto",
        "// ~344 lines and every regeneration would churn. The marker must be the",
        "// whole comment — trailing text on it is not recognised.",
        "// clang-format off",
        f"inline constexpr DurationWeightRun kUpstreamWeightRuns[] = {{",
    ]
    for lo, hi, w in runs:
        lines.append(f"    {{0x{lo:05X}, 0x{hi:05X}, {w!r}}},")
    lines += [
        "};",
        "// clang-format on",
        "",
        f"inline constexpr int kUpstreamWeightRunCount = {len(runs)};",
        "",
        "} // namespace core_omnivoice_duration",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the header is stale")
    ap.add_argument("--check-upstream", action="store_true", help="advisory: compare vendored copy to upstream")
    ap.add_argument("--source", default=VENDORED)
    args = ap.parse_args()

    if args.check_upstream:
        import urllib.request

        with open(args.source, "rb") as fh:
            local = hashlib.sha256(fh.read()).hexdigest()
        remote_bytes = urllib.request.urlopen(UPSTREAM_RAW, timeout=30).read()
        remote = hashlib.sha256(remote_bytes).hexdigest()
        if local == remote:
            print(f"vendored duration.py matches upstream ({local[:12]})")
            return 0
        print(f"::warning::vendored duration.py differs from upstream\n  local  {local}\n  remote {remote}",
              file=sys.stderr)
        return 1

    with open(args.source, "rb") as fh:
        src_sha = hashlib.sha256(fh.read()).hexdigest()
    est = load_estimator(args.source)
    runs = build_runs(est)
    text = render(runs, src_sha)

    if args.check:
        try:
            with open(HEADER, encoding="utf-8") as fh:
                current = fh.read()
        except OSError:
            print(f"::error::{HEADER} is missing — run tools/gen-omnivoice-duration-table.py", file=sys.stderr)
            return 1
        if current != text:
            print(f"::error::{os.path.relpath(HEADER, REPO)} is stale — "
                  f"run python tools/gen-omnivoice-duration-table.py", file=sys.stderr)
            return 1
        print(f"omnivoice duration table OK ({len(runs)} runs, source sha256 {src_sha[:12]})")
        return 0

    with open(HEADER, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {os.path.relpath(HEADER, REPO)} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

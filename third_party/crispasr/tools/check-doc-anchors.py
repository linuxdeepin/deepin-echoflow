#!/usr/bin/env python3
"""Every in-repo Markdown link with a #fragment must point at a real heading.

Anchors rot silently. A heading gets reworded, every link to it keeps rendering
as a link, and the only symptom is that clicking it does nothing — which nobody
notices because nobody clicks their own docs. Found in one pass over README.md
and docs/:

  * README.md          -> #text-to-speech-tts                       (x3; the
                          heading is "Text-to-Speech models")
  * README.md          -> #server-mode-persistent-model-http-api    (moved to
                          docs/server.md)
  * README.md          -> #debug-a-new-backend-against-pytorch-...  (moved to
                          docs/contributing.md)
  * docs/cli.md        -> tts.md#timing-quality-tada_num_candidates (#343)
  * docs/server.md     -> cli.md#strict-pipeline--require-aux-...   (dropped two
                          hyphens; the heading has an em-dash AND a `--flag`)

That last one is why this is a script rather than a careful reader: GitHub's
slug rules turn "— require" into two hyphens and "(`--strict-pipeline`," into
three, and no one derives that by eye.

Slug rules implemented (GitHub-compatible for the cases this repo uses):
lowercase; strip backticks and bold/italic markers; unwrap [text](url) to text;
spaces and tabs become hyphens; every other non-alphanumeric character except
hyphen and underscore is dropped; duplicate headings get -1, -2, … suffixes.

Usage:
    python tools/check-doc-anchors.py                 # README.md + docs/**.md
    python tools/check-doc-anchors.py FILE [FILE...]  # explicit set
"""

from __future__ import annotations

import collections
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fenced code blocks must not contribute headings or links: a ``` block often
# contains shell comments starting with '#', which would otherwise register as
# headings and mask a genuinely missing one.
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def strip_fences(text: str) -> list[str]:
    out, in_fence = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return out


def slug(heading: str) -> str:
    s = heading.strip().lower()
    s = s.replace("`", "")
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)  # [text](url) -> text
    s = s.replace("*", "").replace("~", "")
    return "".join(c if (c.isalnum() or c in "-_") else ("-" if c in " \t" else "") for c in s)


def anchors_of(path: str) -> set[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            lines = strip_fences(fh.read())
    except OSError:
        return set()
    seen: collections.Counter[str] = collections.Counter()
    out: set[str] = set()
    for line in lines:
        m = HEADING.match(line)
        if not m:
            continue
        base = slug(m.group(2))
        n = seen[base]
        seen[base] += 1
        out.add(base if n == 0 else f"{base}-{n}")
    return out


def main() -> int:
    files = sys.argv[1:]
    if not files:
        files = [os.path.join(REPO, "README.md")] + sorted(glob.glob(os.path.join(REPO, "docs", "**", "*.md"),
                                                                    recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("error: no Markdown files to check", file=sys.stderr)
        return 2

    cache: dict[str, set[str]] = {}
    checked = broken = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            body = "\n".join(strip_fences(fh.read()))
        for m in LINK.finditer(body):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")) or "#" not in target:
                continue
            path_part, _, frag = target.partition("#")
            if not frag:
                continue
            tgt = f if path_part == "" else os.path.normpath(os.path.join(os.path.dirname(f), path_part))
            checked += 1
            if not os.path.exists(tgt):
                print(f"MISSING FILE  {os.path.relpath(f, REPO)} -> {target}", file=sys.stderr)
                broken += 1
                continue
            if tgt not in cache:
                cache[tgt] = anchors_of(tgt)
            if frag not in cache[tgt]:
                print(f"BROKEN        {os.path.relpath(f, REPO)} -> {target}", file=sys.stderr)
                broken += 1

    if checked == 0:
        # A checker that examined nothing must not report success.
        print("error: no internal anchors found — has the link regex or the doc layout changed?", file=sys.stderr)
        return 2
    if broken:
        print(f"\n{broken} broken anchor(s) out of {checked} checked in {len(files)} file(s).", file=sys.stderr)
        return 1
    print(f"RESULT: PASS — {checked} internal anchors across {len(files)} files all resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())

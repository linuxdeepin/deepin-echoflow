#!/usr/bin/env python
"""Stdin → stdout wrapper around upstream wetext.Normalizer.

Companion to the C++ INDEXTTS_TEXT_NORMALIZER env hook in
src/indextts.cpp. The IndexTTS-1.5 backend optionally pipes its input
text through whatever shell command is in INDEXTTS_TEXT_NORMALIZER
before running the in-process CJK preprocessor (char_rep_map + CJK
char split). This script is the recommended bridge to upstream
wetext for users who need full TN (numbers→hanzi, dates, pinyin
tones, English contractions). See docs/tts.md "IndexTTS Chinese
text normalization" for the full picture.

Install once:

    pip install wetext

Then run:

    INDEXTTS_TEXT_NORMALIZER="python /path/to/tools/wetext-normalize.py" \\
        crispasr --backend indextts ...

The script auto-detects language via wetext's heuristic (any CJK
char → zh; otherwise → en). Override with --lang. Other wetext
knobs map 1:1 to flags below.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", default="auto", choices=["auto", "en", "zh", "ja"])
    p.add_argument("--operator", default="tn", choices=["tn", "itn"])
    p.add_argument("--fix-contractions", action="store_true")
    p.add_argument("--traditional-to-simple", action="store_true")
    p.add_argument("--full-to-half", action="store_true")
    p.add_argument("--remove-interjections", action="store_true")
    p.add_argument("--remove-puncts", action="store_true")
    p.add_argument("--tag-oov", action="store_true")
    p.add_argument("--enable-0-to-9", action="store_true")
    p.add_argument("--remove-erhua", action="store_true")
    args = p.parse_args()

    try:
        from wetext import Normalizer
    except ImportError:
        print(
            "wetext-normalize.py: wetext is not installed. Run: pip install wetext\n"
            "(falling through unchanged so the calling process can recover.)",
            file=sys.stderr,
        )
        sys.stdout.write(sys.stdin.read())
        return 0

    text = sys.stdin.read().rstrip("\n")
    if not text:
        return 0

    kwargs = {k: v for k, v in vars(args).items() if v is not False and v != "auto"}
    if args.lang != "auto":
        kwargs["lang"] = args.lang
    normalizer = Normalizer(**kwargs)
    sys.stdout.write(normalizer.normalize(text))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

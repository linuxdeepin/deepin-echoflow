#!/usr/bin/env python3
"""Content-coverage scorer for ASR transcripts — char-bigram recall/precision
vs a reference transcript from another (stronger) model.

Built for the issue #89 parakeet-ja long-form audit; general enough for any
"is the backend silently dropping speech?" investigation. Recall ≈ how much
of the reference's content the hypothesis contains; precision ≈ how much of
the hypothesis is supported by the reference. A backend that drops half the
audio shows high precision + low recall — WER alone doesn't separate the two
failure modes.

Both sides are normalized: timestamps/SRT indices stripped, NFKC, whitespace
and punctuation removed. Two extra normalizations matter for Japanese:

  --strip-latin    remove [A-Za-z] from BOTH sides. A JA-only model renders
                   English speech in katakana (correct!), which a latin-script
                   reference (whisper) can never credit — without this flag an
                   English brand name in the audio reads as a coverage loss.
  --reading        hiragana-reading normalization via pykakasi (pip install
                   pykakasi). Erases kanji/kana spelling variants (皆さん vs
                   みなさん, 初め vs 始め) — THE honest coverage metric for JA.

Interpretation guardrail (measured, issue #89): char-bigram agreement between
two *correct* independent systems saturates ~93-95 % raw / ~97 % with
--reading. Calibrate the ceiling by scoring a third model against the same
reference before chasing 100 % — at the ceiling the residual is hearing
variants, not missing content.

Usage:
  python tools/asr_coverage_score.py ref.txt hyp1.txt [hyp2.txt ...] \
      [--strip-latin] [--reading] [--per-line]

  --per-line  also print per-reference-line hit rates (needs [t0 --> t1]
              or SRT-style lines in the reference) — localizes WHERE
              content is lost.

Transcript format: plain text, whisper-style "[hh:mm:ss.mmm --> ...]  text"
lines, or SRT. Everything non-text is stripped.
"""

import argparse
import re
import sys
import unicodedata
from collections import Counter

PUNCT_RE = re.compile(r"[\s、。,.!?！？…・「」『』()（）:;\"'\-—–]")
TS_BRACKET_RE = re.compile(r"\[[^\]]*\]")
SRT_INDEX_RE = re.compile(r"^\d+$", re.M)
SRT_TS_RE = re.compile(r"\d\d:\d\d:\d\d[,.]\d+ --> \d\d:\d\d:\d\d[,.]\d+")


def normalize(text, strip_latin=False, reading=None):
    text = TS_BRACKET_RE.sub(" ", text)
    text = SRT_INDEX_RE.sub(" ", text)
    text = SRT_TS_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = PUNCT_RE.sub("", text)
    if strip_latin:
        text = re.sub(r"[A-Za-z]", "", text)
    if reading is not None:
        text = "".join(item["hira"] for item in reading.convert(text))
    return text


def bigrams(s):
    return Counter(s[i : i + 2] for i in range(len(s) - 1))


def score(ref, hyp):
    rb, hb = bigrams(ref), bigrams(hyp)
    rtot, htot = sum(rb.values()), sum(hb.values())
    recall = sum(min(c, hb.get(g, 0)) for g, c in rb.items()) / max(1, rtot)
    precision = sum(min(c, rb.get(g, 0)) for g, c in hb.items()) / max(1, htot)
    return recall, precision


def ref_lines(path):
    """Yield (timestamp, text) for reference lines that carry timestamps."""
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\[([\d:.]+) --> ([\d:.]+)\]\s*(.*)", line)
        if m:
            yield f"{m.group(1)}-{m.group(2)}", m.group(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="reference transcript (e.g. whisper-large-v3-turbo output)")
    ap.add_argument("hyps", nargs="+", help="hypothesis transcript(s) to score")
    ap.add_argument("--strip-latin", action="store_true", help="drop [A-Za-z] from both sides")
    ap.add_argument("--reading", action="store_true", help="hiragana-reading normalization (needs pykakasi)")
    ap.add_argument("--per-line", action="store_true", help="per-reference-line hit rates (localize losses)")
    ap.add_argument("--per-line-threshold", type=float, default=0.85, help="only print lines below this hit rate")
    args = ap.parse_args()

    reading = None
    if args.reading:
        try:
            import pykakasi
        except ImportError:
            sys.exit("--reading needs pykakasi: pip install pykakasi")
        reading = pykakasi.kakasi()

    def norm(t):
        return normalize(t, args.strip_latin, reading)

    ref = norm(open(args.ref, encoding="utf-8").read())
    print(f"ref: {args.ref}  chars={len(ref)}")
    for h in args.hyps:
        hyp = norm(open(h, encoding="utf-8").read())
        recall, precision = score(ref, hyp)
        print(f"  recall={recall:6.1%}  precision={precision:6.1%}  chars={len(hyp):5d}  {h}")
        if args.per_line:
            for ts, text in ref_lines(args.ref):
                t = norm(text)
                if len(t) < 2:
                    continue
                grams = [t[i : i + 2] for i in range(len(t) - 1)]
                hit = sum(1 for g in grams if g in hyp) / len(grams)
                if hit < args.per_line_threshold:
                    print(f"    {hit:5.0%}  {ts}  {text.strip()}")


if __name__ == "__main__":
    main()

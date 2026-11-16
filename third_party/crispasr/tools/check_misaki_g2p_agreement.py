#!/usr/bin/env python3
"""Measure CrispASR's English G2P against misaki, Kokoro's own G2P (#316).

Kokoro-82M was trained on misaki's output, so misaki IS the reference: any
phoneme string we hand the model that misaki would not have produced is a token
sequence it never saw in training. This runs both over the same running prose
and reports how often they agree.

It compiles a ~40-line dumper against `src/core/g2p_en.h` with one `c++` call —
no CMake target, no library, nothing to keep in sync — and drives misaki
through its Python package. That means it measures the SAME code path the
product uses (`g2p_en::text_to_ipa` + `core_phoneme::convert`), including
whether the misaki conventions are switched on at all, which is precisely the
question that went unanswered through 0.8.24 and 0.8.25.

    pip install misaki                     # plus its spacy model, see misaki
    python tools/check_misaki_g2p_agreement.py --corpus my-prose.txt
    python tools/check_misaki_g2p_agreement.py --corpus my-prose.txt --verbose

Sentences where misaki itself emits `❓` (a word its lexicon does not have and
no fallback was configured for) are counted separately: we produce a
pronunciation there and misaki does not, so they are not our disagreements.

READ THE CLASSIFICATION, NOT JUST THE HEADLINE. The raw agreement number is
pessimistic in a specific, measurable way: a whole line counts as disagreeing
when the two tokenisations diverge, and on ordinary prose most of those
divergences are misaki being WRONG. Its `resolve_tokens` glues the words on
either side of a `--` into one nonsense word (`service--and` -> `sˈɜɹvəsænd`),
which no TTS should say. `--classify` splits the residual so you can see which
side owns it.

`--tagger-value` answers the recurring question "should we port a
part-of-speech tagger / all of misaki?" by running misaki against ITSELF with
the tag withheld. Measured 2026-08-05 on 500 sentences of prose: the tagger
moves 5.42% of misaki's own tokens, but 90% of that is `in`, `a` and `I` —
which core/g2p_ctxwords.h already handles with no tagger at all. The genuinely
tag-dependent remainder is 0.34% (`that`, `read`, `object`, `console`). That is
the entire prize for porting spaCy's en_core_web_sm.

Exit code 0 always — this is a measurement, not a gate. The gate is
tests/test-kokoro-misaki-wiring.cpp.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DUMPER = r"""
#include "core/g2p_en.h"
#include "core/g2p_inflect.h"
#include "core/phoneme_dialect.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

static g2p_en::context C;

int main(int argc, char** argv) {
    const std::string dir = argv[1];
    int total = 0;
    for (const char* w : {"us_gold.json", "us_silver.json"})
        total += g2p_en::load_misaki_json(C.espeak_ipa, C.phrase_final, dir + "/" + w, &C.letters);
    if (total == 0) { fprintf(stderr, "no lexicon under %s\n", dir.c_str()); return 1; }
    C.espeak_ipa.loaded = true;
    C.phrase_final.loaded = !C.phrase_final.entries.empty();
    C.inflect_fallback = [](const std::string& w) -> std::string {
        core_g2p_inflect::Params p;
        p.reduced_vowel = "\xe1\xb5\xbb";
        p.flap = "T";
        return core_g2p_inflect::inflect(w, [](const std::string& stem) -> std::string {
            auto it = C.espeak_ipa.entries.find(stem);
            return it == C.espeak_ipa.entries.end() ? std::string() : it->second;
        }, p);
    };
    const bool misaki = argc > 2 && std::string(argv[2]) == "misaki";
    const g2p_en::style st = misaki ? g2p_en::misaki_style() : g2p_en::style{};
    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && line.back() == '\r') line.pop_back();
        printf("%s\n", core_phoneme::convert(g2p_en::text_to_ipa(C, line, st),
                                             core_phoneme::Dialect::Misaki).c_str());
    }
    return 0;
}
"""


def build_dumper(workdir: Path) -> Path:
    src = workdir / "g2p_dump.cpp"
    src.write_text(DUMPER, encoding="utf-8")
    out = workdir / "g2p_dump"
    cmd = ["c++", "-std=c++17", "-O1", "-I", str(ROOT / "src"), "-o", str(out), str(src)]
    subprocess.run(cmd, check=True)
    return out


def misaki_data_dir() -> Path:
    from misaki import data  # type: ignore

    return Path(data.__file__).parent


def run_misaki(lines: list[str]) -> list[str]:
    from misaki import en  # type: ignore

    g = en.G2P(trf=False, british=False, fallback=None)
    return [g(line)[0].replace("\n", " ") for line in lines]


def run_ours(binary: Path, data_dir: Path, style: str, lines: list[str]) -> list[str]:
    p = subprocess.run(
        [str(binary), str(data_dir), style],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return p.stdout.splitlines()


def report(name: str, ref: list[str], got: list[str], verbose: bool) -> None:
    clean = [(a, b) for a, b in zip(ref, got) if "❓" not in a]
    exact = sum(1 for a, b in zip(ref, got) if a.split() == b.split())
    exact_clean = sum(1 for a, b in clean if a.split() == b.split())
    total_w = matched_w = 0
    scorable_w = scorable_match = 0
    diffs: Counter = Counter()
    for a, b in zip(ref, got):
        A, B = a.split(), b.split()
        total_w += max(len(A), len(B))
        matched_w += sum(1 for x, y in zip(A, B) if x == y)
        for x, y in zip(A, B):
            # A ❓ is a word misaki has no pronunciation for and we do — not our
            # disagreement, and in a novel full of proper names it dominates.
            if "❓" in x:
                continue
            scorable_w += 1
            scorable_match += x == y
        if len(A) == len(B):
            for x, y in zip(A, B):
                if x != y and "❓" not in x:
                    diffs[(x, y)] += 1
    print(
        f"{name:8s}  exact sentences {exact:4d}/{len(ref)} ({100 * exact / max(1, len(ref)):5.1f}%)"
        f"   token agreement {100 * matched_w / max(1, total_w):5.2f}%"
    )
    print(
        f"{'':8s}  ...over the {len(clean)} sentences misaki phonemizes without a "
        f"❓: {exact_clean} ({100 * exact_clean / max(1, len(clean)):5.1f}%)"
    )
    print(
        f"{'':8s}  token agreement over the {scorable_w} tokens misaki DOES "
        f"phonemize: {100 * scorable_match / max(1, scorable_w):5.2f}%"
    )
    if verbose and diffs:
        print(f"{'':8s}  top disagreements (misaki -> ours):")
        for (x, y), n in diffs.most_common(20):
            print(f"{'':10s}  {n:4d}  {x}  ->  {y}")


STRESS_MARKS = "\u02c8\u02cc"
PUNCT_CHARS = set(',.;:!?\u2026\u2014\u201c\u201d\u00ab\u00bb"()')


def classify(lines: list[str], ref: list[str], got: list[str]) -> None:
    """Split the residual by cause. The point is which SIDE owns each bucket."""
    buckets: Counter = Counter()
    total = 0
    misaligned_dash = misaligned_unk = misaligned_other = 0
    for src, a, b in zip(lines, ref, got):
        A, B = a.split(), b.split()
        if len(A) != len(B):
            n = max(len(A), len(B))
            buckets["tokenisation (whole line misaligned)"] += n
            total += n
            if "\u2753" in a:
                misaligned_unk += 1
            elif "--" in src:
                misaligned_dash += 1
            else:
                misaligned_other += 1
            continue
        for x, y in zip(A, B):
            if "\u2753" in x:
                continue
            total += 1
            if x == y:
                continue
            sx = "".join(c for c in x if c not in STRESS_MARKS)
            sy = "".join(c for c in y if c not in STRESS_MARKS)
            px = "".join(c for c in sx if c not in PUNCT_CHARS)
            py = "".join(c for c in sy if c not in PUNCT_CHARS)
            if px == py and sx != sy:
                buckets["punctuation attachment"] += 1
            elif px == py:
                buckets["stress only"] += 1
            else:
                buckets["segments differ"] += 1
    bad = sum(buckets.values())
    print(f"\nresidual, classified ({total} scorable tokens, {bad} disagreeing):")
    for k, v in buckets.most_common():
        print(f"  {v:5d}  ({100 * v / max(1, total):5.2f}%)  {k}")
    print(
        f"\n  of the misaligned LINES: {misaligned_dash} are the `--` convention (misaki glues the\n"
        f"  words either side into one nonsense token — OUR output is the correct one),\n"
        f"  {misaligned_unk} are misaki's own \u2753, {misaligned_other} are anything else.\n"
        f"  Only that last figure is a tokenizer bug worth chasing."
    )


def tagger_value(lines: list[str]) -> None:
    """What would a part-of-speech tagger buy? Ask misaki, with its tag withheld."""
    from misaki import en  # type: ignore

    tagged = [en.G2P(trf=False, british=False, fallback=None)(l)[0] for l in lines]
    orig = en.Lexicon.get_word
    # "" is a tag no rule matches — the closest thing to shipping no tagger.
    en.Lexicon.get_word = lambda self, word, tag, stress, ctx: orig(self, word, "", stress, ctx)
    try:
        untagged = [en.G2P(trf=False, british=False, fallback=None)(l)[0] for l in lines]
    finally:
        en.Lexicon.get_word = orig

    total = changed = 0
    which: Counter = Counter()
    for src, a, b in zip(lines, tagged, untagged):
        A, B = a.split(), b.split()
        if len(A) != len(B):
            continue
        for w, x, y in zip(src.split(), A, B):
            if "\u2753" in x:
                continue
            total += 1
            if x != y:
                changed += 1
                which[w.strip('\u201c\u201d",.;:!?').lower()] += 1
    print("\nmisaki WITH its spaCy tagger vs the SAME misaki with the tag withheld:")
    print(f"  {total} tokens, {changed} changed by the tagger ({100 * changed / max(1, total):.2f}%)")
    # The three misaki needs a tagger for and we do not: core/g2p_ctxwords.h
    # special-cases them off capitalisation and the following vowel instead.
    free = sum(n for w, n in which.items() if w in ("a", "an", "i", "in", "the", "to", "am"))
    print(f"  ...of which {free} ({100 * free / max(1, changed):.0f}%) are a/an/i/in/the/to/am,")
    print("     which core/g2p_ctxwords.h already gets right WITHOUT a tagger.")
    print(f"  Genuinely tag-dependent remainder: {changed - free} tokens "
          f"({100 * (changed - free) / max(1, total):.2f}% of the corpus).")
    for w, n in which.most_common(10):
        print(f"      {n:4d}  {w}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path, help="one sentence per line")
    ap.add_argument("--lexicon-dir", type=Path, help="misaki/data (default: the installed package)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N lines")
    ap.add_argument("--verbose", action="store_true", help="list the top disagreements")
    ap.add_argument("--classify", action="store_true", help="split the residual by cause, and by who owns it")
    ap.add_argument(
        "--tagger-value",
        action="store_true",
        help="run misaki against itself with the POS tag withheld — what a tagger would buy",
    )
    args = ap.parse_args()

    lines = [l.strip() for l in args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        lines = lines[: args.limit]
    if not lines:
        print("empty corpus", file=sys.stderr)
        return 1

    data_dir = args.lexicon_dir or misaki_data_dir()
    with tempfile.TemporaryDirectory(prefix="crispasr-g2p-") as td:
        binary = build_dumper(Path(td))
        ref = run_misaki(lines)
        print(f"{len(lines)} sentences, misaki lexicon at {data_dir}\n")
        ours = run_ours(binary, data_dir, "misaki", lines)
        report("misaki", ref, ours, args.verbose)
        if args.classify:
            classify(lines, ref, ours)
        print()
        report("piper", ref, run_ours(binary, data_dir, "piper", lines), args.verbose)
        print(
            "\n`piper` is the same dictionary read with the ESPEAK consumer's conventions —"
            "\nit is the control, not a target. It should score far lower; when it does not,"
            "\nthe misaki conventions have stopped being applied (which is #316 round 2)."
        )
    if args.tagger_value:
        tagger_value(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())

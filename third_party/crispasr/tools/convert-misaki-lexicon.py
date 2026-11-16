#!/usr/bin/env python3
"""Convert misaki's English lexicon into CrispASR's IPA-dict format (#316).

Kokoro was trained on the output of misaki, its own G2P, so the closest thing to
"correct" Kokoro pronunciation is misaki's lexicon itself. Our CMUdict-based G2P
agrees with misaki on only ~58% of words — not because the conversion is wrong
but because CMUdict makes different stress and unstressed-vowel choices. Shipping
misaki's own entries takes that to ~91% (measured; see docs/tts.md).

  misaki    Apache-2.0, https://github.com/hexgrad/misaki
            data/us_gold.json    ~90 k entries, hand-checked
            data/us_silver.json  ~93 k entries, lower confidence

misaki is Apache-2.0, but DO NOT ASSUME THAT SETTLES THE DATA. Traced
2026-07-28 by comparing the lexicons against espeak-ng 1.52 `en-us` output,
alphabet-normalised, on random samples of 120 words each:

    us_silver.json   87% byte-identical to espeak-ng output
    us_gold.json     48%   (i.e. roughly half hand-corrected away from it)

and the word sets are largely disjoint from CMUdict (only 39% of CMUdict
appears; 72% of misaki does not), so it is not a CMUdict derivative. The
gold/silver naming carries its usual meaning: silver is machine-generated,
gold is the human-verified subset.

**espeak-ng is GPL-3.0, and that includes its pronunciation dictionary.** A
lexicon 87% identical to its output is at least arguably derived from that GPL
data, which upstream may not have had the right to relicense as Apache-2.0.
Nothing here depends on resolving that, because CrispASR does NOT redistribute
this file: you generate it locally from your own `pip install misaki`, and the
runtime falls back to the CMUdict path when it is absent. Publishing it — to
cstr/g2p-dicts or anywhere — is a deliberate decision that needs upstream
clarification first, not a default. Engineering judgement, not legal advice.

Note the practical consequence: since silver ≈ espeak output, a user who has
espeak-ng installed already gets equivalent coverage for those words through
CrispASR's existing espeak path. The lexicon's real value is `gold`.

Gold wins over silver on conflict. POS-dependent entries ({"DEFAULT": …,
"NOUN": …}) collapse to DEFAULT: our G2P has no part-of-speech tagger, and
DEFAULT is what misaki itself falls back to. That is a known, bounded loss —
heteronyms like "read" get one pronunciation.

Values are stored EXACTLY as misaki's lexicon holds them, including `ɾ` where
its runtime output has `T`. The runtime applies the same dialect conversion it
applies to every other G2P path (core/phoneme_dialect.h), which performs `ɾ`→`T`
— so the transform lives in one place instead of being baked in here.

Usage:
    python tools/convert-misaki-lexicon.py --out misaki-us.txt
    python tools/convert-misaki-lexicon.py --misaki-data /path/to/misaki/data
"""

import argparse
import json
import os
import sys


def find_misaki_data() -> str:
    try:
        import misaki  # noqa: F401
    except ImportError:
        sys.exit("misaki not importable — `pip install misaki` or pass --misaki-data")
    return os.path.join(os.path.dirname(sys.modules["misaki"].__file__), "data")


def load(path: str, variant: str = "DEFAULT") -> dict:
    """Read one lexicon.

    A dict-valued entry is keyed by part-of-speech PLUS the pseudo-key "None",
    which misaki selects when `ctx.future_vowel is None` — that is, when NOTHING
    FOLLOWS the word (phrase-final). That is not a POS tag and we can evaluate it
    without a tagger, so it is emitted as a separate lexicon:

        this  ->  ðɪs   mid-phrase          ðˈɪs  phrase-final
        her   ->  hɜɹ                       hˌɜɹ
        been  ->  bɪn                       bˌɪn

    The genuinely POS-keyed variants (DT/VERB/VBD/NOUN/ADJ) still collapse to
    DEFAULT — "that", "live", "read", "desert" keep one pronunciation because we
    ship no tagger.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for word, value in raw.items():
        if isinstance(value, dict):
            # For a variant request, ONLY entries that actually carry that key.
            value = value.get(variant) if variant in value else (value.get("DEFAULT") if variant == "DEFAULT" else None)
        elif variant != "DEFAULT":
            value = None  # a plain string entry has no variants
        if isinstance(value, str) and value:
            out[word] = value
    return out


# Words whose DEFAULT reading loses to a POS-keyed variant in real text.
#
# misaki picks among these with a spaCy tag; we ship no tagger, so the collapse
# must choose ONE. Blindly taking DEFAULT is wrong for a handful of words, and
# which ones is a measurable question, not a matter of taste: each entry below
# won >=75% of its occurrences over 2500 sentences of running prose (n>=3).
#
# Words where DEFAULT already wins were deliberately LEFT ALONE even though they
# are frequent error sources — "that" wants ðˈæt 31% of the time but ðæt 68%, so
# flipping it would lose 2 tokens for every 1 gained. Those need a real tagger.
POS_OVERRIDES = {
    "live": "VERB",       # the verb (lˈɪv) dominates; DEFAULT is the adjective
    "contents": "NOUN",   # the noun (kˈɑntɛnts) dominates
    "am": "None",         # ɐm — the reduced form in running text
    "thee": "None",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misaki-data", default=None, help="misaki's data/ directory")
    ap.add_argument("--out", default="misaki-us.txt")
    ap.add_argument("--gb", action="store_true", help="use the British lexicon instead")
    ap.add_argument(
        "--gold-only",
        action="store_true",
        help="emit only the hand-verified gold lexicon. silver is 87%% identical to "
        "espeak-ng output (GPL-3.0 data); gold is 48%%. Smaller and slower to miss, "
        "but the least espeak-derived subset — see the provenance note above.",
    )
    args = ap.parse_args()

    data = args.misaki_data or find_misaki_data()
    prefix = "gb" if args.gb else "us"
    silver = load(os.path.join(data, f"{prefix}_silver.json"))
    gold = load(os.path.join(data, f"{prefix}_gold.json"))
    with open(os.path.join(data, f"{prefix}_gold.json"), encoding="utf-8") as fh:
        gold_raw = json.load(fh)
    with open(os.path.join(data, f"{prefix}_silver.json"), encoding="utf-8") as fh:
        silver_raw = json.load(fh)
    merged = dict(gold) if args.gold_only else {**silver, **gold}  # gold wins
    # Phrase-final variants ("None" key) — a separate, much smaller lexicon.
    final = {**load(os.path.join(data, f"{prefix}_silver.json"), "None"),
             **load(os.path.join(data, f"{prefix}_gold.json"), "None")}

    # The loader lowercases keys and keeps the FIRST entry per word, so emit a
    # deterministic order and let case-variants collapse predictably.
    # Apply the measured POS overrides before the case-collapse.
    for w, key in POS_OVERRIDES.items():
        for src in (gold_raw, silver_raw):
            v = src.get(w) or src.get(w.capitalize())
            if isinstance(v, dict) and key in v and isinstance(v[key], str):
                merged[w] = v[key]
                break

    rows = {}
    for word, ipa in merged.items():
        key = word.lower()
        # A gold entry must not be shadowed by a silver case-variant.
        if key in rows and word not in gold:
            continue
        rows[key] = ipa

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# CrispASR IPA dict — misaki English lexicon\n")
        f.write("# Source: https://github.com/hexgrad/misaki  (Apache-2.0)\n")
        f.write(f"#   data/{prefix}_gold.json + data/{prefix}_silver.json, gold wins.\n")
        f.write("# Values are misaki's own; the runtime applies the dialect\n")
        f.write("# conversion in src/core/phoneme_dialect.h (notably the flap).\n")
        f.write("# Generated by tools/convert-misaki-lexicon.py — do not hand-edit.\n")
        for key in sorted(rows):
            f.write(f"{key}\t{rows[key]}\n")

    final_path = args.out.replace(".txt", "") + "-final.txt"
    with open(final_path, "w", encoding="utf-8") as f:
        f.write("# CrispASR IPA dict — misaki PHRASE-FINAL variants\n")
        f.write("# Source: https://github.com/hexgrad/misaki  (Apache-2.0)\n")
        f.write("# Selected when nothing follows the word (misaki's 'None' key,\n")
        f.write("# i.e. ctx.future_vowel is None). Generated — do not hand-edit.\n")
        for k in sorted(final):
            f.write(f"{k.lower()}\t{final[k]}\n")
    print(f"wrote {final_path}: {len(final)} phrase-final entries")
    print(f"wrote {args.out}: {len(rows)} entries")
    print(
        "NOTE: this file is a DERIVATIVE of misaki's lexicon (Apache-2.0): gold and\n"
        "      silver are merged, POS-dependent entries collapsed to DEFAULT, keys\n"
        "      lowercased. Apache-2.0 s.4 requires stating those changes and keeping\n"
        "      attribution if you redistribute it — the header above does both. See\n"
        "      THIRD_PARTY_NOTICES.txt before publishing it anywhere.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

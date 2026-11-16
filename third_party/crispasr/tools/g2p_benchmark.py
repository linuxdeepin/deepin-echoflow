#!/usr/bin/env python3
"""
G2P Benchmark: compare built-in G2P output against espeak-ng ground truth.

Two modes:
  1. generate: Run espeak-ng on a word list, save ground truth TSV
  2. compare:  Load ground truth, run built-in G2P, report differences

Usage:
  # Generate ground truth (requires espeak-ng installed):
  python3 tools/g2p_benchmark.py generate \
      --words tools/g2p_benchmark_words.txt \
      --lang en-us \
      --output tools/g2p_ground_truth_en.tsv

  # Compare (no espeak needed — uses pre-generated ground truth):
  python3 tools/g2p_benchmark.py compare \
      --truth tools/g2p_ground_truth_en.tsv \
      --binary build/bin/test-g2p-en \
      --output /tmp/g2p_failures.tsv

  # Or compare using the C++ binary directly:
  python3 tools/g2p_benchmark.py compare-binary \
      --truth tools/g2p_ground_truth_en.tsv \
      --binary build/bin/crispasr \
      --model /mnt/storage/piper/piper-en_US-lessac-medium-f16.gguf
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# Top-1000 English words + common TTS test phrases
DEFAULT_WORDS = [
    # Function words
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "not", "no", "nor", "but", "or", "and", "if", "then",
    # Common content words
    "hello", "world", "audio", "voice", "speech", "language", "computer",
    "artificial", "intelligence", "generated", "system", "model", "text",
    "phone", "phoneme", "dictionary", "pronounce", "pronunciation",
    "english", "german", "french", "spanish", "natural", "neural",
    "network", "machine", "learning", "deep", "data", "digital",
    # Words that commonly fail in G2P
    "though", "through", "thought", "thorough", "cough", "enough",
    "bough", "dough", "rough", "tough", "ought", "bought", "brought",
    "women", "colonel", "queue", "bureau", "recipe", "debris",
    "cafe", "naive", "resume", "unique", "technique", "boutique",
    "choir", "chaos", "character", "psychology", "pneumonia",
    # Numbers and abbreviations
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand", "million",
    # Technical/domain words
    "algorithm", "parameter", "architecture", "transformer", "attention",
    "encoder", "decoder", "embedding", "tokenizer", "inference",
    "quantization", "optimization", "frequency", "amplitude", "waveform",
    # Compound/complex words
    "understand", "information", "communication", "international",
    "responsibility", "environmental", "representative", "approximately",
    "unfortunately", "extraordinary", "congratulations",
]


def generate_ground_truth(words, lang, output_path):
    """Generate espeak-ng IPA for each word. Requires espeak-ng installed."""
    try:
        subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: espeak-ng not installed. Install it first:", file=sys.stderr)
        print("  sudo apt install espeak-ng", file=sys.stderr)
        sys.exit(1)

    results = []
    for word in words:
        r = subprocess.run(
            ["espeak-ng", "-q", "--ipa=3", "-v", lang, word],
            capture_output=True, text=True
        )
        ipa = r.stdout.strip()
        if ipa:
            results.append((word, ipa))
            print(f"  {word:30s} → {ipa}")
        else:
            print(f"  {word:30s} → (empty)", file=sys.stderr)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["word", "espeak_ipa"])
        for word, ipa in results:
            writer.writerow([word, ipa])

    print(f"\nSaved {len(results)} entries to {output_path}")


def load_ground_truth(path):
    """Load TSV ground truth file."""
    entries = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entries[row["word"]] = row["espeak_ipa"]
    return entries


def compare_cmudict(truth, cmudict_path):
    """Compare ground truth against CMUdict → ARPAbet → IPA conversion."""
    # Load CMUdict
    cmudict = {}
    with open(cmudict_path) as f:
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                word = parts[0].split("(")[0].upper()
                if word not in cmudict:
                    cmudict[word] = parts[1]

    # ARPAbet → IPA table (matching our C++ code with the vowel quality fixes)
    arpa_ipa = {
        "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː",
        "AW": "aʊ", "AX": "ə", "AY": "aɪ", "EH": "ɛ",
        "ER": "ɜː", "EY": "eɪ", "IH": "ɪ", "IX": "ɨ",
        "IY": "iː", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ",
        "UW": "uː",
        "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
        "DX": "ɾ", "F": "f", "G": "ɡ", "HH": "h",
        "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n",
        "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
        "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j",
        "Z": "z", "ZH": "ʒ",
    }

    def arpa_to_ipa(arpa_str):
        ipa = ""
        for ph in arpa_str.split():
            base = ph.rstrip("012")
            stress = int(ph[-1]) if ph[-1] in "012" else -1
            if stress == 1:
                ipa += "ˈ"
            # AH0 → ə, IY0 → i, ER → ɜː (no ɹ)
            if base == "AH" and stress == 0:
                ipa += "ə"; continue
            if base == "IY" and stress == 0:
                ipa += "i"; continue
            if base == "ER":
                ipa += "ɜː"; continue
            ipa += arpa_ipa.get(base, "?")
        return ipa

    results = []
    match = 0
    mismatch = 0
    missing = 0

    for word, espeak_ipa in truth.items():
        upper = word.upper()
        if upper in cmudict:
            our_ipa = arpa_to_ipa(cmudict[upper])
            if our_ipa == espeak_ipa:
                match += 1
                results.append((word, espeak_ipa, our_ipa, "MATCH"))
            else:
                mismatch += 1
                results.append((word, espeak_ipa, our_ipa, "MISMATCH"))
        else:
            missing += 1
            results.append((word, espeak_ipa, "", "MISSING"))

    total = match + mismatch + missing
    print(f"\n=== CMUdict → IPA vs espeak-ng ===")
    print(f"  Total:    {total}")
    print(f"  Match:    {match} ({100*match/total:.1f}%)")
    print(f"  Mismatch: {mismatch} ({100*mismatch/total:.1f}%)")
    print(f"  Missing:  {missing} ({100*missing/total:.1f}%)")

    return results


def save_failures(results, output_path):
    """Save all non-matching entries as a TSV for analysis."""
    failures = [r for r in results if r[3] != "MATCH"]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["word", "espeak_ipa", "our_ipa", "status"])
        for word, espeak, ours, status in failures:
            writer.writerow([word, espeak, ours, status])
    print(f"\nSaved {len(failures)} failures to {output_path}")

    # Print first 20 failures
    if failures:
        print(f"\nFirst {min(20, len(failures))} failures:")
        for word, espeak, ours, status in failures[:20]:
            print(f"  {word:25s}  espeak={espeak:20s}  ours={ours:20s}  [{status}]")


def main():
    parser = argparse.ArgumentParser(description="G2P Benchmark")
    sub = parser.add_subparsers(dest="cmd")

    gen = sub.add_parser("generate", help="Generate espeak-ng ground truth")
    gen.add_argument("--words", default=None, help="Word list file (one per line)")
    gen.add_argument("--lang", default="en-us")
    gen.add_argument("--output", required=True)

    cmp = sub.add_parser("compare", help="Compare CMUdict path against ground truth")
    cmp.add_argument("--truth", required=True, help="Ground truth TSV")
    cmp.add_argument("--cmudict", default=os.path.expanduser("~/.cache/crispasr/cmudict.dict"))
    cmp.add_argument("--output", default="/tmp/g2p_failures.tsv")

    args = parser.parse_args()

    if args.cmd == "generate":
        words = DEFAULT_WORDS
        if args.words:
            with open(args.words) as f:
                words = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        generate_ground_truth(words, args.lang, args.output)

    elif args.cmd == "compare":
        truth = load_ground_truth(args.truth)
        results = compare_cmudict(truth, args.cmudict)
        save_failures(results, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

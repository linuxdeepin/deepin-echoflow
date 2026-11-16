#!/usr/bin/env python
"""Token-id parity for the voxtral-tts tokenizer against `mistral-common`.

    pip install mistral-common
    python tools/check-voxtral-tokenizer-parity.py            # ~8k strings
    python tools/check-voxtral-tokenizer-parity.py --n 50000  # longer fuzz

Needs the network (downloads the published `tekken.json`, ~15 MB, cached under
the system temp dir) and a built tree — so it is a tool, not a unit test. The
hermetic half lives in `tests/test-voxtral-pretokenize.cpp`, which pins the
same behaviour at the piece level with generated expectations.

Why both
--------
They fail on different things, and #338 is the reason we know that:

* The unit test pins the SPLIT against the published `config.pattern`. It
  caught that `\\p{N}` is a single codepoint while the runtime grouped digit
  runs — invisible in token ids with this checkpoint's vocabulary, because no
  multi-digit piece exists for BPE to produce, so a run re-splits into the same
  ids. It is still the wrong split and would surface on a vocabulary that had
  one.
* This tool pins the IDS end to end, through BPE and the vocabulary bound. It is
  what found the whitespace and multibyte-punctuation defects in the first
  place, and what proves the #338 bound holds (0 ids >= llm_vocab_size).

The harness compiles `src/voxtral_tts.cpp` so the code under test is the shipped
tokenizer. Do not reimplement it here: a second implementation drifting from the
first is exactly what #338 was.
"""

import argparse
import base64
import json
import random
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEKKEN_URL = "https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main/tekken.json"
CACHE = Path(tempfile.gettempdir()) / "crispasr-voxtral-tekken"

HARNESS_SRC = r'''
#include "voxtral_tts.cpp"
#include <cstdio>
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    if (argc < 4) return 2;
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) return 2;
    std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    voxtral_tts_vocab v;
    v.tekken_vocab_blob = blob;
    v.n_specials = atoi(argv[2]);
    v.n_vocab = atoi(argv[3]);
    v.specials.assign((size_t)v.n_specials, std::string());
    tekken_build_vocab(v, v.n_vocab, 0);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<int32_t> ids;
        for (const auto& pt : tekken_pre_tokenize(line))
            tekken_bpe_encode(v, (const uint8_t*)pt.data(), pt.size(), ids);
        for (size_t i = 0; i < ids.size(); i++) printf("%s%d", i ? " " : "", ids[i]);
        printf("\n");
    }
    return 0;
}
'''


def fetch_tekken() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "tekken.json"
    if not p.exists():
        print(f"downloading {TEKKEN_URL}")
        urllib.request.urlretrieve(TEKKEN_URL, p)
    return p


def build_blob(tekken: dict, out: Path) -> Path:
    if out.exists():
        return out
    with open(out, "wb") as f:
        for e in tekken["vocab"]:
            b = base64.b64decode(e["token_bytes"])
            f.write(struct.pack("<H", len(b)) + b)
    return out


def build_harness(build_dir: Path) -> Path:
    exe = CACHE / "tok_harness"
    src = CACHE / "tok_harness.cpp"
    src.write_text(HARNESS_SRC)
    libdirs = [build_dir / "src", build_dir / "ggml" / "src"]
    for d in libdirs:
        if not d.is_dir():
            sys.exit(f"no build tree at {d} — configure and build first")
    cmd = ["c++", "-std=c++17", "-O1", "-o", str(exe), str(src),
           f"-I{REPO/'src'}", f"-I{REPO/'include'}",
           f"-I{REPO/'ggml'/'include'}", f"-I{REPO/'ggml'/'src'}"]
    for d in libdirs:
        cmd += [f"-L{d}", f"-Wl,-rpath,{d}"]
    cmd += ["-lcrispasr", "-lggml", "-lggml-base", "-lggml-cpu"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"harness build failed:\n{r.stderr[-3000:]}")
    return exe


ATOMS = ["a", "b", "word", "Word", "WORD", "città", "naïve", "Grüße", "日本", "мир", "🎉",
         " ", "  ", "   ", "\t", " \t", "\t ", "1", "12", "345", "0",
         ".", ",", "!", "?", "[", "]", "(", ")", "{", "}", "-", "—", "–", "…",
         "’", "'", '"', "“", "”", "«", "»", "/", ":", ";", "=", "+", "%", "$",
         "€", "°", "@", "#", "&", "*", "_", "~", "^", "|", "\\"]
PROSE = [
    "L’Italia è un paese meraviglioso, con una storia lunghissima.",
    "Nell’azienda dell’amico mio, c’è sempre molto da fare — davvero.",
    "Prices: $1,234.56 (incl. 19% VAT) — see https://example.com/x?q=1",
    "Der Größenwahn des Fürsten führte zur Katastrophe.",
    "¿Cómo estás? ¡Muy bien, gracias!",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="fuzz strings (default 8000)")
    ap.add_argument("--build-dir", default=str(REPO / "build"))
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    try:
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    except ImportError:
        sys.exit("pip install mistral-common")

    tp = fetch_tekken()
    tekken = json.load(open(tp))
    cfg = tekken["config"]
    vs, ns = cfg["default_vocab_size"], cfg["default_num_special_tokens"]
    n_entries, active = len(tekken["vocab"]), vs - ns
    print(f"tekken.json: {n_entries} entries, vocab_size={vs}, specials={ns} "
          f"-> {active} active, {n_entries - active} inactive tail (#338)")

    blob = build_blob(tekken, CACHE / "tekken.blob")
    exe = build_harness(Path(args.build_dir))
    ref = Tekkenizer.from_file(str(tp))

    rng = random.Random(args.seed)
    cases = list(PROSE)
    while len(cases) < args.n:
        s = "".join(rng.choice(ATOMS) for _ in range(rng.randint(1, 9)))
        if "\n" not in s and "\r" not in s:
            cases.append(s)

    proc = subprocess.run([str(exe), str(blob), str(ns), str(vs)],
                          input="\n".join(cases) + "\n", capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"harness rc={proc.returncode}\n{proc.stderr[-2000:]}")
    got = [l.strip() for l in proc.stdout.splitlines()]
    if len(got) != len(cases):
        sys.exit(f"{len(got)} outputs for {len(cases)} inputs")

    bad, oor = [], 0
    for s, g in zip(cases, got):
        mine = [int(x) for x in g.split()] if g else []
        oor += sum(1 for i in mine if not (0 <= i < vs))
        if mine != ref.encode(s, bos=False, eos=False):
            bad.append((s, mine, ref.encode(s, bos=False, eos=False)))

    print(f"\n{len(cases)} strings: {len(bad)} mismatch, {oor} out-of-range ids")
    for s, mine, theirs in bad[:10]:
        dec = lambda ids: [ref.decode([t]) if 0 <= t < vs else "?" for t in ids]
        print(f"\nIN {s!r}\n  crispasr       {dec(mine)}\n  mistral-common {dec(theirs)}")
    if len(bad) > 10:
        print(f"\n… and {len(bad) - 10} more")
    if bad or oor:
        return 1
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

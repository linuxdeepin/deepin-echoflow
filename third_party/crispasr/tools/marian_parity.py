#!/usr/bin/env python3
"""
Parity of the Marian / Opus-MT backend against Hugging Face.

For every sentence:
  (a) the encoder ids our tokenizer produces must equal MarianTokenizer's;
  (b) our greedy translation must equal
      MarianMTModel.generate(num_beams=1, do_sample=False), decoded with
      skip_special_tokens=True.
With --beam N both sides run beam search instead (HF's own semantics).

On top of the sentences, a list of tokenizer stress strings (odd whitespace,
NFKC targets, ligatures, CJK, emoji, pieces absent from vocab.json) is
compared on ids only — that is where a normalizer that is "nearly"
SentencePiece's shows.

  cmake --build build --target marian-parity
  python tools/marian_parity.py \\
      --hf-dir /path/to/opus-mt-de-en --gguf opus-mt-de-en-f16.gguf --lang de

Exit code 0 only when every id sequence and every text is equal. For a
quantized GGUF pass --report-text: text differences are then listed, not
failed (ids must still be equal — quantization does not touch the tokenizer).

The driver prints warm per-sentence timings as well; they are echoed with the
machine's load average, because a timing without the load is not a result.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SENTENCES = {
    "de": [
        "Guten Morgen und herzlich willkommen zur heutigen Sitzung.",
        "Die Arbeitslosenquote ist im letzten Quartal um 3,5 Prozent gesunken, das sind 12 % weniger als im Vorjahr.",
        "Die Konferenz findet am 3. Oktober in Berlin statt.",
        "Können Sie mir bitte sagen, wie spät es ist?",
        "Obwohl die Regierung bereits im vergangenen Jahr angekündigt hatte, dass sie die Steuern senken werde, "
        "sobald die wirtschaftliche Lage es zulasse, ist bis heute nichts geschehen, weil sich die "
        "Koalitionspartner nicht einigen können.",
        "Die Straße ist wegen Bauarbeiten gesperrt, und die Fußgänger müssen über die Brücke gehen, "
        "was viele Bürger ärgert.",
        "Danke.",
        "Zuerst sprechen wir über die Ergebnisse des letzten Quartals.",
        "Danach wird Frau Müller den neuen Kollegen aus München vorstellen.",
        "Der Zug nach Hamburg fährt um 7 Uhr morgens von Gleis 12 ab.",
        "Ich habe gestern Abend ein sehr gutes Buch gelesen.",
        "Wir haben heute drei Punkte auf der Tagesordnung.",
        "Das Wachstum ist vor allem auf den starken Export nach Frankreich und Italien zurückzuführen.",
        "Wenn es morgen regnet, bleiben wir zu Hause und sehen uns einen Film an.",
    ],
    "en": [
        "Good morning and welcome to today's meeting.",
        "Unemployment fell by 3.5 percent last quarter, which is 12% less than a year ago.",
        "The conference will take place in Berlin on October 3rd.",
        "Could you please tell me what time it is?",
        "Although the government announced last year that it would cut taxes as soon as the economy allowed, "
        "nothing has happened so far because the coalition partners cannot agree.",
        "Thank you.",
        "The train to Hamburg leaves from platform 12 at 7 a.m.",
        "First we will talk about the results of the last quarter.",
    ],
}

# Tokenizer only. No line breaks (the driver reads one string per line) and
# none of the literal strings "</s>", "<unk>", "<pad>": Hugging Face cuts those
# out of the text as special tokens before SentencePiece sees it, which the
# runtime does not reproduce (documented in src/core/marian_tokenizer.h).
STRESS = [
    "  führende   und   doppelte    Leerzeichen  ",
    "Tab\tgetrennt\tund geschütztes Leerzeichen",
    "Auslassung… und Gedankenstrich – und Geviertstrich — fertig",
    "„Deutsche Anführungszeichen“ und »französische« und ‘einfache’",
    "½ Liter, 10 % Rabatt, 5 € und 20 °C, § 5 Abs. 2",
    "ﬁnden mit Ligatur, Ｆｕｌｌｗｉｄｔｈ und ①②③",
    "é kombinierend, é vorkomponiert, Å und Å",
    "Weiche­Trennung und Null​breite und ‍Joiner",
    "日本語のテキスト と 中文 和 한국어",
    "Emoji 😀 und 👍🏽 mitten im Satz",
    "peoples | pipe ▁ underline 婺 窪 稼 尭",
    "E-Mail: max.mustermann@example.com, https://example.com/a?b=c&d=e",
    "GROSSBUCHSTABEN und kleinbuchstaben und CamelCase",
    "1.234.567,89 und 1,234,567.89 und 3.14159",
    "a",
    "ß",
    ".",
    "?!…",
    ">>fr<< Bonjour",
    "Ein Zeilentrenner und　ideografisches Leerzeichen",
    "ı İ ſ ẞ ǆ ΐ",
    "\u0001Steuerzeichen\u007f und ﻿BOM",
    "x" * 300,
    "Donaudampfschifffahrtsgesellschaftskapitänsmützenabzeichen",
]


def run_driver(driver: str, gguf: str, lines: list[str], src: str, tgt: str, beam: int, tok_only: bool, reps: int):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    try:
        cmd = [driver, gguf, path, "--src", src, "--tgt", tgt, "--beam", str(beam), "--reps", str(reps)]
        if tok_only:
            cmd.append("--tok-only")
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=1800)
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        sys.exit(f"driver failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    rows = []
    for line in proc.stdout.split("\n"):
        if not line:
            continue
        idx, ids, ms, text = line.split("\t", 3)
        rows.append(([int(x) for x in ids.split()], float(ms), text))
    if len(rows) != len(lines):
        sys.exit(f"driver returned {len(rows)} rows for {len(lines)} lines:\n{proc.stderr[-2000:]}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", required=True, help="Hugging Face checkpoint directory (the converter's --input)")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--lang", required=True, choices=sorted(SENTENCES), help="source language of the sentence set")
    ap.add_argument("--tgt", default=None, help="target language passed to the runtime (default: the other one)")
    ap.add_argument("--driver", default="build/bin/marian-parity")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--report-text", action="store_true",
                    help="list text differences instead of failing on them (quantized GGUFs)")
    ap.add_argument("--sentences", type=Path, default=None, help="one sentence per line, instead of the built-in set")
    args = ap.parse_args()

    import torch
    from transformers import MarianMTModel, MarianTokenizer

    sentences = SENTENCES[args.lang]
    if args.sentences:
        sentences = [l for l in args.sentences.read_text(encoding="utf-8").split("\n") if l.strip()]
    tgt = args.tgt or ("en" if args.lang != "en" else "de")

    tok = MarianTokenizer.from_pretrained(args.hf_dir)
    model = MarianMTModel.from_pretrained(args.hf_dir).eval()
    torch.set_grad_enabled(False)

    ref = []
    for s in sentences:
        enc = tok(s, return_tensors="pt")
        out = model.generate(**enc, num_beams=args.beam, do_sample=False)
        ref.append((enc["input_ids"][0].tolist(), tok.decode(out[0], skip_special_tokens=True), out.shape[1] - 1))
    stress_ref = [tok(s)["input_ids"] for s in STRESS]

    load = os.getloadavg()
    ours = run_driver(args.driver, args.gguf, sentences, args.lang, tgt, args.beam, False, args.reps)
    stress = run_driver(args.driver, args.gguf, STRESS, args.lang, tgt, 1, True, 1)
    load_after = os.getloadavg()

    bad_ids = bad_text = 0
    print(f"\n{Path(args.gguf).name}  vs  {args.hf_dir}   beam={args.beam}")
    print(f"{'#':>2}  {'ids':5}  {'text':5}  {'tok':>4}  {'ms':>8}  sentence → translation")
    for i, (s, (r_ids, r_text, r_new), (o_ids, ms, o_text)) in enumerate(zip(sentences, ref, ours)):
        ids_ok, text_ok = r_ids == o_ids, r_text == o_text
        bad_ids += not ids_ok
        bad_text += not text_ok
        print(f"{i:>2}  {'equal' if ids_ok else 'DIFF':5}  {'equal' if text_ok else 'DIFF':5}  "
              f"{len(o_ids):>4}  {ms:>8.1f}  {s[:48]}{'…' if len(s) > 48 else ''} → {o_text}")
        if not ids_ok:
            print(f"      hf  ids: {r_ids}\n      our ids: {o_ids}")
        if not text_ok:
            print(f"      hf : {r_text}\n      our: {o_text}")
    stress_bad = 0
    for s, r_ids, (o_ids, _, _) in zip(STRESS, stress_ref, stress):
        if r_ids != o_ids:
            stress_bad += 1
            print(f"  stress DIFF {s[:60]!r}\n      hf  ids: {r_ids}\n      our ids: {o_ids}")

    times = sorted(ms for _, ms, _ in ours)
    n = len(sentences)
    print(f"\nsentences: {n - bad_ids}/{n} ids equal, {n - bad_text}/{n} texts equal")
    print(f"tokenizer stress strings: {len(STRESS) - stress_bad}/{len(STRESS)} ids equal")
    print(f"per sentence (warm, median of {args.reps}): min {times[0]:.1f} ms, median {times[n // 2]:.1f} ms, "
          f"max {times[-1]:.1f} ms")
    print(f"load average before/after: {load[0]:.2f} / {load_after[0]:.2f}"
          f"{'   <-- above 8: these timings are NOT a result' if max(load[0], load_after[0]) > 8 else ''}")

    if bad_ids or stress_bad:
        return 1
    if bad_text and not args.report_text:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

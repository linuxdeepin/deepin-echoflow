#!/usr/bin/env python
"""Generate a crispasr-diff reference archive for the deterministic chatterbox
`t3_text_tokens` stage (issue #170 NFKD validation).

Reproduces the upstream ChatterboxMultilingualTTS text path exactly —
`punc_norm` then `MTLTokenizer.encode` (lowercase + NFKD + language-tag prepend
+ SPACE substitution) — and writes the resulting token ids to a GGUF the
crispasr-diff `chatterbox` backend compares its own tokenization against.

Tokenizer-only: needs just grapheme_mtl_merged_expanded_v1.json, not the full
model, so it runs anywhere.

Usage:
  python tools/gen_chatterbox_text_token_ref.py \
      --tokenizer grapheme_mtl.json --lang ar \
      --text "<utterance>" --output ref_text_tokens.gguf
"""
import argparse
import unicodedata

import gguf
import numpy as np
from tokenizers import Tokenizer

SPACE = "[SPACE]"
SENTENCE_ENDERS = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
PUNC_REPLACE = [
    ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "),
    ("—", "-"), ("–", "-"), (" ,", ","),
    ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
]


def punc_norm(text: str) -> str:
    if not text:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for a, b in PUNC_REPLACE:
        text = text.replace(a, b)
    text = text.rstrip(" ")
    if text and text[-1] not in SENTENCE_ENDERS:
        text = text + "."
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lang", default="ar")
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)

    # Upstream MTLTokenizer.preprocess_text + encode for a "grapheme" language
    # (ar/de/fr/es/it/...): lowercase, NFKD, prepend [lang], SPACE-substitute.
    norm = punc_norm(args.text)
    pre = unicodedata.normalize("NFKD", norm.lower())
    enc_in = f"[{args.lang.lower()}]{pre}".replace(" ", SPACE)
    ids = tok.encode(enc_in).ids
    print(f"text -> {len(ids)} ids; first 12: {ids[:12]}")

    writer = gguf.GGUFWriter(args.output, "chatterbox.t3")
    writer.add_string("chatterbox_syn_text", args.text)
    writer.add_string("chatterbox_lang", args.lang)
    writer.add_tensor("t3_text_tokens",
                      np.asarray(ids, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

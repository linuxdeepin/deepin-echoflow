#!/usr/bin/env python3
"""Backfill `cohere_transcribe.supported_languages` into an existing GGUF.

Cohere Transcribe accepts a FIXED language set and answers a wrong language
*fluently* rather than failing. The set lives in the model's `config.json`
(14 codes for the base model, only {en, ar} for the Arabic finetune) and it
CANNOT be recovered from the GGUF itself: the tokenizer carries all 183 ISO
639-1 `<|xx|>` tokens regardless of what the model supports, so `<|de|>` is
well-formed even on a model that has never seen German.

Every GGUF published before the converter learned to write that key therefore
has no way to validate `-l`. This backfills it — tensors are passed through
untouched, so it works on any quantisation and needs no reconversion.

Usage:
  # take the list straight from the source model's config.json on HF
  python tools/gguf-add-cohere-langs.py \
      --hf-config CohereLabs/cohere-transcribe-arabic-07-2026 in.gguf out.gguf

  # or state it explicitly
  python tools/gguf-add-cohere-langs.py --languages en,ar in.gguf out.gguf

Verify afterwards with --verify, which re-reads the output and compares every
tensor's name/shape/dtype/sha256 against the input. Rewriting a published
artifact deserves that check.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gguf
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

KEY_LANGS = "cohere_transcribe.supported_languages"
KEY_MAX_CLIP = "cohere_transcribe.audio.max_clip_s"


def tensor_digest(reader: "gguf.GGUFReader") -> dict:
    """name -> (shape, dtype, sha256) for every tensor, to prove a no-op copy."""
    out = {}
    for t in reader.tensors:
        h = hashlib.sha256(t.data.tobytes()).hexdigest()
        out[str(t.name)] = (tuple(int(x) for x in t.shape), int(t.tensor_type), h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--languages",
                    help="comma-separated ISO-639-1 codes, e.g. 'en,ar'")
    ap.add_argument("--hf-config",
                    help="HF repo id to read `supported_languages` from config.json")
    ap.add_argument("--max-clip-s", type=int, default=None,
                    help="also write audio.max_clip_s (config.json max_audio_clip_s)")
    ap.add_argument("--verify", action="store_true",
                    help="re-read the output and prove every tensor is byte-identical")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the key if the input already has it")
    args = ap.parse_args()

    if args.hf_config:
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(args.hf_config, "config.json")))
        langs = [str(c).strip().lower() for c in cfg.get("supported_languages", [])]
        if args.max_clip_s is None and "max_audio_clip_s" in cfg:
            args.max_clip_s = int(cfg["max_audio_clip_s"])
        if not langs:
            sys.exit(f"{args.hf_config}/config.json has no `supported_languages`")
        print(f"{args.hf_config}: supported_languages = {langs}")
    elif args.languages:
        langs = [c.strip().lower() for c in args.languages.split(",") if c.strip()]
    else:
        sys.exit("need --languages or --hf-config")

    reader = gguf.GGUFReader(args.input)
    arch = reader.get_field("general.architecture")
    arch_str = str(bytes(arch.parts[arch.data[0]]), encoding="utf-8")
    if arch_str != "cohere-transcribe":
        sys.exit(f"{args.input} is '{arch_str}', not a cohere-transcribe GGUF")
    if reader.get_field(KEY_LANGS) is not None and not args.force:
        sys.exit(f"{args.input} already has {KEY_LANGS} (pass --force to replace)")

    before = tensor_digest(reader) if args.verify else None

    # NOTE: no `sub_type=` here. Newer gguf-py grew that field on
    # MetadataDetails; the version installed on this box (and on Kaggle) has
    # only (type, value, description), and infers the array's element type
    # from the first element — verified to produce ARRAY/STRING. Passing
    # sub_type= raises TypeError on the older API, which is how
    # tools/gguf-add-merges.py currently fails.
    new_metadata = {
        KEY_LANGS: MetadataDetails(
            gguf.GGUFValueType.ARRAY, langs,
            "languages the model was trained for",
        ),
    }
    if args.max_clip_s is not None:
        new_metadata[KEY_MAX_CLIP] = MetadataDetails(
            gguf.GGUFValueType.UINT32, int(args.max_clip_s),
            "longest training audio window, seconds",
        )

    endianess = reader.endianess.name
    writer = gguf.GGUFWriter(
        args.output, arch=arch_str,
        endianess=getattr(gguf.GGUFEndian, endianess),
    )
    copy_with_new_metadata(reader, writer, new_metadata, remove_metadata=[])
    print(f"wrote {args.output}")

    if args.verify:
        out = gguf.GGUFReader(args.output)
        got = out.get_field(KEY_LANGS)
        got_langs = [str(bytes(got.parts[i]), encoding="utf-8") for i in got.data]
        if got_langs != langs:
            sys.exit(f"VERIFY FAILED: read back {got_langs}, expected {langs}")
        after = tensor_digest(out)
        if after != before:
            missing = set(before) ^ set(after)
            changed = [k for k in before if k in after and before[k] != after[k]]
            sys.exit(f"VERIFY FAILED: tensors differ "
                     f"(name mismatch: {sorted(missing)[:5]}, changed: {changed[:5]})")
        print(f"verified: {KEY_LANGS}={got_langs}, "
              f"{len(after)} tensors byte-identical (sha256)")


if __name__ == "__main__":
    main()

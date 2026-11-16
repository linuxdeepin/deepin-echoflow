#!/usr/bin/env python3
"""Add `tokenizer.ggml.merges` to an existing GGUF (copy with new metadata).

Backfills BPE merges into GGUFs converted before the converter learned to
write them (#218 glm-asr follow-up: without merges the runtime tokenizer is
specials-only and plain-text instructions cannot be encoded). Works on any
quantisation — tensors are passed through untouched.

Usage:
  python tools/gguf-add-merges.py --tokenizer-json tokenizer.json \
      in.gguf out.gguf
  # or pull the tokenizer straight from HF:
  python tools/gguf-add-merges.py --hf-repo zai-org/GLM-ASR-Nano-2512 \
      in.gguf out.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--tokenizer-json", type=Path, help="local tokenizer.json")
    ap.add_argument("--hf-repo", help="HF repo id to fetch tokenizer.json from")
    args = ap.parse_args()

    if args.tokenizer_json:
        tok_path = args.tokenizer_json
    elif args.hf_repo:
        from huggingface_hub import hf_hub_download
        tok_path = Path(hf_hub_download(args.hf_repo, "tokenizer.json"))
    else:
        sys.exit("need --tokenizer-json or --hf-repo")

    raw = json.load(open(tok_path)).get("model", {}).get("merges", [])
    merges = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in raw]
    if not merges:
        sys.exit(f"no merges found in {tok_path}")
    print(f"{len(merges)} merges from {tok_path}")

    reader = gguf.GGUFReader(args.input)
    arch = reader.get_field("general.architecture")
    arch_str = str(bytes(arch.parts[arch.data[0]]), encoding="utf-8")
    if reader.get_field("tokenizer.ggml.merges") is not None:
        sys.exit(f"{args.input} already has tokenizer.ggml.merges")

    endianess = reader.endianess.name
    writer = gguf.GGUFWriter(
        args.output, arch=arch_str,
        endianess=getattr(gguf.GGUFEndian, endianess),
    )
    # `sub_type` only exists on newer gguf-py. On older versions MetadataDetails
    # is (type, value, description) and passing it raises TypeError — which made
    # this script unusable against the gguf currently installed here. Omitting it
    # is safe there: add_key_value infers the array's element type from the first
    # element (verified: a list[str] lands as ARRAY/STRING). Pass it when the
    # field exists so newer versions keep the explicit typing.
    kwargs = {}
    if "sub_type" in getattr(MetadataDetails, "_fields", ()):
        kwargs["sub_type"] = gguf.GGUFValueType.STRING
    new_metadata = {
        "tokenizer.ggml.merges": MetadataDetails(
            gguf.GGUFValueType.ARRAY, merges,
            description="BPE merges", **kwargs,
        ),
    }
    copy_with_new_metadata(reader, writer, new_metadata, remove_metadata=[])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

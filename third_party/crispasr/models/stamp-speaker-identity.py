#!/usr/bin/env python3
"""Stamp `crispasr.voice.speaker_identity` into an EXISTING GGUF.

Whose voice a preset voice is decides whether CrispASR prepends the EU AI Act
Art. 50(4) audible disclosure. The runtime prefers a stamp inside the file over
its built-in table of file-name patterns, because a stamped file answers for
itself and survives being renamed, re-quantised or moved.

This exists so that answer can be applied to models and voice packs that are
ALREADY PUBLISHED, without re-running a conversion. Re-converting
kartoffel-orpheus (3.5 GB, needs the original safetensors and a torch stack)
just to add one metadata key would be an absurd price for a string, and the
price is why it would not get done.

    python models/stamp-speaker-identity.py \\
        --input  kokoro-voice-df_eva.gguf \\
        --output kokoro-voice-df_eva-stamped.gguf \\
        --speaker-identity real_person \\
        --evidence "HUI-Audio-Corpus-German narrator 'eva'; HUI is built from librivox.org recordings"

Values, and what each one does:

    real_person   The voice is an identifiable individual — a named donor, or a
                  corpus speaker such as VCTK's p225 or HUI's 'bernd'. Output
                  gets the audible AI disclosure. It does NOT gate cloning
                  consent: whether that donor agreed to the model being trained
                  is a licensing matter settled upstream.
    synthetic     A designed or blended voice that is not any one person.

`unknown` is deliberately NOT writable. Absence of the key IS unknown, and
writing "unknown" would turn "nobody has established this" into a claim the
file makes about itself. If you do not know, do not stamp.

Read the provider's model card first. Guessing "synthetic" is the costly error:
it is silent, and it is the direction that removes a disclosure. Of the models
whose provenance this project has resolved so far, the large majority turned out
to be real people.

The tensor data is copied through byte-for-byte; only the KV block changes.
Verify with: `python -c "import gguf; [print(k) for k in gguf.GGUFReader(P).fields]"`
"""

import argparse
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf

# Must match crispasr_voice::speaker_identity_key() in
# examples/cli/crispasr_speaker_identity.h. A drift between writer and reader
# fails open — the stamp is simply never found — so it is spelled once here and
# checked by tests/test-compliance-wiring.cpp.
IDENTITY_KEY = "crispasr.voice.speaker_identity"
EVIDENCE_KEY = "crispasr.voice.speaker_identity_evidence"

WRITABLE = ("real_person", "synthetic")

# GGUFReader surfaces the file HEADER as pseudo-fields alongside the real KV
# pairs. Re-emitting them produces a file with duplicate keys — the reader warns
# ("Duplicate key GGUF.version at offset 76") and then exposes both the real key
# and a mangled `GGUF.version_76`. The writer emits the header itself, so these
# must never be copied. Found by round-tripping a real GGUF and reading it back,
# which is the only way this shows up: the output still loads and the stamp is
# still readable, so nothing downstream fails loudly.
HEADER_PSEUDO_FIELDS = frozenset({"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"})


def _copy_field(writer: "gguf.GGUFWriter", field, skip: set) -> None:
    """Re-emit one KV field from the source file, unchanged."""
    if field.name in skip:
        return
    ftype = field.types[0]
    if ftype == gguf.GGUFValueType.ARRAY:
        elem = field.types[1]
        if elem == gguf.GGUFValueType.STRING:
            vals = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
        else:
            vals = [field.parts[i].tolist()[0] for i in field.data]
        writer.add_array(field.name, vals)
    elif ftype == gguf.GGUFValueType.STRING:
        writer.add_string(field.name, bytes(field.parts[field.data[0]]).decode("utf-8"))
    else:
        val = field.parts[field.data[0]].tolist()[0]
        writer.add_key_value(field.name, val, ftype)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="GGUF to read (model or voice pack)")
    ap.add_argument("--output", required=True, help="GGUF to write (must differ from --input)")
    ap.add_argument("--speaker-identity", dest="identity", required=True, choices=WRITABLE,
                    help="whose voice this is. 'unknown' is not writable — absence means unknown.")
    ap.add_argument("--evidence", default="",
                    help="the provider statement this verdict rests on, recorded alongside it. "
                         "Strongly recommended: a verdict without its source is one the next "
                         "person has to re-derive, or worse, flips because it is inconvenient.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing speaker_identity stamp")
    args = ap.parse_args()

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise SystemExit("--output must differ from --input (this rewrites the file, it does not patch in place)")
    if not os.path.exists(args.input):
        raise SystemExit(f"no such file: {args.input}")

    reader = gguf.GGUFReader(args.input)

    arch_field = reader.fields.get("general.architecture")
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8") if arch_field else "unknown"

    existing = reader.fields.get(IDENTITY_KEY)
    if existing is not None and not args.force:
        cur = bytes(existing.parts[existing.data[0]]).decode("utf-8")
        raise SystemExit(
            f"'{args.input}' already declares {IDENTITY_KEY}={cur!r}.\n"
            "Pass --force to overwrite. Changing a published verdict is a research decision:\n"
            "record why in --evidence."
        )

    print(f"reading  {args.input}  (arch={arch}, {len(reader.tensors)} tensors)")

    writer = gguf.GGUFWriter(args.output, arch, use_temp_file=False)
    skip = HEADER_PSEUDO_FIELDS | {"general.architecture", IDENTITY_KEY, EVIDENCE_KEY}
    for field in reader.fields.values():
        _copy_field(writer, field, skip)

    writer.add_string(IDENTITY_KEY, args.identity)
    if args.evidence:
        writer.add_string(EVIDENCE_KEY, args.evidence)

    for t in reader.tensors:
        writer.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = os.path.getsize(args.output)
    print(f"wrote    {args.output}  ({size} bytes)")
    print(f"stamped  {IDENTITY_KEY} = {args.identity}")
    if args.evidence:
        print(f"evidence {args.evidence}")
    if args.identity == "real_person":
        print("\nOutput synthesized with this file will now carry the spoken AI disclosure.\n"
              "That is the point; it is not a regression.")


if __name__ == "__main__":
    main()

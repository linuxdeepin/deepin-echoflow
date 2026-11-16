"""Shared `--speaker-identity` flag for the model converters.

Whose voice a preset voice is decides whether CrispASR prepends the EU AI Act
Art. 50(4) audible disclosure. The runtime prefers a stamp inside the GGUF over
its built-in table of file-name patterns, because a stamped file answers for
itself and survives being renamed, re-quantised or moved.

One module rather than a copy of the flag in each converter, for the same
reason the verdicts live in one C++ table: the metadata KEY has to match
crispasr_voice::speaker_identity_key() exactly, and seven hand-written copies
are seven chances for one of them to drift. A drift fails OPEN — the stamp is
simply never found — so nothing would fail loudly.

Usage in a converter::

    from _speaker_identity_arg import add_speaker_identity_arg, stamp_speaker_identity

    add_speaker_identity_arg(ap)
    ...
    stamp_speaker_identity(writer, args)

To stamp a GGUF that is already published, use models/stamp-speaker-identity.py
instead — it rewrites the KV block without re-running the conversion.
"""

# Must match crispasr_voice::speaker_identity_key() in
# examples/cli/crispasr_speaker_identity.h, and the same constant in
# models/stamp-speaker-identity.py. Guarded by
# tests/test-compliance-wiring.cpp.
IDENTITY_KEY = "crispasr.voice.speaker_identity"
EVIDENCE_KEY = "crispasr.voice.speaker_identity_evidence"

#: Values a converter may WRITE. "unknown" is deliberately absent: absence of
#: the key IS unknown, and writing it would turn "nobody has established this"
#: into a claim the file makes about itself.
WRITABLE = ("real_person", "synthetic")

_HELP = (
    "whose voice this checkpoint's preset speakers are: real_person or "
    "synthetic. Stamped into the GGUF as crispasr.voice.speaker_identity and "
    "read back by the runtime to decide whether output needs the EU AI Act "
    "Art. 50(4) audible disclosure. Omit when the model card does not say — "
    "unknown is a question, synthetic is a claim, and guessing synthetic "
    "silently removes a disclosure."
)

_EVIDENCE_HELP = (
    "the provider statement this verdict rests on, recorded in the GGUF "
    "alongside it. Strongly recommended: a verdict without its source is one "
    "the next person has to re-derive, or worse, flips because it is "
    "inconvenient."
)


def add_speaker_identity_arg(ap):
    """Add --speaker-identity / --speaker-identity-evidence to a parser."""
    ap.add_argument(
        "--speaker-identity",
        dest="speaker_identity",
        default="",
        choices=("",) + WRITABLE,
        help=_HELP,
    )
    ap.add_argument(
        "--speaker-identity-evidence",
        dest="speaker_identity_evidence",
        default="",
        help=_EVIDENCE_HELP,
    )
    return ap


def stamp_speaker_identity(writer, args, voice_name=None):
    """Write the stamp, if the caller supplied one.

    `voice_name` namespaces the key for a single entry inside a multi-voice
    bank (crispasr.voice.<name>.speaker_identity), mirroring the per-voice
    clone stamp. Omit it for a whole-file verdict.

    Writes nothing when no value was given. That is the point: absence means
    "not established", and a converter must never assert a verdict nobody made.
    """
    value = getattr(args, "speaker_identity", "") or ""
    if not value:
        return False
    key = IDENTITY_KEY if not voice_name else f"crispasr.voice.{voice_name}.speaker_identity"
    writer.add_string(key, value)
    evidence = getattr(args, "speaker_identity_evidence", "") or ""
    if evidence and not voice_name:
        writer.add_string(EVIDENCE_KEY, evidence)
    return True

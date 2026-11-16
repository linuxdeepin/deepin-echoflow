# Reference source — attribution

This directory vendors the MIT-licensed pure-C Voxtral-TTS reference implementation
from **[mudler/voxtral-tts.c](https://github.com/mudler/voxtral-tts.c)** (validated
vs vLLM-Omni), used ONLY as the ground-truth for the `voxtral-tts` diff harness
(`../crispasr-voxtral-diff.py`). See `LICENSE` (MIT) for the upstream terms.

Local additions for the diff harness (small, non-functional instrumentation):
- `voxtral_tts.c`: `VTTS_DUMP=1` prints per-frame `|h|` + 37 codes to stderr.
- `main.c`: `VTTS_CODEC_FROM_FILE=<codes>` decodes codes-from-file (codec-only mode).

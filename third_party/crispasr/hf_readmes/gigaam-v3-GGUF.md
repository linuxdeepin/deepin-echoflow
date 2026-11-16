---
license: mit
language:
- ru
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- russian
- ggml
- gguf
- gigaam
- conformer
- rnn-t
- ctc
library_name: ggml
base_model: ai-sage/GigaAM-v3
---

# GigaAM-v3 — GGUF (ggml conversions)

GGUF conversions of [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3)
for use with the `gigaam` backend in
**[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

GigaAM-v3 is a 220 M-parameter Conformer foundation model for **Russian** ASR,
pretrained with a HuBERT-CTC objective on ~700 K hours of Russian speech. The
upstream repo ships five checkpoints as git revisions; the four ASR ones are
converted here (the `ssl` encoder has no head and produces no transcript).

| File | Size | Head | Vocabulary | Output |
|---|---|---|---|---|
| `gigaam-v3-e2e-rnnt-{f16,q8_0,q4_k}.gguf` | 452 / 249 / 154 MB | RNN-T | SentencePiece 1024 | punctuation + casing + ITN — **best WER (8.4 % avg)** |
| `gigaam-v3-e2e-ctc-{f16,q8_0,q4_k}.gguf` | 449 / 247 / 152 MB | CTC | SentencePiece 256 | punctuation + casing + ITN, faster decode |
| `gigaam-v3-rnnt-{f16,q8_0,q4_k}.gguf` | 449 / 246 / 152 MB | RNN-T | 33 Cyrillic chars | lowercase, no punctuation |
| `gigaam-v3-ctc-{f16,q8_0,q4_k}.gguf` | 449 / 246 / 151 MB | CTC | 33 Cyrillic chars | lowercase, no punctuation |

## Which one to pick

**`gigaam-v3-e2e-rnnt-q8_0.gguf`** unless you have a reason not to — it is the
lowest-WER variant, emits punctuation and casing, and its transcript is
identical to the PyTorch reference.

## Usage

```bash
crispasr --backend gigaam -m gigaam-v3-e2e-rnnt-q8_0.gguf -f audio.wav
# or let the registry fetch it:
crispasr --backend gigaam -m auto --auto-download -f audio.wav
```

Audio is 16 kHz mono. Long inputs are sliced by the CLI's VAD/chunking; the
model itself has a ~25 s practical window (full attention, O(T²)).

## Verification

Every file was checked against a per-stage PyTorch reference dumped from the
upstream `modeling_gigaam.py` (`crispasr-diff gigaam <model> <ref> <wav>`), on
GigaAM's own `example.wav`:

| variant | mel | encoder (cos) | transcript vs PyTorch |
|---|---|---|---|
| f16 (all four) | 1.000000 | **1.000000** | **byte-identical** |
| q8_0 (all four) | 1.000000 | 0.9974 – 0.9988 | **byte-identical** |
| q4_k `ctc`, `rnnt` | 1.000000 | 0.95 – 0.99 | **byte-identical** |
| q4_k `e2e_ctc` | 1.000000 | 0.982 | one spurious trailing `,` |
| q4_k `e2e_rnnt` | 1.000000 | 0.987 | content identical; 4 words lose their capital letter |

So: **q8_0 is the safe quant**; q4_k is fine for the charwise models and costs
a little casing/punctuation fidelity on the two SentencePiece ones.

In every quant the mel filterbank, Hann window, `encoder.pre.*` subsampling
convs and the decode head (`joint.*` / `decoder.*` / `head.ctc.*`) are kept at
source precision — the mel is un-normalized log-mel, so subsampling rounding
error would otherwise cascade through all 16 conformer blocks, and the head is
a blank-vs-token argmax where a flipped decision derails the greedy decode.

## Conversion

```bash
python models/convert-gigaam-to-gguf.py \
    --model ai-sage/GigaAM-v3 --revision e2e_rnnt \
    --output gigaam-v3-e2e-rnnt-f16.gguf
./build/bin/crispasr-quantize gigaam-v3-e2e-rnnt-f16.gguf \
    gigaam-v3-e2e-rnnt-q8_0.gguf q8_0
```

## License

MIT, inherited from [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3).
Please cite the upstream model when you use these weights.

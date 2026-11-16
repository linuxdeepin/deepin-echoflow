---
license: apache-2.0
language:
- ja
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- ggml
- gguf
- parakeet
- ctc
- fastconformer
- japanese
library_name: ggml
base_model: grider-transwithai/parakeet-ctc-1.1b-ja
---

# Parakeet CTC 1.1B (Japanese) — GGUF

GGUF / ggml conversions of [`grider-transwithai/parakeet-ctc-1.1b-ja`](https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja) for use with the `crispasr` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

A 1.1 B-parameter Japanese ASR model:

- **FastConformer-CTC** — a 42-layer FastConformer encoder with a CTC decoder (greedy CTC at inference; one linear head over the SentencePiece vocabulary, no RNNT/TDT predictor).
- Fine-tuned from NVIDIA's English [`nvidia/parakeet-ctc-1.1b`](https://huggingface.co/nvidia/parakeet-ctc-1.1b) on Japanese data.
- **80-mel** front-end, 16 kHz mono, 8× temporal subsampling (50 → 12.5 fps).
- Apache-2.0 licence (the NVIDIA base architecture is CC-BY-4.0).

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `parakeet-ctc-1.1b-ja-f16.gguf`  | 2.13 GB | F16 — highest fidelity, closest to the NeMo reference |
| `parakeet-ctc-1.1b-ja-q8_0.gguf` | 1.26 GB | Q8_0 — **default** download, near-F16 quality |
| `parakeet-ctc-1.1b-ja-q4_k.gguf` | 795 MB  | Q4_K — smallest; some accuracy loss, fine for quick checks |

For a CTC model the Q8_0 quant is robust (CTC is far less sensitive to
quantisation noise than the small JA TDT decoder, which can loop). Use
**Q8_0** for general transcription and **F16** when you want the closest
match to the NeMo Python pipeline.

## Quick start

```bash
# 1. Build the runtime
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target crispasr

# 2. Download the Q8_0 (default) — or swap the filename for the F16 / Q4_K
huggingface-cli download cstr/parakeet-ctc-1.1b-ja-GGUF \
    parakeet-ctc-1.1b-ja-q8_0.gguf --local-dir .

# 3. Transcribe a 16 kHz mono WAV
./build/bin/crispasr \
    -m parakeet-ctc-1.1b-ja-q8_0.gguf -f your-japanese-audio.wav -t 8
```

> **Backend:** this is a **CTC** model — let crispasr auto-detect it (as
> above, no `--backend`) or pass `--backend fastconformer-ctc` explicitly.
> Do **not** pass `--backend parakeet`: that is the RNN-T/TDT *transducer*
> runtime and it will reject a CTC model with *"required tensor
> 'decoder.embed.weight' not found"*.

crispasr can also fetch the model for you by its registry name:

```bash
./build/bin/crispasr -m parakeet-ctc-1.1b-ja \
    --auto-download -f your-japanese-audio.wav
```

## Long-form audio

For clips longer than ~15 s, prefer VAD-bounded chunking — Japanese
FastConformer models drift on long single-pass windows (the safe
single-pass window is ~12 s):

```bash
./build/bin/crispasr -m parakeet-ctc-1.1b-ja-q8_0.gguf \
    -f long-japanese-audio.wav --vad -t 8
```

## Model architecture

| Component | Details |
| --- | --- |
| Encoder     | 42-layer FastConformer, d_model 1024 |
| Subsampling | Conv2d dw_striding stack, 8× temporal (50 → 12.5 fps) |
| Decoder     | CTC — single linear head over the SentencePiece vocab, greedy decode |
| Audio       | 16 kHz mono, **80 mel bins**, n_fft=512, hop=160, win=400 |
| Parameters  | ~1.1 B |

## How this was made

1. The source `.nemo` checkpoint is the **GAL** checkpoint
   (`parakeet-ja-gal.nemo`) from
   [`grider-transwithai/parakeet-ctc-1.1b-ja`](https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja).
   The non-GAL checkpoint in that repo has corrupt F32 weights in
   encoder layers 26–28 (NaN / values > 1e38) and is **not** usable —
   the GAL checkpoint is the converted one.
2. Architecture hyperparameters are read from the checkpoint's
   `model_config.yaml` and cross-checked against the actual tensor
   shapes; the mel filterbank and Hann window are baked into the GGUF
   so the runtime reproduces NeMo's front-end exactly.
3. NeMo state-dict keys are remapped to ggml-friendly names — matmul
   tensors as F16, norms / biases / mel filterbank as F32 — and the
   F16 GGUF is quantised to Q8_0 and Q4_K.
4. The GGUF carries the `canary-ctc` architecture tag; inference runs
   through the shared FastConformer-CTC runtime (`--backend
   fastconformer-ctc`, auto-detected from the filename), **not** the
   RNN-T `parakeet` transducer backend.

## Licence

Apache-2.0, inherited from the
[`grider-transwithai/parakeet-ctc-1.1b-ja`](https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja)
fine-tune. The underlying NVIDIA NeMo FastConformer-CTC architecture
([`nvidia/parakeet-ctc-1.1b`](https://huggingface.co/nvidia/parakeet-ctc-1.1b))
is CC-BY-4.0.

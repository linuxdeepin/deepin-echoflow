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
- nemo
- rnnt
- fastconformer
- reazonspeech
- japanese
library_name: ggml
base_model: reazon-research/reazonspeech-nemo-v2
---

# ReazonSpeech NeMo v2 (Japanese) — GGUF

GGUF / ggml conversions of [`reazon-research/reazonspeech-nemo-v2`](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) for use with the `crispasr` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

A 619 M-parameter Japanese ASR model trained on the
[ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/) v2.0
corpus (~35,000 hours of Japanese audio):

- **FastConformer-RNNT** — pure RNN-Transducer decoder (no TDT duration head).
- **Local relative-position attention** (window 128 + 128, plus 1 global token), so the encoder scales to long audio without quadratic blow-up.
- **80-mel** front-end, 16 kHz mono, 3000-token SentencePiece vocabulary.
- Apache-2.0 licence.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `reazonspeech-nemo-v2-f16.gguf`  | 1.24 GB | F16 — highest fidelity, closest to the NeMo reference |
| `reazonspeech-nemo-v2-q8_0.gguf` | 738 MB  | Q8_0 — **default** download, near-F16 quality |
| `reazonspeech-nemo-v2-q4_k.gguf` | 477 MB  | Q4_K — smallest; some accuracy loss, fine for quick checks |

**Q8_0** is the recommended general-purpose quant; use **F16** when you
want the closest match to the official NeMo Python pipeline.

## Quick start

```bash
# 1. Build the runtime
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target crispasr

# 2. Download the Q8_0 (default) — or swap the filename for the F16 / Q4_K
huggingface-cli download cstr/reazonspeech-nemo-v2-GGUF \
    reazonspeech-nemo-v2-q8_0.gguf --local-dir .

# 3. Transcribe a 16 kHz mono WAV
./build/bin/crispasr --backend parakeet \
    -m reazonspeech-nemo-v2-q8_0.gguf -f your-japanese-audio.wav -t 8
```

crispasr can also fetch the model for you by its registry name:

```bash
./build/bin/crispasr --backend parakeet -m reazonspeech \
    --auto-download -f your-japanese-audio.wav
```

(Both this RNNT model and the sibling
[`cstr/parakeet-ctc-1.1b-ja-GGUF`](https://huggingface.co/cstr/parakeet-ctc-1.1b-ja-GGUF)
run through crispasr's `parakeet` backend — the runtime selects the
RNNT vs. CTC decode path from the GGUF metadata.)

## Long-form audio

The local-attention encoder handles long inputs, but as with the other
Japanese FastConformer models a single long pass can drift; for clips
longer than ~15 s prefer VAD-bounded chunking:

```bash
./build/bin/crispasr --backend parakeet -m reazonspeech-nemo-v2-q8_0.gguf \
    -f long-japanese-audio.wav --vad -t 8
```

## Model architecture

| Component | Details |
| --- | --- |
| Encoder    | FastConformer with **local relative-position attention** (window 128+128, 1 global token) |
| Decoder    | RNN-Transducer (RNNT) — LSTM predictor + joint network; **no TDT durations** |
| Vocab      | 3000 SentencePiece tokens (Japanese) |
| Audio      | 16 kHz mono, **80 mel bins**, n_fft=512, hop=160, win=400 |
| Parameters | ~619 M |

## How this was made

1. The `.nemo` checkpoint from
   [`reazon-research/reazonspeech-nemo-v2`](https://huggingface.co/reazon-research/reazonspeech-nemo-v2)
   is unpacked; architecture hyperparameters (d_model, layers, local-attn
   window, predictor/joint dims, vocab) are read from
   `model_config.yaml` and cross-checked against the tensor shapes. The
   mel filterbank and Hann window are baked into the GGUF so the runtime
   reproduces NeMo's front-end.
2. NeMo state-dict keys are remapped to ggml-friendly names — matmul
   tensors as F16, norms / biases / mel filterbank as F32 — and the
   F16 GGUF is quantised to Q8_0 and Q4_K.
3. Inference runs through `src/parakeet.{h,cpp}` in CrispASR, which
   handles the local relative-position attention and the RNNT
   predictor/joint loop.

## Licence

Apache-2.0, inherited from
[`reazon-research/reazonspeech-nemo-v2`](https://huggingface.co/reazon-research/reazonspeech-nemo-v2).
Please also see the
[ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/)
project for details on the training corpus.

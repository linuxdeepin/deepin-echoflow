---
license: openmdw-1.1
language:
  - ar
  - bg
  - cs
  - da
  - de
  - el
  - en
  - es
  - et
  - fi
  - fr
  - he
  - hi
  - hr
  - hu
  - it
  - ja
  - ko
  - lt
  - lv
  - nb
  - nl
  - nn
  - pl
  - pt
  - ro
  - ru
  - sk
  - sl
  - sv
  - th
  - tr
  - uk
  - vi
  - zh
tags:
  - asr
  - speech-recognition
  - gguf
  - streaming
  - fastconformer
  - rnnt
  - multilingual
  - crispasr
pipeline_tag: automatic-speech-recognition
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
---

# Nemotron-3.5-ASR-Streaming-0.6B GGUF

GGUF conversion of [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Model details

**Architecture:** Cache-Aware Streaming FastConformer encoder (24 layers, d=1024, 8 heads) + RNN-T decoder (2-layer LSTM, hidden=640) + joint network (640 → 13088 vocab).

**Languages:** 39 languages, selected via `prompt_kernel` MLP conditioning:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| ar-AR | Arabic | fr-CA | French (CA) | nn-NO | Norwegian (NN) |
| bg-BG | Bulgarian | fr-FR | French (FR) | pl-PL | Polish |
| cs-CZ | Czech | he-IL | Hebrew | pt-BR | Portuguese (BR) |
| da-DK | Danish | hi-IN | Hindi | pt-PT | Portuguese (PT) |
| de-DE | German | hr-HR | Croatian | ro-RO | Romanian |
| el-GR | Greek | hu-HU | Hungarian | ru-RU | Russian |
| en-GB | English (GB) | it-IT | Italian | sk-SK | Slovak |
| en-US | English (US) | ja-JP | Japanese | sl-SL | Slovenian |
| es-ES | Spanish (ES) | ko-KR | Korean | sv-SE | Swedish |
| es-US | Spanish (US) | lt-LT | Lithuanian | th-TH | Thai |
| et-EE | Estonian | lv-LV | Latvian | tr-TR | Turkish |
| fi-FI | Finnish | nb-NO | Norwegian (NB) | uk-UA | Ukrainian |
|  |  | nl-NL | Dutch | vi-VN | Vietnamese |
|  |  |  |  | zh-CN | Chinese |

**Key properties:**
- Sample rate: 16 kHz
- 128 mel filterbank features, n_fft=512, hop=160 (10ms), win=400 (25ms)
- 8× time downsampling (causal) → 80ms frame duration
- Streaming: cache-aware attention with `att_context_size=[[56,3],[56,0],[56,6],[56,13]]`
- Vocab: 13087 SentencePiece tokens + 1 blank (pure RNN-T, no TDT durations)
- License: [OpenMDW-1.1](https://github.com/linux-foundation/open-model-developer-weight-license/blob/main/LICENSE.md) (permissive)

## Files

| File | Size | Description |
|------|------|-------------|
| `nemotron-3.5-asr-streaming-0.6b-f16.gguf` | 1.3 GB | F16 weights (full precision, F32 pre-encode) |
| `nemotron-3.5-asr-streaming-0.6b-q8_0.gguf` | 686 MB | Q8_0 quantized (high precision, low footprint) |
| `nemotron-3.5-asr-streaming-0.6b-q4_k.gguf` | 458 MB | Q4_K quantized (recommended, ~2x faster) |

## Usage with CrispASR

```bash
# Download (Q4_K recommended — 458 MB, ~2x faster than F16)
huggingface-cli download cstr/nemotron-3.5-asr-streaming-GGUF \
  nemotron-3.5-asr-streaming-0.6b-q4_k.gguf --local-dir models/

# Transcribe (English, default)
crispasr --backend nemotron \
  -m models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf \
  -f audio.wav

# Transcribe in German
crispasr --backend nemotron \
  -m models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf \
  -f audio.wav --language de-DE
```

## Conversion

Converted from the original NeMo `.nemo` checkpoint using:

```bash
python models/convert-nemotron-to-gguf.py \
  --nemo nvidia/nemotron-3.5-asr-streaming-0.6b \
  --output nemotron-3.5-asr-streaming-0.6b-f16.gguf
```

Quantized variants can be produced with:

```bash
crispasr-quantize models/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
  models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf Q4_K
```

## Architecture

```
Audio (16kHz) → Mel (128 bins, 10ms hop)
  → Pre-encode (3× causal Conv2d, 8× downsample, Linear 4352→1024)
  → 24× Cache-Aware FastConformer block:
      FFN1(½) → MHA(rel_pos, cache-aware) → DWConv(k=9, causal, LN) → FFN2(½) → LN
  → Prompt kernel (MLP: concat(enc[1024], lang_onehot[128]) → 2048 → 1024)
  → RNN-T decoder:
      Prediction: Embed(13088, 640) + 2-layer LSTM(640)
      Joint: enc(1024→640) + pred(640→640) → ReLU → Linear(640→13088)
  → Greedy / beam search decode
```

## Original model

- **Paper:** [NVIDIA NeMo documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- **Source:** [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- **License:** OpenMDW-1.1

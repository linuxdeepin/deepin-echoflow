---
license: other
license_name: funasr-model-license-v1.1
license_link: https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE
language:
- zh
- yue
- en
- ja
- ko
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- ggml
- gguf
- funasr
- sensevoice
- sanm
- multilingual
- language-identification
- audio-event-detection
library_name: ggml
base_model: FunAudioLLM/SenseVoiceSmall
---

# SenseVoiceSmall — GGUF (ggml-quantised)

GGUF / ggml conversion of [`FunAudioLLM/SenseVoiceSmall`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) for use with the `sensevoice` backend in **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

SenseVoiceSmall is Alibaba's **multi-task encoder-only ASR**: one forward pass through a 70-block SANM encoder emits the full transcript **plus** the spoken language ID and audio-event tags through a single CTC head. Non-autoregressive design → **15× faster than Whisper-Large** (70 ms for 10 s of audio in upstream's measurements).

> **Note on the emotion tag.** Upstream SenseVoice is also an emotion classifier — its CTC head emits an `<|HAPPY|>` / `<|ANGRY|>` / … marker in the annotation prefix. **CrispASR does not expose it.** Inferring emotions from voice makes a system an "emotion recognition system" under EU AI Act Art. 3(39), which is prohibited in workplace and education settings (Art. 5(1)(f)) and high-risk otherwise (Annex III(1)(c)). CrispASR parses the marker only to strip it out of the transcript, then discards the value. The weights are Alibaba's and unmodified — the classifier head is still in the GGUF; it is the *runtime* that does not surface it. See [docs/eu-ai-act.md](https://github.com/CrispStrobe/CrispASR/blob/main/docs/eu-ai-act.md).

- **70-block SenseVoiceEncoderSmall** (1 entry block @ 560→512 + 49 main blocks + 20 tp blocks, all 512-dim, 4 heads, FSMN k=11 depthwise convolution branch — the same encoder body Fun-ASR-Nano-2512 ships, just here paired with a CTC head instead of an LLM decoder)
- **4 query embeddings** (language / event / emotion / textnorm) prepended to the LFR fbank features so the encoder can emit rich annotations at those positions
- **CTC head** (`ctc.ctc_lo`, 25055 SentencePiece pieces)
- **50+ languages** with native LID (no whisper-tiny pre-step needed)
- **Three quants shipped** (May 2026): F16 (448 MB), Q8_0 (240 MB), **Q4_K (129 MB — recommended default)**. All three produce byte-identical transcripts on English (JFK) and Japanese (JSUT) clips end-to-end on M1 Metal. 72 tensors stay F16 in the Q4_K/Q8_0 quants because their leading dim isn't quant-block-aligned: 70× `attn.fsmn.w` (kernel=11 depthwise convolution) and 2× `attn.qkv.w` (560-dim input from the SANM context concat); the other ~280 weight matrices quantize cleanly.

## What you get in the output

By default, stdout shows the clean transcript:

```text
And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

With `-oj` the JSON output exposes the four rich-annotation tags as
explicit fields:

```json
{
  "text":        "And so my fellow Americans...",
  "language":    "en",
  "audio_event": "Speech",
  "itn_flag":    "withitn"
}
```

`sensevoice_transcribe()` returns the transcript with the annotation
prefix stripped. It used to return the raw prefixed string —

```text
<|en|><|HAPPY|><|Speech|><|withitn|>And so my fellow Americans...
```

— which reached every binding, since the session ABI routes SenseVoice
through it. Both entry points now drop the prefix; use
`sensevoice_transcribe_structured()` when you want language / audio-event /
ITN back as `struct sensevoice_result` fields.

Tag value sets:

- Languages: `zh` / `en` / `yue` / `ja` / `ko` / `nospeech`
- Emotions: `HAPPY` / `SAD` / `ANGRY` / `NEUTRAL` / `EMO_UNKNOWN`
- Audio events: `Speech` / `Music` / `Applause` / `Laughter` / `Cry` / `BGM` (and more — the upstream set is open-ended)
- Text norm: `withitn` (Arabic digits, punctuation) or `woitn` (raw)

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `sensevoice-small-q4_k.gguf` | 129 MB | **Recommended default.** 2× faster on M1 vs F16; byte-identical transcript on tested clips. Auto-download target for `--backend sensevoice -m auto`. |
| `sensevoice-small-q8_0.gguf` | 240 MB | Larger but slightly closer to F16 numerically on borderline annotation-tag argmax cases. |
| `sensevoice-small-f16.gguf` | 448 MB | F16 reference weights. Use when you want bit-stability against the upstream PyTorch reference for diff testing. |

## Quick Start

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target crispasr-cli

./build/bin/crispasr \
    --backend sensevoice \
    -m /path/to/sensevoice-small-q4_k.gguf \
    -f samples/jfk.wav -l en

# Or auto-download (resolves to Q4_K by default):
./build/bin/crispasr --backend sensevoice -m auto -f samples/jfk.wav -l en
```

## Verification

`crispasr-diff sensevoice` is 76/76 PASS, byte-identical `generated_text`,
on Alibaba's own example `zh.mp3`; 75/76 PASS on `samples/jfk.wav` with
the single difference being the emotion-tag argmax flipping between
`<|ANGRY|>` and `<|EMO_UNKNOWN|>` (F16/op-order pushes that one slot
across a near-tied boundary; the transcript itself is byte-identical
in both runs). That slot is stripped and discarded by the runtime, so
the flip is not observable in any CrispASR output. On Apple M1 Metal the runtime hits **15-22× realtime**.

## Licence + attribution

Upstream **FunAudioLLM/SenseVoiceSmall**:

- **Code** (the `funasr` Python package): Apache-2.0.
- **Model weights**: [**FunASR Model License v1.1**](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE) (Alibaba) — commercial use OK with attribution. Confirmed on the upstream-tracking discussion in [CrispStrobe/CrispASR#99](https://github.com/CrispStrobe/CrispASR/issues/99).

These GGUF files are a quantised / repackaged distribution of the upstream weights and inherit the FunASR Model License v1.1. Please attribute Alibaba / FunAudioLLM in downstream products.

> If you use this model, please also cite the upstream FunASR work.
> See the [upstream model card](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) for the canonical citation.

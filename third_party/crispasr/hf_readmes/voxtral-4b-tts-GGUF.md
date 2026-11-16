---
license: cc-by-nc-4.0
language:
- en
- fr
- de
- es
- it
- pt
- nl
- ar
- hi
pipeline_tag: text-to-speech
tags:
- audio
- text-to-speech
- tts
- ggml
- gguf
- voxtral
- mistral
- flow-matching
- multilingual
library_name: ggml
base_model: mistralai/Voxtral-4B-TTS-2603
---

# Voxtral-4B-TTS-2603 — GGUF

GGUF / ggml conversions of [`mistralai/Voxtral-4B-TTS-2603`](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) for use with the `voxtral-tts` backend of **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** — one C++ binary, no Python.

Text-to-speech across 9 languages (en, fr, de, es, it, pt, nl, ar, hi) with 20 preset voices, 24 kHz output.

## Architecture

A three-component pipeline, all implemented as ggml compute graphs:

1. **LLM backbone** — Ministral-3B autoregressive decoder (26 layers, GQA 32/8, NORMAL/adjacent-pair RoPE θ=1e6). Conditioned on a preset voice prefix + text, it emits one hidden state per audio frame.
2. **Flow-matching acoustic transformer** — 3-layer bidirectional transformer (no positional encoding). Per frame: a semantic token (greedy argmax) plus 36 acoustic FSQ codes via an 8-step (7-interval) Euler ODE with classifier-free guidance (α=1.2).
3. **Voxtral codec decoder** — 292-d input (256-d semantic VQ + 36-d FSQ) → causal conv → 4× [2-layer ALiBi transformer + ConvTranspose1d upsampling] → 240 PCM samples/frame at 24 kHz.

## Files

| File | Precision | Size | Notes |
|------|-----------|------|-------|
| `voxtral-4b-tts-q4_k.gguf` | Q4_K | ~2.4 GB | default; fits comfortably in 8 GB RAM |
| `voxtral-4b-tts-q8_0.gguf` | Q8_0 | ~4.3 GB | higher quality |
| `voxtral-4b-tts-f16.gguf` | F16 | ~8.2 GB | reference precision |

The semantic VQ codebook (`codec.semantic_cb.weight`) and preset voice embeddings are kept at F32 in every file.

## Usage

```bash
# auto-downloads the Q4_K on first run
crispasr --backend voxtral-tts -m auto --auto-download \
    --tts "Bonjour le monde." --voice fr_female --tts-output out.wav
```

`--voice` accepts any preset (e.g. `neutral_female`, `neutral_male`, `casual_female`, `cheerful_female`, `fr_female`, `de_male`, `es_female`, `it_male`, `pt_female`, `nl_male`, `ar_male`, `hi_female`, …). Omit it for the default voice. Add `--seed N` for a reproducible acoustic sample.

## License

The GGUF conversions inherit the base model's **CC-BY-NC-4.0** license (non-commercial). See [`mistralai/Voxtral-4B-TTS-2603`](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) for the original weights and terms.

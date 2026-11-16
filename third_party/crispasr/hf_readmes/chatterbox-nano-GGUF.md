---
license: mit
language:
- en
base_model:
- ResembleAI/chatterbox-nano
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- chatterbox
- chatterbox-nano
- flow-matching
- meanflow
- hifi-gan
- gguf
- crispasr
library_name: ggml
---

# Chatterbox-Nano TTS — GGUF (ggml)

GGUF / ggml conversion of [`ResembleAI/chatterbox-nano`](https://huggingface.co/ResembleAI/chatterbox-nano) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** (v0.8.31+, issue #382).

Chatterbox-Nano is the small sibling of Chatterbox-Turbo: the same AR text-to-speech pipeline with a **GPT2-small T3 backbone (12 layers, 768 hidden, 12 heads)** instead of Turbo's GPT2-medium (24L, 1024D). Upstream's `s3gen_meanflow.safetensors`, `ve.safetensors` and `conds.pt` are **byte-identical** to `ResembleAI/chatterbox-turbo` (same LFS blobs), so this repo ships only the T3 — **use the S3Gen from [`cstr/chatterbox-turbo-GGUF`](https://huggingface.co/cstr/chatterbox-turbo-GGUF) as the companion.** CrispASR's registry does this automatically. Distributed under **MIT license**.

## Files

| File | Size | Notes |
|---|---:|---|
| `chatterbox-nano-t3-f16.gguf`  | 476 MB | T3 GPT-2 AR model (12L, 768D) + VE + baked conds |
| `chatterbox-nano-t3-q8_0.gguf` | 345 MB | Quantized T3, recommended deployment default |
| `chatterbox-nano-t3-q4_k.gguf` | 283 MB | Smaller T3 quant for memory-constrained use |

The GGUF carries `chatterbox.t3.n_kv_heads = 12` explicitly (GPT-2 is MHA; CrispASR's built-in default of 16 fits Turbo only). Token ids match upstream inference (`T3Config`: start_text 255, stop_text 0, speech 6561/6562) — the `stop_text_token: 50256` in `t3_nano_v1.yaml` is a training-time artifact that upstream `generate()` never reads.

## Usage

```bash
# auto-download T3 + Turbo S3Gen companion
crispasr --backend chatterbox-nano -m auto --auto-download \
  --tts "Hello from Chatterbox Nano." --tts-output nano.wav

# explicit files
crispasr --backend chatterbox-nano -m chatterbox-nano-t3-q8_0.gguf \
  --codec-model chatterbox-turbo-s3gen-q8_0.gguf \
  --tts "Hello from Chatterbox Nano." --tts-output nano.wav
```

Voice cloning (`--voice reference.wav`), the 19 `[emotion]` style tokens, and the sampler defaults are identical to the Turbo backend; CFG/min_p/exaggeration are not supported by the Turbo/Nano pipeline (upstream disables them).

## Conversion

Converted with [`models/convert-chatterbox-to-gguf.py --variant nano`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-chatterbox-to-gguf.py) from upstream revision `71ccd1d0`; quantized with `crispasr-quantize`.

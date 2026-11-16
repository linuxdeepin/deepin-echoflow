---
license: cc-by-4.0
language:
- en
- de
- es
- it
- pt
- fr
base_model:
- kyutai/pocket-tts
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- voice-cloning
- pocket-tts
- mimi
- kyutai
- gguf
- crispasr
library_name: ggml
---

# Pocket TTS — GGUF (ggml-quantised)

GGUF / ggml conversion of [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Pocket TTS is a lightweight (~100M param) continuous-latent autoregressive TTS model from Kyutai, based on the CALM paper (arXiv:2509.06926). Unlike codebook-based TTS models, Pocket TTS emits continuous float vectors — no discrete tokens, no softmax sampling:
- **FlowLM backbone** — causal transformer (1024D, 16 heads, 6 layers; 24 layers for the French preview, RoPE, GELU) operating at 12.5 Hz
- **Consistency head** — SimpleMLPAdaLN (512D, 6 ResBlocks) with timestep embedding → one-step LSD decode → 32-dim continuous latent vectors
- **Mimi VAE decoder** — SEANet upsample convolutions + 2-layer transformer → 24 kHz PCM
- **Mimi VAE encoder** *(voice-cloning builds only)* — SEANet downsample + 2-layer transformer + speaker projection → reference conditioning
- **Text tokenizer** — SentencePiece BPE (4000 vocab, embedded in GGUF)

Single GGUF file — no separate codec companion needed (Mimi weights and the tokenizer are embedded).

Released under **CC-BY-4.0** license.

## Voice cloning

Pocket TTS is zero-shot: it clones the timbre of a short reference clip. The
**voice-cloning builds** (the files *without* `novc` in the name) embed the Mimi
VAE **encoder** and speaker projection needed to condition on a reference; the
English-only `novc` builds omit that encoder and are ~20 MB smaller.

```bash
# clone the timbre of ref.wav (any sample rate; mono is used)
./build/bin/crispasr --backend pocket-tts -m pocket-tts-english-f16.gguf \
    --voice ref.wav \
    --tts "The quick brown fox jumps over the lazy dog." \
    --tts-output fox.wav --seed 42
```

Pocket TTS produces near-silence without voice conditioning, so if you omit
`--voice` a built-in default reference is used automatically. Use a clean,
single-speaker reference of a few seconds for best results.

## Files

The multilingual Q8_0/F16 files are full voice-cloning models. German, Spanish,
Italian, and Portuguese are Kyutai's distilled 6-layer releases; French is the
larger, undistilled 24-layer preview. The decoder-only **`novc`** alternatives
are currently English-only.

| File | Voice clone | Quant | Size | Notes |
|---|:---:|---|---:|---|
| `pocket-tts-english-f16.gguf`       | ✅ | F16  | 219 MB | Reference quality, cloning |
| `pocket-tts-english-q8_0.gguf`      | ✅ | Q8_0 | 124 MB | Near-F16, cloning |
| `pocket-tts-english-q4_k.gguf`      | ✅ | Q4_K |  73 MB | Smallest with cloning |
| `pocket-tts-english-novc-f16.gguf`  | — | F16  | 200 MB | Decoder only |
| `pocket-tts-english-novc-q8_0.gguf` | — | Q8_0 | 110 MB | Decoder only |
| `pocket-tts-english-novc-q4_k.gguf` | — | Q4_K |  62 MB | Decoder only, smallest |
| `pocket-tts-german-q8_0.gguf`        | ✅ | Q8_0 | 124 MB | German, distilled 6L |
| `pocket-tts-spanish-q8_0.gguf`       | ✅ | Q8_0 | 124 MB | Spanish, distilled 6L |
| `pocket-tts-italian-q8_0.gguf`       | ✅ | Q8_0 | 124 MB | Italian, distilled 6L |
| `pocket-tts-portuguese-q8_0.gguf`    | ✅ | Q8_0 | 124 MB | Portuguese, distilled 6L |
| `pocket-tts-french_24l-q8_0.gguf`    | ✅ | Q8_0 | 365 MB | French, undistilled 24L preview |

F16 equivalents of every multilingual Q8_0 file are also available (about
219 MB for each 6-layer model and 673 MB for French).

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr-cli

# 2. Download a model (voice-cloning F16 shown)
huggingface-cli download cstr/pocket-tts-GGUF pocket-tts-english-f16.gguf --local-dir .

# 3. Synthesize
./build/bin/crispasr --backend pocket-tts -m pocket-tts-english-f16.gguf \
    --tts "Hello, how are you today?" \
    --tts-output hello.wav --seed 42
```

Or with auto-download. `-l de`, `es`, `it`, `pt`, or `fr` selects and caches
the matching Q8_0 checkpoint; omit `-l` for English:
```bash
./build/bin/crispasr --backend pocket-tts -m auto --auto-download -l es \
    --accept-license pocket-tts-terms \
    --voice ref.wav --i-have-rights \
    --tts "Hola, este modelo ya habla español." \
    --tts-output hola.wav
```

## Python binding

```python
from crispasr import Session

sess = Session("pocket-tts-english-f16.gguf")
sess.set_tts_seed(42)
pcm = sess.synthesize("Hello world.")
sess.write_wav("hello.wav", pcm)
```

## Conversion

Converted with `models/convert-pocket-tts-to-gguf.py` from the CrispASR repo
(`--voice-cloning` bakes in the Mimi encoder + speaker projection). The Mimi
codec and SentencePiece tokenizer are embedded in the single GGUF.

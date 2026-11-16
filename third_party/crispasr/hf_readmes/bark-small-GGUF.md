---
license: mit
tags:
  - text-to-speech
  - tts
  - bark
  - gguf
  - crispasr
language:
  - en
  - de
  - es
  - fr
  - it
  - ja
  - ko
  - pl
  - pt
  - ru
  - tr
  - zh
---

# Bark Small — GGUF

[Suno Bark](https://github.com/suno-ai/bark) (MIT license) converted to GGUF for native C++ inference with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Model details

- **Architecture**: 3-stage hierarchical transformer (semantic → coarse → fine) + EnCodec decoder
- **Parameters**: ~300M total across 3 GPT-2 sub-models
- **Output**: 24 kHz mono PCM
- **Languages**: 13 languages with pre-trained speaker prompts
- **German speakers**: `v2/de_speaker_0` through `v2/de_speaker_9`
- **License**: MIT

## Quantization table

| File | Quant | Size | Quality |
|------|-------|------|---------|
| `bark-small-f16.gguf` | F16 | 809 MB | Reference |
| `bark-small-q8_0.gguf` | Q8_0 | 435 MB | Near-lossless |
| `bark-small-q4_k.gguf` | Q4_K | 235 MB | Good for real-time |

All variants pack the 3 sub-models (text/semantic, coarse acoustic, fine acoustic) + EnCodec decoder into a single GGUF file. No companion model needed.

## Usage with CrispASR

```bash
# Auto-download and synthesize
crispasr --backend bark -m auto --tts "Hello, how are you today?" --tts-output hello.wav

# With a specific quantization
crispasr --backend bark -m bark-small-q4_k.gguf --tts "The quick brown fox" --tts-output fox.wav

# With a German speaker prompt (when supported)
crispasr --backend bark -m bark-small-q8_0.gguf --tts "Hallo Welt" --voice v2/de_speaker_3 --tts-output hallo.wav
```

## Conversion

Produced with:
```bash
python models/convert-bark-to-gguf.py --output bark-small-f16.gguf
crispasr-quantize bark-small-f16.gguf bark-small-q8_0.gguf q8_0
crispasr-quantize bark-small-f16.gguf bark-small-q4_k.gguf q4_k
```

## Architecture details

### Stage 1 — Semantic model
- GPT-2 (12 layers, 768-d) generating semantic tokens from text
- BERT WordPiece tokenizer (119547 vocab)
- Output: up to 768 semantic tokens

### Stage 2 — Coarse acoustic model
- GPT-2 (12 layers, 1024-d) converting semantic → coarse EnCodec codes
- Alternates codebook 0/1 prediction
- Output: 2 × ~384 coarse tokens

### Stage 3 — Fine acoustic model
- Non-causal GPT-2 (12 layers, 1024-d)
- Fills codebooks 2-7 from codebooks 0-1
- Output: 8 codebooks × 384 timesteps

### EnCodec decoder
- 8-codebook RVQ (1024 entries each)
- SEANet CNN decoder with ELU activation
- Upsample ratios [8, 5, 4, 2] → 24 kHz

## Credits

- Original model: [Suno AI](https://github.com/suno-ai/bark) (MIT)
- GGUF conversion + C++ runtime: [CrispASR](https://github.com/CrispStrobe/CrispASR)

## Voice provenance (EU AI Act Art. 50(4))

**Not established.** Suno's README documents 100+ speaker presets and a prompt library, and says nothing about where those voices came from. Third-party write-ups describe them as "fully synthetic"; that phrasing is not in Suno's own documentation, and a summary is not a provider statement.

CrispASR records this as `speaker_identity=unknown`. CrispASR warns once per model and adds **no** disclosure. If you know these presets reproduce identifiable people, pass `--speaker-identity real_person`. Do not pass `synthetic` to silence the warning unless you have a source — that is the direction that silently removes a disclosure.

Override per run with `--speaker-identity`, or stamp a file permanently with
`models/stamp-speaker-identity.py`. See
[`docs/eu-ai-act.md` §6.2a](https://github.com/CrispStrobe/CrispASR/blob/main/docs/eu-ai-act.md).

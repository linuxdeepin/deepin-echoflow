---
license: cc-by-4.0
language:
- en
base_model:
- nvidia/tts_en_fastpitch
- nvidia/tts_hifigan
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- fastpitch
- hifigan
- non-autoregressive
- gguf
- crispasr
library_name: ggml
---

# FastPitch (English) + HiFi-GAN — GGUF (ggml)

GGUF / ggml conversion of [`nvidia/tts_en_fastpitch`](https://huggingface.co/nvidia/tts_en_fastpitch) + [`nvidia/tts_hifigan`](https://huggingface.co/nvidia/tts_hifigan) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

FastPitch is a **non-autoregressive** parallel TTS model that generates the entire mel spectrogram in a single forward pass (no sampling, no KV cache), making it very fast. The HiFi-GAN vocoder converts the mel to 22050 Hz PCM audio.

- **Text encoder**: 6-layer Transformer (384-d, 1-head, post-norm, Conv1d FFN)
- **Duration predictor**: 2-layer Conv1d stack + linear projection
- **Pitch predictor**: 2-layer Conv1d stack + linear projection
- **Mel decoder**: 6-layer Transformer (same architecture as encoder)
- **HiFi-GAN vocoder**: conv_pre + 4 upsample stages (rates 8,8,2,2) with MRF resblocks + conv_post

Single speaker, English. ~60M parameters total (FastPitch + HiFi-GAN combined in one GGUF).

Released under **CC-BY-4.0** (NeMo model license).

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `fastpitch-en-f16.gguf` | F16 | ~230 MB | Reference quality |
| `fastpitch-en-q8_0.gguf` | Q8_0 | ~120 MB | Near-lossless |
| `fastpitch-en-q4_k.gguf` | Q4_K | ~70 MB | Best size/quality balance |

> **The F16 build was rebuilt (2026-08-03), after being briefly withdrawn.**
> The file previously published here was corrupt: the converter handed a
> float32 array to `add_tensor(raw_dtype=F16)`, which labels bytes rather than
> converting them, so it held half the weights reinterpreted as garbage
> (943,872 NaNs) and would not open.
>
> Rebuilding it exposed a second issue that briefly looked fatal — every run
> aborted in ggml graph compute. It was two tensors: ggml's Metal binary ops
> require the second operand to be F32, and `enc.pos_emb` / `dec.pos_emb` are
> added straight to the hidden state, so converting them to F16 killed the
> graph. They are kept F32 now; a matmul weight may be F16, an addend may not.
>
> The current f16 synthesises correctly and ASR-round-trips clean.


## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr-cli

# 2. Download model (auto-download also works: -m auto --backend fastpitch)
hf download cstr/fastpitch-en-GGUF fastpitch-en-q8_0.gguf --local-dir .

# 3. Synthesize
./build/bin/crispasr --backend fastpitch -m fastpitch-en-q8_0.gguf \
    --tts "Hello there, how are you doing today?" \
    --tts-output hello.wav

# 4. Verify (ASR roundtrip)
./build/bin/crispasr -m models/ggml-base.en.bin -f hello.wav
```

## Conversion

```bash
python models/convert-fastpitch-to-gguf.py \
    --hf-model nvidia/tts_en_fastpitch \
    --hf-vocoder nvidia/tts_hifigan \
    --output fastpitch-en-f16.gguf --ftype f16
```

## Limitations

- Single speaker only (the English model has `n_speakers=1`)
- Character-level tokenization (no G2P phoneme conversion yet; proper ARPABET G2P would improve pronunciation of uncommon words)
- Deterministic output (no temperature/seed controls — same input always produces same output)
- 22050 Hz sample rate

## Voice provenance (EU AI Act Art. 50(4))

The upstream `nvidia/tts_en_fastpitch` card states it is *"trained on LJSpeech"*. LJSpeech is 13,100 clips of a **single narrator** — Linda Johnson, recorded 2016–17 for LibriVox — and this is the single-speaker English checkpoint (`n_speakers=1`). The voice you hear is one identifiable person.

CrispASR records this as `speaker_identity=real_person`. Output synthesized with it carries a **spoken AI disclosure**, because audio resembling an identifiable person is a deep fake under Art. 3(60) whether or not any cloning took place. It does **not** require `--i-have-rights`: the donor's agreement to the training is a licensing matter settled upstream, which a downstream operator cannot attest to.

Override per run with `--speaker-identity`, or stamp a file permanently with
`models/stamp-speaker-identity.py`. See
[`docs/eu-ai-act.md` §6.2a](https://github.com/CrispStrobe/CrispASR/blob/main/docs/eu-ai-act.md).

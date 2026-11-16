---
license: apache-2.0
language:
- ar
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- conformer
- arabic
- crispasr
base_model: CohereLabs/cohere-transcribe-arabic-07-2026
---

# cohere-transcribe-arabic-07-2026 — GGUF

GGUF weights for **[CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)** — Cohere Labs' 2B-parameter Arabic ASR model (a FastConformer encoder + Transformer decoder), released July 2026 (~11% WER on FLEURS Arabic per the source card).

These GGUFs run on CPU/Metal/CUDA/Vulkan via **[CrispASR](https://github.com/CrispStrobe/CrispASR)** — a C++ runtime for the Cohere Conformer-encoder / Transformer-decoder architecture.

> **License**: Apache 2.0 (inherited from source model). See the [original model card](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) for full terms.

---

## Files

| File | Size | Type |
|------|------|------|
| `cohere-transcribe-arabic-f16.gguf` | 4.1 GB | F16 (reference precision) |
| `cohere-transcribe-arabic-q8_0.gguf` | 2.4 GB | Q8_0 |
| `cohere-transcribe-arabic-q4_k.gguf` | 1.5 GB | Q4_K |
| `cohere-transcribe-arabic-q4_k-imatrix.gguf` | 1.5 GB | Q4_K + importance matrix (Arabic-calibrated) |
| `cohere-transcribe-arabic-ref.gguf` | small | per-stage reference activations for `crispasr-diff` |

All quants keep LayerNorm weights / biases at F32. The `-imatrix` build is
calibrated on CC0 Common Voice Arabic (`fsicoli/common_voice_17_0`, `ar/dev`) and
recovers quality at the 4-bit budget; prefer it over plain `q4_k`.

On an M1 (Metal), end-to-end encode+decode on an 11 s clip is on par with the
`transcribe.cpp` reference runtime (~1.5 s, load excluded).

---

## Quick start

```bash
# Build CrispASR (see the repo for full instructions)
git clone --recursive https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Shorthand: auto-downloads the recommended imatrix GGUF and defaults
# the language to Arabic — no -m / --hf-repo / -l needed.
build/bin/crispasr --backend cohere-ar audio.wav
```

`cohere-ar` is a CLI alias for the `cohere` backend: it routes to the same
runtime, resolves `cohere-transcribe-arabic-q4_k-imatrix.gguf` via `-m auto`,
and sets `-l ar` unless you pass an explicit `-l` (which always wins — useful
if you want to run this model with the LID pre-step or force another
language for testing). It's equivalent to:

```bash
build/bin/crispasr --backend cohere \
    --hf-repo cstr/cohere-transcribe-arabic-07-2026-GGUF:cohere-transcribe-arabic-q4_k-imatrix.gguf \
    audio.wav -l ar
```

Or point `-m` at a locally downloaded GGUF:

```bash
build/bin/crispasr --backend cohere -m cohere-transcribe-arabic-q4_k.gguf audio.wav -l ar
```

---

## Supported languages: `en` and `ar` only

This finetune's `config.json` lists exactly two: `en`, `ar`. That matters more
than it sounds, because **nothing else can tell you**. The finetune keeps the
base tokenizer, so all 183 ISO-639-1 `<|xx|>` tokens are present in the vocab —
`<|de|>`, `<|ru|>`, `<|ja|>` all decode without error. And Cohere Transcribe
answers a wrong language *fluently* rather than failing. On one 8 s Arabic
clip, `-l ru` added a hallucinated leading word, `-l ja` swapped the quotation
marks for brackets, and `-l de` changed the diacritics — all plausible, none
flagged.

So CrispASR reads the whitelist from the GGUF and substitutes loudly:

```
cohere: language 'de' is not supported by this model — using 'en' instead. Supported: en, ar
```

The GGUFs here carry `cohere_transcribe.supported_languages`. For any Cohere
GGUF converted before that key existed, declare it at runtime instead:

```bash
CRISPASR_COHERE_LANGS=en,ar build/bin/crispasr --backend cohere -m old.gguf audio.wav -l auto
```

With the list present and `-l auto`, CrispASR identifies the language by
**probing this model itself** — one short decode per candidate, no whisper-tiny
download — and can therefore only ever return `en` or `ar`:

```
cohere[lid]: en  len=64   agree=1.00 div=0.79 score=158  :: The city is located in the city of Jerry, a large city of Je
cohere[lid]: ar  len=82   agree=1.00 div=1.00 score=328  :: العاصفة شبه الاستوائية "جيري" تغا
crispasr: LID -> language = 'ar' (cohere-probe, p=0.675)
```

---

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | 48-layer FastConformer, d=1280, heads=8, head_dim=160, ffn=5120, conv_kernel=9 |
| **Decoder** | 8-layer causal Transformer, d=1024, heads=8, head_dim=128, ffn=4096 (ReLU), max_ctx=1024 |
| **Vocab** | 16,384 SentencePiece tokens |
| **Audio** | 16 kHz mono, 128 mel bins, n_fft=512, hop=160, win=400, 8× time subsampling |
| **Parameters** | ~2B |

Prompt tokens (decoder): the runtime prepends `decoder_start_token_id` (`▁`,
13764) to the control-token prompt, matching the reference `decoder_input_ids`.
Mel frame count is `floor(n/hop)+1` (NeMo FilterbankFeatures), giving `T_enc =
calc_length(T_mel)`; both are required for the cross-attention context to line up.

---

## Validation (`crispasr-diff`)

`cohere-transcribe-arabic-ref.gguf` holds per-stage reference activations dumped
from the transformers model by `tools/dump_reference.py`. Reproduce the
per-layer parity check with:

```bash
build/bin/crispasr-diff cohere cohere-transcribe-arabic-f16.gguf \
    cohere-transcribe-arabic-ref.gguf audio.wav
```

---

## Related

- **Source model**: [CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
- **English sibling**: [cstr/cohere-transcribe-03-2026-GGUF](https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF) — Cohere Transcribe 2B (lowest English WER)
- **C++ runtime**: [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)

## Provenance and EU AI Act Art. 53 note

- **Upstream model:** [CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) — published by `CohereLabs`.
- **Upstream licence:** `apache-2.0`. This repository redistributes under the same terms; it grants no rights the upstream licence does not.
- **What was done here:** format conversion and/or quantisation only (GGUF). No training, no fine-tuning, no merging, no distillation, no change to architecture, vocabulary or capability. Only the numeric representation of the upstream weights differs.
- **Training data:** documented — where it is documented at all — by the upstream provider; see the upstream model card. No training data was used, added or selected by this repository. No training-content summary was found on the upstream model card at the time of writing; that documentation gap is upstream's and is not filled here.
- **Provider status:** under Regulation (EU) 2024/1689 the upstream authors remain the provider of this model. Converting the serialisation format does not make this repository the provider of a new general-purpose AI model, and no such claim is made. Questions about training content, copyright policy or model capability belong upstream.

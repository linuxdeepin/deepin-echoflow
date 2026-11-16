---
license: apache-2.0
language:
- en
- zh
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- ggml
- gguf
- qwen3
- qwen3-omni
- speech-llm
library_name: ggml
base_model: OpenMOSS-Team/MOSS-Transcribe-preview-2B
---

# MOSS-Transcribe-preview-2B — GGUF (ggml-quantised)

GGUF / ggml conversions of [`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B) for use with the `crispasr` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

MOSS-Transcribe is OpenMOSS's **speech-LLM** ASR model (~2.4 B params, Apache-2.0):

- **Stock Qwen3-Omni-MoE audio encoder** (the full 1280-dim / 32-layer tower) feeds frames into a **Qwen3-1.7B LLM** via embedding splice in a ChatML prompt.
- **4.87 % average WER** (reported by the authors).
- Runs **on CPU or GPU** (Metal/CUDA) through the CrispASR runtime, with a persistent KV cache for O(1) per-token decode.

It is a close sibling of CrispASR's `moss-audio` backend (same author) but **ASR-dedicated**: no DeepStack, a `conv_out`/`proj1`/`proj2` encoder head, and a smaller 1.7 B decoder.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `moss-transcribe-preview-2b-f16.gguf`  | 4.51 GB | F16 |
| `moss-transcribe-preview-2b-q8_0.gguf` | 3.28 GB | Q8_0, near-lossless |
| `moss-transcribe-preview-2b-q4_k.gguf` | 2.63 GB | **Q4_K — recommended default** |

The Q4_K and Q8_0 builds keep the **audio encoder, the adapter, and the tied token-embedding / output head at F16** (only the LM's attention and FFN matmuls are quantised), so transcript quality is preserved.

All quantisations produce the correct transcript on `samples/jfk.wav`:
> and so my fellow americans ask not what your country can do for you ask what you can do for your country

(The model outputs lowercase, lightly punctuated text.)

## Quick Start

```bash
# 1. Build the runtime
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target crispasr

# 2. Download a quantisation
hf download cstr/MOSS-Transcribe-preview-2B-GGUF \
    moss-transcribe-preview-2b-q4_k.gguf --local-dir .

# 3. Transcribe
./build/bin/crispasr -m moss-transcribe-preview-2b-q4_k.gguf your-audio.wav
```

Audio must be 16 kHz mono. Pre-convert with:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

The CrispASR model registry also auto-downloads the Q4_K build on demand (`-m moss-transcribe`).

## Architecture

| Component | Details |
| --- | --- |
| Audio encoder | Qwen3-Omni-MoE audio tower: 32-layer pre-LN Transformer, d=1280, heads=20, head_dim=64, FFN=5120, **windowed (block-diagonal) attention** |
| Conv front-end | 3 × Conv2D stride-2 (1→480→480→480) → `conv_out` (480·16=7680 → 1280, no bias) → sinusoidal positions |
| Encoder head | `ln_post` → `proj1` (1280→1280) → GELU → `proj2` (1280→2048) |
| Adapter | Gated-MLP (SwiGLU): 2048 → 8192 → 2048 |
| LLM | Qwen3-1.7B: 28 layers, hidden=2048, **16 Q / 8 KV heads (GQA)**, head_dim=128, FFN=6144, SwiGLU, RMSNorm, **per-head Q-norm / K-norm**, NEOX RoPE θ=1e6, **tied embeddings** |
| Vocab | 151 936 tokens (Qwen BPE, GPT-2 byte encoding) |
| Audio | 16 kHz mono, 128 mel bins, n_fft=400, hop=160 (matches `WhisperFeatureExtractor`) |
| Prompt | `chat_template_default.py` ChatML: `<\|im_start\|>user\n<\|audio_start\|>` · audio · `<\|audio_end\|><\|im_end\|>\n<\|im_start\|>assistant\n` → transcript → `<\|im_end\|>` |
| Audio injection | audio placeholder positions in the prompt get their token embedding replaced with the adapter output frames |
| Parameters | ~2.4 B |

## Implementation notes (correctness)

The C++ runtime is verified against the PyTorch reference at every architectural boundary on `samples/jfk.wav` via the `crispasr-diff` harness:

| Stage | Diff metric | Result |
| --- | --- | --- |
| Mel (C++ STFT vs `WhisperFeatureExtractor`) | per-bin cosine | 1.000000 |
| Encoder layer 0 (conv + windowed attention) | per-row cosine | 1.000000 (all rows) |
| Full encoder + adapter | per-row cosine | ~0.98 (F16 weight precision) |
| First decode token | argmax vs reference | match (`and`) |
| End-to-end transcript | vs bf16 reference | **verbatim** |

### Non-obvious gotchas the port handled

1. **Prompt template is mandatory.** Inference must use the `chat_template_default.py` ChatML framing (`user` / `assistant` markers around the audio). The bare audio layout makes the model emit garbage instead of transcribing.
2. **Whisper drops the trailing STFT frame** (`stft[..., :-1]`), giving exactly `n_samples / hop` mel frames; the runtime truncates to match (otherwise the audio-token count drifts by one).
3. **The token embedding is tied to the output head**, so it is pinned at F16 in the quantised builds — quantising it corrupts both the input embeddings and every output logit.

## License

Apache-2.0, inherited from the base model.

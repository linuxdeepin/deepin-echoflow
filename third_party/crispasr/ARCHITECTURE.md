# CrispASR — Architecture

One-page dependency map for contributors. Start here if you need to
know which file to edit for a given change, or which models would be
affected by a given `src/core/` refactor.

For user-facing docs see `README.md`. For pending work see `PLAN.md`.
For the reasoning behind design choices see `LEARNINGS.md`.

---

## Layer overview

```
┌───────────────────────────────────────────────────────────────────┐
│ examples/cli/  — the crispasr binary                              │
│ ──────────────────────────────────────────────────────────────── │
│   cli.cpp                   crispasr entry + --backend branch  │
│   whisper_params.h          shared params struct                  │
│                                                                    │
│   crispasr_backend.{h,cpp}  interface + factory + GGUF detect     │
│   crispasr_backend_*.cpp    75 per-model adapter files            │
│     ASR:  crispasr (whisper)  parakeet  canary  cohere  granite   │
│           voxtral  voxtral4b  qwen3  nemotron  gigaam  …          │
│     TTS:  kokoro  orpheus  chatterbox  qwen3_tts  moss_tts  …     │
│     other: htdemucs  crepe  tabcnn  btc  m2m100  t5  …            │
│                                                                    │
│   crispasr_run.cpp          top-level pipeline dispatch           │
│   crispasr_vad_cli.{h,cpp}  Silero VAD slicing                    │
│   crispasr_output.{h,cpp}   TXT/SRT/VTT/CSV/JSON/LRC writers      │
│   crispasr_model_mgr_cli.{h,cpp}  -m auto via curl/wget           │
│   crispasr_aligner_cli.{h,cpp}  canary_ctc alignment wrapper      │
│   crispasr_llm_pipeline.h   shared LLM decode loop (CLI-side)     │
├───────────────────────────────────────────────────────────────────┤
│ src/  — per-model C runtimes (public headers in include/)         │
│ ──────────────────────────────────────────────────────────────── │
│   crispasr.cpp             OpenAI Whisper (reference impl;        │
│                            public header include/crispasr.h)      │
│   parakeet.{h,cpp}         NVIDIA Parakeet TDT                    │
│   canary.{h,cpp}           NVIDIA Canary 1B v2                    │
│   canary_ctc.{h,cpp}       Canary auxiliary CTC aligner           │
│   cohere.{h,cpp}            Cohere Transcribe 2B                  │
│   qwen3_asr.{h,cpp}        Qwen3-ASR 0.6B (speech-LLM)            │
│   voxtral.{h,cpp}           Voxtral-Mini-3B (speech-LLM)          │
│   voxtral4b.{h,cpp}         Voxtral-Mini-4B-Realtime              │
│   granite_speech.{h,cpp}    Granite 4.0-1B Speech                 │
│   wav2vec2-ggml.{h,cpp}     Wav2vec2 CTC (cohere-align)           │
│   … plus ~65 more model runtimes (TTS, translation, music,        │
│     diarization, VAD, LID) — see docs/architecture.md             │
├───────────────────────────────────────────────────────────────────┤
│ src/core/  — shared primitives (static library crispasr-core)    │
│ ──────────────────────────────────────────────────────────────── │
│   ~118 headers; the four load-bearing ones:                       │
│   mel.{h,cpp}          log-mel spectrogram (NeMo + HF clusters)   │
│   ffn.h                SwiGLU / plain-SiLU FFN (header-only)      │
│   attention.h          Llama-style MHA + flash-attn (header-only) │
│   gguf_loader.{h,cpp}  GGUF open + weight mmap + name lookup      │
│   (plus fft, cpu_ops, ctc, fastconformer, conformer_ibm,          │
│    granite_llm, qformer, bpe, sanm, greedy_decode, cqt, …)        │
├───────────────────────────────────────────────────────────────────┤
│ ggml/  — tensor library + backend dispatch + quantisation         │
└───────────────────────────────────────────────────────────────────┘
```

---

## `src/core/` consumption map

Which model uses which shared helper, today. The four original
"founding" models are shown; the helpers have since spread far wider
(`mel.h` 30 consumers, `ffn.h` 36, `attention.h` 42,
`gguf_loader.h` 99 — `grep -rl '#include "core/<x>.h"' src/*.cpp`
is the live answer).

|               | `mel` | `ffn` | `attention` | `gguf_loader` |
|---|:---:|:---:|:---:|:---:|
| parakeet      |  ✔  |     |     |  ✔  |
| canary        |  ✔  |     |  ✔  |  ✔  |
| canary_ctc    |  ✔  |     |     |  ✔  |
| cohere        |  ✔  |     |  ✔  |  ✔  |
| voxtral       |  ✔  |  ✔  |  ✔ (LLM block) |  ✔  |
| voxtral4b     |  ✔  |  ✔  |  ✔ (encoder + LLM) |  ✔  |
| qwen3_asr     |  ✔  |  ✔  |  ✔  |  ✔  |
| granite_speech|  ✔  |  ✔  |  ✔  |  ✔  |
| wav2vec2-ggml |     |     |     |  ✔  |
| **whisper**   |     |     |     |     |

**Whisper is intentionally not migrated** — it's the battle-tested
reference and the test gate for every other refactor. (`src/crispasr.cpp`
does pull a few non-numeric core headers — `gpu_backend_pref.h`,
`whisper_special_tokens.h`, `ggml_cpu_backend.h` — but none of the four
numeric primitives above.)

**`wav2vec2-ggml`** is called from the legacy `cohere-align` path; it
has a minimal model structure, so only `core_gguf::` was worth adopting.

---

## Dependency graph (edge direction: "depends on")

```
             crispasr binary
                   │
    ┌──────────────┴──────────────┐
    │                              │
 cli.cpp                   crispasr_backend_*.cpp
    │                              │
    │                              ├─→ crispasr.cpp (whisper)
    │                              ├─→ parakeet.{cpp,h} ──┐
    │                              ├─→ canary.{cpp,h}    │
    │                              ├─→ canary_ctc.*     ─┤
    │                              ├─→ cohere.{cpp,h}   │
    │                              ├─→ qwen3_asr.*      ─┤
    │                              ├─→ voxtral.{cpp,h}  ─┼──→ crispasr-core
    │                              ├─→ voxtral4b.*     ──┤    (src/core/)
    │                              ├─→ granite_speech.*─ ┘       │
    │                              └─→ canary_ctc (aligner)      │
    │                                                             │
    ├─→ common (crispasr example lib)                          │
    ├─→ crispasr-lib (alias `whisper`; the whisper-backend path)  │
    └─→ crispasr_{vad_cli,output,model_mgr_cli,aligner_cli,run}   │
                                                                   │
                                                                   ▼
                                                                 ggml
```

Every non-whisper model links `crispasr-core`. `crispasr-lib` (the
whisper runtime, aliased as `whisper`) links it too — for the
non-numeric helpers (GPU backend preference, cache, special-token
tables) — but deliberately keeps its own mel / attention / GGUF code.

---

## How to find the code for a feature

| Looking for | Look in |
|---|---|
| `--backend` CLI flag parsing | `examples/cli/cli.cpp` (look for `"--backend"`) |
| Backend auto-detection from GGUF | `examples/cli/crispasr_backend.cpp` → `crispasr_detect_backend_from_gguf` |
| Feature capability matrix / warnings | `examples/cli/crispasr_run.cpp` → `warn_unsupported` |
| `-m auto` download | `examples/cli/crispasr_model_mgr_cli.cpp` |
| VAD slicing | `examples/cli/crispasr_vad_cli.{h,cpp}` |
| SRT / VTT / JSON writers | `examples/cli/crispasr_output.{h,cpp}` |
| CTC alignment for LLM backends | `examples/cli/crispasr_aligner_cli.{h,cpp}` |
| Whisper code path (historical) | `examples/cli/cli.cpp` main(), post-dispatch |
| Model-specific transcribe logic | `examples/cli/crispasr_backend_<X>.cpp` |
| Model-specific mel / encoder / LLM | `src/<model>.cpp` |
| Shared log-mel spectrogram | `src/core/mel.{h,cpp}` |
| Shared SwiGLU FFN helper | `src/core/ffn.h` |
| Shared Llama self-attention helper | `src/core/attention.h` |
| Shared GGUF loading + weight map | `src/core/gguf_loader.{h,cpp}` |
| Streaming session API (whisper / kyutai-stt / moonshine-streaming) | `src/crispasr_c_api.cpp` → `crispasr_session_stream_open` + `crispasr_stream_*` dispatch |
| Native voxtral4b streaming (PLAN #7: incremental encoder + speculative prefill + live captions + decoder thread) | `src/voxtral4b.cpp` → `voxtral4b_stream_*` |
| Mic capture (cross-platform via miniaudio) | `src/crispasr_mic.{h,cpp}` |

---

## Adding a new backend

Five files, ~200-300 LOC total (most of it in the src/ model file).
Step-by-step with worked examples is in `README.md` →
"Adding a new backend". Short version:

1. Implement `src/yourmodel.{h,cpp}` with a C API. Prefer
   `core_mel::compute`, `core_ffn::swiglu`, `core_attn::…`, and
   `core_gguf::…` over hand-rolling the equivalents.
2. Wrap it in `examples/cli/crispasr_backend_yourmodel.cpp` (~120
   LOC, see `crispasr_backend_parakeet.cpp` as the minimal template).
3. Register in `examples/cli/crispasr_backend.cpp` factory + list, and
   add the architecture string(s) to `src/core/arch_backend_map.h`
   (`core_arch::backend_for_arch`) — the single table shared by the CLI
   and by `crispasr_detect_backend_from_gguf` in `src/crispasr_c_api.cpp`
   (#335). Adding it in only one place breaks every non-CLI binding.
4. Link in `src/CMakeLists.txt` (new library) and
   `examples/cli/CMakeLists.txt` (add to crispasr target).
5. Optional: register the default quantised HF repo in
   `src/crispasr_model_registry.cpp` so `-m auto` works.

Regression-test by running `crispasr --backend yourmodel -m model.gguf
-f samples/jfk.wav` before AND after your change and `diff`-ing the
output. Bit-identical is the gate.

---

## What's intentionally NOT shared

Some code looks duplicated but isn't worth extracting:

- **Each model's custom Cooley-Tukey FFT.** ~28 near-identical
  radix-2 implementations still live in `src/*.cpp` (a further ~17
  models have adopted `src/core/fft.h`). Unifying the rest would save
  a few hundred LOC at the
  cost of making the FFT function pointer indirection mandatory.
  `core_mel::FftR2C` accepts any of them via a thread-local scratch
  wrapper (see `LEARNINGS.md` → "In-place recursive FFTs are
  const-unsafe"). Not a blocker.

- **Each model's GGUF tensor naming scheme.** These are genuinely
  model-specific (e.g. `encoder.layers.0.attn.q.weight` vs
  `model.audio.encoder.layer.0.self_attn.q_proj.weight`) and have to
  live in the per-model loader. `core_gguf::` handles the scaffolding
  around them; the per-field assignment loop stays.

- **Each model's forward graph structure.** The `src/core/attention.h`
  and `src/core/ffn.h` helpers cover the common building blocks, but
  the overall graph topology (how many encoder layers, where the
  projector sits, how the KV cache is threaded, what the prompt
  template looks like) is model-specific and belongs in `src/<model>.cpp`.

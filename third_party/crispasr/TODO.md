# CrispASR — TODO

**Live work tracking moved to `PLAN.md`** (priority table + per-item
sections). Historical milestones: `HISTORY.md`. Technical deep-dives:
`LEARNINGS.md`. Upstream blockers: `UPSTREAM.md`.

This file keeps only the distilled long-tail backlog that predates the
PLAN.md workflow (items below last reviewed 2026-07-11; everything else
from the old tracker was verified shipped or superseded).

## Backlog (unscheduled)

- **core/attention.h sliding-window knob** — voxtral4b's audio encoder uses
  750-token SWA; needs a `sliding_window` parameter (caller-built mask).
- **core/attention.h µP scale tricks** — granite uses `attention_multiplier`
  (1/128) instead of 1/√d and `residual_multiplier` 0.22; support natively.
- **parakeet TDT decoder on GPU** — LSTM predictor + joint head are pure CPU
  float* loops; per-token stepping is sequential, so GPU win is uncertain
  (see PLAN §232 profiling: decode is cblas-bound).
- **voxtral4b native streaming protocol** — model supports 240 ms–2.4 s
  configurable-delay realtime streaming; we run chunk-and-transcribe.
- **voxtral right-padding 17 → 10 tokens** to match reference `voxtral.c`.
- **speech-translation quality validation at scale** (regression-tested on
  German only).
- **canary SRT/VTT subtitle output** (plain transcript only; CTC alignment
  already works via `-am`).
- **gemma4**: mel/attention hparams into GGUF (multi-flavor); LLM-side diff
  stages; Q2_K long-context PLE quality test; extract the USM Conformer
  audio tower into a shared lib.
- **granite encoder OpenMP annotations** — linked but no `#pragma omp` on
  the per-layer hot loops yet.
- **granite mel onto `core_mel`** — the last holdout: granite stacks two
  80-mel frames into `(160, T/2)`; needs a `core_mel::Params::stacked_frames`
  knob (referenced from ARCHITECTURE.md).
- **tools/reference_backends/voxtral.py** — LLM-half reference so
  `voxtral-test-llm` stops reporting `[SKIP]`.
- **tests/CMakeLists.txt** — migrate test target references to
  `$<TARGET_FILE:crispasr>`.

## Upstream (see UPSTREAM.md)

- ggml x86 AVX-VNNI / AVX512-VNNI dispatch for Q8_0 dot products (closes
  the gap to ONNX INT8 on x86 servers).
- NeMo Forced Aligner auxiliary CTC model standalone release (not blocking —
  our converter extracts it from the `.nemo` tarball).

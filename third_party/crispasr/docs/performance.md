# Performance — KV Cache & State Cache Survey

Comprehensive survey of caching strategies across all CrispASR and CrispEmbed backends.

## KV Cache Implementations

### CrispASR — Autoregressive ASR/LLM Decoders

Most autoregressive decoders use the shared `core_attn::kv_self_attn` helper; a few
(canary, cohere, granite_speech, kugelaudio, pocket_tts) hand-roll an equivalent
cache instead. Either way the KV tensors are persistent, allocated via
`ggml_backend_alloc_ctx_tensors`, and shaped as a 4D tensor
`(head_dim, max_ctx, n_kv_heads, n_layers)` for both K and V. The element type is
F16 by default and overridable per-half with `CRISPASR_KV_QUANT` /
`CRISPASR_KV_QUANT_K` / `CRISPASR_KV_QUANT_V` (`src/core/attention.h`).

| Backend | KV Cache | Conv/Other State | Allocation | Notes |
|---------|----------|-----------------|------------|-------|
| voxtral | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Mistral-based GQA 32/8 |
| voxtral4b | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Fused QKV |
| qwen3_asr | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | GQA expansion, RoPE NEOX |
| granite_speech | Own 4D K/V (`core_attn::kv_dtype_pair_from_env`) | — | `ggml_backend_sched` | µP residual_multiplier |
| granite_nle | None (NAR, `kv_k`/`kv_v` = `nullptr`) | Conv1d cache | `ggml_backend_sched` | NAR with conv streaming |
| cohere | Own 4D K/V + per-layer `cross_kv_k`/`cross_kv_v` | — | `ggml_backend_sched` | Encoder-decoder cross-attn KV |
| canary | Own 4D K/V (`core_attn::kv_cache_write`) | — | `ggml_backend_sched` | FastConformer enc + Transformer dec |
| glm_asr | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | GLM decoder |
| mimo_asr | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | MIMO encoder-decoder |
| gemma4_e2b | Dual cache (sliding+full) | — | `ggml_backend_sched` | Separate K/V for each attention type |
| funasr | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Q/K/V merged for efficiency |
| mini_omni2 | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | 8-stream multimodal |
| moss_audio | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Whisper enc + Qwen3 LM |

### CrispASR — TTS/Speech Generation

| Backend | KV Cache | Conv/Other State | Allocation | Notes |
|---------|----------|-----------------|------------|-------|
| csm_tts | Dual (backbone+depth) | — | `ggml_backend_sched` | 2 separate KV caches |
| qwen3_tts | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Post-proj Q/K norm |
| cosyvoice3_tts | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | AR speech-token LM |
| pocket_tts | Own `pocket_tts_kv_cache` (backbone + Mimi dec) | — | `ggml_backend_sched` | Llama-1B backbone |
| tada_tts | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Per-token flow matching |
| zonos_tts | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | CFG-guided, 9-codebook DAC |
| kugelaudio | Own 4D K/V (inline `ggml_view_4d` slices) | — | `ggml_gallocr` (direct, §209) + sched for VAE | 7B Qwen2.5 + DiT diffusion. LM/diffusion compute directly on ctx->backend (avoids the §206 weight-less-first-op cross-backend-copy bug); only the VAE decoder stays on the sched (its `ggml_pad` is Metal-unsupported → CPU fallback) |
| voxcpm2_tts | `core_attn::kv_self_attn` | — | `ggml_gallocr` | Flow-matching diffusion |
| vibevoice | `core_attn::kv_self_attn` | Conv cache | `ggml_backend_sched` | σ-VAE streaming |
| **lfm2_audio** | `core_attn::kv_self_attn` | **Conv state cache** | `ggml_gallocr` (direct, §206) | **Hybrid conv+attn backbone; backbone computes directly on ctx->backend — not the sched — to avoid the weight-less-first-op cross-backend-copy bug** |
| **nemotron** | cache_last_channel + cache_last_time | **Conv + attn cache** | `ggml_backend_sched` | **Cache-aware streaming FastConformer** |
| bark_tts | None (3× non-cached forward) | — | `ggml_backend_sched` | Could benefit from KV cache |
| chatterbox | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | T3 AR + S3Gen flow |

### CrispASR — Non-Autoregressive (no KV cache needed)

| Backend | Reason no cache needed |
|---------|----------------------|
| parakeet | CTC/TDT encoder — single forward pass |
| fastconformer_ctc | CTC head — encoder-only |
| wav2vec2 | CTC head — encoder-only |
| sensevoice | CTC multi-task — encoder-only |
| paraformer | NAR decoder — single pass with CIF predictor |
| kokoro | StyleTTS2 — non-autoregressive (has LSTM but runs fresh each call) |
| f5_tts | DiT flow-matching — iterative denoising, not autoregressive |
| fastpitch | Non-autoregressive parallel TTS |
| melotts | VITS2 — flow-based, non-autoregressive |
| piper | VITS — same as melotts |
| speecht5 | Encoder-decoder but mel output is parallel |
| dia_tts | Single-forward with CFG |
| parler_tts | T5 encoder + MusicGen — greedy but short |

### CrispEmbed — Encoders Only

No KV caches needed. All models are encoder-only (BERT, ViT, CNN).
Three models compute position biases per-forward: `bttr_ocr`, `ppformulanet_ocr`, `posformer_ocr`.

## LFM2-Audio: Hybrid Conv + Attention Cache

The LFM2 backbone uses two distinct cache types:

### Attention KV Cache (6 of 16 layers)
- Standard `core_attn::kv_self_attn` with F16 persistent tensors
- Cache: `(head_dim=64, max_ctx=2048, n_kv_heads=8, n_attn_layers=6)`
- Written during prefill, read+extended during decode

### Conv State Cache (10 of 16 layers)
- CPU-side float arrays: `(hidden=2048, kernel-1=2)` per layer
- Total: 10 layers × 2048 × 2 × 4 bytes = **160 KB**
- Stores last K-1=2 Bx columns from the ShortConv computation
- During decode: cached columns + new Bx → depthwise conv → take last output
- Shifted left by 1 after each step

### Depthformer Cache
- 6-layer transformer generating 8 codebooks per audio frame
- **Done** — manual CPU-side K/V cache, one step per codebook: O(codebooks) = 8
  passes per frame (was O(codebooks²) = 36 before the cache landed)
- Per layer `[max_codebooks][kv_dim] = [8][256]`; total 6 × 2 × 8 × 256 × 4 = **96 KB**
- Cached positions `0..c-1` are concatenated with the freshly computed K/V at
  position `c` (`lfm2_depthformer_sample_frame`, `src/lfm2_audio.cpp`)

## Allocation Strategies

| Strategy | Used by | Pros | Cons |
|----------|---------|------|------|
| `ggml_backend_sched` | 61 runtime files exclusively (+17 mixed) | GPU offload, automatic tensor placement | More complex setup |
| `ggml_gallocr` | 16 runtime files exclusively (+17 mixed) | Simple, exact allocation | CPU-only, manual tensor_set/get |

Counts are per `src/*.cpp` runtime file (some runtimes use both — e.g. a
sched'd main graph plus a gallocr'd codec/vocoder subgraph). Re-derive with
`grep -l ggml_backend_sched_new src/*.cpp` / `grep -l ggml_gallocr_new src/*.cpp`.
| Fixed buffer (`no_alloc=false`) | Legacy / simple paths | Simplest code | Wasteful, page-fault overhead |

## Performance Benchmarks (LFM2-Audio, 4-core CPU)

| Configuration | ASR (11s JFK, 23 tok) | TTS (short) | Notes |
|--------------|----------------------|-------------|-------|
| F16 no cache | ~480s | N/A | Original |
| F16 KV only | ~200s | N/A | 2.4× |
| F16 KV+conv | ~47s | ~60s | 10× |
| Q4_K KV+conv | ~31s | ~45s | 15× |
| F16 gallocr | ~2m20s | — | Reduced sys overhead |
| Q4_K gallocr+256MB | ~1m8s | — | Best CPU perf |
| GPU (M1 Metal, §206) | ~15s | — | Correct now (gallocr direct compute); AR decode is dispatch-bound so currently ~slower than threaded CPU. GPU-decode graph caching is the perf follow-up. |

## CrispASR vs transcribe.cpp — Head-to-Head Benchmark

Systematic evaluation against [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
(ASR-only, ggml-based, 16 model families, 60+ variants) on Kaggle P100 (sm_60, CUDA 12.8).

Test audio: `jfk.wav` (11s, 16 kHz mono). WER computed against reference transcript.
RTF = real-time factor (lower = faster; < 1.0 means faster than real-time).

### GPU mode (both engines on CUDA, P100 sm_60) — v16, 2026-07-11

| Family | CA RTF | TC RTF | CA WER | TC WER | Notes |
|--------|--------|--------|--------|--------|-------|
| SenseVoice Small | **0.016** | 0.019 | 0% | 0% | CA 1.2x faster |
| Whisper base | **0.022** | 0.020 | 0% | 0% | Parity |
| Canary 1B v2 | **0.045** | 0.049 | 0% | 0% | CA 1.1x faster |
| FunASR Nano 2512 | **0.045** | 0.140 | 0% | 100% | CA 3x faster; TC GPU bug |
| Cohere Transcribe | **0.045** | FAIL | 0% | — | CA wins; TC crashes on GPU |
| Moonshine Tiny | 0.051 | **0.012** | 9.1% | 0% | TC 4.3x; head_dim=36 no fast flash kernel |
| Whisper Large v3 Turbo | 0.061 | **0.050** | 0% | 0% | TC 1.2x faster |
| Qwen3-ASR 0.6B | **0.086** | 0.115 | 0% | 0% | CA 1.3x faster |
| Parakeet TDT 0.6B | 0.100 | **0.030** | 0% | 0% | TC 3.3x; CPU cblas TDT decoder |
| Moonshine Streaming Tiny | 0.238 | **0.012** | 0% | 0% | TC 20x; sliding-window masks |
| Nemotron 3.5 ASR 0.6B | 0.345 | **0.053** | 0% | 0% | TC 6.5x; CPU cblas RNNT decoder |

### CPU mode (both engines, no GPU) — v13

| Family | CA RTF | TC RTF | CA WER | TC WER |
|--------|--------|--------|--------|--------|
| Moonshine Tiny | 0.069 | **0.068** | 9.1% | 0% |
| SenseVoice Small | 0.156 | **0.118** | 0% | 0% |
| Whisper base | 0.233 | **0.135** | 0% | 0% |
| Moonshine Streaming Tiny | 0.244 | **0.032** | 0% | 0% |
| Parakeet TDT 0.6B | 0.385 | **0.266** | 0% | 0% |
| FunASR Nano 2512 | 0.417 | **0.232** | 0% | 0% |
| Canary 1B v2 | 0.526 | **0.384** | 0% | 0% |
| Qwen3-ASR 0.6B | 0.625 | **0.459** | 0% | 0% |
| Nemotron 3.5 ASR 0.6B | 0.625 | **0.232** | 0% | 0% |
| Cohere Transcribe | 0.909 | **0.803** | 0% | 0% |
| Whisper Large v3 Turbo | 5.000 | **2.294** | 0% | 0% |

### transcribe.cpp-only models (CrispASR coverage gaps)

| Family | TC RTF | TC WER | Notes |
|--------|--------|--------|-------|
| GigaAM v3 E2E-CTC | 0.074 | 27.3% | Russian-focused + EN; no CrispASR equivalent |

### Key findings

- **GPU**: CrispASR wins 6/11, ties 1, loses 4. Wins on SenseVoice, Whisper,
  Canary, FunASR, Cohere, Qwen3. Loses on Parakeet (CPU cblas decoder), Nemotron
  (CPU cblas RNNT), Moonshine (encoder im2col), Moonshine Streaming (architectural).
  Both require `-DGGML_CUDA_NO_VMM=ON` on Kaggle for `CUDA::cuda_driver` resolution.
- **CPU**: transcribe.cpp 1.3-3x faster across the board — leaner runtime with
  less dispatch overhead. CrispASR's unified backend path includes VAD, segment
  merging, and post-processing that transcribe.cpp skips.
- **WER**: Both engines produce identical transcripts (0% WER) on 7/9 models.
  Moonshine Tiny shows a minor Q4_K vs Q8_0 quant difference.
  FunASR Nano has a transcribe.cpp GPU inference bug (100% WER on GPU, 0% on CPU).
- **Feature gap**: transcribe.cpp is ASR-only; CrispASR adds TTS, S2S, diarization,
  LID, forced alignment, translation, and streaming — with GPU acceleration.
- **Architecture**: transcribe.cpp optimises for per-model performance with
  architecture-specific dispatch; CrispASR trades some per-model speed for a
  unified backend that supports 40+ model families across ASR/TTS/S2S/LID/MT.

Benchmark script: `tools/kaggle/transcribe-cpp-bench/transcribe_cpp_bench.py`
Kernel: `${KAGGLE_ACCOUNT}/crispasr-vs-transcribe-cpp-bench`
Results: `cstr/crispasr-kaggle-progress` dataset, prefix `transcribe-cpp-bench/`

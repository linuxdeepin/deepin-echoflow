# CUDA enablement and CrispASR refresh — 2026-11-02

## Scope and revisions

- Previous EchoFlow runtime: `5d7d8dd00c533213376fa7f910852ffc48762393`.
- EchoFlow integration merge: `7f8296d73`.
- Official upstream merged from `https://github.com/CrispStrobe/CrispASR.git`:
  `830ca6024ece6477da959236ee4f825a4d085300` (2026-11-02). This includes the
  planned `04200fae791c8d5113fa47d5896ccd13f18f6bad` plus the Metal backend
  name fix from PR #565.
- Nested ggml: `c36dab89b662838f0f5d4826c399198c0b90bbfc` →
  `e2ae679a563014f46c546ce6c6448e8601e4dea6`.
- Preserve downstream slim static embedding (`5d1e0adc`), VAD compute-error
  handling (`c2415dd9`), native GQA (`823812c4`), and slim C API guards
  (`5d7d8dd0`). New backend includes receive the same Qwen3-only guards.
- The integration commit is available on the downstream fork's
  `echoflow/cuda-runtime-refresh` branch for reproducible submodule checkout.

## Relevant upstream changes

The shared mel frontend adds SIMD windowing, fused preprocessing/padding and
parallel projection/log/normalization paths (`9fa9c3531`, `465d7d0d1`). Qwen3
calls this frontend, so this is the most relevant optimization to retain.
Shared resampling and worker scheduling also changed, with follow-up audit
repairs (`84b0a513b`, `d71788aa7`). These are code-review findings; this update
has no controlled before/after latency or CER benchmark.

Qwen3 now accepts free-form transcription context (`777e8ff23`). This is a
possible next step for terminology and domain hints. This update retains the
API but does not change EchoFlow's prompt settings or transcription behavior.

The ggml delta consists of CANN weight-layout/cache work and an API declaration;
it contains no CUDA-source changes. CUDA inference already existed in the
pinned runtime. There is no evidence here of a new NVIDIA-specific speedup.
Additional ASR/TTS/separation model families remain excluded from EchoFlow's
linked runtime. No service-side HTTP download is added.

## EchoFlow CUDA behavior

`-DGGML_CUDA=ON` enables the upstream CUDA backend. Default configuration and
Debian packages remain CPU-only. ASR already selects an available GPU backend;
transcript polish now keeps the chat API's default all-layer offload request
instead of forcing `n_gpu_layers=0`. CPU-only builds keep using CPU. Silero VAD
continues to set `use_gpu=false`.

The user installer forwards `GGML_CUDA` (default OFF) and optional
`CMAKE_CUDA_ARCHITECTURES`. On RTX 5070 Laptop, CUDA 12.8 accepts architecture
`120`; ggml refines it to `120a`. A CUDA executable requires the CUDA runtime,
cuBLAS and driver libraries at startup, so it is not a universal CPU binary.

## Validation

CPU configure/build and all 28 EchoFlow CTests pass. Unix-socket tests were
rerun outside the filesystem/network sandbox after four sandbox-only failures.
The service self-test also passes outside the sandbox. Default config printing,
the bundled `paraformer_zh.wav` transcription with Qwen3-ASR-0.6B, and a short
Chinese transcript through Qwen3-1.7B polish complete successfully. These are
smoke tests, not a corpus-level CER comparison.

Installer specs verify CPU defaults, CUDA opt-in, architecture propagation and
`--no-start` behavior. Shell syntax and the EchoFlow diff whitespace checks pass.
The downstream delta against the upstream tip also passes whitespace checks;
unrelated whitespace already in upstream Java bindings/model cards is retained.

The CUDA build uses NVIDIA's SHA-256-verified 12.8.1 redistribution components
(nvcc 12.8.93), with a retained local toolkit under ignored `build/cuda-toolkit`.
GPU execution was subsequently verified on RTX 5070 Laptop with NVIDIA
595.91.07 under kernel 7.3.0. ASR logs identify the CUDA device and CUDA graph
execution. A separate, fully preloaded polish process appears in NVIDIA's
compute-process list with 1730 MiB of GPU memory. The CPU and CUDA CTest suites
were each rerun on 2026-11-03 and pass all 28 tests.

## Same-machine CPU/CUDA sample comparison

On 2026-11-02, compare the two RelWithDebInfo/native builds on Intel Core Ultra
9 275HX, using the same models, Chinese language setting and eight CPU threads.
Run the bundled 13.052-second `samples/paraformer_zh.wav` five times through
`voice_latency_benchmark --transcribe-file ... --iterations 5`. For polish, feed
five short Chinese sentences to `echoflow-service --polish-text -`. Exclude the
first call from each series and take the median of the remaining four.

| Operation | CPU median | CUDA median | CPU/CUDA ratio |
| --- | ---: | ---: | ---: |
| Qwen3-ASR-0.6B Q4_K transcription | 1270.997 ms | 167.419 ms | 7.59x |
| Qwen3-1.7B Q4_K_M short-text polish | 616.5 ms | 110.5 ms | 5.58x |

The samples and timings are preserved in
[`cuda-upstream-benchmark.json`](cuda-upstream-benchmark.json). These are
model-resident operation timings, excluding model loading, microphone capture,
VAD and Fcitx commit latency. CUDA and CPU ASR results differ in punctuation;
the five polish outputs agree. This small comparison does not establish corpus
CER, long-utterance quality, end-to-end latency or a release performance gate.

# Learnings — CrispASR vs transcribe.cpp Evaluation

Lessons from the systematic head-to-head benchmark against
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (July 2026).

## Build & Infrastructure

1. **GGML_CUDA_NO_VMM=ON is essential on Kaggle**: Both CrispASR and transcribe.cpp
   use ggml's CUDA backend. Without this flag, cmake's `FindCUDAToolkit` fails to
   create the `CUDA::cuda_driver` imported target because `libcuda.so` only exists
   in `/usr/local/cuda/lib64/stubs/` on Kaggle. The flag gates the
   `target_link_libraries(... CUDA::cuda_driver)` call entirely. Neither symlinks
   nor `CMAKE_LIBRARY_PATH` nor `-DCUDA_DRIVER_LIBRARY` fix it — those target
   different cmake subsystems than `FindCUDAToolkit`'s internal lookup.

2. **ccache must come from the same environment**: A ccache snapshot from the VPS
   (different g++ version) gives 100% cache miss on Kaggle (g++ 11.4.0).
   ccache keys on preprocessed source + compiler path + flags. Always refresh the
   `crispasr-ccache` dataset from a successful Kaggle build, never from the VPS.
   Warm cache cuts builds from ~25 min to ~3 min.

3. **git submodules need explicit init after --depth 1**: `git clone --depth 1`
   does not fetch submodules. CrispASR's ggml is a submodule since 2026-07-07;
   cmake fails at `add_subdirectory(ggml)` without
   `git submodule update --init --recursive --depth 1`.

## CLI & API

4. **CrispASR uses --backend, not -b**: There is no short form. Using `-b whisper`
   causes "unknown argument" → usage text → exit(0). Exit code 0 (not nonzero)
   makes it look like success with empty output. Always check stderr even when
   exit code is 0.

5. **Companion files must be beside the GGUF**: Moonshine backends need
   `tokenizer.bin` in the same directory as the model GGUF. `--auto-download`
   puts companions in the cache dir, not next to a manually-placed model file.
   When downloading GGUFs manually, also download companions from the same HF repo.

6. **transcribe.cpp's --batch-jsonl gives structured timing**: Single-file mode
   outputs human-readable text; `--batch FILE --batch-jsonl` gives JSON with
   per-utterance `mel_ms`, `encode_ms`, `decode_ms`, `load_ms`. More precise
   than wall-clock timing for RTF comparison.

## Performance

7. **GPU-vs-GPU is competitive, not a blowout**: With both engines on CUDA (P100),
   CrispASR wins 3/9 (SenseVoice, Qwen3, Canary, FunASR), transcribe.cpp wins
   4/9 (Parakeet, Moonshine, Moonshine-Streaming, Nemotron), near-parity on
   Whisper. The earlier "CrispASR 7-10x faster" result was comparing CUDA vs
   CPU-fallback — transcribe.cpp's CUDA build was broken until we fixed it with
   `-DGGML_CUDA_NO_VMM=ON`.

8. **CPU overhead is CrispASR's main disadvantage**: 1.3-3x slower than
   transcribe.cpp on CPU. CrispASR's unified backend path includes VAD slicing,
   segment merging, post-processing (punctuation, truecasing), and LID detection
   that transcribe.cpp doesn't do. This overhead is amortised on GPU but
   dominates on CPU.

9. **Quantisation affects transcription**: Moonshine Tiny Q4_K (CrispASR) produces
   "american asked" while Q8_0 (transcribe.cpp) produces "americans ask" — a real
   accuracy difference from lossy quantisation, not a code bug.

## Performance — Deep Dive (§232 optimisation attempts)

14. **GGML_LLAMAFILE (tinyBLAS) doesn't help Q4_K models on x86**: A/B on Intel
    Xeon Skylake showed neutral results. tinyBLAS optimises the F32 GEMM kernel,
    but Q4_K inference is dominated by dequantisation, not the GEMM itself.
    transcribe.cpp's "~29% faster encoder" claim may apply to F16 or ARM NEON.
    Keep ON as default (zero risk, may help other architectures).

15. **Moonshine decoder is compute-bound, not overhead-bound**: The per-token
    graph rebuild + sched alloc takes ~1-2ms, but the actual matmul compute
    takes ~315ms/token (CPU, Q4_K tiny). The graph CAN'T be cached because
    `cur_pos` is baked into KV cache view byte offsets (`ggml_view_3d` uses
    compile-time offsets). This is a fundamental ggml API limitation — views
    don't support tensor-computed offsets.

16. **In-graph argmax saves GPU transfer, not CPU time**: Adding `ggml_argmax`
    to the decode graph reduces device→host transfer from 128 KB (full vocab
    logits) to 4 bytes (one int32) per token. On CPU this is meaningless
    (shared memory). On GPU it eliminates 3.3 MB of PCIe traffic per
    transcription (26 tokens × 128 KB). Needs Kaggle GPU benchmark to measure.

17. **Sliding-window attention masks cannot be skipped for offline files**:
    Moonshine-streaming's encoder was trained with specific window sizes
    (wl=16, wr=4 per layer). Removing masks for offline processing (to match
    transcribe.cpp's approach) produces degenerate output (repeating tokens).
    The model's learned feature distribution depends on the windowed attention
    pattern. The 21x speed gap requires either:
    - A sparse/banded flash attention kernel that natively skips masked positions
    - Using non-streaming moonshine for offline files (recommend `--backend
      moonshine` instead of `--backend moonshine-streaming` for files)
    - Accepting the gap (streaming model = streaming overhead)

18. **Nemotron streaming attention has the same constraint**: The cache-aware
    FastConformer uses L=56 R=3 attention windows. Like moonshine-streaming,
    switching to full attention for offline would likely break output. The 8.4x
    gap is architectural — streaming models pay streaming overhead even offline.

19. **RTF timing decomposition is critical for diagnosis**: CrispASR's stderr
    RTF excludes model load and LID but includes VAD + inference + post-proc.
    transcribe.cpp's `--batch-jsonl` gives mel/encode/decode separately. On the
    Kaggle benchmark, the subprocess wall-clock (`ca_infer_s`) includes
    everything — this is what matters for the user experience but not for
    engine-level comparison. Always report both.

20. **Batched blank-scan helps modestly on GPU, hurts on CPU**: The batched
    TDT/RNNT decode pre-computes joint logits for 32 frames in one sgemm,
    then scans for the first non-blank. On P100 GPU: Parakeet decode 955ms→828ms
    (13%), Nemotron 385ms→345ms (10%). On CPU: SLOWER (sgemm overhead for small
    matrices exceeds N×sgemv). Gate batched path behind GPU detection or env var.

21. **Transducer decoders are fundamentally CPU-bound in CrispASR**: The LSTM
    predictor + joint network use host-side `cblas_sgemv` with sequential state
    updates. Even with batched blank-scanning, the LSTM step between token
    emissions runs on CPU. The path to matching transcribe.cpp's 29ms Parakeet
    decode is porting LSTM+joint to a ggml graph on GPU. Small matrices (640×640)
    make this challenging — GPU kernel launch latency may dominate.

22. **Moonshine encoder gap is im2col on raw audio**: CrispASR processes raw
    176K audio samples through 3 Conv1d layers via `ggml_im2col + ggml_mul_mat`,
    creating 45.6 MB of F32 intermediates. transcribe.cpp likely uses `ggml_conv_1d`
    directly or a pre-computed mel spectrogram, avoiding the large intermediate.
    The encoder produces 2737 frames (not 550 — moonshine-streaming subsamples
    more aggressively). This accounts for the 5.8x GPU gap.

23. **Cohere works and CrispASR wins**: Fixed URL (repo is `cstr/cohere-transcribe-
    03-2026-GGUF`, not `cstr/cohere-transcribe-GGUF`). CA 0.046 vs TC 0.070 =
    CrispASR 1.5x faster on GPU. Cohere's encoder-decoder architecture benefits
    from CrispASR's GPU-accelerated cross-attention path.

## Model Coverage

10. **CrispASR coverage gaps**: GigaAM v3 family (Russian+EN ASR, 4 variants) and
    MedASR (gated medical) exist only in transcribe.cpp.

11. **transcribe.cpp coverage gaps**: No TTS, no diarization, no LID, no forced
    alignment, no translation (m2m100, madlad), no S2S. CrispASR covers all of
    these plus many unique ASR backends (cohere, granite, voxtral, glm, mimo,
    vibevoice, lfm2-audio, etc.).

## Benchmark Methodology

12. **RTF measurement differs**: CrispASR reports wall-clock including audio I/O,
    VAD, and post-processing. transcribe.cpp's `--batch-jsonl` reports
    mel+encode+decode only. For fair CPU comparison, use wall-clock on both sides.
    For GPU comparison, CrispASR's stderr RTF is the authoritative number.

13. **Normalisation matters for WER**: SenseVoice emits `<|TAG|>` tokens, Nemotron
    emits inline `en-us` language codes. Strip both before WER computation.
    Use lowercase + strip punctuation + normalise whitespace as the baseline.

24. **Batched blank-scan makes GPU WORSE, not better**: v14 showed Parakeet
    decode 0.095→0.833 (8.8x slower) and Nemotron 0.345→1.667 (4.8x slower)
    with CRISPASR_TDT_BATCH=1. Root cause: the "batch" runs a CPU sgemm
    (32×8198 logits) while the GPU sits idle. The sequential path runs 1×8198
    sgemv per step and terminates at the first blank — much cheaper because
    most frames ARE blank. Batching only helps when the sgemm itself runs on
    GPU (i.e., the LSTM+joint is a ggml graph), not when it's a CPU-side
    cblas call between GPU encoder passes. **Lesson: don't batch CPU work
    that feeds a GPU pipeline — batch the GPU work itself.**

25. **A CPU-pinned decode graph must keep its weights on CPU too (moonshine,
    M1 Metal)**: moonshine's self-attn KV cache lives on a CPU buffer, so
    `ggml_backend_sched` runs the whole decode step on the CPU even in GPU
    mode. With the decoder weights loaded onto the GPU (the naive all-GPU
    load), the sched then re-copies every GPU-resident decoder weight
    (incl. the 18-36 MB embed/lm_head) GPU→CPU on *each* per-token graph
    rebuild — the copy can't be cached because each step builds a fresh
    graph. Measured on a *quiet* M1 (jfk, 26 tok): Metal decode q8 39→23 ms
    (−40%), f16 119→50 ms (−58%), bit-identical transcript; f16 hurts most
    because its weights are 2× q8. Fix: `load_weights_split` routing
    `encoder.*`→GPU, `decoder.*`→CPU (moonshine default; `MOONSHINE_ALL_GPU=1`
    restores the old load). The per-token copy also explains the plan's
    440-660 ms/decode-*under-load* figures — the copy balloons under GPU
    contention. **Lesson: when the KV cache pins a decode graph to CPU,
    co-locate that graph's weights on CPU; a GPU weight buffer feeding a
    CPU split is a per-step cross-backend copy, not free residency.**
    (Corollary to the §232 CP_DIRECT finding: for these tiny models the win
    is avoiding cross-backend traffic, not sched-free dispatch.)

26. **For moonshine *tiny* on Apple Silicon at idle, pure CPU beats GPU
    outright**: measured totals (jfk, q8) — CPU 79 ms (enc 47 + dec 30) vs
    all-GPU 110 ms (enc 67 + dec 39). The model is small enough that Metal
    launch + the per-layer encoder attention Metal↔CPU permute bounces cost
    more than they save. The hybrid load (learning 25) recovers the decode
    half; the encoder half (GPU 67 vs CPU 47) remains a launch/bounce tax,
    left as-is (fixing the flash-attn layout bounce is a shared-code, higher-
    risk change). Net: GPU-mode moonshine-tiny is now enc-bound, not the
    earlier "GPU decode is 440 ms" story (that was the CPU-run-mislabeled-as-
    GPU bug × the weight-copy tax, both now addressed).

27. **Keeping a small-model encoder fully on Metal (manual attn) is SLOWER
    than the flash_attn CPU-bounce**: moonshine's encoder head_dim=36 has no
    Metal flash_attn kernel, so `ggml_flash_attn_ext` runs on CPU and the
    sched bounces each layer MTL→CPU→MTL (Q/K/V copies ≈514 KB each). Replacing
    it with manual `mul_mat + soft_max_ext + mul_mat` (Metal-supported at any
    head_dim) keeps the whole encoder on-backend — but measured ~40% SLOWER on
    M1 (enc 162 vs 114 ms, even with the flash path's CPU work under load).
    The T² scores tensor ([T,T,nh] ≈ 6.8 MB/layer) + 3 `cont`s spawn many small
    Metal kernels whose launch cost exceeds the cheap bounce copies — the same
    "death by kernel count" that sinks sched-free dispatch for these tiny
    graphs. Transcript identical. **Lesson: a cross-backend bounce of a few
    hundred KB is often cheaper than materialising T² attention on the GPU for
    a small model; don't assume "keep it all on one backend" wins — measure.**
    Kept opt-in (`MOONSHINE_ENC_ATTN=manual`) for CUDA/Vulkan/base re-test.

28. **The GPU→CPU decoder-weight copy isn't just slow — it drifts the output**:
    applying the learning-25 hybrid split to `moonshine_streaming` (same
    CPU-KV-pinned decode) in forced-GPU mode (`MOONSHINE_STREAMING_GPU=1`), the
    hybrid path reproduces the pure-CPU transcript **exactly**, while the legacy
    all-GPU path stably drops a comma ("so my" vs "so, my", deterministic across
    reps). The per-token GPU→CPU weight copy perturbs a borderline decode logit
    enough to shift a token boundary → different punctuation. So co-locating a
    CPU-pinned graph's weights on CPU is a *correctness* win too, not only perf.
    (moonshine_streaming stays CPU-by-default — the encoder is launch-bound on
    GPU; the fix is latent until the §232 Fix-2 batch encoder lands.)

29. **Gate a GPU transducer-decode port on Kaggle BEFORE building it — the naive
    per-step version loses.** Scoped the Parakeet TDT decode port (target: TC's
    29 ms P100 decode). Three pieces of evidence all point the same way: (a)
    LEARNINGS 24 — batched GPU joint measured 8.8× WORSE (CPU sgemm, idle GPU);
    (b) LEARNINGS 25-27 — per-step GPU dispatch of ≤8198×640 matmuls is
    launch-bound on Metal and loses to CPU cblas; (c) M1 CANNOT measure the win
    (parakeet is already encoder-bound + 6× RT on Metal; the decode gap is a
    CUDA-vs-CUDA competitiveness issue). TC's 29 ms almost certainly comes from
    **CUDA-graph capture** (whole step loop as one replayable graph), not per-step
    ggml dispatch. **Lesson: for a transducer/AR GPU-decode port whose only
    payoff is on CUDA, don't hand-write a large speculative ggml decode from an
    M1 session — scope the exact math (mind stale comments: the parakeet joint is
    ReLU, not the tanh the header claims), record the design, and build+A/B it on
    Kaggle where it can actually be measured. Correctness (transcript/WER parity)
    is HW-independent and can be pre-validated on M1; perf cannot.** Design in
    HISTORY.md §232 "RNNT/TDT GPU decode".

30. **The Parakeet "955 ms P100 decode gap" is substantially a slow-CPU-BLAS
    artifact, not pure GPU-idle.** After actually implementing the ggml-graph
    TDT decode (LSTM predictor + joint on `ctx->backend`, opt-in
    `PARAKEET_GGML_DECODE=1`), the M1 numbers reframe the whole gap: cblas decode
    is only **~60 ms on M1** (Apple Accelerate) vs the **955 ms** the §232 Kaggle
    kernel reported (OpenBLAS on the Kaggle CPU) — same code, same audio, ~16×
    difference purely from CPU BLAS quality. The ggml GPU decode is **55-59 ms on
    M1 Metal** — neutral/slightly faster, and **transcript-identical** to cblas
    (jfk + multispeaker, CPU & Metal), so the port is correct and safe to ship
    gated. **Lessons:** (a) before attributing a CPU-path "gap" to GPU-idle /
    architecture and building a GPU port, measure the baseline on the SAME BLAS
    the comparison used — a large chunk of the parakeet gap may close by linking a
    faster CPU BLAS, not by a GPU decode. (b) A correct hand-written ggml
    transducer decode is a modest, contained amount of code (one LSTM-layer helper
    + two step builders) and validates on M1 for correctness — so "build it, gate
    it, let Kaggle judge perf" was tractable after all; the reservation in
    LEARNINGS 29 was about not FLIPPING THE DEFAULT unvalidated, which still holds.
    Kaggle P100 A/B: `tools/kaggle/parakeet-ggml-decode-ab/`.

31. **Per-step ggml decode overhead scales with STEP COUNT — same code, opposite
    M1 verdict on parakeet vs nemotron.** The identical `core_rnnt_ggml` decode
    (predictor LSTM + joint as per-step ggml graphs, shared by both after the DRY
    extract) is M1-neutral on parakeet (cblas ~60 ms vs ggml ~57 ms) but a **2×
    REGRESSION on nemotron** (cblas ~755 ms vs ggml ~1613 ms) — both
    transcript-identical. The difference is decode length: nemotron runs far more
    per-step dispatches, so the per-step `ggml_init`/build/`sched_reset`/alloc/free
    overhead (fixed per step) dominates when there are many steps. **Lessons:** (a)
    a per-step ggml dispatch pattern's cost is `n_steps × per-step-overhead` — it
    can be neutral on a short decode and a big regression on a long one; measure on
    the backend with the MOST steps, not the fewest. (b) This is exactly why the
    real fix is a **persistent graph** (build once, reuse — amortises the per-step
    build/alloc) and/or in-graph argmax; the naive per-step version is a
    correctness-validated stepping stone, not the shippable-default perf win.
    Both backends stay gated (`PARAKEET_GGML_DECODE` / `NEMOTRON_GGML_DECODE`),
    default cblas; the Kaggle P100 A/B covers both.

32. **The persistent-graph decoder confirms LEARNING 31 and WINS on M1 — build the
    step graph once, reuse it.** `core_rnnt_ggml::Decoder` builds the predictor +
    joint graphs once, gallocr-allocates each once on the backend, and dispatches
    sched-free per step (tensor_set → `ggml_backend_graph_compute` → read),
    amortising the per-step `ggml_init`/build/`sched_alloc` that killed the naive
    version. Measured on M1 Metal (jfk, transcript-identical, RNNT_GGML_PERSTEP to
    A/B): **nemotron decode cblas ~510-806 ms | persistent ~261-307 ms | per-step
    ~318-421 ms** — persistent is ~2× faster than cblas AND beats per-step,
    fixing the LEARNING-31 regression. Parakeet (short decode) is within noise for
    all three (per-step overhead is small when there are few steps — consistent
    with 31). **Lessons:** (a) for a reused per-token/-frame GPU graph, persistent
    gallocr + sched-free `ggml_backend_graph_compute` is the pattern; per-step
    rebuild is only a correctness scaffold. (b) It beats M1's *fast* Accelerate
    cblas, so on P100 (slow OpenBLAS) the win should be much larger — but flip the
    default only after the clean P100 bench (the noisy M1 load here inflates the
    absolutes; the ordering is stable). (c) Watch the §234 gallocr aliasing gotcha
    — re-set ALL inputs before every compute (state + token/proj each step do).

33. **P100 A/B CLOSES the §232 transducer decode gap — persistent ggml decode is
    5-12× faster than cblas, DEFAULT FLIPPED.** The Kaggle P100 (sm_60) run
    (`tools/kaggle/parakeet-ggml-decode-ab`, both transcript-IDENTICAL): parakeet
    decode **763.5 → 145.2 ms (5.26×)**, nemotron decode **2589.3 → 209.1 ms
    (12.38×)**. This is the win LEARNING 30 predicted (Kaggle's slow OpenBLAS cblas
    vs the GPU) and 32 set up (persistent graph). Default flipped to persistent
    ggml decode **when the decode backend is a GPU** (`!ggml_backend_is_cpu(
    ctx->backend)`), cblas on CPU. **Two gotchas that mattered for the flip:**
    (a) detect GPU with `ggml_backend_is_cpu(backend)`, NOT `backend !=
    backend_cpu` — parakeet's `pick_backend(false)` returns a *distinct* CPU
    backend instance, so the `!=` test mis-enabled ggml on `--no-gpu` (caught by
    the decode-timer printing "(ggml)" under `--no-gpu`). (b) The whole arc
    validates the "build it, gate it, let the TARGET hardware judge, then flip"
    discipline (29→33): per-step was a stepping stone; persistent won; P100
    confirmed; only then flip. nemotron's 12× shows the payoff scales with decode
    length (LEARNING 31's step-count point, inverted once the per-step overhead is
    gone).

34. **The P100 decode win is a slow-OpenBLAS effect — flipping "ggml when GPU"
    REGRESSED Metal.** After flipping (LEARNING 33), a total-RTF check on M1
    caught it: parakeet total ~16× cblas vs ~11× ggml — cblas FASTER on Metal.
    Apple Accelerate cblas does the small transducer matmuls faster than the
    per-step GPU decode dispatch, so the P100 5-12× (vs slow OpenBLAS) does NOT
    generalise to Metal. Fix: gate the default to CUDA/Vulkan, keep cblas on
    Metal + CPU — `#if defined(GGML_USE_METAL): if
    (ggml_backend_is_metal(backend)) ggml_dec = false;` (env override still forces
    either way). **Two lessons:** (a) a GPU-vs-CPU "win" measured on one platform's
    BLAS must be re-checked on the others before a blanket flip — the CPU baseline
    quality (Accelerate vs OpenBLAS) flips the verdict (ties back to LEARNING 30).
    (b) `ggml_backend_name()` for Metal returns **"MTL0"**, not "Metal" — a
    `strstr(name, "Metal")` check silently never matches; use
    `ggml_backend_is_metal()` (guarded by `GGML_USE_METAL` so CUDA builds compile
    it out). Caught only because I measured TOTAL RTF, not just the decode A/B.

35. **A strided `ggml_get_rows` index tensor aborts on CUDA but silently works on
    CPU/Metal.** Enabling the dia TTS GPU path, the encoder + cross-attn ran fine
    on a P100 but the AR decoder produced **0 tokens** and aborted with
    `GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type)) failed`.
    `build_dia_decoder_embedding` built a per-codebook index by hand-striding a
    view (`view->nb[0] = n_output_heads * elsize`) and passing it straight to
    `ggml_get_rows`. CUDA's `get_rows` kernel requires the **index tensor
    (`src1`) to be contiguous in dim 0** (`nb[0] == type_size`); the CPU and Metal
    kernels tolerate an arbitrary stride, so it worked on both dev boxes and the
    bug only surfaced on the Kaggle CUDA A/B. Fix: `ggml_cont` the index before
    `get_rows` — output-neutral, negligible cost, correct on every backend (also
    covers Vulkan, same requirement). **Two lessons:** (a) any `ggml_get_rows`
    whose index comes from a `view`/`permute`/hand-set `nb[]` must be `ggml_cont`'d
    before it touches a GPU — audit for `nb[0] =` assignments feeding `get_rows`
    (dia was the ONLY offender across src/; the other ~126 call sites pass
    contiguous I32 or offset-only `view_1d` slices, which are fine). (b) "correct
    on CPU AND Metal" is NOT sufficient GPU validation — CUDA has stricter
    contiguity asserts on several ops; the decoded-output roundtrip must run on a
    real CUDA box (Kaggle) before a GPU default flips there. This is the mandated
    CUDA A/B (rule #4/#5) earning its keep: it stopped dia shipping empty audio on
    CUDA. Ties back to the GPU-portability gotchas in the dev guide.

36. **A benchmark RTF is meaningless until you prove it did the work — a
    crash/no-op mints a fake "win."** The issue-#81 fleet bench reported CrispASR
    parakeet-ctc at "102.4× / 127.7× RT" and moonshine at "103×" — all bogus. Root
    cause: the harness timed a CLI subprocess and computed `rtf = audio_dur /
    walltime` with **no return-code or transcript check**, and the parakeet-CTC
    model was pointed at the *transducer* backend, which rejects it and exits in
    ~0.5 s. `audio_dur / 0.5 s` = a spectacular fake RTF. The tell was in the log:
    the "55 s" run took the SAME ~0.5 s as the "11 s" run (fixed time ⇒ not
    processing the audio; real inference scales with duration). Three defenses,
    now standard in the kernel: (a) **FAIL-guard** — a non-zero exit or empty
    transcript is recorded as FAIL, never timed; (b) **proof-of-work word count** —
    the 55 s clip is the 11 s clip ×5, so a real transcript must have ~5× the words
    (this is how the onnx-CUDA **220×** was confirmed REAL — 22 words → 110 words —
    vs the failed-load fakes); (c) **per-shape warmup + median + absolute ms** —
    the same onnx run first showed "174× / 2.3×" (short fast, long "collapsed"),
    which was a cold-CUDA-JIT artifact on the un-warmed 55 s shape, not an O(T²)
    blowup. Also: quote absolute time next to RTF — a 0.06 s denominator magnifies
    noise. And keep the comparison honest about asymmetry: CrispASR was timed as a
    fresh subprocess **including model load per call** while onnx loaded once, so
    short-clip RTF favours onnx; the load-amortised long column is the fair one.
    Net #81 truth (real varied audio, load-excluded): onnx-asr on the **GPU** beats
    CrispASR on parakeet (ctc 207× / tdt 112× vs CrispASR 25× / 35×), while CrispASR
    beats onnx on **CPU** (25-35× vs 4.6-4.8×). (d) **Never loop one short clip to
    fake "long" audio** — a 55 s clip made of jfk×5 lets the model/CUDA caches reuse
    work across the identical repeats and inflates the RTF (onnx-CUDA 220×→**207×**
    on real varied 134 s LibriSpeech). Use genuinely varied speech (concatenate
    DISTINCT utterances). (e) Report the engine's **load-excluded** RTF when
    comparing a subprocess-per-call engine against an in-process one — CrispASR's
    own CLI RTF (excludes load, #19) was 25× vs 18-20× from subprocess wall.

37. **`kaggle kernels output` is page-capped at 500 files and does NOT
    auto-continue — anything that sorts late is unreachable.** Refreshing the
    `crispasr-ccache` dataset requires pulling the kernel's `ccache.tar` back out,
    but the kernel `git clone`s the whole repo into `/kaggle/working` AND ccache
    wrote a loose `.ccache/` tree there — thousands of files that sort before
    `ccache.tar`, filling page 1 forever, so `ccache.tar` never downloaded and
    every build stayed cold (0 % hits). Two fixes: put the git clone under
    **`/kaggle/temp`** (not `/kaggle/working`) and relocate **`CCACHE_DIR`** out of
    `/kaggle/working` too, so the only ccache artifact left in the output is the
    single `ccache.tar` (reachable in page 1). After that the documented `kaggle
    datasets version` refresh works and the next build warmed to **92.9 % hits**
    (~3 min vs ~21 min cold). Lesson: keep `/kaggle/working` down to just the
    artifacts you need to retrieve; stage the repo, models, and build dirs in
    `/kaggle/temp`. (And `onnxruntime-gpu`: pin the **CUDA-12** wheel `==1.19.2` on
    Kaggle P100 — the default now links libcudart.so.13 and ImportErrors on 12.8;
    import-guard the onnx phase so it can't crash the whole run.)

## Multi-surface dispatch & long audio (issue #257 + improvements program)

38. **A fix to a backend must land in THREE places, not one — the CLI adapter,
    the HTTP server, AND the session C-ABI (`crispasr_c_api.cpp`).** The session
    reimplements each backend's transcribe inline; it does NOT call the CLI
    `CrispasrBackend` adapter (dev-guide HARD RULE #6). Issue #257's segmentation
    fix had to be applied in all three; the JA-by-vocab-size misdetection was
    patched in ~5 spots. The server is a fourth wrinkle: it uses the *adapter*
    (`backend->transcribe`) but had its own slicing that ignored
    `CAP_INTERNAL_CHUNKING`, so it per-slice-transcribed a full-attention encoder
    until the CLI's `backend_self_chunks_on_explicit` gate was mirrored into it.
    The durable fix for the class is to HOIST the orchestration into the library
    (`parakeet_orchestrate.{h,cpp}`) so every surface calls one implementation —
    the CLI adapter dropped 310 LOC becoming a thin wrapper. Before hoisting a
    backend speculatively, AUDIT first: `tests/test-surface-parity.sh`
    (`CRISPASR_PARITY_BACKEND=<be>`) proved only parakeet actually diverged
    CLI-vs-session; qwen3/moonshine/nemotron already agreed, so their ~200-line
    inline blocks did NOT need the (risky) refactor.

39. **The parity harness must compare TOTAL concatenated content, not
    per-segment, and be punctuation/case-insensitive.** Two traps: (a) the CLI
    ran with `--no-punctuation` (which *strips* a model's native punctuation)
    while the session API has no equivalent → punctuation-only "failures" that
    are cosmetic, not dispatch bugs — normalize punctuation/case out. (b) On long
    audio the CLI dispatcher and the session auto-chunker cut at DIFFERENT energy
    minima, so segment COUNTS legitimately differ (CLI 2 vs session 3) while the
    transcript is identical — compare the joined transcript with a word-overlap
    threshold, not segment-by-segment. Only 16 kHz clips are safe to feed both
    surfaces raw; for other rates resample ONCE to a shared 16 kHz WAV or the two
    resamplers produce different mel and the comparison is meaningless.

40. **The session `crispasr_session_transcribe` is a LOW-LEVEL "transcribe this
    buffer" primitive — it does NOT auto-chunk long audio (parakeet is the lone
    exception, with bespoke inline chunking).** So short-segment models (moonshine,
    whisper) degrade and HANG on one long pass, while the CLI/server add
    dispatcher chunking on top. Fix (`transcribe_autochunk`, gated
    `CRISPASR_SESSION_AUTOCHUNK`, default on): slice long audio at energy minima,
    transcribe each piece, shift timestamps to absolute. Verified moonshine/60 s:
    1 seg / hung → 3 segs / completes. Gate off backends that self-chunk
    (parakeet/reazonspeech), the `return_logits` path (per-slice CTC grids can't
    merge), and explicit chunk requests.

41. **Chunking a loop-prone greedy decoder EXPOSES the loop; fix it in two layers
    — cleanup AND decode-time break.** A hard slice sends moonshine's greedy loop
    into a short token cycle ("I'm sorry" ×15). Post-hoc `core_ngram::fix_loops`
    (which cohere/granite/glm adapters already apply, moonshine did NOT) cleans the
    output. But the decoder still BURNS every step generating the loop before it's
    trimmed — a decode-time break (`core_repeat::tail_is_repetition`: a period-≤8
    block repeated ≥4× stops generation) cut moonshine/60 s from 57 s → 11 s (5×),
    identical output. Both are gated, default-on (a runaway loop is never wanted).

42. **The O(T²) memory estimate for a heuristic gate can be validated against a
    reported real allocation.** parakeet's single-pass rel-pos bias
    (`parakeet_est_singlepass_peak_mb`, coeff 8.0) estimated **1931 MiB** at
    T≈2812 / 8 heads — matching the reporter's measured **1911.98 MiB** cudaMalloc
    almost exactly. So a proactive `CRISPASR_PARAKEET_VRAM_BUDGET_MB` policy can
    pick streamed BEFORE allocating the bias it can't afford, layered over the
    reactive OOM fallback. (CUDA no-OOM proof and the server worker-pool GPU
    concurrency proof can't run on M1 — they ship as Kaggle kernels under
    `tools/kaggle/{parakeet-mem-policy-cuda,server-workers-cuda}/`.)

43. **Server request concurrency (`CRISPASR_SERVER_WORKERS=N`) is workload-bound,
    not a free win — and unsafe unless the SHARED state is pooled too.** `model_mutex`
    guards not just the backend but the non-re-entrant post-processing contexts
    (punctuation, truecaser, LID model, aligner), so a pool must route only
    "pure-ASR" requests (explicit language, no aligner, no post-processing) to
    pooled workers and keep the rest serialized. And the throughput: on an M1 CPU
    with a memory-bandwidth-bound model, 2 concurrent requests were *slower* than
    serial (62 s vs ~32 s — the instances contend for bandwidth). It only helps
    where a single request under-utilises the box (spare cores, an unsaturated
    GPU, small models). Kept default-off; report the null honestly rather than
    claim a speedup that isn't there.

44. **A concurrent commit can add a NEW git submodule — a rebased worktree then
    fails `cmake` config with "Cannot find source file" until you
    `git submodule update --init --recursive <path>`.** Saw it when a `c2pa-audio`
    submodule landed on main mid-session; the worktree's build.ninja regen failed
    on the missing `third_party/c2pa-audio/src/*.cpp`. Also: heavy concurrent I/O
    on the external SSD wedges *git itself* (2-min timeouts on `git status`), not
    just model access — the tell it's the disk, not a hang.

45. **A new path whose FIRST A/B regressed on one clip → keep it GATED (off), not
    revert (F4).** Per-backend session chunk window: the hypothesis (short-segment
    models want a sub-30 s slice, mirroring the CLI's `vad_slice_cap_seconds()`)
    REGRESSED the one long clip measured — moonshine/60 s song CLI-vs-session
    content overlap `15 s→0.58 | 20 s→0.56 | 30 s→0.75 | 40 s→0.75` (more slices =
    more chunk-boundary artifacts on hard audio; 30 s is the plateau). The reflex
    was to revert as "no measured benefit" (HARD RULE #4). Corrected: #4 forbids
    shipping an unverified path AS DEFAULT — it does not mean delete it. A one-clip
    regression is evidence to gate-OFF, not to erase; the path may win on other
    clips/models and flipping the default is then a one-liner. Shipped opt-in
    (`CRISPASR_SESSION_PERBACKEND_CHUNK=1`, default flat 30 s). Codified as
    dev-guide A/B rule 3a. Implement the decision pure with the gate as a PARAMETER
    (`session_default_chunk_seconds(backend, perbackend_enabled)`) so both modes
    unit-test without env.

46. **firered's greedy decoder runs away exactly like moonshine — but only greedy,
    and the audit stops there (F1/F6).** Extended the `core_repeat` decode-time
    break beyond moonshine, evidence-gated. On the loop-prone 60 s song, firered at
    `--beam-size 1` SATURATES `max_len=150` with a period-1 cycle (`OOH ×35`),
    burning ~350 s, and firered's CLI adapter has no `core_ngram::fix_loops` so the
    garbage tail reaches the output — a speed AND quality bug. Wired
    `core_repeat::tail_is_repetition` into the `beam_size==1` branch ONLY (beam=3,
    the default, self-terminated at 59 tokens → not wired, Phase 1b), gated
    `CRISPASR_FIRERED_NO_REPEAT_BREAK` default on. A/B on the SAME binary (token
    count is the load-invariant proof, not wall-clock on a contended box): break
    OFF → 150 tokens/`OOH ×35`, break ON → 59 tokens/`OOH ×4`, coherent content
    byte-identical. Audit discipline held: the OTHER guard-less greedy backends
    (glm-asr, cohere-transcribe) transcribed the same song CLEANLY (EOS-terminated,
    no saturation) → NOT wired; kyutai_stt has no local model + a different
    streaming-decoder shape → NOT wired. Only wire where the runaway is DEMONSTRATED.

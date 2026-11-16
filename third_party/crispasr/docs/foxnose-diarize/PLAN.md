# #324 — FoxNoseTech/diarize port

Issue #324 (external reporter) asks for FoxNoseTech/diarize as an alternative
to pyannote. It is not a model port: it is ~1,350 lines of Python glue over
pip packages. The genuinely new work was one embedding model plus ~800 lines
of numerics.

## NOW — active work

Embedder, clustering, smoothing, pipeline, DER harness, CLI wiring and
word-aligned segment splitting all landed. **Automatic speaker counting is the
open problem** (below).

The old "not yet done" list here was stale — re-verified 2026-08-03, all five
had in fact landed:

| claimed missing | actual state |
|---|---|
| WeSpeaker GGUF upload + CC-BY attribution | uploaded; `api.model_info("cstr/wespeaker-resnet34-lm-GGUF").cardData["license"]` returns `cc-by-4.0` |
| THIRD_PARTY_NOTICES entry | present (7 WeSpeaker mentions) |
| docs/architecture.md section | present (6 foxnose mentions) |
| session ABI | `foxnose_embedder_path` in `crispasr_c_api.cpp` (`crispasr_diarize_opts_abi`) |
| registry auto-download | `{"wespeaker", "wespeaker-resnet34-lm.gguf", …}` in `crispasr_model_registry.cpp` |

Only **real-audio DER** was genuinely open, and it is now the root `PLAN.md`
NOW item (§1: route the pyannote+embedder path through the same
`core_spectral::cluster_speakers` that gets foxnose to 7.32%).

### Post-release server fixes (reporter follow-up on v0.8.25)

The reporter hit two SERVER-only defects — neither reachable from the CLI,
both instances of the multi-surface dispatch trap (the server re-implements
the runner's diarize orchestration instead of calling it).

1. **Transcript loss around every speaker change.** The CLI diarizes inside
   its per-slice loop, handing the slice its own segment vector; the server
   transcribes all slices first and then re-walks the merged list, giving each
   slice a copied sub-range. `crispasr_apply_diarize` **grows** that sub-range
   whenever it splits a segment at a speaker turn (pyannote, foxnose and
   sherpa all do word-range splitting), and the copy-back wrote back only the
   element count it started with — so every sub-segment past the first was
   dropped. One segment split three ways lost two thirds of its text. With VAD
   the slices are short and often hold a single segment, so the loss is
   "large portions"; without VAD a 30 s chunk holds many segments and only the
   ones straddling a turn are lost, which is why the reporter saw *"almost"
   full transcription, some parts where there are overlapping speakers /
   laughter didn't get transcribed* — those parts ARE the split ones.
   Fixed by hoisting the re-walk into `crispasr_diarize_merged_by_slice`
   (`crispasr_diarize_cli.{h,cpp}`), which rebuilds the list instead of
   copying a fixed count back. Guarded by `tests/test-diarize-slice-rewalk.cpp`
   (`[issue324]`), whose last case drives the REAL splitter via a hand-built
   global-sherpa cache so the guard is anchored to production behaviour.
2. **Foxnose ran per slice on the server.** `diarize_foxnose_global` was never
   set there and `crispasr_apply_foxnose_global` was never called, so each VAD
   slice clustered independently — numbering restarted at 0 every few seconds
   and the WeSpeaker embedder was re-resolved per slice. The server now
   mirrors `crispasr_apply_global_speaker_stages`: stand the per-slice path
   down, run one global pass, and skip the TitaNet remap when it owns the
   labels.

## What was reused vs. built

| stage | source |
|---|---|
| VAD | already in-tree (Silero) — not rebuilt |
| speaker embedding | **new**: WeSpeaker ResNet34-LM, `src/wespeaker.{h,cpp}` |
| clustering | **new**: `src/core/spectral_diarize.{h,cpp}` |
| temporal smoothing | **new**: `src/core/diarize_smooth.{h,cpp}` |
| orchestration | **new**: `src/core/foxnose_pipeline.{h,cpp}` |
| metric | **new**: `src/core/der.h` |

## Licensing

The repo stays MIT. Three separate questions, only one a constraint:

* **WeSpeaker weights are CC-BY-4.0**, not Apache-2.0 as FoxNose's README
  states. Weights are not code, so this does not touch our source licence —
  but redistributing the GGUF REQUIRES attribution in the model card and
  THIRD_PARTY_NOTICES.txt.
* **FoxNose and wenet-e2e/wespeaker code are Apache-2.0.** Apache is
  compatible with MIT but does not become MIT; copied expression would stay
  Apache and make the repo mixed-licence.
* **Clean-room keeps it uniformly MIT**, and here that was nearly free:
  `clustering.py` is a sequence of scikit-learn CALLS, not implementations, so
  there was nothing to translate. What was taken is the recipe and the tuned
  constants — ideas and parameters, not copyrightable expression. The upstream
  is imported by the reference dumper as an ORACLE only, never vendored.

## Acceptance

Bit-exact sklearn parity is unachievable and always would be: k-means++
seeding, GMM init and the ARPACK eigensolver all ride sklearn's RNG stream.
So the gates are known-answer unit tests plus DER, not label equality.

WeSpeaker embedder, per-stage vs the upstream oracle on samples/jfk.wav:

| stage | cos_mean |
|---|---|
| fbank | 0.999999 |
| stem / layer1-4 | 0.99997 - 0.999995 |
| stats | 0.999999 |
| embedding | 0.999997, cosine(emb, ref) = **0.99999747** |

Clustering + smoothing + pipeline: **375 assertions over 57 hermetic cases**
(no model, no audio, no network), including an end-to-end synthetic
2-speaker timeline at **DER 0.0**.

## ⚠ Real-data benchmark — and a reversed conclusion

**8 VoxConverse dev files, human labels, 0.25 s collar, optimal 1:1 mapping.**
Pooled over 1109 s of scored reference speech:

| system | miss | FA | confusion | **DER** |
|---|---|---|---|---|
| upstream Python `diarize` 0.1.2 | 0.6 | 4.7 | 29.0 | **3.1 %** |
| this port, `bic` (upstream estimator) | 0.0 | 26.1 | 32.7 | **5.3 %** |
| this port, `eigengap` | 0.0 | 26.1 | 101.3 | **11.4 %** |

### The eigengap default was wrong and has been reverted

An earlier revision made eigengap the default on the strength of five
synthetic blob configurations (5/5 exact vs BIC's 4/5) and one 31.5 s clip.
Both were unrepresentative — `samples/multispeaker.wav` is the same speech
re-read by different speakers, an unusually easy case — and the real benchmark
reversed the verdict.

Eigengap systematically UNDER-counts on real speech:

    reference speakers   4  7  2  5  5  4  4  5
    eigengap             3  5  2  3  3  2  2  3
    bic                  4  6  3  5  4  3  2  5

and the confusion term triples. It is retained behind
`CRISPASR_DIARIZE_COUNT=eigengap` — it is genuinely better on well-separated
data and costs less — but it is NOT the default, and synthetic evidence must
not be used to make it one again. The unit test now pins the default.

### With VAD, this port is at parity with upstream

The first benchmark handed our pipeline WHOLE FILES as a single speech region
while upstream ran Silero VAD first, charging us 26.1 s of false alarm the real
CLI path never incurs (it takes the caller's ASR/VAD segments). Re-run with the
same Silero VAD and upstream's parameters (threshold 0.45, min speech 200 ms,
min silence 50 ms, pad 20 ms):

| system | miss | FA | confusion | **DER** |
|---|---|---|---|---|
| upstream Python `diarize` 0.1.2 | 0.6 | 4.7 | 29.0 | **3.07 %** |
| **this port, `bic` + Silero VAD** | 0.0 | 9.0 | **26.5** | **3.18 %** |
| this port, `bic`, no VAD | 0.0 | 26.1 | 32.7 | 5.27 % |

0.11 points apart, and our speaker CONFUSION is actually lower (26.5 s vs
29.0 s) — the residual gap is false alarm, not diarization. Estimated speaker
counts now differ on one file of eight:

    reference   4 7 2 5 5 4 4 5
    this port   4 6 2 5 4 4 2 5
    upstream    4 7 2 5 4 4 2 5

### Reproducing the benchmark

```bash
# 1. audio + human labels (one dev shard, 44 files; 8 used here)
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download('diarizers-community/voxconverse','data/dev-00000-of-00005.parquet',
                repo_type='dataset', local_dir='voxconverse')
EOF
# 2. the upstream reference implementation, in its own venv
python -m venv foxvenv && ./foxvenv/bin/pip install diarize==0.1.2
# 3. score both with tools/der_score.py (0.25 s collar, optimal 1:1 mapping)
```

## Blueprint parity — measured end to end

The upstream Python pipeline (`pip install diarize==0.1.2`) was run on the same
`samples/multispeaker.wav` and compared against this port.

**With the speaker count pinned to 2 on both sides**, the speaker assignment is
identical and the boundaries agree within ~1 s:

| upstream Python | this port |
|---|---|
| 0.30-10.60 SPEAKER_00 (3 pieces, VAD-split) | 0.00-10.50 SPEAKER_00 |
| 11.50-15.60 SPEAKER_01 | 10.50-15.90 SPEAKER_01 |
| 16.30-26.60 SPEAKER_00 (3 pieces) | 15.90-26.70 SPEAKER_00 |
| 27.50-31.50 SPEAKER_01 | 26.70-31.50 SPEAKER_01 |

Scored with the DER harness (0.25 s collar, optimal 1:1 mapping), treating the
upstream output as reference:

    missed        0.00 s
    false alarm   1.05 s
    confusion     0.00 s
    DER           3.93 %

**Zero speaker confusion**: wherever both assign a speaker, they agree. The
entire residual is false alarm, and it is explained — upstream runs its own
Silero VAD and drops silence gaps, while this port tiles the caller's speech
regions contiguously. That is a difference in where the speech segmentation
comes from, not a diarization disagreement.

**On automatic counting this port is better than upstream.** On the same clip
upstream emits 11 speakers across 25 segments (its default `max_speakers=20`);
this port emits 2. The gated `CRISPASR_DIARIZE_COUNT=bic` path reproduces
upstream's failure mode (7-8 speakers), which is what confirms the port is
faithful — the improvement comes from the eigengap switch, not from a
divergence in the shared parts.

### What would still settle it properly

A DER number on labelled audio. There is none in the repo and none in
`cstr/crispasr-regression-fixtures` — this was checked. VoxConverse dev (what
upstream benchmarks on) needs a ~1-2 GB audio download plus RTTM wiring, which
is its own task. Until then the honest recommendation is a **conservative
`--diarize-max-speakers` default (4-6, not 20)**, and pinning
`--diarize-num-speakers` when the count is known.

## Wiring

`--diarize-method foxnose --diarize-embedder <wespeaker.gguf>`, with
`--diarize-max-speakers` / `--diarize-num-speakers`.

The method plugs into the existing `crispasr_diarize_segments` contract: the
caller's segments ARE the speech regions (they come from ASR/VAD upstream), so
FoxNose deliberately runs no VAD of its own — re-segmenting would duplicate
work and desynchronise labels from the segments they attach to.

Segment splitting IS implemented (`split_segments_on_foxnose_turns`): the
method now returns its derived turns through `crispasr_diarize_segments`'s
`out_turns`, and the CLI splits any caller segment spanning several speakers
at word-aligned boundaries. It reuses the same `group_words_into_speaker_runs`
grouping and sub-segment emission as the pyannote splitter — only the per-word
labelling differs (turn-interval lookup instead of posterior scoring).
Measured effect on `samples/multispeaker.wav`: before, both ASR segments
collapsed to `(speaker 0)`; after, the first slice correctly reads
0 -> 1 -> 0 across the ~11 s boundary.

### Cross-slice speaker identity (fixed)

Per-slice diarization cannot give consistent identities: each slice clusters
independently and restarts numbering at 0, so `speaker 0` in one slice is a
different person from `speaker 0` in the next. On `samples/multispeaker.wav`
(2 slices) the final turn came out `speaker 0` where the whole-file pipeline
says SPEAKER_01.

Fixed by running FoxNose in ONE global pass after transcription
(`crispasr_apply_foxnose_global`), using the final segment list as its speech
regions. The pyannote path solves the same problem with a pre-computed
posterior cache (#107); FoxNose does it afterwards instead, because it needs
the segments as speech regions and they do not exist beforehand. The per-slice
path stands down via `params.diarize_foxnose_global`, so the embedder is
loaded once rather than per slice.

Result on `samples/multispeaker.wav` — CLI output now matches the whole-file
pipeline's turns exactly:

| CLI | pipeline truth |
|---|---|
| 0.28-10.84 speaker 0 | 0.00-10.50 SPEAKER_00 |
| 11.64-15.88 speaker 1 | 10.50-15.90 SPEAKER_01 |
| 17.04-26.80 speaker 0 | 15.90-26.70 SPEAKER_00 |
| 26.52-31.52 speaker 1 | 26.70-31.50 SPEAKER_01 |

Note this runs on the `crispasr_run` unified path. The legacy `cli.cpp`
whisper path still diarizes per slice; it is the same fallback situation the
pyannote cache has there.

## Env gates

| var | effect |
|---|---|
| `CRISPASR_DIARIZE_BIC_WINDOW=1` | score silhouette only in `[k-2, k+3]` around the BIC anchor instead of the full `[min,max]` range (the default) |
| `CRISPASR_WESPEAKER_BENCH=1` | per-stage embedder timings |
| `CRISPASR_WESPEAKER_DEBUG=1` | embedder diagnostics |

## Performance — what was measured, and what not to bother trying

All numbers: M1, Release, `esrit.wav` (215.1 s, 3 VAD regions, 352 embedding
windows), through the real `crispasr --diarize-method foxnose` path. The
machine was noisy early on, so every A/B below was re-run interleaved on a
quiet machine; the paired numbers repeat to ~1%.

Where the time goes on a 215 s file (~45 s wall, ~4.8x realtime):

| stage | share |
|---|---|
| WeSpeaker ResNet34 forward | ~70% (94% of the embedder) |
| speaker-count estimation (GMM/BIC + silhouette) | ~12% (58 s -> 51 s when `--diarize-num-speakers` pins it), before the parallel sweep below |
| Kaldi fbank | ~4% |
| VAD, clustering, smoothing, I/O | remainder |

**Speaker counting was single-threaded and recomputed its inputs (2026-08).**
`cluster_speakers` rebuilt the same O(n^2 d) cosine affinity up to ~10 times
per call — once inside the `run(k)` lambda and at three more sites — and then
ran both k-sweeps serially: the GMM/BIC sweep (`n_init=5`, `max_iter=300` per
k) and the spectral+silhouette sweep (a dense Laplacian, a subspace iteration
and `10x300` k-means per k, for every k in `[2, max_speakers]`). Every k is
independent — `gmm_bic` seeds its own RNG per init, and the spectral runs take
the same `seed` per candidate — so the sweeps parallelise with the reductions
left serial in ascending k, which is what keeps the tie-breaks (and therefore
the chosen count) bit-identical.

The fan-out is a file-local `parallel_tasks` in `spectral_diarize.cpp`, kept
local to match every other parallelism helper in the tree
(`firered_parallel_for`, `marblenet_parallel_for`, `ov_parallel_for`, and the
inline loops in `core/mel.cpp` and `core/foxnose_pipeline.cpp` — none of them
sit in a shared header). It differs from those four in handing out task
INDICES from an atomic counter rather than splitting [0, n) into contiguous
per-thread chunks: they parallelise uniform-cost items like audio frames,
whereas a k=8 spectral run costs many times a k=2 one, so a static split
leaves the thread holding the cheap candidates idle. Unifying all five behind
one helper is a reasonable cleanup, but it has its own testing surface (two
VAD models plus omnivoice) and does not belong in a diarization perf change.

⚠ Note for anyone extending this: neither of the "just use the standard/
platform" routes works here, and both were tested rather than assumed.
`std::execution::par` is in-standard for core (C++17) but does not build on
the Apple toolchain — libc++ gates `<execution>` behind
`_LIBCPP_HAS_EXPERIMENTAL_PSTL`, and force-enabling it fails to LINK
(`__pstl::__libdispatch::__dispatch_apply` is not in the shipped dylib);
libstdc++ implements it over Intel TBB, which is nowhere in this tree. OpenMP
is used heavily elsewhere (~28 pragma sites, including `schedule(dynamic)` at
`mel_band_roformer.cpp:666`), but every site is `#ifdef _OPENMP`-guarded and
OpenMP is NOT found on the stock macOS toolchain, so a pragma-based sweep
would have delivered 0% of the speedup below on the machine it was measured
on — the same trap the firered-asr note in `src/CMakeLists.txt` records.

Measured on an M-series box, `-t 8`, esrit.wav (215 s), warm runs, arms
interleaved base/PR: ASR-only 2.39 s; foxnose end-to-end **10.82 s -> 9.63 s**,
i.e. **diarization delta 8.43 s -> 7.24 s (1.16x)**. Labels are bit-identical
(259 spectral assertions unchanged, shard DER identical per file at 7.32%).
The remaining estimation cost is real work, not redundancy: pinning the count
with `--diarize-num-speakers` still saves ~1 s on top of this.

### Levers that were tried and LOST — do not re-litigate without new evidence

| lever | result |
|---|---|
| **Metal / GPU** (`use_gpu = true`) | **2.0x SLOWER** — 89.6 / 89.6 s vs 45.1 / 44.6 s CPU, interleaved. The graph is submitted 352 times, once per 1.2 s window, and each one is tiny (80 mels x ~120 frames). Per-dispatch overhead and the host<->device round trip swamp the conv work. This is why `crispasr_diarize.cpp` deliberately does NOT set `cp.use_gpu`. It would likely flip if the windows were batched — see below. |
| **F16 conv kernels** | **2.2x SLOWER** — 297 vs 133 ms/window. Correctness is fine now (our fork's `00285218` removed the `src1->type == F32` assert; cosine 0.99999724 vs the oracle) and the GGUF drops 23.9 -> 13.3 MB, but ggml's CPU conv path is far slower on F16 input. The converter keeps 4-D kernels at F32 and says so. |
| **Persistent graph + `gallocr`** (the `bananamind_tts` / `beat_this` trick) | **~0.1% available.** Instrumented build/alloc/compute per call: **0.050 ms build, 0.049 ms alloc, 103.430 ms compute.** There is no per-call overhead to remove. The graph is already rebuilt-and-thrown-away for free. |
| **More threads** | wash. `-t 4` 44.72 / 44.58 s vs `-t 8` 45.05 / 44.60 s. The M1's 4 E-cores contribute nothing; the default `n_threads = 4` is already right. |

BLAS *was* not a factor on this path, and the reason turned out to be fixable:
the embedder ran on a plain `ggml_backend_cpu_init()` backend and never
scheduled to the BLAS backend even though the binary reports
`backends: cpu,metal,blas`. That was not a scheduler quirk — `ggml_conv_2d_direct`
emits ONE `GGML_OP_CONV_2D` node that does its im2col + GEMM internally, so
there is no `MUL_MAT` for the BLAS backend to claim (it supports `MUL_MAT` /
`OUT_PROD` and nothing else). Lowering the convs to explicit IM2COL + MUL_MAT
nodes and putting the BLAS backend in the scheduler is what the 2026-08 round
below does, and it is the single biggest win recorded on this model.

### The lever that is actually left: stop embedding the same audio twice

`kEmbeddingWindowSeconds = 1.2` with `kEmbeddingStepSeconds = 0.6` means every
sample except the first and last half-window goes through ResNet34 **twice** —
352 windows x 1.2 s = 422 s of audio pushed through the net for 211 s of
speech, exactly 2x by construction.

ResNet34 here is fully convolutional in time, so in principle the net could run
**once per VAD region** and TSTP could then pool over each window's slice of
the layer4 time axis. That is close to a 2x cut of the dominant 70%.

It is NOT a free refactor and must not be assumed equivalent:

* conv padding at a *region* edge is not the padding at a *window* edge, so
  embeddings near boundaries will change;
* layer4 is downsampled 8x in time, so a 1.2 s window is only ~15 frames there
  and window boundaries no longer land on frame boundaries;
* it changes memory from O(window) to O(region), which matters on long files.

So it needs a DER run on the VoxConverse shard (see the benchmark section
above), not a cosine check. A cheaper variant with the same shape: batch N
windows into one graph along `ne[3]` — the `parakeet` / `nemotron` batched
decode trick — which changes no arithmetic at all, improves CPU GEMM shapes,
and is the change most likely to make the GPU path win instead of lose.

### 2026-08 round — reaching Accelerate (different machine: M-series, `-t 8`)

Baselines on this box, esrit.wav (215 s) through the unified runner, warm
runs only, arms always interleaved base/PR/base/PR. ASR-only **2.39 s**.

⚠ This is a desktop, not a quiet bench, and the ratio moves with load — so
both conditions are recorded rather than the flattering one. Interleaving is
what makes them comparable at all; a non-interleaved arm on this machine is
worthless.

| condition | base delta | PR delta | speedup |
|---|---|---|---|
| loadavg ~1 (4 samples/arm) | 8.43 s | 5.96 s | **1.41x** |
| loadavg ~7, desktop apps busy (3 samples/arm) | 9.13 s | 6.77 s | **1.35x** |

Output JSON is byte-identical to base in every row below, and the 8-file
VoxConverse shard scores identically per file (mean 7.32%).

| lever | result |
|---|---|
| **im2col conv lowering + BLAS backend in the sched** (`CRISPASR_WESPEAKER_CONV`, now the DEFAULT) | **WON — 1.35-1.41x on the diarization delta** (see the table above), end-to-end 10.82 → 8.35 s quiet. See the corrected BLAS note above for why the direct-conv op could never reach Accelerate. Embeddings cosine 1.0 vs direct (`test_wespeaker_live.cpp` "im2col conv matches direct conv"), DER identical per file. |
| **Batch N windows along `ne[3]`** (`wespeaker_embed_batch`, `CRISPASR_DIARIZE_BATCH_EMBED`) | **CPU: LOST** — 12.1 → 14.9 s wall at 8 workers, and cost-neutral single-threaded (43.0 s vs 44.0 s of embed compute). ggml's CPU conv loops `ne[3]`, so fusing changes no GEMM shape, and 352 windows → 11 chunks starves the worker pool. The prediction above that batching would "improve CPU GEMM shapes" was WRONG for ggml's CPU conv — it is right for the GPU. **GPU: the enabler**, see the next row. Arithmetic-identical either way (cosine 1.0, batch-vs-loop live test); defaults ON only under GPU. |
| **Metal, revisited with im2col + batching** (`CRISPASR_WESPEAKER_GPU=1`) | **parity, not a win** — 8.5 s wall (batch 32, one GPU context) vs 8.4 s for the 8-worker CPU schedule. The 2.0x loss recorded above is GONE: im2col gets real `mul_mm` kernels instead of the naive scalar `kernel_conv_2d`, and batching cuts ~350 dispatches to ~11. Ships as an opt-in env; CPU stays the default. Mixed GPU-main + CPU-workers **aborts by construction** — workers borrow weights that now live in a Metal buffer — so `use_gpu` clamps the pool to one context. One-GPU-many-CPU would need dual-resident weights. |
| **Kaldi fbank SIMD/Accelerate** | **skipped on measurement** — 697 ms out of 43 s of single-threaded embed compute (~1.6%) on this box, not the ~4% the M1 table above shows. Below the effort line; the `kaldi_fbank.cpp` scalar mel projection and per-call FFT scratch remain unoptimised. |

**Which half does the work?** The lowering and the BLAS backend arrive
together, so they were separated with a throwaway probe that skipped the
`ggml_backend_init_by_name("BLAS")` call while keeping im2col. Interleaved,
3 samples/arm, loadavg ~3, esrit.wav, ASR-only 2.22 s:

| arm | end-to-end | diarization delta | vs direct |
|---|---|---|---|
| `direct` | 11.14 s | 8.92 s | — |
| `im2col`, BLAS backend **withheld** | 11.22 s | 9.00 s | **1.00x — no gain** |
| `im2col` + BLAS | 8.21 s | 5.99 s | **1.49x** |

So the lowering on its own buys **nothing**: ggml's CPU `MUL_MAT` (llamafile
tinyBLAS) is no faster here than the GEMM already inside `GGML_OP_CONV_2D`.
The entire win is Accelerate, and the lowering is purely the mechanism that
lets the scheduler hand it the GEMM. That also means the win is
platform-conditional — on a build with no BLAS backend this change is
performance-neutral, not a regression (outputs are byte-identical either way).

⚠ The `~70% / ~12% / ~4%` split at the top of this section is the M1 measurement
and is now stale for the embedder: with the conv lowering in, the ResNet
forward no longer dominates by that margin. Re-measure before using those
shares to pick the next lever.

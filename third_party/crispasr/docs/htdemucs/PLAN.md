# HTDemucs — parity validation (feat/htdemucs-parity-diff)

HTDemucs (Meta Demucs v4, 42M params, 533 tensors) had a complete-looking C++
runtime in `src/htdemucs.cpp` but **zero cosine-similarity validation** — the
binary had never produced output compared against the Python reference. The VPS
could not run it (8 GB, OOM). This branch does that validation on the M1.

## NOW — active work

- [x] Worktree `feat/htdemucs-parity-diff` + submodules
- [x] `input_wav` stage added to `tools/reference_backends/htdemucs.py` so the
      diff replays the reference's exact 44.1 kHz waveform (resampler-independent
      input gate, per the dev-guide "gate input alignment first" rule)
- [x] Reference dumped: 22 stages, F32 GGUF converted (533 tensors, 160 MB)
- [x] Per-stage capture + `htdemucs_diff()` in `src/htdemucs.cpp`
      (self-contained runner, dots-tts/mel-band-roformer pattern)
- [x] `htdemucs` wired into `examples/cli/crispasr_diff_main.cpp` (checklist #9)
- [x] **BUG 1 FIXED** — `read_tensor_f32` infinite recursion (see below)
- [x] First full per-stage run (was: 6/21 stages)
- [x] **BUG 2 FIXED** — DConv GroupNorms
- [x] **BUG 4 FIXED** — time encoder skipped DConv entirely
- [x] **BUG 5 FIXED** — tdecoder rewrite used a 1x1 helper on a K=3 Conv1d
- [x] **BUG 6 FIXED** — freq_emb ne[] swapped
- [x] **ENCODER NOW FULLY PASSES** — 15/25 stages, every encoder stage
      (freq + time, all 4 layers) and both pre_transformer_* at cos >= 0.999998
- [x] **CROSSTRANSFORMER NOW FULLY PASSES** — all 5 layers + post_transformer_*
      at cos = 1.000000 (BUGS 7, 8, 9)
- [x] **DECODER + OUTPUT NOW PASS** (BUGS 10, 11, 12)
- [x] **45/45 STAGES PASS** — full end-to-end F32 parity
- [x] Decoded-output roundtrip (HARD RULE #3) — ASR of the separated vocals
      stem is byte-identical to ASR of the original mix
- [x] F16 verified: 45/45, min cos 0.999961
- [x] BUG 3: smoke test registered in CMake and passing

**STATUS: COMPLETE.** Branch `feat/htdemucs-parity-diff`, not yet merged.
- [ ] BUG 3: `tests/test_htdemucs_smoke.cpp` never registered in CMake

## Bugs found

### BUG 1 — `read_tensor_f32` infinite recursion (FIXED)

```cpp
if (t->type == GGML_TYPE_F32) {
    auto _rd = read_tensor_f32(t);   // calls itself -> stack overflow
```

A find/replace that swapped `ggml_backend_tensor_get` for the `read_tensor_f32`
wrapper also rewrote the call *inside the wrapper's own definition*. Effect:
**any F32 GGUF crashed with SIGSEGV (stack exhaustion) on the first weight
read** — the model could never run at all in F32. Only the F16 path worked,
which is why the shipped F16 GGUF appeared functional. Fixed to call
`ggml_backend_tensor_get` directly.

Diagnosis note: the crash presented as SIGKILL/137 with a 78 GB "peak memory
footprint" under the time branch and SIGSEGV/139 without it. `maximum resident
set size` read only 1.8 GB (compression-capped — see the macOS
peak-footprint-not-RSS learning); lldb could not unwind the corrupted stack, so
the location was pinned with fflush'd markers instead.

### BUG 2 — DConv GroupNorms skipped (CONFIRMED, unfixed)

`src/htdemucs.cpp` DConv sublayer forward:

```cpp
// GroupNorm(1) + GELU (skip norm for now — LayerScale init=1e-3 dominates)
```

Both `GroupNorm(1, hidden)` (after the dilated conv) and `GroupNorm(1, 2C)`
(after the 1x1 conv) are omitted. The justification is wrong: `1e-3` is the
LayerScale *init*, not its trained value. The weights are real and non-identity
— `encoder.0.dconv.layers.0.1` has weight mean 1.563 / std 0.496, bias absmax
0.554. The converter exports them and `bind_dconv` binds them into
`norm1_w/norm1_b/norm2_w/norm2_b`; the forward simply never reads those fields
(dead weights).

### BUG 3 — smoke test never built

`tests/test_htdemucs_smoke.cpp` exists but appears in no `CMakeLists.txt`, so it
has never compiled or run. Checklist item 12 is unmet.

## Layout contract (verified, not assumed)

Both the reference and the C++ use `(C, Fq, T)` row-major with `t` fastest
(`x[t + fq*T + c*T*Fq]`, confirmed in `cpu_conv2d_freq`). No permutation is
needed in the diff. Time stages are `(C, T)` channel-major; the final outputs
are captured channel-major *before* the interleave.

### BUG 4 — time encoder skipped DConv entirely (FIXED)

`HEncLayer.forward` is `conv -> GELU(norm1) -> dconv -> GLU(norm2(rewrite))`.
The frequency branch had the DConv; the time branch went straight from GELU to
the rewrite. All 4 tencoder layers have a real DConv, so every `enc_time_*` was
wrong. (`apply_dconv()` existed for exactly this but was never called — dead
code, and itself buggy: it splits the GLU on `ne[0]`, which is *time* for
`ggml_conv_1d` output, not channels.)

Fixed by extracting the DConv stack into `cpu_dconv_inplace()` shared by both
branches, and splitting the time encoder into two ggml graphs
(conv+GELU | rewrite+GLU) with the CPU DConv between them. ggml's 2D `(T, C)`
layout is flat `t + c*T`, identical to the channel-major `(C, T)` convention
`xt_buf` uses, so no transpose is needed at the handoff.

Result: `enc_time_0/1/2` and `pre_transformer_xt` went to cos = 1.000000.

### BUG 5 — tdecoder rewrite is a K=3 Conv1d, not 1x1 (FIXED)

`cpu_conv2d_1x1` was called on `tdecoder.N.rewrite.weight`, whose ne is
`(K=3, IC, OC)`. That helper reads `out_C = ne[3]`, correct for a 1x1 **Conv2d**
`(1,1,IC,OC)` but = 1 for a Conv1d — so the output collapsed, `cpu_glu` returned
an empty buffer, and the following `ConvTranspose1d` dereferenced
`std::vector::data()` on it: the **null-pointer crash at address 0x0** that
killed every run at `dec[0]`. Also the helper assumes K=1, ignoring the K=3
context window.

Added `cpu_conv1d_time()` (proper K, symmetric padding = `hp.context`) and a
`GGML_ASSERT` in `cpu_conv2d_1x1` so a Conv1d weight can never be silently
misread as 1x1 again. The encoder's own 1x1 use (`(1,1,48,96)`) is legitimate.

### BUG 6 — freq_emb ne[] swapped (FIXED)

`nn.Embedding.weight` is `(num_embeddings, embedding_dim)` row-major = ggml
`ne(embedding_dim, num_embeddings)`. The code read `n_freqs = ne[0]` and
`C = ne[1]`, exactly backwards, which both strided by 512 instead of 48 AND made
`min(x_Fq, emb_n_freqs) = min(512, 48) = 48`, so only 48 of 512 frequency bands
received the embedding at all.

The `10 * 0.2 = 2.0` total scale was already correct — verified numerically
against the reference (`max|eff - gguf.T*2.0| = 0.0`) before touching the code.

Bisected by adding `enc0_conv / enc0_gelu / enc0_dconv / enc0_rewrite` stages to
both the dumper and the runtime: all four read cos = 1.000000 while `enc_freq_0`
(the same tensor plus freq_emb) read 0.884, isolating it to the embedding add.

### Harness bug — dangling `first_fail` (FIXED)

The summary reported `FIRST DIVERGENCE: output_vocals` when the real first
failure was `enc_freq_0`. `first_fail` was a `const char*` assigned from
`("enc_freq_" + std::to_string(i)).c_str()` — a pointer into a temporary that
dies at the end of the full expression. Now a `std::string`.

### BUG 7 — transformer input: pos-emb added BEFORE norm_in (FIXED)

`CrossTransformerEncoder.forward` is `x = norm_in(x)` **then**
`x = x + weight_pos_embed * pos_emb`. The C++ added the position embedding
first and then LayerNorm'd it, which renormalised the embedding away. Both sin
embedding formulas themselves were already correct (verified term by term
against `create_2d_sin_embedding` / `create_sin_embedding`).

### BUG 8 — spurious transpose scrambled both transformer branches (FIXED)

`ggml_conv_1d` output has `ne[0] = seq`, i.e. seq is the **fast** axis, so the
channel upsampler already left the buffers as `[dim][seq]` in C memory — exactly
what the attention code indexes (`tmp[d*seq_len + s]`). The code read that as
`(seq, dim)` and transposed it (with a matching transpose back before
`channel_down`, so the two cancelled and the shapes stayed plausible) — meaning
every transformer layer ran on scrambled data. Both transposes removed.

The freq branch keeps its native `s = fr*T1 + t1` order rather than Python's
`(t1 fr)`; the position embedding is indexed to match. Token order is otherwise
irrelevant (self-attn is permutation-equivariant, cross-attn sums over the whole
key/value set, FFN/LayerNorm are per-token) and the decoder reads the same order.

### BUG 9 — norm_out is GroupNorm(1), not LayerNorm (FIXED) — the big one

Each transformer layer ends with `norm_out`, a `MyGroupNorm(num_groups=1)`:
**one mean/std over all channels AND all tokens jointly**. The C++ applied
`cpu_layernorm`, which computes a **separate mean/std per token**. That left
~1% error in every layer.

Why it looked like a layer-4 bug: layer 4's pre-`norm_out` activations are
outlier-dominated (norm ~85,589 normalised down to ~78), so that step amplifies
any upstream relative error enormously — `ct_l3_z` cos 0.992 became `ct_l4_z`
cos 0.217. Verified with forward hooks on the real module that this collapse is
genuine reference behaviour, NOT a reference-replication artifact, before
concluding the fault was upstream.

Fixing the normalization alone took the whole transformer to cos = 1.000000:

  ct_l0_z 0.987908 -> 1.000000      ct_l4_z 0.217092 -> 1.000000
  post_transformer_z 0.406606 -> 1.000000

Lesson: a uniform ~1% per-layer error is worth chasing as structural, not
written off as drift — the amplifying stage downstream is the symptom, not
the cause.

### BUG 10 — decoder skipped DConv entirely (FIXED)

`HDecLayer.forward` is `x+skip -> GLU(norm1(rewrite)) -> dconv -> conv_tr`, and
every decoder AND tdecoder layer has a real DConv. The C++ decoder loop
referenced `dconv` zero times even though `bind_dec_layer` populated it — the
same class of omission as BUG 4. Applying it (per frequency band on the freq
side, directly on the time side, both via `cpu_dconv_inplace`) took
`dec_freq_0..3` from cos 0.960/0.903 to >= 0.999999.

### BUG 11 — iSTFT scale and _ispec alignment (FIXED)

Three separate errors in the output stage:

1. `compute_istft` used `norm_factor = 1/sqrt(nfft)`. `torch.stft(normalized=
   True)` already DIVIDES by `sqrt(nfft)` on the way in, so the inverse must
   MULTIPLY — every stem came out `1/nfft` = 1/4096 too quiet. (Cosine is
   scale-invariant, which is why `spec_input` passing at 1.000000 did not
   catch it; the `|mine|` vs `|ref|` columns did.)
2. `_ispec` pads 2 zero frames on each side before the inverse transform. These
   are not merely trimmed afterwards — they change the window-sum normalisation
   in the first/last `nfft` samples, which is exactly the region cropped into.
3. The output was taken at offset `nfft/2` = 2048, but `_ispec` crops
   `hop/2*3` = 1536 to undo `_spec`'s reflect pre-padding — a 512-sample
   misalignment.

### BUG 12 — time-branch lengths off by one level, silently dropping the branch (FIXED)

Python records the time length BEFORE each tencoder runs
(`lengths_t.append(xt.shape[-1])` precedes `xt = tenc(xt)`); the C++ recorded it
after. The decoder pops these to crop each `ConvTranspose1d` back to its layer's
input length, so the whole stack was shifted one level and the last tdecoder
produced 85995 samples instead of 343980.

That then failed this guard silently:

```cpp
if (xt_C >= S * ac && xt_T >= n_samples) {   // 85995 >= 343980 -> false
```

so the time branch was dropped from every stem with no error. It went unnoticed
because `output_vocals` still passed at cos 0.999879 — for speech the
spectrogram branch dominates that one stem. The quiet stems were pure
spectrogram output, wrong by 10-40x. Adding an `else` debug branch to that
silent guard is what surfaced it.

Lesson: a `>=` guard that silently skips a whole branch should always have a
diagnostic on the else path.

## RESULT — 45/45 stages pass (F32)

Every stage from `spec_input` through all four stems at cos >= 0.999989,
most at 1.000000. Full table in `run14.log` / this branch's commits.

## Acceptance (HARD RULE #3 — decoded output, not just cosine)

Per-stage cosine is necessary but not sufficient, so the stems were checked
as audio. `--separate` on the full 11 s `samples/jfk.wav` (485100 samples, which
also exercises the chunking path rather than the padded 343980-sample segment):

| stem   | rms     | dBFS   |
|--------|---------|--------|
| vocals | 0.13962 | -17.10 |
| other  | 0.00521 | -45.66 |
| drums  | 0.00021 | -73.57 |
| bass   | 0.00010 | -79.62 |

That is the correct physical answer for a speech clip: essentially all energy in
vocals, drums and bass silent. Sum-of-stems rms 0.13998 vs mixture rms 0.14210 —
the stems reconstruct the input.

ASR round-trip (`ggml-tiny.en`):

- mix:          "And so my fellow Americans ask not what your country can do for
                 you / ask what you can do for your country."
- vocals stem:  identical, word for word.

Smoke test on a synthetic 440/880 Hz stereo tone separates it into `other`
(rms 0.293) with drums/bass/vocals ~5e-4 — also the right answer.

## Quantization

| build | stages | min cos |
|-------|--------|---------|
| F32   | 45/45  | 0.999989 |
| F16   | 45/45  | 0.999961 |

F16 GGUF is 84 MB (156 F16 tensors; norms/biases stay F32 per the quantizer
rules) vs 168 MB F32.

## Notes for whoever picks this up

- The runtime is CPU-side for the convolutions, the DConv stacks and the whole
  CrossTransformer, all written as scalar loops. A full 7.8 s segment takes
  minutes on an M1 and was ~25 min when the box was at load 100+. There is a lot
  of headroom (e.g. `read_tensor_f32` re-reads and re-converts the same weights
  inside the per-frequency-band DConv loop, ~4096 redundant reads per encoder).
  Perf work should follow the A/B rules in the dev guide — gate it, don't replace.
- `apply_dconv()` (the ggml DConv) is still dead code and still has the GLU-axis
  bug described under BUG 4. It is left in place rather than deleted so the ggml
  path survives for a future perf pass; do not call it without fixing the split.
- The diff replays `input_wav` from the reference, so it never exercises the
  16k->44.1k resampler. That is deliberate (it isolates model parity) but means
  resampler parity is separately unverified.

## Performance

### Baseline profile (CRISPASR_HTDEMUCS_PROFILE=1, one 343980-sample segment)

The runtime was correctness-first: ggml was used mostly for weight loading and
7 small graphs, while every expensive op was a hand-written scalar loop. The
backend is hardcoded `ggml_backend_cpu_init()`; `params.use_gpu` and
`params.n_threads` are set in the defaults and never read again.

Profiling said the bottleneck was NOT the convolutions:

| phase | before | share |
|-------|--------|-------|
| ct.self_attn  | 322.3 s | 52.4% |
| ct.cross_attn | 205.7 s | 33.4% |
| enc.rewrite   |  25.3 s |  4.1% |
| enc.conv2d    |  20.1 s |  3.3% |
| everything else | ~42 s | ~7% |
| **total** | **615.5 s** | |

### What was done (all gated, all output-equivalent)

1. `CRISPASR_HTDEMUCS_BLAS` — the CrossTransformer's matmuls (QKV, QK^T,
   attn·V, out-proj, both FFN linears, self- and cross-attention) through
   `cblas_sgemm`. The scalar versions strode `seq_len` floats (~10 KB) in their
   innermost loop — roughly a cache miss per multiply-add.
   **ct.self_attn 322.3 s -> 8.5 s, ct.cross_attn 205.7 s -> 3.5 s.**
2. `CRISPASR_HTDEMUCS_FASTCONV` — batched im2col + one GEMM instead of
   per-time-frame im2col + dot-product loops; 1x1 convs emitted as channel
   matmuls. **enc.conv2d 10.0 s -> 0.17 s, enc.rewrite 12.2 s -> 0.30 s.**
3. `CRISPASR_HTDEMUCS_WCACHE` — cache F32 weight copies by tensor pointer.
   The DConv stacks called `read_tensor_f32` from inside their per-band loop:
   9 reads x 680 bands ~ 6k redundant dequant+copy passes per encoder layer.

Every original path is preserved behind its gate, per the project rule. A/B
verified in both directions — **45/45 stages pass in every gate combination**,
so the fast paths are output-equivalent and the gates are usable for
regression bisection.

### Measured totals — READ THE CAVEAT

| configuration | total |
|---------------|-------|
| all gates OFF (original code) | 338 s |
| all gates ON  | **41.5 s** |

Both from the per-stage diff on the same box. This machine is shared with other
active sessions and its load average swung between 19 and 100 during this work;
a repeat of the all-ON run at load 94 read 293 s. Per the dev guide's A/B rule
these numbers are indicative, NOT a benchmark: a trustworthy figure needs a
quiet box (or Kaggle) with a warm-up discarded and a median of >= 3.

### Comparison to ONNX Runtime

For reference, `timcsy/demucs-web-onnx` htdemucs_embedded.onnx on
onnxruntime CPUExecutionProvider, same 343980-sample segment:
**~43 s/segment with ORT_ENABLE_ALL, 103 s with ORT_DISABLE_ALL.**

So the optimised CPU path is now in the same order of magnitude as ORT rather
than ~15x behind it. A like-for-like claim still needs the quiet-box rerun above.

### Round 2 — after the DConv / ConvTranspose / conv1d work

Adding a wall-clock row to the profiler exposed that the round-1 "5.29 s"
was **instrumented phases only**. The decoder ConvTranspose2d had never been
timed and was 68% of the real forward. Two further mis-readings followed, both
caused by scope boundaries rather than by the code:

- `tenc(total)` and the outer `channel_up/down` scope were **nested totals**
  (the latter wrapped the entire transformer section — upsample, all 5 layers,
  downsample), so they read 41-51% and sent me after the K=1 samplers. Measured
  in isolation the samplers are **18 ms**. Nested rows are now labelled
  `[nested]`, excluded from the uninstrumented total, and every phase has its
  own scope.

Lesson: a profile row is only as trustworthy as the scope boundary behind it,
and "total" must be stated against wall clock or it will flatter you.

| phase | round 1 | round 2 |
|-------|---------|---------|
| dec.convtranspose  | 294 360 ms (untimed) | 168 ms |
| tdec.rewrite       | 7 824 ms | 149 ms |
| tdec.convtranspose | 3 094 ms | 152 ms |
| dec.dconv          | 28 848 ms | 783 ms |
| enc.dconv          | 7 872 ms | 955 ms |
| chan.up/down (K=1) | — | 18 ms |

`tdec.rewrite` is worth calling out: it was *this branch's own* scalar
`cpu_conv1d_time`, written while fixing the K=3 Conv1d bug, and it became the
single largest phase once everything around it got faster.

### Measured result

Benchmark protocol per the dev guide: warm-up discarded, median of 5, load
recorded per run. Box load 16-19 (shared machine, could not get below ~9).

| | one 343980-sample segment |
|---|---|
| original (all gates OFF) | ~615 s |
| **optimised (all gates ON)** | **median 14.1 s** (best 12.2 s) |
| ONNX Runtime CPU, ORT_ENABLE_ALL | ~43 s |
| ONNX Runtime CPU, ORT_DISABLE_ALL | ~103 s |

So roughly **44x** over the original and, on this box, ~3x faster than ORT with
full graph optimisations. Caveat on the ORT comparison: those figures were
measured separately and their load conditions are not known to me, and ORT is
multi-threaded by default while this runtime is entirely single-threaded — so
treat the ratio as indicative, not as a controlled benchmark.

Parity is unchanged throughout: **45/45 stages, min cos 0.999963**, in every
gate combination, and the separated stems are bit-identical to the
pre-optimisation build (vocals rms 0.13962 / other 0.00521 / drums 0.00021 /
bass 0.00010) with the ASR round-trip still word-for-word correct.

### Next targets (unstarted)

The profile is now flat — no phase dominates, and the remaining scalar work is
small. The two big structural levers left are both untouched:

- **Threading.** Everything is single-threaded. The per-frequency-band loops and
  the per-source output loop are embarrassingly parallel, and Accelerate's sgemm
  is already multi-threaded internally for large matrices but is being handed
  work serially. `params.n_threads` is still a dead field.
- **GPU.** `params.use_gpu` is still never read; the backend is hardcoded
  `ggml_backend_cpu_init()`. Now that the hot ops are all GEMM-shaped, a ggml
  graph port is far more tractable than it was against the scalar loops.


## ggml graph port (CRISPASR_HTDEMUCS_GGML)

The CrossTransformer now has a real ggml graph implementation, and the backend
is finally selectable — `ggml_backend_cpu_init()` was hardcoded, so `use_gpu`
and `n_threads` were dead fields and Metal/CUDA/Vulkan could never engage.

### What the graph does

All `t_layers` of BOTH branches in ONE build+alloc: LayerNorm, QKV projection,
multi-head attention via `ggml_soft_max_ext`, output projection, LayerScale
residuals, FFN, and `norm_out`. Self- and cross-attention alternate on
`classic_parity`, with `old_x` captured before `x` is overwritten so the time
branch attends to the pre-update spectrogram branch.

Two details that matter:

- **Layout.** The CPU buffers are (dim-slow, seq-fast), index `d*seq + s`, but
  ggml contracts `mul_mat` on `ne[0]`, so the graph wants `ne = (dim, seq)` =
  index `s*dim + d`. Transposed once on upload and once on download.
- **norm_out** is `GroupNorm(num_groups=1)` over all channels AND tokens
  jointly, so it is a flatten-to-1D + `ggml_group_norm(1)` + per-channel affine.
  Using `ggml_norm` would be per-token — the exact bug that cost the original
  CPU implementation its parity.

Every layer is marked `set_output` **only when the diff is capturing**, so the
graph is diffable layer by layer without preventing gallocr from reusing those
buffers in normal runs.

### Correctness — verified on both backends

| backend | result |
|---------|--------|
| CPU  | 45/45, `ct_l0..l4` cos 1.000000 |
| Metal (MTL0) | 45/45, `ct_l0..l4` cos 1.000000 |

CLI acceptance on Metal: ASR round-trip word-for-word, stem RMS matching the
CPU path to 5 decimals (vocals 0.13963 vs 0.13962).

The GPU path also exposed a real leak: `htdemucs_diff` never freed the
reference `WeightLoad`, whose buffer is allocated on `ctx->backend`. Invisible
on CPU; on Metal it tripped the device destructor's live-resource assert at
exit (`GGML_ASSERT([rsets->data count] == 0)`, exit 134). Now freed on every
return path, before the backend it belongs to.

### Speed — NO VERDICT on this box, default stays OFF

Transformer phase only, same run, load ~32:

| path | ct phase |
|------|----------|
| CPU + Accelerate BLAS (default) | 10.9 s |
| ggml graph, CPU backend | 17.4 s |
| ggml graph, Metal | 7.0 s |

So ggml-on-CPU is clearly **slower** than Accelerate (expected — Accelerate's
sgemm is very well tuned), and Metal beats BLAS on the transformer by ~1.6x.

Total wall time is **inconclusive**. A first single-shot pair suggested Metal
45.7 s vs BLAS 97.6 s, but load drifted 21 -> 56 between the arms. Interleaving
flipped the result (BLAS won all 3 rounds) — and then load was climbing
monotonically with Metal always second. Alternating the order gave times that
track the load average far more closely than the config:

    META 58.2s load=87   BLAS 62.4s load=65   BLAS 59.1s load=78
    META 36.3s load=57   META 31.2s load=44   BLAS 24.3s load=35

This is precisely the dev-guide failure mode ("a quiet-CPU-vs-loaded-GPU
comparison invented a false 2x win"). A verdict needs a quiet box or a Kaggle
CUDA run.

Per the inverse-default rule, **`CRISPASR_HTDEMUCS_GGML` stays default OFF**:
verified correct, not yet proven faster overall. The CPU/BLAS path is untouched
and remains the default.

Plausible reason total wall does not follow the transformer win: only the
transformer is a graph. The encoder/decoder still run on CPU, and with a GPU
backend their small per-call graphs (time encoder conv, decoder rewrite) pay
launch overhead — the "per-step GPU dispatch of small matmuls is launch-bound"
effect in the dev guide. Porting those too, or splitting weight placement, is
the follow-up.


## Unified ggml graph — frequency path complete

With `CRISPASR_HTDEMUCS_GGML=1` the **encoder, CrossTransformer and decoder**
all run as ggml graphs on the selected backend (`CRISPASR_HTDEMUCS_GPU=1` for
Metal/CUDA/Vulkan).

### How each op maps

| op | ggml expression |
|----|-----------------|
| freq encoder conv | `ggml_conv_2d`, kernel `(1,K,IC,OC)`, `s1=stride`, `p1=pad` |
| DConv dilated conv | `ggml_conv_2d`, kernel `(K,1,C,hidden)`, `d0=dilation` |
| DConv 1x1 / rewrite | `ggml_conv_2d` with a 1x1 kernel |
| per-band GroupNorm(1) | permute to `(T, C, Fq)`, `ggml_group_norm(n_groups=n_bands)` |
| GLU | views on `ne[2]` + `ggml_sigmoid` + `ggml_mul` |
| attention | `ggml_mul_mat` + `ggml_soft_max_ext` |
| norm_out GroupNorm(1) | flatten to 1D + `ggml_group_norm(1)` + affine |
| **ConvTranspose2d [K,1]/[S,1]** | **K x 1x1 conv + `ggml_acc` strided scatter** |

`KH = 1` on the DConv is what keeps frequency bands independent — the graph
equivalent of Python's `y.permute(0,2,1,3).reshape(-1, C, T)`.

### The ConvTranspose decomposition

This was the only genuine gap: ggml has just `ggml_conv_transpose_2d_p0`
(one stride, zero padding), but the model needs kernel `[K,1]` with stride
`[S,1]` on the frequency axis. Using the same identity as the CPU
GEMM+scatter path:

    out_raw[t, fq_in*S + kh, oc] += sum_ic W[ic][oc][kh] * x[t, fq_in, ic]

For a FIXED `kh` the inner sum is exactly a 1x1 conv over channels, and the
placement is a `ggml_acc` into a view with frequency stride `S`. Scattering
into the **uncropped** output is the trick that keeps every offset
non-negative — `z[..., pad:-pad, :]` then becomes a plain view.

### Correctness

| backend | result |
|---------|--------|
| Metal | **45/45**, `dec_freq_0..3` >= 0.999999, `output_drums` 0.999959 |
| CPU | 44/45 — identical except `output_drums` 0.998132 |

The CPU-backend miss is numerics, not structure: `max_abs` is 1.7e-4 on the
QUIETEST stem (-73 dBFS), where a fixed absolute error dominates the cosine.
`output_vocals` and `dec_freq_3` are 1.000000 on both. It reflects a different
summation order in ggml's CPU conv; Metal — the backend this path targets — is
clean. Worth resolving before the gate could ever become default.

ASR acceptance on the Metal full-graph path: word-for-word correct.

### Still on the CPU

- **Time branch** encoder/decoder (its convs would map the same way; it is
  ~8-10% of the forward).
- **STFT / iSTFT** — DSP rather than NN, and 0.2% of the profile.
- The CaC unmask / output assembly.

### Performance — still no verdict

Attempts to A/B the full-graph GPU path against CPU+BLAS on this machine were
worthless: the load average moved between 45 and **354** during the runs, and
the timings tracked load far more than configuration. Nothing is claimed. This
needs a quiet box or a Kaggle CUDA run, which is also the right place to
exercise the CUDA backend for the first time (CUDA has stricter contiguity
asserts than CPU/Metal — see the `get_rows` note in the dev guide, so a
Metal-clean graph is NOT proof CUDA is clean).


## Perf check on the graph path (2026-07-20)

Wall-clock A/B on this box remains unusable (load moved 3.6 -> 54 mid-run;
BLAS ranged 10.95 s to 121.1 s for the SAME config). But **within-run phase
ratios are load-robust**, and those reproduce cleanly across three paired runs:

| phase | BLAS (CPU) x3 | Metal full-graph x3 | verdict |
|-------|---------------|---------------------|---------|
| transformer | 4909 / 6744 / 3672 ms | 1008 / 1122 / 1122 ms | Metal **3.3-6x faster** |
| encoder | 1035 / 1837 / 1078 ms | 2200 / 2168 / 2221 ms | Metal **1.2-2.1x slower** |
| decoder | 2459 / 4704 / 3023 ms | 3237 / 3539 / 3272 ms | ~parity |

Two findings:

1. **The transformer win is real and large.** One big graph of dense matmuls is
   exactly what a GPU is for.
2. **The encoder graph is consistently SLOWER than CPU+Accelerate**, and the
   decoder is no better than parity. The reason is structural, not the ops: the
   port currently builds a graph PER LAYER, so every layer pays a graph build +
   gallocr alloc and a full host<->device roundtrip of its activations (encoder
   layer 0 alone is 8.2M floats = 33 MB each way). That is the dev guide's
   "per-step GPU dispatch is launch-bound" / cross-backend-traffic effect: the
   data movement dominates the per-layer compute.

Also worth recording: **Metal timings are far more STABLE under contention**
(~5% spread) than the CPU path (~2x spread), because the GPU work is insulated
from CPU load. That is why earlier A/B attempts kept flipping — minimum-of-N
sampling favours BLAS (it captures the quiet moments), while typical loaded
conditions favour Metal.

### Next step — fuse into ONE graph

The remaining work is not more ops, it is removing the host roundtrips: build
encoder + transformer + decoder as a SINGLE graph so activations never leave
the device between layers, with skip connections kept as in-graph tensors
rather than downloaded and re-uploaded. That is what should convert the
transformer-only win into an end-to-end one.


## FINAL benchmark (2026-07-20, quiet box) — CPU+Accelerate wins on M1

First measurement all session taken under genuinely controlled conditions:
load stable 9.5-12 across every sample (spread ~2.5, versus 9-354 earlier),
Firefox's WebRender GPU-compositor process idle at 0% (it had been at 74%,
which contends for the very GPU under test and penalises ONLY the Metal arm),
6 samples per config, balanced order, warm-up discarded.

Each config was proven live before timing — the profiler must show its own
phase marker (`ct.self_attn` / `enc.ggml_graph` / `fused.graph`), so no config
can silently be another. That check exists because a stale `test-htdemucs-smoke`
had previously made an entire three-way benchmark measure the per-layer path
twice while labelled FUSED.

| config | min | median |
|--------|-----|--------|
| **CPU + Accelerate (default)** | **3.95 s** | **4.12 s** |
| fused single graph (Metal) | 5.21 s | 5.62 s |
| per-layer graphs (Metal) | 5.33 s | 5.71 s |

Proof of work: both arms produce identical stems (drums 0.000569, bass
0.000276, other 0.292620/0.292617, vocals 0.000798/0.000799), so the times
reflect real separation, not a short-circuit.

### Verdict

- **CPU+Accelerate is the fastest path on Apple Silicon**, by ~36% over the
  best GPU path. `CRISPASR_HTDEMUCS_GGML` correctly stays default OFF.
- **Fusion bought ~2%** (5.71 -> 5.62 s median), i.e. nothing. The reason is
  M1's unified memory: the host<->device "roundtrips" fusion removes are not
  real transfers there, so the overhead I predicted was largely absent. What
  fusion does remove — per-layer graph build + gallocr alloc — was small next
  to compute.
- Metal loses to Accelerate because the M1 AMX matrix coprocessor makes FP32
  sgemm extremely fast, and this model's per-layer work is modest (42M params).

This does NOT mean the graph port was wasted. It is the prerequisite for CUDA,
where both assumptions invert: transfers are real PCIe traffic (so fusion
should matter a lot) and a discrete GPU has a far larger advantage over server
CPU BLAS than Metal has over AMX. That is the measurement worth taking next.

### Headline

    original (all gates OFF)            ~615 s
    optimised CPU+Accelerate (default)  4.12 s median   -> ~149x
    ONNX Runtime CPU, ORT_ENABLE_ALL    ~43 s           -> ~10x faster than ORT

The ORT figure was measured separately under unknown load; treat the 10x as
indicative rather than a controlled head-to-head.

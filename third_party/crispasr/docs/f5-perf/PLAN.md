# F5-TTS performance — issue #294

## NOW — status

**Quality follow-ups DONE (2026-07-25), audio-confirmed** — after the speed work
the reporter hit two quality bugs; both fixed and TTS→ASR-roundtripped on M1 Metal
(F16 GGUF + whisper-large-v3-turbo):
- **Chinese g2p** (`9a2fd7dc` + `3167d014`): `convert_to_pinyin` was ASCII-only →
  Chinese was silence. Added `src/core/pinyin_g2p.*` (jieba-min max-match +
  pypinyin TONE3 + 不/一/third-tone sandhi, data embedded via
  `tools/gen_pinyin_data.py`), parity 99.8% vs pypinyin (`tests/test-pinyin-g2p`).
  Then the real audio fix: f5 was char-splitting each pinyin syllable — now maps
  each token to its vocab id directly. ZH roundtrip exact.
  ⚠ [[LEARNINGS: token-parity ≠ working audio]] — the g2p test was green while
  audio was garbled; only the roundtrip caught the downstream char-split.
- **English truncation** (`9a2fd7dc`): removed the CrispASR-only rate clamp
  (`fixed_rate*2.5`, no upstream equivalent) that cut the tail of a slow ref;
  now asymmetric (loose upper). `CRISPASR_F5_DURATION_CLAMP=0` = exact upstream.

## Speed work (original #294)

Branch `perf/f5-speedups` (worktree `CrispASR-f5perf`). Reporter: F5-TTS slow on
RTX 5060 Ti (sm_120) + Ryzen 5 2600, F16 model, 32 ODE steps.

**Hard constraint:** F16 is the ONLY viable format for F5 (flow-matching:
every weight used 1408×/synth, q8's ~0.5% error compounds → unintelligible;
see `hf_readmes/f5-tts-GGUF.md`). So **no quantization** — all wins come from
compute efficiency and fewer/smaller forward passes.

**Methodology (mandatory, dev doc §"A/B every perf optimization"):** every change
gated + default OFF; negative control = gate-off must stay byte-identical
(md5) to baseline; judge the ON path by TTS→ASR roundtrip, not cosine. M1 Metal
GPU timing is dispatch-bound/noisy → **speed verdict + any default-flip needs a
CUDA A/B (reporter box or Kaggle)**, not M1.

### Baseline (M1 Metal, F16, jfk ref + "quick brown fox…", seed 42, 32 steps)

- ref_T=1032 duration=1662 (ref = 62% of T)
- text_embed 87 ms
- **ode_solve 60.9 s** = host_embed (CPU) **15.6 s / 26%** + dit_graph (GPU) **45.3 s / 74%**
- vocos 0.4 s (**negligible** — vocos only decodes the generated frames)
- md5 `e249b19c8822b5b061d302839ef65678`; roundtrip = exact ("The quick brown fox
  jumps over the lazy dog and then it ran away.")

### Where time goes → priorities

1. **dit_graph 74%** → F16 activations (#4). [IN PROGRESS]
2. **host_embed 26%** (bigger on reporter's slow CPU) → move input-embed into the
   GPU graph (omnivoice-style). [TODO]
3. **NFE reduction** (deterministic, box-independent): fewer steps (`--tts-steps`,
   shipped), interval-CFG (`CRISPASR_F5_CFG_INTERVAL`, shipped), higher-order ODE
   solver (new). [TODO]
4. Shorter reference clip → smaller T on every forward (user-side, free). [TELL REPORTER]
5. ~~Vocos GPU/FASTCONV~~ — DROPPED, vocos is <1%.

### Validated NFE levers (M1 Metal, roundtrip intact — DETERMINISTIC, carries to CUDA)

| Config | ode_solve | Speedup | Roundtrip |
|--------|-----------|---------|-----------|
| baseline (32 steps) | 60.9 s | 1.0× | perfect |
| `--tts-steps 16` | 30.3 s | **2.01×** | perfect ✅ |
| `CRISPASR_F5_CFG_INTERVAL=2` | 46.6 s | **1.31×** | perfect ✅ |
| 16 steps + interval 2 | 23.5 s | **2.59×** | perfect ✅ |

These use existing knobs — the win is validating + recommending them. Fewer/skipped
forward passes ⇒ speedup is box-independent (unlike the GPU-compute changes below,
which need a CUDA verdict).

### Ecosystem research (how F5 runs elsewhere) — reshapes priorities

- Upstream (SWivid) = torchdiffeq **Euler**, nfe **32** (16 offered), **CFG as ONE
  2×-batch forward** (our `F5_BATCH_CFG`, i.e. our *default two-forward path is the
  non-standard one*), sway −1.0, Vocos. cfg_strength 2.0.
- **EPSS (arXiv 2505.19931)** training-free non-uniform step pruning → **~4× at 7 NFE**,
  quality stable to 7–12 NFE. **We already ship these schedules** in
  `get_epss_timesteps` (n=5/6/7/10/12/16). So the headline win is already coded —
  just needs validation + recommendation, no new kernel.
- **Guidance-free / interval CFG** halves per-step cost; interval form is portable and
  we already have `CRISPASR_F5_CFG_INTERVAL`. Paper RTF 0.31→0.17 by dropping uncond.
- **Layer caching across steps** (DiTReducio 2509.09748) — implemented the temporal-skip
  half (`CRISPASR_F5_DIT_SKIP=K`): reuse the cached full step-velocity, recompute every
  K steps + first/last. MEASURED @32 steps: K=2 → 1.9× perfect, K=3 → 2.7× minor artifact.
  BUT ≈ equivalent to just using fewer EPSS steps (K=2@32 ≈ `--tts-steps 16`): same
  forward-count reduction, so validated + gated but not strongly additive to EPSS. The
  branch-skip half (skip attention/FFN within blocks) — see below.
- Reference length is a real lever (joint ref+gen DiT sequence). Confirmed.

### DiTReducio branch-skip (in progress)

**Goal:** skip the FFN (or attention) branch within blocks on some steps, reusing the
cached branch delta — surgical vs. temporal-skip's whole-step reuse. Potential value:
push past temporal-skip's quality wall (K=3 artifacted) in the aggressive regime.

**Architectural finding:** a static ggml graph executes every node, so branch-skip can't be
a runtime `if` — it needs either (a) a 2nd "no-FFN" graph variant reading cached branch
deltas kept as **persistent on-device tensors** shared between graphs (host round-trip of
22×[dim,T] ≈ 150 MB/step would exceed the compute saved), or (b) the cheap probe below.
Ceiling per skipped branch ≈ 25–30% (vs temporal-skip's 50%).

**Plan (chosen: prototype cheaply first):** gated `CRISPASR_F5_FFN_SKIP=K` probe that kept the
FFN matmuls running (NO speed win) but blended a cached FFN delta on skip steps via graph I/O
(fresh delta out, cached delta in, `use_cache` scalar), host round-trip. Purpose: measure the
QUALITY of FFN-skip vs temporal-skip at aggressive K before committing to the on-device
dual-graph build.

**Findings (M1 Metal, 32 steps, quality probe — decisive NEGATIVE):**

| K | temporal-skip (DIT_SKIP) | FFN-skip (branch) |
|---|--------------------------|-------------------|
| 2 | perfect | perfect |
| 3 | "quick-**brand**" minor artifact | "**Quickback**" similar artifact |
| 4 | — | "park junk, oh, they lay lazy" badly degraded |

FFN-skip does **not** hold quality past the point temporal-skip does — it hits the same wall at
K=3 and degrades worse at K=4 — AND its speed ceiling is lower (~25–30% vs 50%). So the expensive
on-device dual-graph build is **NOT worth it**: strictly worse than the temporal-skip already
shipped. Probe reverted (diagnostic only, no speedup; fast path stays byte-identical). Decision:
**do not build branch-skip.** Temporal-skip + EPSS + EMBED_GPU are the levers to ship.
- No verified Candle/burn F5 port, no vLLM/SGLang. MLX port ~8× RT on M3 Max.

**Conclusion: F5 is already well-optimized; the real wins are configuration
(EPSS low steps + interval-CFG + short ref + batched-CFG on CUDA), not new kernels.**

### Changes tried

| # | Change | Gate | Status |
|---|--------|------|--------|
| 4 | F16 activations in DiT matmuls | `CRISPASR_F5_F16_ACT` | built + gated. **Metal: byte-identical + ~17% SLOWER** (ggml already casts RHS to F16 internally). Committed, default OFF, CUDA-A/B-only. |
| 6 | stable-alloc (skip per-step re-alloc for CUDA-graph replay) | `CRISPASR_F5_STABLE_ALLOC` | **REVERTED — correctness bug**: pos_in clobbered after step 0 → garbage ("(wind blowing)"). Proper fix needs persistent input tensors on a dedicated buffer (omnivoice §245 pattern); CUDA-only value, unverifiable on Metal. Not worth it now. |
| 2 | host-embed → GPU graph | `CRISPASR_F5_EMBED_GPU` | **DONE + WINS.** Cached embed graph (input_proj + 2× grouped conv-pos + Mish + residual) on ctx->backend, reusing the cosyvoice3 grouped-conv-pos pattern (symmetric pad=15). MEASURED M1: host_embed 15.6→3.9 s, **ode_solve 60.9→48.3 s (1.26×)**, roundtrip perfect, gate-off byte-identical (e249). Wins even on M1 (removes the serial CPU stall between GPU dispatches); bigger expected on fast-GPU/slow-CPU. Default OFF (CPU-only builds would run the graph on CPU); recommend with a GPU backend, flip default after CUDA confirm. |
| batched CFG default (CUDA) | `CRISPASR_F5_BATCH_CFG` | exists; validating correctness. Matches upstream. Candidate CUDA default. |
| EPSS low-NFE + interval | (knobs) | validating quality at n=7/10/12 (+interval). Primary recommendation. |

### Reporter comms
- Posted knobs (`--tts-steps 16`, `CRISPASR_F5_CFG_INTERVAL=2`), `-nfa`-is-a-no-op,
  and `CRISPASR_F5_BENCH=1` request. issue #294 comment.
- TODO: follow up with the validated 2.6× numbers + shorter-reference tip.

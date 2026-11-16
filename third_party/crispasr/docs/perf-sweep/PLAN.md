# CrispASR perf campaign — fleet-wide codec/CFG optimization

Cross-cutting speedups discovered during the OmniVoice #254 work, generalized to
the whole backend fleet. Each item ships **env-gated** (A/B toggle, never a silent
default), **A/B'd** against ground truth (byte-equivalence + ASR/decoded-output
roundtrip on F16 AND a quant), and **unit-tested** where a model-free test is
possible. Default flips only on a proven speed AND quality win.

## NOW — active work

**✅ [OPUS-1M-c2pa] DONE 2026-07-17 — openvoice2 reference-SE disk cache (TODO-6).**
⚠ Re-targeted from f5_tts (f5's `set_reference` is only `compute_mel_spectrogram` — pure
DSP, sub-ms, no win). openvoice2's `ref_enc_forward` (STFT + 6-conv reference encoder →
256-d `target_se`) IS a neural encode, re-run on every `openvoice2_convert` /
`openvoice2_extract_speaker_embedding`. Both routed through a new `openvoice2_target_se()`
that wraps the SHARED `crispasr_ref_cache` content-addressed helper (tag `openvoice2-se`,
key = fnv1a(ref_pcm) ^ fnv1a(ref_sr)) — same helper as irodori-latent / indextts-cond;
resample+STFT+ref_enc are skipped entirely on a hit. Cache dir = `CRISPASR_TTS_REF_CACHE_DIR`
→ `$TMPDIR/crispasr-tts-refcache/`; global `CRISPASR_TTS_REF_CACHE=0` disables (helper-owned
env, NOT the OVC1 `CRISPASR_*_VOICE_CACHE`). **A/B VERIFIED** via `test-openvoice2-hifi`
(clean-WAV harness, jfk src=ref, `openvoice2-tcc-f16`): run1 miss → run2 HIT (log-confirmed)
→ run3 `CRISPASR_TTS_REF_CACHE=0` no-hit; **output byte-IDENTICAL across all 3 runs**
(cache returns bit-exact SE; the disabled fresh-encode matches too). Cache blob 1061 B
(256×f32 + header); `CRISPASR_TTS_REF_CACHE_DIR` honored. Landed on branch
`perf/f5-refvoice-cache`. (dots_tts NOT viable here — speaker CAM++ GGUF not local.)

**✅ [OPUS-1M] DONE (impl) 2026-07-16 — chatterbox interval-CFG (`a4a8f64de`, opt-in).**
`CRISPASR_S3GEN_CFG_INTERVAL` in s3gen's 10-step CFM Euler solver: K>1 forces the
sequential single-UNet path and skips the uncond pass every K steps; batched-B2 default
(K=1) byte-unchanged by construction. **Verification (honest):** default byte-safe;
K>1 ENGAGES + valid non-crash output; but K>1 now M1-VERIFIED via a fast CFM-only
harness (`tests/chatterbox-s3gen-cfg-interval-ab`, skips T3 AR): K1a==K1b byte-identical
(default B2 deterministic), K1-vs-K2 mel cos 0.980 / K3 0.984 — strong content proxy.
Only a real-speech ASR round-trip + CUDA remain as off-box nice-to-haves. (Sonnet owns TODO-B k=1→matmul.)

**In flight (2026-07-16, this session): remaining locally-doable items.** ✅ TODO-C
resolved — **kokoro FASTCONV landed** (`323e96f23`, 89 F16 kernels, byte-identical);
**indextts_voc not applicable** (F16 tensors are custom-CPU-op AA filters, not
ggml_conv_1d). ✅ TODO-D **diagnosed** — fastpitch f16 is a MULTI-bug dead-end (loader
= converter offset gaps, fixable by re-pack; but then Metal + CPU f16-path runtime
asserts) — not worth pursuing (q8_0 default is a FASTCONV no-op anyway). Full
diagnosis in the TODO-D section. **All locally-doable items in this batch are now
resolved.**


**Status (2026-07-16): FASTCONV landed + A/B-verified (byte-identical, default ON)
for 7 backends — omnivoice, irodori, zonos, speecht5, chatterbox_s3gen, cosyvoice3, kokoro.
Interval-CFG (opt-in, default OFF): 6 landed+verified (cosyvoice3, f5-tts,
voxcpm2, dots-tts, irodori, tada) + chatterbox impl (off-box content verify open). All on `main`, all green.**

**➡ FRESH AGENT: go to the "TODO QUEUE FOR A FRESH AGENT" section near the bottom
of this file. FASTCONV is done for every local F16 target; the active thread is
TODO-2 interval-CFG (6 landed). Only chatterbox (CFM — clean pattern but ~1–3 h synth
on this M1, so verify on a quieter box / CUDA) remains as a clean flow candidate. ⚠
NOT amenable: dia/zonos/voxtral (AR — uncond shares a batched B=2 KV cache, skipping
its forward corrupts the KV; measured dia); vibevoice (dominant CFG is that same
AR-neg-LM path — its cheap diffusion pred-head CFG is intervallable but low-value,
~20 DPM steps, coarse). TODO-B/C/D/3/4 remain as scoped below.**
Remaining: TODO-B chatterbox k=1→matmul (⚠ NOT cosyvoice3 — only 1 of its 85 hift
kernels is k=1, measured) · TODO-C indextts/kokoro · TODO-D fastpitch loader ·
TODO-2 interval-CFG (✅ cosyvoice3 `63a91a6a5` + f5-tts `678ee5ce1` + voxcpm2
`2c7cc5df4` + dots `675498cb3` + irodori `d04620cba` + tada `3c8180cdc` landed opt-in;
only chatterbox left, off-box) · TODO-3
Metal q4_k (needs registry alt-quant schema — not a quick win) · TODO-4 CI perf gate (🟡 compare logic+test landed; nightly benchmark-emit + committed baseline pending).
Coverage triage below (only F16-kernel models benefit). ⚠ There is no
`handover-prompts/fastconv-fleet-sweep-round2.md` on disk — this NOW section + the
TODO QUEUE below ARE the round-2 handover.

**Interval-CFG (TODO-2) — cosyvoice3 (`63a91a6a5`) + f5-tts (`678ee5ce1`) + voxcpm2
(`2c7cc5df4`) + dots (`675498cb3`) + irodori (`d04620cba`) + tada (`3c8180cdc`) landed
opt-in, default OFF.** Recompute the uncond CFG forward only every K steps, reuse the
cache in between; cond fresh; first+last always recompute. Default K=1 is
byte-identical to legacy.
- **cosyvoice3** `CRISPASR_COSYVOICE3_CFG_INTERVAL` (`cv3_run_solve_euler`): K=1
  byte-exact (cos=1.0 twice via `cosyvoice3-flow-cfg-interval-ab`); K=2 mel cos
  0.9994, K=3 0.9915. Full ASR round-trip PENDING (synthetic-input harness).
- **f5-tts** `CRISPASR_F5_CFG_INTERVAL` (32-step ODE sampler): verified end-to-end
  via the CLI — K=1 twice PCM byte-IDENTICAL; **content preserved (ASR(K1)==ASR(K2),
  whisper base.en, word-overlap 1.0)**; K=2 acoustic divergence log-STFT cos 0.945 /
  envelope corr 0.78 (PCM corr is phase-noise per [[tts-parity-not-by-audio-corr]]).
- **voxcpm2** `CRISPASR_VOXCPM2_CFG_INTERVAL` (per-patch CFM `cfm_euler_solve`):
  verified via CLI — K=1 twice PCM byte-IDENTICAL; content preserved (ASR(K1)==ASR(K2)).
  ⚠ voxcpm2 is AR at the patch level, so K=2 shifts the stop predictor by a patch
  (3.36 s → 3.52 s) — same words, slightly different duration.
- **dots-tts** `CRISPASR_DOTS_CFG_INTERVAL` (per-patch flow-matching `dots_flow_match_core`):
  verified via CLI (dots-tts-soar-f16 + BigVGAN vocoder, 4.4 GB) — K=1 twice PCM
  byte-IDENTICAL; content preserved — ASR(K1)==ASR(K2) word-for-word on an unambiguous
  sentence (an earlier fox/box diff was a whisper mishearing of the K=1 baseline, not
  interval); K=2 log-STFT cos 0.940.
- **irodori** `CRISPASR_IRODORI_CFG_INTERVAL` (40-step RF-ODE, up to 3 uncond forwards
  per step: text/speaker/caption — the heaviest CFG): verified via CLI (irodori-500m-v3
  + dacvae-ja, JA text, seed 42) — K=1 twice PCM byte-IDENTICAL; K=2 log-STFT cos 0.971.
  ⚠ ASR round-trip INCONCLUSIVE (multilingual ggml-base garbles both the exact K=1 and
  K=2 — weak JA ASR, not interval); rests on byte-exact K=1 + high STFT cosine.
- **tada** `CRISPASR_TADA_CFG_INTERVAL` (FM Euler `fm_euler_solve`; neg is a fixed
  per-solve uncond, so cleanly skippable — the neg-KV is the separate AR LLM):
  verified via CLI (tada-tts-1b-q4_k + tada-codec, text-only, seed 42) — K=1 twice PCM
  byte-IDENTICAL; content preserved ASR(K1)==ASR(K2); K=2 log-STFT cos 0.978. Covers
  the default single-candidate AND ranked paths.
All approximate → stay opt-in. NATURALNESS at aggressive K needs a HUMAN EAR — NOT
claimed for any.

**⚠ vibevoice — investigated, NOT worth interval-CFG.** Its dominant CFG is an AR
negative-LM path (`run_lm_step kv_sel=1`, per-frame KV-cached — same non-amenable
class as dia). It also has a batched cond+uncond CFG in the DPM-Solver++ pred-head
(`get_pred_head_graph(ctx,2)`, ~20 steps) which IS intervallable, but that head is
cheap and its neg_condition is fixed per frame (uncond varies only via z/t), so the
win is negligible while the expensive neg-LM CFG can't be intervalled. Skipped.

Commits (pushed to `main`):
`203f28f01` shared core_dac cache+conv1d · `191a7ebe4` omnivoice migrate ·
`8f2b17e4a` irodori · `8d8e6d9c8` zonos · `8231e8144` core_hifigan overload ·
`1c558b4f0` speecht5 · `41dbc88cc` chatterbox_s3gen wire + default flip (branch
`perf/fastconv-hifigan-fleet`).

### ⚠ HiFi-GAN family — coverage reality (measured 2026-07-16, GGUF-parsed)
The handover's "one overload sets up 3 backends" is over-optimistic. FASTCONV
only engages when the vocoder conv kernels are **F16** (the cast it kills). Actual
shipped dtypes (voc conv1d-routed kernels, ups.* excluded):
- **speecht5** — registry default `-f16` → **F16 ×74 → ENGAGES.** ✅ wired+verified.
- **fastpitch** — default `-q8_0` → F32 ×74 → **no-op**; only the non-default
  `-f16` variant has F16 ×74. Wired locally but NOT committed: its `-f16` GGUF hits
  a *pre-existing* loader bug (`gguf_init_from_file_ptr: failed to read tensor data`,
  in the GGUF reader before any graph code — file is byte-valid, md5-stable across
  two downloads) so it can't be run end-to-end here; and the q8_0 default is a no-op
  anyway. Low value until the f16 loader is fixed. **Discipline: don't ship what you
  can't run.**
- **bananamind** — ships only `-q8_0` and `-f32` (NO f16 variant in either
  en/de repo) → F32 ×74 → FASTCONV can **never** engage. Do NOT wire (dead code)
  unless an F16 build is published.

### Recipe — wire FASTCONV into one backend (proven 3×)
1. Add `core_dac::fastconv_cache <name>_fc;` to the backend's context struct.
2. At codec/vocoder load (after weights resolved, on the codec compute backend):
   collect all decode conv-kernel `ggml_tensor*` into a vector and
   `ctx-><name>_fc.bake(codec_backend, convs, env_on);` gated
   `CRISPASR_<BACKEND>_FASTCONV` (default on — the change is numerically equivalent).
3. In the decode graph, route each conv through the fc-aware overload:
   `core_dac::conv1d(g,x,w,b,K,dil,&fc)` / `dec_block(...,&fc)` /
   `build_decode_graph(...,&fc)` (DAC family) or `core_hifigan::conv1d(...,&fc)`
   (HiFi-GAN family). For a bespoke local lambda, replace `w` with `fc.get(w)`.
4. `<name>_fc.free();` in the backend's free().
5. **A/B (MANDATORY, seed-aware):** build, then synth fastconv ON vs OFF. If the
   model is stochastic (flow-matching / AR sampling), PASS `--seed N` — else the
   RNG dominates and a huge diff is FALSE (the irodori trap below). Gate:
   ON-vs-ON @same seed must be 0 (deterministic) BEFORE trusting ON-vs-OFF.
   Expect ON-vs-OFF ≈ 0 (byte-identical) or ≤~F16-codec drift; ASR/decoded output
   must be intact. Confirm fastconv ENGAGED (kernels are F16, not a no-op).


- ✅ **Item 1 — shared `core_dac::fastconv_cache` + `conv1d`/`res_unit`/`dec_block`
  fast path** (`203f28f01`). Unit test `test-fastconv` (206 assertions): fast ≈
  legacy (K=7 + k=1), byte-identical when disabled. `res_unit`/`dec_block` take an
  optional `fc` (default nullptr = legacy, so all existing callers untouched).
- ✅ **Item 6 — OmniVoice migrated to the shared helper** (`191a7ebe4`) — deleted
  its ~90-line local copy; verified equivalent (max|d| 23/32768, ASR exact).
- ✅ **Item 7 pilot — irodori_tts** (`CRISPASR_IRODORI_FASTCONV`, default on):
  bakes 92 F16 decode conv kernels → F32; wired `decode_dac_window` through
  `core_dac::conv1d(...,&fc)` + `dec_block(...,&fc)`. **Codec-only A/B (seed 42,
  isolating the flow-matching RNG): BYTE-IDENTICAL (0/32768).**
  - ⚠ **Lesson:** irodori is non-deterministic without a seed (`std::mt19937` from
    `random_device`, flow-matching noise). A naive on/off byte-diff showed
    max|d|=45607 (RNG, NOT fastconv). ON-vs-ON @seed42 = 0 confirmed determinism;
    the seeded on/off = 0 confirmed the codec change. [[tts-parity-not-by-audio-corr]].
- ✅ **zonos_tts** (`CRISPASR_ZONOS_FASTCONV`, default on): threaded the cache
  through `core_dac::build_decode_graph(...,&fc)` (extended with an optional fc
  param; nullptr = legacy). Bakes the dac-44khz F16 decode kernels. **Codec A/B
  (seed 42): BYTE-IDENTICAL (0/32768), non-silent.** So all three core_dac-family
  backends (omnivoice/irodori/zonos) now share one FASTCONV impl.
- ✅ **Shared `core_hifigan::conv1d` FASTCONV overload** — the HiFi-GAN vocoder is
  shared by fastpitch/speecht5/bananamind, so one overload (reusing
  `core_dac::fastconv_cache`) sets up 3 backends. Time-major layout so no
  k=1→matmul, but the F16-cast kill (the main win) applies. Unit test extended
  (`core_hifigan` case, cos>0.99999); 210 assertions total. `nullptr`-default so
  nothing changes until a backend bakes + passes `&fc`.
- ✅ **chatterbox_s3gen** (`CRISPASR_S3GEN_FASTCONV`, **default ON**) — `41dbc88cc` (wire,
  opt-in) + default flip. Cast-kill only (bitwise-identical): bakes F32 copies of the
  275 F16 conv kernels (100 K>1 + 175 K=1) and pointer-swaps `c->tensors` (same idiom as
  the ctx_f16 Metal fix), so `ggml_conv_1d` skips its F16→F32 cast. Split-load aware (two
  `fastconv_cache`, one per backend); `.ups.` conv_transpose excluded (permute path).
  **A/B (seed 42, Metal, `chatterbox-s3gen-q8_0`): ON vs OFF = 0/32768 across all 17280
  samples — audio BYTE-IDENTICAL** (only the trailing AI-provenance metadata chunk differs
  per-run, unrelated). For a flow-matching+AR pipeline that byte-identity also subsumes the
  determinism gate. Engagement proven (ON 275 baked/swapped, OFF 0; `CRISPASR_S3GEN_FASTCONV_DEBUG`).
  ⚠ Each synth is ~1–3 h on the contended M1 (load 300+), so this was one full ON+OFF pair,
  not 3 arms. ⚠ Only F16 conv kernels benefit; the 79 F32 + 3 ups kernels are untouched (correct).
  **k=1→matmul NOT done** (175 K=1 kernels) — a further win, but changes reduction order so it
  needs its own A/B on the drift-prone GPU path; left for later.
- ✅ **speecht5_tts** (`CRISPASR_SPEECHT5_FASTCONV`, default on) — `1c558b4f0`.
  Threaded the fastconv cache through `core_hifigan::forward()`/`resblock_forward()`
  (one change wires the whole family) + `collect_fastconv_kernels()` helper (excludes
  ups.*). Bakes 74 F16 vocoder convs → F32. **A/B on `speecht5-tts-f16` (deterministic,
  no RNG): ON-vs-ON 0/32768, ON-vs-OFF 0/32768 (byte-identical).** Engagement proven
  via `CRISPASR_SPEECHT5_FASTCONV_DEBUG`: ON bakes 74/74, OFF bakes 0. ASR roundtrip
  intact.
- ✅ **cosyvoice3_tts** (`CRISPASR_COSYVOICE3_FASTCONV`, default on) — `edbe64d28`.
  Cast-kill for the 85 F16 hift-vocoder conv kernels via the re-point idiom (idiom
  (a)): bake one F32 copy of each F16 kernel at hift load, re-point the named
  `cv3_hift` fields (`conv_pre_w`, `conv_post_w`, `ups_w[3]`, `resblocks[9].{c1,c2}_w[3]`,
  `src_down_w[3]`, `src_resblocks[3].{c1,c2}_w[3]`, `f0_condnet_w[5]` = 85) to the
  baked copies — zero graph change (every hift graph reads via `const auto& h =
  ctx->hift`). ⚠ **ups_w[] ARE baked** (cosyvoice3 upsamples with a regular
  `cv3_causal_conv1d`, not conv_transpose). 2D linears (m_source.l_linear,
  f0.classifier) left untouched (the CPU source path reads l_linear as raw F32).
  GGUF-verified 85 F16 conv kernels, 0 F16 non-conv. **A/B (new deterministic
  harness `tests/cosyvoice3-hift-fastconv-ab.cpp`: fixed mel+noise →
  `run_hift_inference`, no seed needed): ON vs OFF vs ON-rerun all hash
  `127fb40eeec32d8f` — BYTE-IDENTICAL, non-silent (max|a|=0.99, no NaN).**
  Engagement proven via `CRISPASR_COSYVOICE3_FASTCONV_DEBUG`: ON bakes 85/85, OFF 0.
  test-fastconv still green (210 assertions; core_dac untouched). ⚠ k=1→matmul NOT
  done (reduction-order change, its own A/B) — same as chatterbox, left for later.
### Bespoke-lambda triage (GGUF-parsed 2026-07-16 — F16 conv kernels required)
Parse conv kernels by dtype (⚠ name suffix varies: HiFi-GAN uses `.weight`,
cosyvoice3 uses `.w`) before wiring — only F16 benefits:
- **cosyvoice3-hift-f16** → ✅ **DONE** (`edbe64d28`, re-point idiom (a), 85/85 baked,
  byte-identical A/B). Original scope kept below for reference. Exact
  wiring (turnkey): `cv3_hift` (src/cosyvoice3_tts.cpp:339) holds the kernels in
  named fields — `conv_pre_w`, `conv_post_w`, `ups_w[3]`, `resblocks[9].{c1_w,c2_w}[3]`,
  `src_down_w[3]`, `src_resblocks[3].{c1_w,c2_w}[3]`, `f0_condnet_w[5]` — AND a
  `cv3_hift::tensors` map. ⚠ **cosyvoice3 ups are regular conv1d (via
  `cv3_causal_conv1d`), NOT conv_transpose** — so unlike HiFi-GAN/chatterbox, INCLUDE
  `ups_w` (do NOT exclude `.ups.`). Two viable idioms: (a) re-point every named
  conv-kernel field to a baked F32 copy at hift load (zero graph change, but must
  enumerate all ~85 fields); (b) thread `fc.get(w)` through the 3 helpers
  (`cv3_lookahead_conv1d`, `cv3_causal_conv1d`, `cv3_causal_grouped_conv1d`) + the
  raw resblock `ggml_conv_1d` sites. Gate `CRISPASR_COSYVOICE3_FASTCONV`.
  ⚠ Flow-matching (CFM) → seed-aware A/B, and each synth is slow like chatterbox;
  the hift vocoder is deterministic given mel, so the FAST verify path is the
  existing `cv3_extract_hift_decode_stage` harness driven with a fixed mel — A/B
  just the hift decode, no slow flow stage. **Left for a next cycle** (large edit +
  slow full-pipeline verify; not startable half-verified at session tail).
- **chatterbox_s3gen** — ✅ DONE (q8_0 default, 275 F16). Note its `-mtl` variant is
  all-F32 (no-op) — the q8_0 is the one that benefits.
- **indextts_voc (9 convs), kokoro (9)** — no local main model on this box; parse +
  wire on a box that has them.
- ⏭ **Next:** all local F16 FASTCONV targets are now landed (cosyvoice3 was the last).
  Remaining FASTCONV work needs a different box (TODO-C indextts/kokoro — no local
  model here) or is a separate GPU-A/B item (TODO-B chatterbox/cosyvoice3
  k=1→matmul). Then items 2 (interval-CFG), 3 (Metal q4_k), 4 (CI perf gate).

---

**Kickoff notes (superseded above): 26 backends, only 2 had FASTCONV.**

### Evidence (grep of `src/*.cpp`, 2026-07-16)
- **26 TTS backends** route their vocoder/codec through `core_dac::conv1d` /
  `core_convt::*` (which cast the F16 kernel → F32 inside EVERY graph).
- **Only 2** (omnivoice, qwen3_tts) have a baked-F32 fast path. **24 un-migrated.**
- `PERFORMANCE.md` claims FASTCONV "landed for qwen3-tts, voxtral-tts, omnivoice,
  tada, chatterbox" — but only 2 actually have it. Same overclaim class as the
  omnivoice one corrected this session; **coverage doc is unreliable, audit it.**
- **11 backends** use classifier-free guidance (chatterbox, cosyvoice3, dia, dots,
  f5, irodori, tada, vibevoice, voxcpm2, voxtral, zonos) — interval-CFG candidates.
- Rich benchmark harness already exists (`tools/benchmark_asr_engines.py`,
  `tests/benchmark_*`) but is NOT a CI gate; PERFORMANCE.md is hand-maintained + drifts.

---

## Item 1 — Shared FASTCONV in `core_dac` (24 backends) ★ highest ROI
The fork's `ggml_conv_1d` does `ggml_cast(F16→F32)` on the kernel inside every graph
when activations are F32 (~dozens of casts per codec decode), plus a pure-copy im2col
for k=1 convs. OmniVoice proved baking F32 kernels once at load + k=1→matmul = **2.9×
decode, output-equivalent**.

**Design (shared, no per-backend reinvention):**
- `core_dac::fastconv_cache` — owns a `ctx_f32` + backend buffer + `unordered_map<const
  ggml_tensor*, ggml_tensor*>`; `bake(backend, {conv weights})` at load converts each
  F16 kernel → F32 once; `get(w)` returns the baked F32 (or `w` if not F16/absent).
- `core_dac::conv1d(ctx, x, w, b, K, dil, const fastconv_cache* fc)` overload — when
  `fc` present+enabled: k=1 → `ggml_mul_mat` (skip im2col), K>1 → `ggml_conv_1d` with
  the baked F32 kernel (cast becomes a no-op). `fc == nullptr` → identical to today.
- Per backend: add a `fastconv_cache` member, `bake(...)` the codec convs at load
  (gated `CRISPASR_<BACKEND>_FASTCONV`, default per-backend after A/B), pass `&fc_`.
- **Unit test** (`tests/test-fastconv.cpp`, model-free): random F16 conv kernel + input,
  assert `conv1d(...,&fc)` ≈ `conv1d(...)` within F32 tol, and k=1 path exact.

**Rollout:** shared helper → migrate omnivoice (dogfood/de-dup) → pilot kokoro/piper/
melotts → remaining ~21, each byte+ASR A/B'd.

# ============================================================================
# PENDING — consolidated status board (OPUS-1M, 2026-07-16→17)
# ============================================================================
# Single place to see EVERY remaining task + its blocker. Detail in the TODO
# sections below. ✅ = landed this session · 🔨 = in progress · ⛔ = blocked.
#
# FASTCONV (default-on, byte-identical): ✅ 7 landed (omnivoice, irodori, zonos,
#   speecht5, chatterbox_s3gen, cosyvoice3, kokoro). No local F16-through-conv1d
#   targets remain (indextts F16 = custom-CPU-op filters; fastpitch f16 = dead-end).
# Interval-CFG (opt-in): ✅ 6 landed (cosyvoice3, f5, voxcpm2, dots, irodori, tada);
#   ✅ chatterbox opt-in, M1-VERIFIED (mel cos 0.980 via CFM-only harness); real-speech ASR + CUDA = off-box nice-to-haves. ⛔ dia/zonos/voxtral (AR batched-KV,
#   not amenable); vibevoice (low-value). NATURALNESS ear for ALL 7 = ⛔ human-only.
#
# STILL OPEN (by blocker):
#   • TODO-3 Metal q4_k→prefer-q8 on Apple — VERIFIABLE HERE (registry schema + test-
#     registry). Quick tier needs per-entry alt-quant metadata + HF q8-URL check. OPEN.
#   • TODO-B chatterbox k=1→matmul [SONNET] — needs CUDA A/B (drift-prone GPU). OPEN.
#   • TODO-2 chatterbox interval-CFG — DONE + M1-verified (mel cos 0.980, CFM-only
#     harness); real-speech ASR round-trip + CUDA = off-box nice-to-haves only.
#   • TODO-4 nightly perf-emit + committed baseline [SONNET] — needs GPU-in-CI +
#     noise-floor calibration; compare-logic+test already landed (unit-tested here).
#   • TODO-5 fused-step-graph rollout [FABLE] · TODO-6 voice-cache rollout [claimed] ·
#     TODO-7 CUDA-graph audit [SONNET→FABLE] — graph/CUDA work, off-box or FABLE-tier.
#   • TODO-3 deep (ggml Metal q4_k dequant kernel) + TODO-D fastpitch f16 re-convert
#     — larger, low-priority; documented dead-ends/kernels.
#   • NATURALNESS ears for the 6–7 interval-CFG backends at aggressive K = human-only.
# ============================================================================
# TODO QUEUE FOR A FRESH AGENT — fully scoped, do in this order

**Owner tags:** `[FABLE]` = top-tier model, graph/runtime math written by hand
against the diff harness (dev-guide rule: never delegate compute-graph code).
`[SONNET]` = delegable to a smaller model — pattern-proven, no graph math, has
a mechanical pass/fail recipe; verification (byte-A/B + roundtrip) is still
NON-NEGOTIABLE. Mixed tags split design/graph work from scaffolding.
# ============================================================================
# Every item below: work in a git worktree off origin/main, gate behind an env
# var, keep BOTH paths, A/B before flipping any default, format with
# tools/format.sh --fix (clang-format 18), rebase onto origin/main before every
# push (main moves constantly). See "Discipline" at the very bottom + the
# handover in handover-prompts/fastconv-fleet-sweep-round2.md.

## TODO-A — cosyvoice3_tts FASTCONV  ✅ DONE (`edbe64d28`, 2026-07-16)
Landed via re-point idiom (a): 85/85 F16 hift conv kernels baked+swapped,
byte-identical A/B (hash `127fb40eeec32d8f` ON==OFF==ON-rerun) with the new
deterministic `tests/cosyvoice3-hift-fastconv-ab.cpp` harness. Original plan below.

**Goal:** cast-kill the 85 F16 hift-vocoder conv kernels (same idea as speecht5/
chatterbox). GGUF-confirmed: `cosyvoice3-hift-f16.gguf` has 85 F16 3D conv kernels.
**Files:** `src/cosyvoice3_tts.cpp` only. Struct `cv3_hift` at :339 (fields listed
in the Bespoke-lambda triage above); conv helpers `cv3_lookahead_conv1d` (:2559),
`cv3_causal_conv1d` (:2577), `cv3_causal_grouped_conv1d` (:2515); main graph
`cv3_build_hift_decode_graph` (:4079); f0 graph `cv3_build_hift_f0_graph` (:3599).
**Steps:**
  1. Add `core_dac::fastconv_cache hift_fc;` to `cosyvoice3_tts_context` (or to
     `cv3_hift`). `#include "core/dac_decoder.h"`.
  2. At hift load (where `cv3_hift` named fields are assigned from `cv3_hift::tensors`):
     collect every F16 3D conv kernel (`conv_pre_w`, `conv_post_w`, `ups_w[3]`,
     `resblocks[9].{c1_w,c2_w}[3]`, `src_down_w[3]`, `src_resblocks[3].{c1_w,c2_w}[3]`,
     `f0_condnet_w[5]` — NOT the 2D `f0_classifier_w`/`m_source_l_linear_w`), then
     `hift_fc.bake(hift_backend, kernels, on)` gated `CRISPASR_COSYVOICE3_FASTCONV`.
  3. Re-point each named field to its baked copy: `f = hift_fc.get(f)`. ⚠ INCLUDE
     `ups_w` — cosyvoice3 upsamples with REGULAR conv1d (`cv3_causal_conv1d`), not
     conv_transpose, so it benefits (do NOT copy the chatterbox/HiFi-GAN `.ups`
     exclusion). Re-point is cleaner than threading `fc` through every helper.
  4. `hift_fc.free();` in the context free, BEFORE the hift backend is freed.
  5. Add a `CRISPASR_COSYVOICE3_FASTCONV_DEBUG` one-liner printing baked/swapped counts.
**Verify (FAST path — avoids the slow flow stage):** the hift vocoder is
deterministic given the mel. Drive `cv3_extract_hift_decode_stage(ctx, mel, T_mel,
s_stft, "hift_decode", &n)` with a FIXED mel (dump one from a short synth, or a
constant ramp) ON vs OFF → must be byte-identical (cast-kill). Full-pipeline synth
A/B (seed-aware, flow-matching) is the belt-and-braces but is slow (~min–h on M1).
Local models: `cosyvoice3-{llm-q4_k,flow-q8_0,campplus-f16,hift-f16,s3tok-f16,
voices}.gguf` all in `/Volumes/backups/ai/crispasr-gguf/`. **⚠ Stage models to the
internal disk first** — the external SSD reads at ~13 MB/s (near-full). Default ON
once byte-identical.

## TODO-B [SONNET] — chatterbox_s3gen k=1→matmul (further win, on top of the landed cast-kill)
The landed `CRISPASR_S3GEN_FASTCONV` does cast-kill only. 175 of its 275 F16 conv
kernels are K=1 — a K=1 conv is a channel matmul, so routing them through
`ggml_mul_mat` instead of `ggml_conv_1d`/im2col skips a pure-copy im2col
(materialises hundreds of MB at audio-rate T). BUT this changes reduction order, so
it is NOT bitwise-identical and needs its OWN seed-aware A/B on the drift-prone GPU
path (chatterbox has a documented GPU mul_mat drift history — see
handover-prompts/chatterbox-gpu-mul-mat-drift.md). Gate separately
(`CRISPASR_S3GEN_FASTCONV_MATMUL`), default OFF until A/B'd on Metal AND CUDA.
Same for TODO-A's cosyvoice3 K=1 kernels once cast-kill lands.

## TODO-C — kokoro ✅ DONE · indextts_voc ✅ investigated (not applicable)
- **kokoro** ✅ **DONE** (`323e96f23`, `CRISPASR_KOKORO_FASTCONV`, default on). GGUF
  finding: despite the `-q8_0` default, **all 89 3D conv kernels are F16** (q8_0 hits
  matmul/linear weights only) — FASTCONV engages. Re-point idiom (bake F32 + swap
  c->tensors); ConvTranspose ups use a separate F32 `ups_w_perm` (swap harmless);
  only 3D F16 baked (1D/2D linears read raw on CPU untouched). **A/B: ON vs OFF PCM
  BYTE-IDENTICAL, 89/89 baked, ASR intact.** ⚠ Lesson: the F16-kernel gate is
  PER-TENSOR — a q8_0 model can still keep conv kernels F16.
- **indextts_voc** ✅ **investigated — NOT a FASTCONV target.** `indextts-bigvgan` has
  161 F16 3D tensors, but they are BigVGAN anti-aliasing `us`/`ds` filters consumed by
  a **custom CPU op** (`aa_snake_beta_op`, read into `std::vector<float>` at load) —
  NOT through `ggml_conv_1d`, so there is no F16→F32 cast to kill. The learned convs
  (conv_pre/post/resblocks, 218 tensors) are already F32 → no-op. ⚠ Refined gate:
  "F16 kernels ROUTED THROUGH ggml_conv_1d", not just "F16 present". Do NOT wire.

## TODO-D — fastpitch f16 ✅ DIAGNOSED — a MULTI-bug dead-end (not a single loader fix)
Investigated end-to-end (2026-07-16). The loader error was mis-scoped as a reader bug;
it is actually **THREE separate bugs**, and fastpitch's `-f16` variant was clearly
never run (the q8_0 default masks them):
1. **Loader = CONVERTER offset bug, not a reader bug.** `gguf_init_from_file_ptr:
   tensor 'dec.layer.0.attn.qkv.bias' has offset 98304, expected 49152` — the f16
   GGUF has **non-sequential tensor-data offsets** (~49 KB gaps; 98304 = expected +
   one `attn.out.weight` = 49152×2). The ggml reader correctly rejects the gaps; the
   lenient Python `gguf` lib reads it fine via per-tensor `data_offset`. **A re-pack**
   (GGUFReader → GGUFWriter, preserving exact KV `field.types` + `raw_dtype=tensor_type`)
   rewrites sequential offsets and **LOADS**. So the file data is intact; only its
   layout is malformed → the real fix is re-CONVERTING from source with a correct
   writer (the uploaded `cstr/fastpitch-en-GGUF` f16 is malformed).
2. **Then Metal aborts:** `ggml-metal-ops.cpp:3355 GGML_ASSERT(op->src[1]->type ==
   F32)` — an F16 activation feeds a Metal op that requires F32.
3. **And CPU aborts:** `ggml_compute_forward_sub` — an F16/shape mismatch in a `sub`.
So the f16 code path has latent type bugs beyond the loader. **Verdict: not worth it**
— fastpitch's default is q8_0 (F32 kernels → FASTCONV no-op anyway), and unblocking
f16 needs (a) a re-converted GGUF AND (b) fixing 2 f16-path runtime type bugs. Left as
a documented dead-end; re-pack recipe above is the starting point if ever pursued.

## TODO-2 [OPUS-1M ACTIVE — chatterbox impl'd + M1-verifying; off-box CUDA verify OPEN] — Interval-CFG for guidance backends (~20–40%, biggest raw win, APPROXIMATE)
⚠ **Coordination:** OPUS-1M has already IMPLEMENTED chatterbox interval-CFG
(`CRISPASR_S3GEN_CFG_INTERVAL`, s3gen `cfm_euler_solve`) and is verifying on M1 (see
NOW claim). Fable/others: the only OPEN part is the **off-box CUDA verify** of the
chatterbox K>1 approximation (drift-prone GPU path). Do NOT re-implement.
✅ **cosyvoice3 landed opt-in** (`63a91a6a5`, `CRISPASR_COSYVOICE3_CFG_INTERVAL`,
default 1). `cv3_run_solve_euler` gains the uncond-skip-every-K path; K>1 forces the
separate 2-forward path (the batched `COSYVOICE3_CFG_BATCH` fuses cond+uncond, nothing
to skip). Verified via `tests/cosyvoice3-flow-cfg-interval-ab` (fixed-input flow
solver): K=1 byte-identical to legacy (cos=1.0 twice); K=2 mel cos 0.9994, K=3 0.9915
vs exact; non-silent, no NaN. ⚠ mel-cosine (synthetic inputs) is a content PROXY —
full real-text ASR round-trip + naturalness ear PENDING (slow CLI synth + human
listener); no naturalness verdict claimed.

✅ **f5-tts landed opt-in** (`678ee5ce1`, `CRISPASR_F5_CFG_INTERVAL`, default 1).
Same pattern in f5's 32-step ODE sampler; the separate 2-forward path is already the
default (batched `F5_BATCH_CFG` is opt-in/off), interval overrides it when enabled;
the CFG combine reads uncond via `v_unc_ptr` so K=1 is byte-unchanged. **Verified
end-to-end via the CLI (f5-tts-v1-base-f16, jfk voice, seed 42):** K=1 twice → PCM
byte-identical; content preserved — ASR(K1)==ASR(K2) exact (whisper base.en);
K=2 acoustic divergence log-STFT cos 0.945 / envelope corr 0.78 (⚠ PCM corr 0.089 is
phase-noise, NOT content — [[tts-parity-not-by-audio-corr]]). No new harness — used
`crispasr --tts … --voice … --seed`.

✅ **voxcpm2 landed opt-in** (`2c7cc5df4`, `CRISPASR_VOXCPM2_CFG_INTERVAL`, default 1).
Per-patch CFM denoiser `cfm_euler_solve` — the cond/uncond forwards are already two
separate `locdit_call()`s, so interval simply skips the uncond call every K steps.
**Verified via CLI (voxcpm2-q4_k, jfk voice, seed 42):** K=1 twice PCM byte-identical;
content preserved — ASR(K1)==ASR(K2). ⚠ voxcpm2 is AR at the patch level, so K=2
shifts the stop predictor one patch (3.36 s → 3.52 s output) — same words. ~8 CFG
backends still to do (below).

✅ **irodori landed opt-in** (`d04620cba`, `CRISPASR_IRODORI_CFG_INTERVAL`, default 1).
40-step RF-ODE with up to THREE independent uncond forwards per in-window step
(text/speaker/caption) — the heaviest CFG in the fleet, so the biggest per-step win.
Interval caches all three uncond velocities and recomputes them every K CFG-active
steps. **Verified via CLI (irodori-500m-v3 + dacvae-ja, JA text, seed 42):** K=1 twice
PCM byte-IDENTICAL; K=2 log-STFT cos 0.971. ⚠ JA ASR round-trip inconclusive (base
multilingual whisper garbles both K=1 and K=2 — weak JA ASR, not interval).

✅ **tada landed opt-in** (`3c8180cdc`, `CRISPASR_TADA_CFG_INTERVAL`, default 1).
FM Euler `fm_euler_solve`: the neg is a fixed per-solve uncond (the neg-KV cache is
the separate AR LLM, not the FM head), and `vel_neg` persists across iterations, so a
skip step just leaves its stale value — no cache buffer. Covers the default
single-candidate AND ranked paths; only the a_cfg!=1 (cosine-decayed prefix) steps
count. **Verified via CLI (tada-tts-1b-q4_k, text-only, seed 42):** K=1 twice PCM
byte-IDENTICAL; content preserved ASR(K1)==ASR(K2); K=2 log-STFT cos 0.978.

**Remaining TODO-2 candidates — only 1 clean (chatterbox), 4 non-amenable:**
- **chatterbox** (CFM s3gen) — clean flow pattern, but a full synth is ~1–3 h on this
  contended M1, so verify on a quieter box / CUDA. The one remaining clean copy.
- ⚠ **dia / zonos / voxtral** — AR with a BATCHED B=2 KV cache (cond+uncond share one
  cache); skipping the uncond forward corrupts its KV, so the simple interval skip
  does NOT apply (confirmed by reading dia's decoder). NOT amenable.
- ⚠ **vibevoice** — dominant CFG is the same AR neg-LM path (`run_lm_step kv_sel=1`);
  its DPM pred-head CFG is intervallable but cheap + coarse (fixed per-frame
  neg_condition, ~20 steps) → negligible ROI. Skipped (investigated).

Port OmniVoice's `OMNIVOICE_CFG_INTERVAL=K`: recompute the UNCOND classifier-free-
guidance forward only every K steps and reuse its cached logits; the COND forward
stays fresh every step; ALWAYS recompute the first + last step. Reference impl: grep
`OMNIVOICE_CFG_INTERVAL` in `src/omnivoice.cpp` for the caching pattern to copy.
**Candidates (11, all CFG):** f5 (ODE), chatterbox (CFM), vibevoice (DPM), voxcpm2,
cosyvoice3, dia, zonos, tada, dots, voxtral, irodori — do one at a time.
**Per backend:** gate `CRISPASR_<BACKEND>_CFG_INTERVAL` (or `<BACKEND>_CFG_INTERVAL`
to match omnivoice), default **1 (exact)** — this path is APPROXIMATE so it stays
OPT-IN forever, never a default flip. **Validation:** ASR-roundtrip the decoded
audio at K=1 vs K=2/3 — content must stay intact (word overlap ~1.0). ⚠ NATURALNESS
at aggressive K needs a HUMAN EAR (an agent can't listen) — so ship it enabled-but-
default-off with a doc note, do NOT claim a naturalness verdict. Document each env
var in this PLAN.

## TODO-5 [FABLE — graph math by hand; SONNET for scaffolding/bench] — Fused step graph rollout (the omnivoice #254 2.3× CUDA pattern)

Omnivoice's stage0 win (LEARNINGS 2026-07-16 entry; reporter-verified 2.3× /
RTF 0.07 on CUDA, byte-identical) came from killing per-step HOST detours in a
fixed-shape iterative loop: embeds computed in-graph from an ids-only upload,
target-slice-only logits readback, threaded scoring, ONE persistent graph per
arm so CUDA-graph capture engages. Survey of the fleet for the same shape
(fixed-T multi-step loop with per-step host work between forwards):

1. ✅ **dots_tts DiT/FM solver — LANDED (`53941ccc8`, 2026-07-17).** Persistent
   per-arm per-patch graphs, in-graph coordinate_proj, constants (prefix/mask/
   pos/g_cond) in a dedicated buffer, noise-slot-only velocity readback; shared
   dots_build_dit_body so legacy/fused cannot drift. Flow-match latents
   BYTE-IDENTICAL on CPU and Metal (CRISPASR_DOTS_FM_AB in-process A/B).
   Gate CRISPASR_DOTS_FUSED_STEP: default ON CPU/Metal (strict subset of legacy
   host work + identical output), OFF on CUDA pending TODO-7. Quiet-box timing
   + the voice-clone (g_cond) arm exercise ride with the TODO-7 audit.
   Original scoping (kept for reference): **dots_tts DiT/FM solver.** `dots_dit_forward`
   (src/dots_tts.cpp:903) builds a fresh ggml_context + graph + gallocr and
   frees them EVERY ODE step, for BOTH CFG arms (2 × num_steps rebuild+alloc
   per synthesis); `dots_linear` (:1461) additionally spins a one-shot graph
   per step for coordinate_proj on the host, and the full seq_c/seq_u embeds
   re-upload every step when only the noise-slot patch changes. Fixed shapes
   across steps → persistent per-arm graphs + in-graph coord_proj + noise-slot-
   only upload; optionally fuse cond+uncond via seq-concat + block-split attn
   (the omnivoice unified mode — dots already carries an additive attn mask, so
   the masking fits). Expect the omnivoice-class CUDA win; Metal likely neutral.
2. **vibevoice token-embed round-trip — AUDITED 2026-07-17, scoped.**
   `run_token_embedding_lookup` builds a fresh graph + sched_reset +
   sched_alloc + host round-trip PER AR TOKEN (call sites :1617 and :3535,
   n_ids=1) for a single get_rows. ⚠ tok_emb is QUANTIZED in shipped ggufs
   (the function exists for exactly that), so folding the lookup into the LM
   step graph would disable CUDA-graph capture (TAG_GET_ROWS_CUDA_GRAPHS) —
   the right design is a persistent embed MICRO-graph (built once, dedicated
   gallocr) whose output tensor is device-resident and referenced by the step
   graph, no host round-trip. Models are local (0.5b f16/q8, 7b q4_k) so the
   byte-A/B is runnable on this box. ✅ **LANDED (`736b70762`, 2026-07-17)**
   as a persistent 1-token embed micro-graph (own gallocr pinned to the
   weight's backend, sched-independent — pred_sched isolation rationale;
   quantized-GET_ROWS capture rule respected by NOT fusing into the step
   graph). A/B: ASR 7b decode transcripts identical across ~200+ per-token
   lookups (engagement print verified); realtime-TTS wav data byte-identical.
   Gate VIBEVOICE_PERSIST_EMBED (default ON). ⚠ Side-finding: the LOCAL
   vibevoice-7b-q4_k.gguf transcribes jfk.wav as garbage on BOTH arms
   (pre-existing) and vibevoice-realtime-0.5b-tts-f16.gguf is TRUNCATED
   (mmap bounds check fails) — likely one bad download batch; re-download
   and re-verify both [SONNET].
3. ✅ **tada FM — AUDITED 2026-07-17, already persistent, no action.**
   `fm_batch_gf` is built once in a dedicated meta arena (`fm_batch_meta`,
   rebuilt only when B changes); per step is alloc+set+compute. Remaining
   per-step gallocr/sched alloc is the standard idiom, low value.
4. **kugelaudio DPM pred head — AUDITED 2026-07-17, scoped, PARKED (no local
   model).** `build_pred_head_graph(ctx, 1)` is rebuilt INSIDE the 20-step
   DPM-SDE loop (× every generated frame; shared compute_meta arena, shared
   galloc). Fix: hoist build+alloc above the step loop (per frame — the
   shared meta arena is clobbered by the LM step graph between frames, so
   per-synthesis persistence needs its own arena like tada's fm_batch_meta);
   the 3 inputs are already re-set every step. [SONNET-capable once a
   kugelaudio gguf is present — lifecycle hoist + byte-cmp recipe; design
   decided here.]
4. **CUDA-graph capture audit (cheap, fleet-wide):** on a CUDA box, count
   `CUDA graph warmup` prints per synthesis for every iterative backend — more
   than one per distinct graph, or any `warmup reset` mid-loop, means capture
   is thrashing (omnivoice showed 4+reset pre-fix, 1 post-fix). ⚠ Before fusing
   any embedding lookup in-graph, check the embd table dtype in the SHIPPED
   quants: the fork disables CUDA-graph capture for GET_ROWS on a quantized
   table (TAG_GET_ROWS_CUDA_GRAPHS); omnivoice survives only because the
   quantizer keeps its embd tables F16 — zonos/parler/moss may not.
5. **NOT amenable:** pure-AR KV-cache decoders (dia/zonos/voxtral/moss) — a
   1-token step has no bulk embed prep or full-seq logits to fuse; their
   pattern is the persistent AR step graph (already the dev-guide default).
   Threaded scoring is likewise omnivoice-specific: MaskGIT scores all
   n_cb×T positions per step; AR backends score ONE position per step.
6. **Platform caveat (from the #254 M1 A/B):** these host-detour wins are
   CUDA-shaped. On unified-memory Metal the identical change measured NEUTRAL
   (embeds cost 0.9 s of ~94 s legacy gen) — the M1 cannot SHOW the win, so
   candidates must be judged on a discrete-GPU box (Kaggle / reporter-class).

**Model-tier triage (who does what):** the step-graph rework itself — block-
split attention fusion, gallocr aliasing discipline, bitwise-identity design
(order-matched adds, rng-stream-preserving threading), CUDA-graph capture
semantics — is runtime compute-graph work: **top-tier model, by hand, against
the diff harness** (dev-guide rule: never delegate graph math to agents).
Delegable to a smaller model (Sonnet-class): the surrounding scaffolding —
env-gate boilerplate, OMNIVOICE_BENCH-style stage timers, A/B runner scripts,
codes-dump/cmp verification runs, ASR roundtrips, README/env-var doc sync.

## TODO-6 [SONNET] — Reference-voice disk cache rollout (omnivoice OVC1 / pocket PVL1 pattern)

Deterministic, expensive reference encodes re-run on EVERY CLI invocation for
voice-cloning backends. pocket_tts (PVL1 latents) and omnivoice (OVC1 codes,
`007f82357`) now cache content-addressed (FNV-1a over PREPROCESSED audio +
encoder-weight fingerprint, atomic tmp+rename, CRISPASR_CACHE_DIR →
CRISPASR_MODELS_DIR → ~/.cache/crispasr, env-gated OFF switch). ~80-line
mirror per backend. Candidates (each has a ref-audio encode in
set-voice/prompt paths): qwen3_tts, chatterbox (campplus speaker emb + s3gen
prompt tokens), indextts (persist its in-memory `cond_latents`/`ref_cached`
to disk), f5_tts (re-encodes the ref mel every call), voxcpm2, dots_tts,
vibevoice, tada_tts, openvoice2, moss/dia (codec ref codes, same shape as
omnivoice). Verify per backend: run 2 logs a cache-hit line + decoded output
identical (compare WAV `data` chunks — the C2PA chunk is timestamped).

**Model-tier triage: Sonnet-friendly.** No graph math — an ~80-line mirror of
a twice-proven pattern (pocket_tts PVL1 `src/pocket_tts.cpp:3481`, omnivoice
OVC1 `omnivoice_set_voice_prompt`) with a mechanical verification recipe. The
two things a smaller model must get right (spell them out in the task prompt):
(a) hash the PREPROCESSED audio (post resample/RMS/trim), never the raw file
bytes; (b) fingerprint an encoder weight tensor so a re-converted model
re-encodes. One backend per worktree/commit; live-test each before the next.

## TODO-7 [SONNET phase 1 → FABLE phase 2] — CUDA-graph capture audit, fleet-wide (data collection → verdicts)

Two-phase. **Phase 1 (Sonnet-friendly, mechanical):** on a CUDA box run one
synthesis per iterative backend (dots, tada, kugelaudio, vibevoice, cosyvoice3,
f5, voxcpm2, irodori, chatterbox) with GGML_LOG_DEBUG visible; tabulate per
backend: #`CUDA graph warmup complete`, #`warmup reset`, and #distinct graphs
expected. Also record each shipped quant's embd-table dtype (gguf-dump) — a
quantized embd table + in-graph GET_ROWS disables capture entirely
(TAG_GET_ROWS_CUDA_GRAPHS). **Phase 2 (top-tier):** interpret thrash patterns
(reset mid-loop = unstable graph; warmups >> graphs = per-step rebuild) and fix
— that is step-graph restructuring per TODO-5.

## TODO-3 [FABLE design → SONNET impl] — Metal q4_k → prefer q8 on Apple Silicon
Measured earlier this campaign: q4_k is BOTH slower AND lower-quality than q8 on
Metal (Apple's q4_k dequant path). Two tiers:
- **Quick (config):** `src/crispasr_model_registry.cpp` is currently a STATIC table
  with ONE hardcoded quant per entry (no platform hook). Add a platform-aware
  preference: on Apple Silicon (`#ifdef __APPLE__` / runtime `ggml_backend_metal`),
  when both a q8 and q4_k variant exist for a model, PREFER q8 — but as a
  preference + stderr WARNING, NOT a forced flip (q8 is ~2× the download/RAM, a real
  tradeoff). `--model-quant` already exists as a manual override; respect it. Verify
  by checking the resolved filename on an Apple run vs a `--model-quant q4_k` override.
- **Deep (kernel, upstream ggml):** write a faster Metal q4_k dequant kernel in the
  ggml fork (`ggml/` submodule, branch `crispstrobe-ops`) — helps EVERY quantized
  backend on M-series. Bench with `tools/benchmark_asr_engines.py --json`. Upstream-
  able to ggml-org/llama.cpp (disclose only mechanical AI use, per the dev guide).

## TODO-4 [SONNET] — Perf-regression CI gate (highest leverage, but CI-only-verifiable)
🟡 **Compare logic + test LANDED** (`tools/perf_baseline_compare.py` +
`tests/test_perf_baseline_compare.py`, wired into the nightly `unit-tests` job).
`compare(baseline, current, factor=2.0)` is a pure function → HARD issues (a
baseline (engine,quant,mode,audio) MISSING / realtime_factor<=0 / empty transcript in
current — the LEARNINGS-4a crash-mints-a-fake-win case) + SOFT warnings (RTF below
baseline/2× — coarse, runner-noise-tolerant). CLI: `--strict` exits 1 only on a HARD
issue; default always exits 0 (informational). 9 unit tests green locally (pytest +
unittest); `--selftest` for a dependency-free smoke check. **Remaining (needs a
GPU-in-CI nightly run + a committed baseline JSON, calibrated over a few nights):**
add a nightly step that runs `benchmark_asr_engines.py --json`, commits the first run
as `tests/regression/perf_baseline.json`, and runs the compare informationally; also
add TTS backends via a TTS→ASR word-overlap gate (the manifest is ASR-only today).
This slice ships the verifiable core; the nightly wiring proves out only on the cron.

PERFORMANCE.md demonstrably drifts (this campaign corrected its FASTCONV overclaims
twice). Wire a real gate. **⚠ Design pitfall:** tight RTF gating on shared GitHub
runners is a FALSE-ALARM generator — the regression manifest itself notes "GH runner
decode diverges significantly." So gate the DETERMINISTIC signal hard and the noisy
one loose:
- **Building blocks that already exist:** `tools/benchmark_asr_engines.py` emits a
  rich JSON snapshot via `--json` (RunResult: engine, quant, `realtime_factor`,
  `transcript_sample`, wer, load_s). `.github/workflows/regression.yml` +
  `tests/regression/manifest.json` already do per-backend decoded-output goldens
  (transcript + CER/WER tolerances) for ASR, nightly. TTS backends are NOT in the
  manifest (it's ASR-only).
- **Robust design:** extend the nightly regression job (do NOT add a PR-blocking
  perf job) to also (a) HARD-gate the decoded output (exact / CER as today — this is
  the high-signal deterministic check), and (b) capture RTF into a committed
  baseline JSON and SOFT-warn (log, don't fail) only on a coarse ≥2× regression
  (catches algorithmic blowups, ignores runner noise). For TTS backends, the "decoded
  output" gate = TTS→ASR roundtrip word-overlap.
- **Verify:** the compare-script logic is locally unit-testable; the CI wiring itself
  only proves out on push (watch the Actions run). Land it as informational first,
  tighten thresholds after a few nightly runs establish the noise floor.

---

## Discipline (every item — NON-NEGOTIABLE)
1. Env-gated (`CRISPASR_<BACKEND>_<FEATURE>`), both paths kept, never remove gates.
2. A/B vs ground truth: byte/near-equivalence + ASR/decoded roundtrip, F16 AND quant.
   For stochastic (flow-matching/AR) backends: PASS A SEED; prove ON-vs-ON=0 (or that
   ON-vs-OFF=0 across all samples, which subsumes it) before trusting ON-vs-OFF.
3. Unit test where model-free (the shared conv helper especially — `test-fastconv`).
4. Flip default only on speed AND quality win; approximate paths (interval-CFG,
   k=1→matmul on GPU) stay opt-in until A/B'd.
5. Prove FASTCONV ENGAGED, not a silent no-op: a `*_FASTCONV_DEBUG` bake-count print
   (ON bakes N/N, OFF bakes 0). Byte-identity ALONE can't tell "equivalent" from
   "no-op" — GGUF-parse the shipped quant to confirm F16 conv kernels first.
6. Checkpoint: update this PLAN + push to main at each landed backend.

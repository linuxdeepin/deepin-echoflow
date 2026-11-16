# Music transcription in CrispASR

Porting the CometBeat / mus-textbook "transcription → SOTA" model roster
(`docs/TRANSCRIPTION_SOTA_HANDOFF.md` in that repo) from ONNX to CrispASR
ggml/GGUF backends.

## NOW — active work

- **Done**: feasibility triage of all 7 handoff workers (below); fixed a live
  `CAP_SEPARATE`/`CAP_STREAMING` bit collision (both were `1u << 22`; CLI builds
  clean after the move to bit 23). **CREPE converter landed and validated**:
  `models/convert-crepe-to-gguf.py` + `tools/crepe_numpy_parity.py`, cos=1.0 vs
  torchcrepe on both capacities (tiny 1.0 MB, full 44.5 MB at f16).
- **Done**: `src/crepe.{h,cpp}` runtime — **cos = 1.0 vs the numpy spec** on a
  real tone sweep, decoding 220.6 / 440.4 / 881.4 Hz at 0.95–0.97 confidence.
  `tests/test_crepe_parity.cpp` is the acceptance gate.
- **Done**: CREPE wired through the 12-point checklist — `CAP_PITCH = 1u << 24`,
  `examples/cli/crispasr_backend_crepe.cpp` (redirect shim, mirroring the
  htdemucs one), the `--pitch` early dispatcher
  (`examples/cli/crispasr_pitch_cli.{h,cpp}`, mirroring `--separate`), factory /
  roster / arch auto-detect (`crepe`) / filename heuristic in
  `crispasr_backend.cpp`, the session C ABI (`crispasr_session_pitch*` in
  `src/crispasr_c_api.cpp` + `include/crispasr_session.h`), registry entries for
  `cstr/crepe-GGUF` (**tiny is the default**), CMake linkage, README + docs/cli.md.
- **Done**: ✅ **BTC chord recognition** (branch `feat/cqt-and-chords`), full
  12-point checklist. `models/convert-btc-to-gguf.py` (213 tensors, both
  checkpoints), `tools/btc_torch_parity.py` (numpy spec, cos 0.99999995 /
  1.00000004 vs torch), `src/btc_chords.{h,cpp}`, the `--chords` early
  dispatcher (`examples/cli/crispasr_chords_cli.{h,cpp}`), arch auto-detect
  (`btc` → `btc-chords`) in `crispasr_detect_backend_from_gguf`, session C ABI
  (`crispasr_session_chords*`), wasm bindings (`sessionChords`), Go cgo
  LDFLAGS, registry entries, CMake linkage, README + docs/cli.md,
  `tests/test_btc_chords_live.cpp` + `CRISPASR_MODEL_BTC_CHORDS`.
  **`crispasr-diff btc` = 13/13 stages at cos 1.000000**; live session test
  41 assertions. Weights are CC-BY-NC-SA and ship behind
  `--accept-license cc-by-nc-sa-4.0`. Defaults to the **170-class** vocabulary
  (`CRISPASR_BTC_MAJ_MIN=1` collapses to maj/min — the reverse is impossible).
- **Done**: 🐞 **`core/cqt.h` was missing librosa's `scale=True`** — every bin
  came out low by `sqrt(N_k)` (152× at bin 0), so BTC read the features as
  near-silence and emitted `N` for every frame. Fixed by folding `sqrt(N)` into
  the kernel normalisation; verified against librosa (per-bin ratio median
  1.0002, magnitude cos 0.999941). **`tools/cqt_librosa_parity.py` could not
  see this**: correlation and peak-bin match are both scale-invariant, exactly
  like the htdemucs iSTFT scale bug that cosine let through. It now asserts on
  the median per-bin magnitude ratio; reverting the fix drives that median to
  0.0131, a 76× margin.
- **Done**: ✅ **piano-transcription per-stage validated — and it was BROKEN.**
  `piano_transcription_diff()` + registration in the diff binary (the reference
  dumper already existed but nothing consumed it). First run: mel PASS
  cos 1.000000, `conv_block_output` FAIL cos 0.810. Root cause: upstream passes
  momentum POSITIONALLY to `nn.BatchNorm2d`, where the second positional
  parameter is `eps` — so all 33 BatchNorm2d run at **eps=0.01**, while the 4
  BatchNorm1d (built with an explicit keyword) keep 1e-5. We hardcoded 1e-5
  everywhere; with running variances ~0.003 that is a 2.09x error per layer.
  Now **8/8 stages at cos 1.000000**. The weights were trained with the slip,
  so it is reproduced deliberately, like BTC's trailing ReLU.
- **Done**: ✅ **Real-music acceptance + two front-end parity bugs fixed.** The
  diff harness replays `input_feat` by design, so it never tested our CQT.
  Running the torch reference end-to-end on its own 257 s test clip exposed
  both: (1) `audio_file_to_features` CQTs each 10 s chunk INDEPENDENTLY and
  concatenates (2778 frames vs our continuous 2770); (2) frame duration is
  `inst_len/timestep` = 0.0925926 s, not `hop/sample_rate` = 0.0928798 s
  (0.79 s drift over 4 minutes). Our features scored cos 0.9993 vs a continuous
  librosa CQT but only **0.8815 vs the reference pipeline**; now 0.9993 vs the
  pipeline with an exact frame-count match. `mir_eval` agreement with the torch
  reference went **86.63 % → 98.56 %** (tetrads), 99.17 % root. Guarded by
  `[geometry]` tests; GGUFs carry `btc.inst_len_sec` and were re-uploaded.
- **Published**: `cstr/btc-chords-GGUF` — 4 GGUFs (170/25-class x f16/f32) +
  model card, all 13/13 at cos 1.000000. Licence gate verified live.
- **Done**: `cstr/crepe-GGUF` is PUBLISHED (verified 2026-07-20: f16 + q8_0 +
  q4_k for both tiny and full, 7 files). The earlier "not published yet" note
  was stale -- another session landed it. NOTE: the registry only lists the f16
  variants; the q8_0/q4_k uploads have no registry entry, so `-m auto` can
  never select them.
- **Done**: ✅ registry/packaging gaps closed. The four crepe quants were on HF
  with no registry entry (`-m auto` could never pick them) — now listed, each
  measured against the f16 of its capacity: tiny q8_0 cos 0.999993, tiny q4_k
  0.998757, full q8_0 0.999999, full q4_k 0.999933. BTC gained **q8_0 only**
  (4.5 MB, root 99.17 % / tetrads 98.52 % vs f16's 99.17 / 98.56 on real
  music). **q4_k deliberately NOT published**: 95.46 % tetrads for 0.6 MB
  saved. Quantize BTC from the f16 — only 73/213 tensors are quantizable, so a
  q8_0 from f32 lands at 7.5 MB, larger than the f16.
- **Done**: 📋 **Guitar tablature scoped — `GUITAR_TAB_SPEC.md` (§GT1).** A caller
  asked whether CrispASR should ship a `(string, fret)` **emission scorer** while
  their Viterbi/DP applies the hard constraints. Verdict is split by arm.
  **Audio → tab: adopt as proposed.** TabCNN (ISMIR 2019, ~0.8 M params) *is*
  already an emission scorer — six independent per-string softmaxes over 21 fret
  classes, no coupling, no CRF, no temporal model, **no decoding of any kind** —
  so our Viterbi over its output is a strict improvement on the published argmax.
  GuitarSet 6-fold player-wise: tab F1 0.748, TDR 0.899, multipitch 0.826.
  **But GuitarSet massively overstates**: zero-shot on EGSet12 (real electric
  guitar) tab F1 collapses 0.748 → **0.447**, TDR 0.899 → 0.695, and the fix is
  **data not architecture** — re-rendering with real tones/effects recovers to
  0.585 / 0.819 with the architecture held constant. So the shippable artifact is
  the **GuitarProFX-augmented** TabCNN, whose weights are public. FretNet does
  NOT supersede it on the headline tab metric (0.727 vs 0.717 like-for-like, and
  marginally worse on frame multipitch); its win is note-level via an onset head,
  and ⚠ its tablature head is *not* a per-string softmax.
  **Symbolic → tab: do NOT ship.** DadaGP is a dataset+tokenizer with *no*
  playability metric and *no* baseline comparison — it is zero evidence that
  neural beats classical. Its encoding is lossy where it matters (93.7 % of key
  signatures auto-assigned C major; 3/4 vs 6/8 indistinguishable; rare tunings
  dropped). And it is a **live licensing blocker**: Zenodo request-gated "FOR
  RESEARCH PURPOSES", *no* license on the 26,181 scraped GuitarPro files, MIT
  covers only the codec code, CC-BY-4.0 is arXiv's manuscript license, commercial
  fair use explicitly unresolved. Same shape as the BTC provenance problem —
  except BTC at least had a nameable license to gate on; **an `--accept-license`
  tag cannot launder an unlicensed scrape**. The two strongest symbolic models
  (MIDI-to-Tab ISMIR 2024, Fretting-Transformer ICMC 2025) are both ggml-sized
  but release **no weights** and both depend on DadaGP.
  **The interesting move**: the canonical HMM fingering decoder has *degenerate
  0/1 emissions* — all playability knowledge sits in hand-designed transitions —
  so there is an empty slot that learned per-(string,fret) emissions drop into
  without changing the DP. Nobody in the surveyed literature has run that
  experiment, and the 2016 data-scarcity reason for hand-designing is gone.
  ✅ **Both blockers now CLEARED (§GT1 §0) — the audio arm is GO.**
  (1) **Licensing is clean end-to-end**: GuitarSet v1.1.0 (Zenodo 3371780),
  EGSet12 (11406378), Guitar-TECHS and GOAT are **all CC BY 4.0**, open access,
  no NC/SA. Better still, the **EGSet12 record ships the trained model** — the
  GuitarProFX-augmented TabCNN this spec recommends — under CC BY 4.0. So the
  shippable weights have clean provenance and need only attribution. ⚠ Mind the
  code/weights gap: FretNet's repo is MIT but **TabCNN's has no licence file**,
  so build via the repo's existing **clean-room protocol** (Transcoda precedent)
  — weights from Zenodo, graph implemented from the ISMIR paper (§3.3–3.5 fully
  specify it), MIT FretNet as the readable reference, never transcribe the
  unlicensed source. (2) **Nothing in the 2024–26 wave supersedes TabCNN.**
  TART (2510.02597) turned out not to be a competitor on this axis at all: no
  GuitarSet 6-fold CV (plain 80/20), **no tablature F1 and no TDR whatsoever**,
  no comparison to TabCNN/FretNet, a *rule-based* final tab stage validated on a
  synthetic example — and its string/fret stage **is** the DadaGP-trained
  Fretting-Transformer, so it inherits the licensing blocker and scores 42.1 %
  tab accuracy. Guitar-TECHS and GOAT are datasets, not models. TART actually
  reproduces this spec's central finding independently: even the newest
  "comprehensive audio-to-tab tool" delegates string/fret to an autoregressive
  symbolic model. (3) Bonus: those four CC-BY-4.0 corpora together are a
  genuine **clean-room training set that avoids DadaGP entirely** — which did
  not exist when the DadaGP-dependent symbolic models were trained.
  Method note: 104 claims extracted from 21 primary sources, 25 adversarially
  verified, **5 killed** — including this session's own initial reading that
  DadaGP's baked-in `string:fret` tokens make the split *fatally* adverse
  (refuted 0-3). The opposite claim also died 0-3, so the verdict is "lossy and
  awkward", not "impossible". ⚠ Never table symbolic and audio numbers together —
  they share no metric, and MIDI-to-Tab *consumes GuitarSet's training split*.
- **Next**: `core/stft.h` extraction is independent (CREPE needs no STFT).
- **Done**: 🎸 **TabCNN ground-truth harness + converter landed.** The
  crispasr-diff regime works here — no golden oracle needed.
  **Provenance correction**: the EGSet12 weights are a pickled
  `amt_tools.models.tabcnn.TabCNN`, NOT the Keras `andywiggins/tab-cnn`. So the
  reference implementation is **amt-tools (MIT)** and **no clean-room constraint
  applies** — better than §GT1 §0 assumed. Weights stay CC BY 4.0.
  `tools/reference_backends/tabcnn.py` dumps 13 stages (audio → cqt_db → conv ×3
  → pool → dense → logits → `[T,6,21]`), registered in `dump_reference.py`.
  **Two front-end traps, both found by reading the MIT source instead of
  inferring**: (1) `model.frontend` is an **EMPTY Sequential** — the CQT lives
  outside the model, so a diff starting at features would never test our CQT,
  exactly the blind spot that cost BTC 86.63 %→98.56 % and broke
  piano-transcription; the dumper therefore emits `audio` and `cqt_db` as stages
  so the C++ is diffed from the waveform. (2) `post_proc` does **not** stop at
  `amplitude_to_db(ref=np.max)` — it affinely rescales `[-80,0]` dB to `[0,1]`
  via `/80 + 1`. My first reimplementation omitted it and measured **cos =
  −0.544**, median per-bin magnitude ratio **0.0047**, inputs ~180× out of
  range. Cosine alone would have partly hidden a pure scale error; the
  **|mine| = 15895 vs |ref| = 88** magnitudes made it obvious — HARD RULE 2b
  working as designed. Now **cos = 1.0000000000, max|diff| = 0.0, magnitude
  ratio = 1.0000000000** vs `amt_tools.features.CQT`.
  Geometry pinned from the object + amt_tools sources, not guessed: 833,982
  params, `dim_in=192`, `frame_width=9`, GuitarProfile E2-A2-D3-G3-B3-E4,
  `num_pitches=20` → SoftmaxGroups(6×21)=126, and `(192−6)//2 = 93` matches the
  observed pool height. **`sample_rate` is 44100, not 22050** — 192 bins at
  24/oct from E2 reaches 21096 Hz and only fits under Nyquist at 44.1 kHz; an
  earlier draft assumed 22.05 kHz and librosa correctly refused.
  `models/convert-tabcnn-to-gguf.py` → 10 tensors, **1.78 MB f16**, round-trip
  verified per tensor (conv + all biases **exact**, the two f16 matrices at
  2.6e-4 / 3.3e-4). It refuses to write on any geometry disagreement. Front-end
  constants, `silent_class` and the tuning array are written into the GGUF so
  the runtime cannot drift from the dumper.
  ⚠️ **Open design constraint**: `ref=np.max` is a **per-clip** normalisation, so
  the feature cannot be computed chunked without changing it. Settle the
  long-audio strategy BEFORE writing the runtime.
- **Done**: ✅ 🎸 **UNBLOCKED — root cause was a wrong `fmin`, and `core/cqt.h`
  is fine.** The end-to-end evaluation had been scoring **F1 0.0008 (TP=1 of
  1251)** against a published 0.447. Everything I suspected was innocent: the
  harness (rerun amt_tools-native end to end — their CQT, `pre_proc`,
  `finalize_output`, GT pipeline — reproduced my numbers exactly), the stereo
  (EGSet12 is dual-mono, channel corr 0.99999), normalisation (cancels under
  `ref=np.max`), the output layout, and string order.
  The fault was a **front-end constant I inferred instead of read**: I used
  `fmin = E2` (the guitar's lowest string — physically motivated and wrong) at
  44.1 kHz. DAFx-24 states plainly *"resampled to the 22050Hz sampling rate
  expected by TabCNN"*, and 192 bins at 24/oct only fit under that Nyquist from
  a much lower `fmin`. Measured on EGSet12 track 01 vs its JAMS ground truth:

  | config | tablature F1 | TP |
  |---|---|---|
  | sr 44100, fmin E2 (what I had) | 0.0008 | 1 |
  | sr 22050, fmin E1 | 0.0403 | 29 |
  | **sr 22050, fmin C1 (correct)** | **0.7708** | **533** |

  Note `fmin = C1` is *already* `core/cqt.h`'s default — TabCNN's front end is
  BTC's config except `n_bins` 192 vs 144 and `hop` 512 vs 2048.
- **Done**: ✅ **`core/cqt.h` needs NO librosa-compatible rewrite.** Re-taking the
  comparison at the correct params, through the real model against ground truth:
  librosa F1 **0.7708** vs core/cqt.h **0.7732** (**ΔF1 +0.0024**, core marginally
  better, within noise), prediction agreement **98.57 %**, feature cos 0.99880.
  The earlier "core/cqt.h changes 6.7 % of played notes, unusable" verdict was an
  artifact of running at the wrong front-end config — it is **withdrawn**. The
  planned option-1 DSP rewrite (librosa recursive per-octave downsampling) is
  **cancelled as unnecessary**; it would have been days of work to fix a
  non-problem.
  Constants corrected in `tools/reference_backends/tabcnn.py`,
  `models/convert-tabcnn-to-gguf.py`, `tools/tabcnn_torch_parity.py` and
  `tests/test-core-cqt.cpp`. Dumper front end re-verified bit-exact vs
  `amt_tools.features.CQT` (cos 1.0000000000, max|diff| 0.0); parity gate still
  11/11 PASS.
  **Durable lesson**: every wrong `fmin` still RAN and produced plausible-looking
  tensors. Only the task-level score against ground truth exposed it — cosine,
  shapes and stage parity were all green throughout. HARD RULE #3.
- **Next (guitar tab)**: blockers cleared, so the audio arm can start. Order:
  (1) ✅ done: weights pulled + tensor layout pinned; (2) `models/convert-tabcnn-to-gguf.py` +
  `tools/tabcnn_torch_parity.py` — and per the BTC/CQT lesson assert on the
  **median per-bin magnitude ratio**, not just cosine, since the front end is a
  CQT and cosine is scale-blind; (3) `src/tabcnn.{h,cpp}` emitting `[T, 6, 21]`
  **log-probs**; (4) `--tab` task surface per `contributing.md` §7. Acceptance is
  **EGSet12 zero-shot**, not GuitarSet — GuitarSet is the training protocol and
  flatters by ~0.30 F1. Symbolic arm stays parked (§GT1 §4.2/§2.3).

### Performance — measured, M1, quiet box (load 4.0), 10 s audio, median of 3

| model | Metal | CPU |
|---|---|---|
| full (44.5 MB f16) | 20.0 s — **RTF 2.0** | ~400 s — RTF 40 |
| tiny (1.0 MB f16) | 2.8 s — **RTF 0.28** | ~24 s — RTF 2.4 |

CREPE is genuinely expensive: at the reference 10 ms hop it is **1409 MMAC per
frame → 282 GFLOP per second of audio** for `full`, and 36.7 MMAC/frame →
7.3 GFLOP/s for `tiny` (38× cheaper). So **tiny is the shipping default** — it
is also what the handoff asks for ("smallest that hits accuracy"). `full` stays
available and is the right choice offline. Neither is close to real-time on CPU;
the GPU path is not optional here.

Three graph decisions got it from the first working version (RTF 31) to here:

1. **Batching (the big one).** One frame per dispatch wastes the GPU on a model
   this small per-frame. `kBatch = 64` makes each layer one large GEMM.
2. **Channel-fastest layout throughout.** `ggml_conv_1d` ends by permuting back
   to (OL, OC, N), materializing the whole activation every layer. We keep the
   mul_mat's native (OC, OL, N), do bias/relu/BN there — where a plain (OC)
   vector broadcasts along ne[0], ggml's fast path, instead of a stride-0
   (1, OC, 1) broadcast — and pool with `ggml_pool_2d(k0=1, k1=2)`. The one
   transpose im2col forces is deferred until *after* the pool, so it moves half
   the bytes, and the last layer skips it entirely because (OC, OL, N) already
   *is* the channel-fastest flatten the classifier wants.
3. **F32-baked conv kernels** (`ggml_conv_1d` casts an F16 kernel to F32 inside
   the graph — in a persistent graph that re-casts 44 MB per 10 ms frame).
   Gated `CRISPASR_CREPE_NO_BAKE_F32=1`. Honest note: this one measured
   **neutral** here, unlike qwen3-tts CODEC_FASTCONV. Kept gated-on because it
   is provably redundant work, but it was not the win.

Gates: `CRISPASR_CREPE_NO_GPU=1`, `CRISPASR_CREPE_NO_BAKE_F32=1`,
`CRISPASR_CREPE_DEBUG=1`.

### `ggml_conv_1d` returns a tensor whose declared shape contradicts its data for N > 1

**Status: fixed in the fork (`ggml/src/ggml.c`), upstream PR drafted at
`tools/upstream-prs/24-conv-1d-batch-reshape.md` + a standalone repro. NOT yet
merged to main — one audit item is open, see below.**

The im2col is the FIRST `ggml_mul_mat` argument, so the result's ne is
`[N*OL, OC]` (OC slowest). The final `ggml_reshape_3d` declares `[OL, OC, N]`
(N slowest). Those expressions coincide **exactly when N == 1** and differ
otherwise — which is why every shipping caller is correct and this was invisible.

Repro (`tools/upstream-prs/24-conv-1d-batch-reshape.repro.cpp`, standalone,
vs a hand-rolled direct convolution), before the fix:

```
N=1  cos=1.00000000  OK        N=2  cos=0.41129104  MISMATCH        N=3  cos=0.05935857  MISMATCH
```

After: all three `cos=1.0`. Fix reshapes to the true `[OL, N, OC]` then permutes;
the `N == 1` branch is the *unmodified original statement*, so batch-1 callers
are bit-identical **by construction**, not merely by test.

Corroborating facts:

- **Upstream `llama.cpp` has byte-identical code.** Not a fork regression, and
  not fixed upstream.
- **Upstream `test-backend-ops.cpp` has ZERO `conv_1d` cases.** It covers
  `IM2COL` and `MUL_MAT` as ops, but `ggml_conv_1d` is a composite graph
  builder, so the reshape between them is untested. That is the mechanism by
  which this survived.

#### ✅ AUDIT COMPLETE — landed in the fork (`CrispStrobe/ggml@662b05fb`)

The open question was whether any existing caller passes N > 1. Answered, and
**my original safety argument was wrong**:

- **CrispEmbed: zero `ggml_conv_1d` callers.** Unaffected entirely. (Its only 1-D
  conv use is two `ggml_conv_1d_dw` calls, a different function, both N == 1.)
- **CrispASR: 141 call sites** — 11 more than my `grep` found, because
  `ggml_conv_1d_ph` forwards to `ggml_conv_1d` without matching the literal
  string. **136 pass N == 1.** **2 pass N > 1.** 0 unknown.

The two batched callers are `aa_snake_beta_native` in `src/indextts_voc.cpp`
(:508 and :551), which deliberately maps **channels onto the batch axis** so one
depthwise FIR runs across all C channels at once. So "the N == 1 branch is
unmodified, therefore every caller is bit-identical" was **false** — those two
take the new branch.

They are safe for a *different* reason: their filter is `[K,1,1]`, i.e.
**OC == 1**, and with OC == 1 both branches produce the identical flat layout
`n*OL+ol` *and* the identical declared `ne`. Confirmed from the source (the
shape is documented at `indextts_voc.cpp:459-460` and enforced by a downstream
`ggml_reshape_2d` nelements assert) and verified empirically on that exact shape
class at N = 1..4. Neither site compensates for the old transpose, so nothing
depended on the broken layout.

**The branches diverge only when N > 1 AND OC > 1** — which no caller in either
repo does. CREPE would have been the first, which is why it surfaced here.

Gates run: standalone repro (both shape classes, all N) cos = 1.0; CrispASR unit
suite **1032/1032**; CREPE parity unchanged at cos = 1.0.

**Companion, now also landed (`CrispStrobe/ggml@655c14e4`): `ggml_conv_1d_dw`
batch support.** The first description of this (mine, repeating an agent's) was
wrong: it does NOT silently drop the batch dim. It reshapes to `[T,1,C,N]` and
hits `GGML_ASSERT(b->ne[3] == 1)` at `ggml.c:4476`, i.e. it **aborts** — a safe
failure, an unsupported case rather than a correctness bug. Verified by probing
it rather than reading it. Fixed by folding the batch into the channel axis and
tiling the kernel with `ggml_repeat`; verified N = 1..4 at cos = 1.0,
max_abs = 0.0 exactly. No existing caller changes (N == 1 path untouched, and
nothing can depend on an abort).

### CREPE weights are published

**https://huggingface.co/cstr/crepe-GGUF** — all six files (f16/q8_0/q4_k ×
tiny/full), `license: mit` verified present on the card via
`model_info(expand=["cardData"])`, public, ungated. So `-m auto` /
`--auto-download` now resolves. Published deliberately *before* the accuracy
eval, so that eval can be run on real music from the published artifacts.

#### ✅ ACCURACY EVAL ON REAL MUSIC — the octave concern did NOT reproduce

Run on 10 monophonic instrumental recordings (violin arco + pizz, piano, glock,
carillon, cello, flute, three folk melodies, brass), `tools/crepe_music_eval.py`.
No hand-labelled F0, so two load-independent proxies: **tiny-vs-full octave
disagreement** (|log2(a/b)| ~ 1) and **in-tessitura rate** on voiced frames
(voiced_prob >= 0.5).

| | tiny | full |
|---|---|---|
| in-tessitura (weighted) | **89.6%** | 89.0% |
| voiced frames | 8166 | 9165 |
| octave disagreement tiny-vs-full | **2.3%** | — |

**Conclusion: `crepe-tiny` is NOT meaningfully worse than `crepe-full` on real
monophonic music** — 0.6 pt apart on in-tessitura, 2.3% octave disagreement, and
on `10_amazing_brass` tiny is actually *better* (92.8% vs 83.9%). Given tiny is
38x cheaper (RTF 0.28 vs 2.0), **tiny stays the default.** The earlier
`samples/jfk.wav` octave worry was archival *speech* — out of domain for a model
being shipped for music — and it did not generalize.

Per-clip, the failures are domain limits shared by both capacities, not capacity
defects:
- `02_violin_pizz` — 49% / 52% in range. Plucked, fast-decaying transients; most
  frames have no sustained pitch. Worst clip by far for both.
- `05_carillon_ode` — only 39 (tiny) / 117 (full) of 1501 frames voiced at all.
  Bells are inharmonic; CREPE is trained on harmonic pitch. Correctly abstains
  rather than inventing pitch.
- Clean cases (fur_elise, cello, flute, row_boat, old_macdonald, glock) sit at
  96–100% in range with 0–6% octave disagreement.

⚠️ **Caveats on this eval, stated so it is not over-trusted.** (a) The tessitura
bounds are hand-guessed per instrument, so the *absolute* in-range numbers are
soft — `01_violin_scale` reads 86%/80%, which is more likely my bounds than real
error. The tiny-vs-full *comparison* is the robust part, since both are scored
identically. (b) A first version of this script also reported "fraction within
+/-50 cents of the nearest semitone" at exactly 100.0% for every clip and both
models — that metric is **vacuous by construction** (deviation from the *nearest*
semitone is bounded to +/-50c) and was removed. It is not evidence of anything.
(c) Real per-frame ground truth (a labelled MIR dataset) is still the honest way
to get an absolute note-F number against the handoff's "note-F >= 0.9" gate.

### Two measurement traps hit while benchmarking (both in the dev doc already)

- A run piped to `head -2` reported **0.79 s for 30 s of audio** (RTF 0.026,
  which would have been ~10 TFLOP/s — above M1's FP32 peak). SIGPIPE had killed
  it after two lines. The "too good, and the arithmetic disagrees" smell is what
  caught it; the frame-count-scales check (101 / 1001 / 3001) is what confirmed
  the real runs.
- Load average hit **253** mid-session, making every timing meaningless. Numbers
  above were all re-taken at load 4.0.
- **Branch**: `feat/music-transcription`, worktree
  `.claude/worktrees/music-transcription`.

### CREPE blueprint — the geometry the C++ must hit

Traced from `torchcrepe/model.py` + `core.py` + `convert.py` (the *source*, see
the warning below). Input is a 1024-sample 16 kHz frame, per-frame normalized
(`-= mean`, `/= max(std, 1e-10)`); hop is 10 ms; `pad=True` zero-pads
`WINDOW_SIZE//2` each edge.

Per layer: `F.pad -> conv -> F.relu -> batch_norm -> max_pool2d(2)`.

| layer | K | stride | pad (l, r) | out ch (full / tiny) | T out |
|---|---|---|---|---|---|
| conv1 | 512 | 4 | 254, 254 | 1024 / 128 | 1024 → 256 → 128 |
| conv2 | 64 | 1 | 31, **32** | 128 / 16 | 128 → 64 |
| conv3 | 64 | 1 | 31, **32** | 128 / 16 | 64 → 32 |
| conv4 | 64 | 1 | 31, **32** | 128 / 16 | 32 → 16 |
| conv5 | 64 | 1 | 31, **32** | 256 / 32 | 16 → 8 |
| conv6 | 64 | 1 | 31, **32** | 512 / 64 | 8 → 4 |

Then permute to (T, C) — **C is the fast axis** — flatten to `in_features`
(4 × 512 = 2048 full, 4 × 64 = 256 tiny), `classifier` Linear → 360, sigmoid.
Decode: `cents = 20 * bin + 1997.3794084376191`, `Hz = 10 * 2**(cents/1200)`.

Three traps, all now pinned by `tools/crepe_numpy_parity.py`:

1. **ReLU is BEFORE BatchNorm.** So the conv+BN fold is *invalid*. BN ships as a
   standalone per-channel affine (`_BN.scale`, `_BN.offset`, computed in f64).
2. **conv2..6 padding is asymmetric (31, 32)** and Metal rejects an asymmetric
   `GGML_OP_PAD` — use symmetric `p=32` and drop output column 0.
3. **`torchcrepe.convert.bins_to_cents` applies dithering** (triangular noise),
   so the reference is *non-deterministic*. Disable it when dumping parity
   fixtures, and do not implement it in C++. Also note torchcrepe's default
   decoder is **Viterbi**, not the handoff's weighted-average-around-argmax —
   implement `local_average` (original CREPE) and treat Viterbi as optional.

> ⚠️ **Lesson (HARD RULE #1, the expensive way).** The first converter folded BN
> into the conv, because a fetched *summary* of `model.py` listed the ops as
> "Batch Norm ... ReLU activation" in that order. The real source has the relu
> first. The failure looked like plausible numerics, not a structural bug: layer
> 1 at cos=0.83 with ~2× the reference magnitude — because least-squares fitting
> an affine through a *rectified* signal recovers about half the true scale. What
> caught it in one run was printing `|mine|` and `|ref|` per stage and noticing
> `|mine|` was **identical across four different input frames**. A fetched
> summary of source is not reading the source.

---

## Verdict: yes for the neural models, no for two of the seven

The handoff lists 7 workers. They are not the same kind of thing — four are
neural models (a CrispASR port makes sense), two are pure score-level algorithms
(they belong in Dart), and one is already shipped here.

| Worker | CrispASR? | Status / why |
|---|---|---|
| **W-SEP** | ✅ **already done** | HTDemucs (`src/htdemucs.cpp`, §248 full parity, `cstr/htdemucs-GGUF`) + Mel-Band RoFormer (`src/mel_band_roformer.cpp`, waveform bit-exact 2.4e-7). Both shipped with `--separate`, auto-download, C ABI, Python `Session.separate()`. **Don't export Open-Unmix to ONNX — call CrispASR.** |
| **W-CREPE** | ✅ port — start here | 6-layer 1D CNN on raw 16 kHz audio → 360-bin activation. No STFT, no attention, MIT. The single easiest port in the repo's history. |
| **W-PIANO** (slice 1) | ✅ port | Kong/ByteDance high-res piano CNN + biGRU on log-mel. `core/mel.h` covers the front-end; needs a **GRU** in `core/` (only LSTM exists today). |
| **Basic Pitch** | ✅ port | Already ONNX in the app; Apache-2.0, ~4 MB CNN over a harmonic-CQT stack. Needs a **CQT** front-end (absent). |
| **W-HARMONY** | ⚠️ port, licence-gated | Small CRNN/CQT chord model. Architecture is easy; the work is finding a checkpoint whose **licence** is actually permissive. Timebox the checkpoint hunt before the port. |
| **W-DRUMS** | ⚠️ mostly DSP | Onset + band-energy classification is DSP, and DSP belongs where the app is. Only worth a backend if a permissive drum-transcription CNN is chosen. |
| **W-MT3** (slice 2) | ⚠️ frontier, timebox | T5 encoder-decoder over spectrogram frames → MIDI-like tokens, Apache-2.0. The *architecture* is well-trodden in ggml (easier than ONNX, honestly). The risk is the **checkpoint format** — T5X/JAX gin, not HF safetensors — so the converter is the whole job. Feasibility memo before committing. |
| **W-METRE** | ❌ **not CrispASR** | Downbeat DP + metrical quantisation. No model, no tensors. Pure algorithm over a `RhythmGrid`. Keep in Dart. |
| **W-NOTATION** | ❌ **not CrispASR** | Voice separation, staff split, enharmonic spelling — operates on `crisp_notation` score types, not audio. Keep in Dart. |

So: **5 of 7 are worth porting, 1 is already done, 2 should stay in Dart.**

### Why port at all, given ONNX works

1. **W-SEP is the handoff's "biggest lever" and it already exists here**, at
   higher quality than the Open-Unmix fallback the handoff proposes, with
   per-stage cosine parity already validated. That alone justifies the seam.
2. **One runtime for the whole chain.** Separation → F0 → notes currently means
   ONNX Runtime *plus* whatever runs the stems. CrispASR already owns the audio
   IO, resampling, chunking, and model auto-download.
3. **Quantization.** `crispasr-quantize` gives q8_0/q4_k for free; these models
   ship as f32 ONNX. CREPE-full at q8_0 is a phone-sized model.
4. **Metal / CUDA / WASM** come from ggml, not from a per-model ONNX EP story.

The counter-argument is honest and worth stating: for **Basic Pitch and CREPE
specifically**, ONNX already works in the app today, and porting buys speed and
packaging, not capability. The capability wins are W-SEP (done), piano, and MT3.

---

## Architecture: a new task surface, not a `transcribe()` overload

`docs/source-separation-surface.md` already settled this argument for stems: a
task that returns something other than `crispasr_segment`s must **not** be
layered onto `transcribe()`; it gets its own early dispatcher before the ASR
backend is constructed. Music transcription (audio in → note events out) is the
same shape, so it copies that design:

- `src/core/note_events.h` — the result surface, mirroring
  `src/core/separation_io.h` (header-only, unit-testable without linking a
  backend). Carries the Dart-side seam types: `{midi, onMs, offMs, confidence}`
  note events, `{timeMs, f0Hz, voicedProb}` pitch frames.
- `examples/cli/crispasr_music_cli.{h,cpp}` — early route, mirroring
  `crispasr_separate_cli.{h,cpp}`, hooked once from `cli.cpp`.
- `CAP_MUSIC_TRANSCRIBE = 1u << 24` (bit 23 now belongs to `CAP_STREAMING`
  after the collision fix; 22 stays `CAP_SEPARATE`).
- A MIDI writer in `core/` so the CLI can emit `.mid` directly. MusicXML
  engraving stays in Dart — that's `crisp_notation`'s job, not a C runtime's.

**Contract compatibility.** The handoff freezes `contracts.dart`
(`PitchFrame` / `NoteEvent` / `RhythmGrid`). `core/note_events.h` is designed to
be a 1:1 memory-layout match so the Dart FFI binding is a reinterpret, not a
marshal. That is the whole point of the seam — an engine swaps behind it.

---

## Phase 0 — infrastructure (blocks everything else)

The survey turned up three real gaps. None is hard; all are prerequisites.

1. **`core/stft.h` — forward STFT.** `core/istft.h` exists but covers only the
   inverse. HTDemucs rolls its own (`src/htdemucs.cpp:548` `compute_stft`) and
   mel-band-roformer has a second copy. A music backend would be the **third**
   copy. Extract now, before adding to the pile.
   ⚠️ This refactors two *shipped* backends → per the A/B rule, it needs
   byte-identical stem output on both before it lands, gated if not.
2. **`core/cqt.h` — constant-Q / harmonic-CQT.** Absent entirely. Basic Pitch
   and every chord model want log-frequency bins. Built on (1).
3. **`core/gru.h`.** `core/lstm.h` has uni/bidirectional LSTM; the piano model
   needs biGRU. Mirror the LSTM file's structure.

Ordering: (1) → (3) can proceed in parallel with CREPE, which needs neither.

## Phase 1 — CREPE (recommended first backend)

Why first: it needs **zero** new infrastructure. Raw 16 kHz waveform in
(1024-sample frames), 6 conv+batchnorm+maxpool blocks, one 360-unit dense layer
out. No STFT, no attention, no autoregression, no tokenizer. It exercises the
entire new music surface end-to-end — CLI flag, capability bit, note-event
result type, converter, registry, C ABI, bindings — against the simplest
possible model, which is exactly how you want to debug a new surface.

- `models/convert-crepe-to-gguf.py` — from the MIT Keras/`torchcrepe` weights.
- `src/crepe.{h,cpp}` + the 12-point checklist in `docs/contributing.md`.
- Parity: `tools/reference_backends/crepe.py` → `crispasr-diff crepe`, per-stage
  cos ≥ 0.999 vs `torchcrepe`.
- **Acceptance is the decoded output, not cosine** (HARD RULE #3): synth a
  C-major scale → `crepe` → note segmentation → note-F ≥ 0.9 with **zero octave
  errors**, which is the specific failure the handoff wants fixed.

## Phase 2+ — piano, Basic Pitch, harmony, drums, MT3

Sequenced after phase 1 proves the surface. Each follows the same regime:
blueprint read line-by-line → converter → per-stage diff → decoded-output gate →
registry + 12-point checklist. MT3 gets a feasibility memo (checkpoint
conversion viability) **before** any C++ is written.

---

## Licence scoping of the remaining roster (2026-07-20)

Every candidate below was checked against the CometBeat HARD RULE — patent-free
and MIT/Apache-2.0-compatible — by reading the actual LICENSE file or HF card,
not from memory. **Code licence and WEIGHTS licence are tracked separately**,
because for chords they diverge and that divergence is the whole problem.

| Component | Code | Weights | Verdict |
|---|---|---|---|
| CREPE (marl, torchcrepe) | MIT | MIT | ✅ shipped |
| onnxcrepe (yqzhishen) | MIT | converted from torchcrepe + TF CREPE | ✅ useful as an ONNX cross-check |
| mangio-crepe (Mangio-RVC-Fork) | MIT | — same CREPE weights | ✅ **nothing to port** — see below |
| RMVPE (Dream-High) | **Apache-2.0** | **MIT** (`lj1995/VoiceConversionWebUI`) | ✅ clean — best quality tier |
| FCPE (CNChTu/TorchFCPE) | MIT | MIT repo | ✅ clean — cheapest tier |
| w-okada/voice-changer | MIT (6 holders incl. RVC, yxlllc) | mixed; **Beatrice v2 is a custom licence** | ⚠️ integration *reference* only — do NOT vendor |
| anyf0 (SoulMelody) | MIT | wraps crepe/fcpe/rmvpe | ✅ good reference implementation |
| Basic Pitch (Spotify) | Apache-2.0 | Apache-2.0 | ✅ clean — blocked on CQT, not licence |
| piano_transcription (Kong) | MIT | MIT | ✅ clean — in flight |
| **BTC-ISMIR19 (chords)** | **MIT**, ships `btc_model{,_large_voca}.pt` | trained on **Isophonics = CC BY-NC-SA** | ⚠️ **THE GATE** |
| MT3 | Apache-2.0 | T5X/JAX gin checkpoint | ⚠️ converter is the whole job |
| madmom / Essentia / aubio / Vamp | GPL/AGPL + Böck patents | — | ❌ excluded by the hard rule |

### mangio-crepe needs no port

It is **not a different model**. Mangio-RVC-Fork's contribution is a
*configurable `crepe_hop_length`* on the same MIT CREPE weights; its own README
recommends upstream RVC's CREPE for artifact handling. Our `src/crepe.cpp`
already exposes hop as a parameter, so this is covered. Worth stating plainly so
nobody spends a week on it.

### The chord problem is DATA provenance, not code

BTC-ISMIR19 is the obvious port — MIT code, pretrained checkpoints committed to
the repo, architecture we can already build (bi-directional self-attention over
CQT; every op exists in the CrispASR ggml stack). The catch is upstream of the
code: its checkpoints were trained on **Isophonics annotations, which are
CC BY-NC-SA** (non-commercial, share-alike), as are Robbie Williams and
UsPop2002.

Whether NC-licensed *annotations* encumber the resulting weights is legally
unsettled, and the repo ships them under MIT. But "unsettled" is not the bar
this project set. Three options, in order of preference:

1. **Retrain on ChoCo's permissive subset.** ChoCo aggregates 18 chord corpora
   under **CC BY 4.0**, with only three NC exceptions to exclude (Chordify
   Annotator Subjectivity, Mozart Piano Sonata, JAAH). That leaves Billboard,
   Real Book, RWC-Pop, Weimar Jazz, Wikifonia, iReal Pro, Band-in-a-Box, When in
   Rome, Rock Corpus, Nottingham, Schubert-Winterreise — ample for a small CRNN,
   with commercially-clean provenance we can state in the model card.
2. **Synthetic audio.** There is recent work on training chord recognisers on
   artificially generated audio (arXiv 2508.05878). Rendering progressions from
   permissive symbolic sources gives *fully* clean provenance and pairs well
   with option 1 as augmentation.
3. **Ship the chroma-template path** (already in `crisp_notation`
   `chroma_analysis.dart` / `analyze()`) as the default and treat the neural
   chord model as a later premium tier.

### DECISION (2026-07-20): port BTC now, gate the weights non-commercially

Superseding the "train first" recommendation above. We ship the BTC port with
its upstream checkpoints, treated as **non-commercial weights behind a
download-time attestation** — the same posture the repo already takes for
Voxtral-4B-TTS (CC-BY-NC-4.0) and the German moonshine models (CC-BY-NC-SA-4.0).

Why this is sound:

- **Our code stays MIT.** The BTC architecture is written from the paper into
  the CrispASR ggml stack; nothing GPL/AGPL is copied. CrispASR and CometBeat
  can both be commercial.
- **The restriction rides with the WEIGHTS, not the software.** Users obtain
  NC weights only after attesting they will not use them commercially, and the
  restriction is stated in the registry, the model card and the CLI.
- **Commercial users are not blocked** — they get the chroma-template path
  (already in `crisp_notation`), and later the ChoCo-trained Apache-2.0 model.

Training a clean model on the ChoCo CC-BY subset (option 1 above) remains the
target for a *commercially usable* chord tier. It is now a follow-up, not a
prerequisite, and the BTC port is what proves the architecture + surface first.

#### Required mechanism — MIRROR CrispEmbed, do not invent one

**`--i-have-rights` is the WRONG flag.** It attests *speaker consent for voice
cloning* — a third-party-rights question. Licence compliance is unrelated, and
one flag must not silently grant two different permissions. A separate mechanism
is required.

**CrispEmbed already has the right one** (`examples/cli/model_mgr.{h,cpp}`), and
it is stricter than anything CrispASR does today. Mirror it rather than inventing
a parallel design — the two repos should behave identically:

| CrispEmbed (existing) | CrispASR (to add) |
|---|---|
| `license_requires_acceptance(spdx)` — `cc-by-nc*`, `gemma`, `llama*`, `lfm1.0`, `other` | same predicate, same tag list |
| `resolve_model(arg, auto_download, accepted_license)` | extend `crispasr_resolve_model()` with the same parameter |
| `--accept-license <spdx>` | `--accept-license <spdx>` |
| `CRISPEMBED_ACCEPT_LICENSE` env fallback | `CRISPASR_ACCEPT_LICENSE` |
| accepts the exact SPDX tag, or `all` / `*` | same |
| TTY: prints licence + model-card URL, prompts `[y/N]` | same |
| non-TTY without acceptance: **refuses** | same |
| **`auto_download` alone is NOT sufficient** | same — this is the key property |

Two things CrispEmbed's design gets right that a blanket NC flag would not:

1. **Acceptance is per-licence, not blanket.** The user attests to a *specific*
   SPDX tag; `all` exists but is opt-in.
2. **SPDX tags, not substring matching.** CrispASR currently tests
   `license.find("NC")`, and that same NC-detection logic is duplicated in three
   places (`crispasr_model_registry.cpp` + two spots in
   `crispasr_model_mgr_cli.cpp`). Moving to SPDX tags de-duplicates it.

What CrispASR must change:

- Registry `license` becomes an SPDX tag (`cc-by-nc-sa-4.0`) with the prose
  reason kept separately, so the predicate is exact rather than a substring hit.
- `print_license_note()` currently fires AFTER `ensure_cached_file()` — an NC
  model is already on disk when the warning prints. The gate must precede the
  fetch, and must also cover the **cached-hit early return**, which today skips
  the notice entirely.
- Put the predicate + acceptance check in the library so CLI, session C-ABI and
  server all inherit it (the multi-surface rule).

CometBeat mirrors the same gate in its model store before fetching, and states
the restriction in the UI at the point of download — not buried in an About box.

Follow-up once the ChoCo-trained model exists: it is Apache-2.0, needs no gate,
and becomes the default; BTC stays as the opt-in higher-accuracy NC tier.

### CQT is the shared unlock

Basic Pitch and the chord CRNN both need a **constant-Q transform**, which
`core/` does not have (only `core/mel.h` and `core/fft.h`). Building
`core/cqt.h` once unblocks BOTH, and is the highest-leverage remaining
infrastructure item — ahead of either model port.

### F0 tier — CREPE is shipped, RMVPE is the quality upgrade

The handoff already flags RMVPE as "the quality tier after CREPE", and the
licence check confirms it is clean (Apache-2.0 code, MIT weights). It is also
what w-okada's guide recommends for all-purpose use, and it is robust to
accompaniment — which matters because our W-SEP stems are not perfectly clean.
FCPE is the cheap tier if CREPE-tiny proves too slow on low-end hardware.
Priority: RMVPE > FCPE, and neither is urgent while CREPE-tiny hits RTF 0.28.

---

## Additional CrispASR tasks from the cross-runtime scoping (2026-07-20)

Fell out of scoping which models CometBeat's **pure-Dart** ONNX runtime can
carry. That runtime can afford ~10–15 min for an offline whole-song analysis
job, which is a very different budget from interactive use — and it changes what
is worth having on the CrispASR side too.

### A cheap-separator tier

CrispASR has the two *best* separators (HTDemucs, Mel-Band RoFormer) and neither
of the *cheap* ones. That is a real gap for low-end hardware and for the
pure-Dart path:

| Candidate | Licence | Architecture | Why |
|---|---|---|---|
| **Spleeter 4-stem** (Deezer) | **MIT** | 12-layer U-Net on magnitude spectrograms — all convs | Cheapest separator that exists; 100x realtime on GPU. All-conv maps straight onto the existing im2col/GEMM path — no new op families. |
| **Open-Unmix** (`umx`/`umxhq`) | **MIT** | 3-layer BiLSTM on magnitude spectrograms | Named in the handoff. Cheaper than HTDemucs but SEQUENTIAL, so it parallelises badly; expect it to lose to Spleeter despite fewer FLOPs. |

Task: port **Spleeter first** (`core/lstm.h` is not even needed — it is pure
conv), measure against HTDemucs on the same clip, and register it as the
low-resource separation default. Open-Unmix only if Spleeter's TF-checkpoint
conversion turns out awkward.

Reference point for why this matters: HTDemucs costs ~103 s per 7.8 s segment in
a pure-Dart runtime — ~45 min for a 3.5-minute song, over any usable budget —
while an all-conv U-Net is roughly an order of magnitude cheaper.

### Pitch tiers below and above CREPE

- **`CRISPASR_CREPE_HOP` knob.** CREPE's cost is linear in frame count, so
  doubling the hop from the reference 10 ms to 20 ms halves the work for a
  modest resolution loss. Cheapest possible quality/speed lever on an already
  shipped backend; expose it and document the tradeoff.
- **FCPE** (CNChTu/TorchFCPE, **MIT**) — explicitly designed to be fast; the
  tier BELOW crepe-tiny for constrained devices.
- **RMVPE** (Dream-High, **Apache-2.0** code / **MIT** weights via
  `lj1995/VoiceConversionWebUI`) — the quality tier ABOVE CREPE, robust to
  accompaniment, which matters because our W-SEP stems are not perfectly clean.
  Non-autoregressive, so it fits an offline budget.

### Not worth porting

- **MT3.** Already flagged as frontier; the scoping sharpens *why*: it is
  autoregressive seq2seq, and token-by-token decoding is the one shape where a
  generous offline budget does not help. Deprioritise below everything above.
- **mangio-crepe.** Not a distinct model — a configurable hop on the same MIT
  CREPE weights (see the licence scoping). The `_HOP` knob above covers it.

### Division of labour with the pure-Dart runtime

CrispASR is the native/performance path; the Dart runtime is the web/WASM path
and the fallback. They should NOT both chase the same models. CrispASR keeps the
heavy, highest-quality engines (HTDemucs, Mel-Band RoFormer, RMVPE, and MT3 if
it ever lands); the Dart runtime takes the small permissive ones it can actually
finish in-budget (Basic Pitch, CREPE-tiny, BTC chords, Spleeter). `core/cqt.h`
and a Dart CQT are the one piece both need — worth keeping the two
implementations diff-checkable against each other.

## Open questions

- **Where does the app call this from?** CrispASR has Dart/Flutter bindings, so
  the seam can be FFI. But the handoff's engines are `!kIsWeb`-guarded with a
  pure-Dart web fallback — CrispASR's WASM build could actually *remove* that
  caveat. Worth confirming with the app author before designing the binding.
- **Model hosting.** Existing convention is `cstr/<name>-GGUF` on HF with a
  `license:` YAML tag that must be verified post-upload. CREPE (MIT), RMVPE
  (MIT weights), Basic Pitch (Apache-2.0) and piano_transcription (MIT) are all
  clean. The chord checkpoint is the one that does NOT survive vetting — see the
  licence scoping above; the plan is to train rather than port those weights.
- **Does the chord model need to be neural at all for v1?** The chroma-template
  path already exists in the app. If the ChoCo permissive subset proves thin,
  shipping DSP chords + a documented "premium tier later" may be the better
  trade than a weak model with clean provenance.
- ~~**Guitar tab — GuitarSet's licence**~~ ✅ **ANSWERED 2026-07-20**: CC BY 4.0,
  as are EGSet12, Guitar-TECHS and GOAT. EGSet12 additionally ships the trained
  augmented-TabCNN weights under CC BY 4.0. Unlike the chord problem, this one
  vets clean — port, don't retrain.
- ~~**Guitar tab — does TART supersede TabCNN?**~~ ✅ **ANSWERED**: no. No
  GuitarSet 6-fold, no tab F1/TDR, no TabCNN/FretNet comparison, rule-based
  final stage, DadaGP-trained string/fret stage at 42.1 % tab accuracy.
  Guitar-TECHS and GOAT are datasets. Still unread: arXiv 2510.10619 (symbolic
  only, cannot move the audio recommendation).
- **Guitar tab — is the clean-room corpus SUFFICIENT?** Licensing is solved
  (GuitarSet + EGSet12 + Guitar-TECHS + GOAT, all CC BY 4.0), so the open part
  is now purely empirical: is that enough data to train a symbolic emission
  scorer, and does SynthTab offer a licensable synthesis path? Note this is the
  *opposite* outcome to the chord problem — there the licence forced "train
  rather than port"; here the weights vet clean, so port first and treat
  training as the symbolic-arm option.
- **Guitar tab — does a learned emission scorer actually beat the hand-designed
  DP?** The classical HMM's emissions are degenerate 0/1, so the slot is empty
  and the experiment is architecturally trivial. Nobody has run it, and there is
  no agreed playability metric to score it with — which is itself the blocker.

## Suggested order (highest leverage first)

0. ~~**Licence-acceptance gate, ported from CrispEmbed**~~ ✅ **DONE** — land BEFORE any NC
   weights are registered, so there is never a window in which they are
   downloadable ungated. Also de-duplicates CrispASR's three copies of
   substring-based NC detection.
1. **`core/cqt.h`** — unblocks Basic Pitch AND the chord model. Infrastructure,
   no licence risk, reusable.
2. **BTC chords** — architecture from the paper, weights gated NC.
3. **Finish piano_transcription** — currently cos 0.971, below the 0.999 gate.
   It is the closest thing to a finished port that is not yet finished.
4. **Basic Pitch** — Apache-2.0 end to end, and the app already depends on it
   via ONNX, so this is a like-for-like replacement with a known-good oracle.
5. **RMVPE** — clean licence, real quality win on sung f0 over accompaniment.
6. **ChoCo-trained chord model** — the commercially-clean tier, Apache-2.0.
7. **MT3** — feasibility memo on the T5X checkpoint conversion FIRST.

Everything above is CPU/Metal-verifiable locally. The **Kaggle/CUDA run should
wait until this roster is complete**, so one clean CUDA session covers every
backend at once rather than being repeated per port.

---

## CometBeat Q&A — answers as of 2026-07-20 (§251)

Four asks came back from the CometBeat agent. Answers, with the facts checked
against `origin/main` rather than assumed:

**1. `separate()` Dart binding — DONE, stop shelling out to the CLI.**
Landed on main (`05ee77b17`). `CrispasrSession.separate(Float32List pcmStereo)
-> List<Stem>` where `Stem = ({String name, Float32List pcm})`, plus a
`separateSampleRate` probe. Verified end-to-end through the real FFI against
`cstr/htdemucs-GGUF` q4_k: 4 stems (drums/bass/other/vocals), interleaved
lengths exactly matching the input, ~4 s for 2 s of audio.
⚠️ Two contract points: input is **interleaved** stereo (`L,R,L,R…`) at 44100 Hz,
and the native side counts samples **per channel** — an odd-length buffer now
throws `ArgumentError` rather than being misread. To chain into `pitch()`,
downmix to mono and resample 44100→16000 first; the dartdoc carries the recipe.
**Not on pub.dev yet — it needs an 0.8.17 release.**

**2. Piano — bindable TODAY, but through `transcribe()`, not a note-event API.**
`piano-transcription` is session-openable now (`crispasr_session_open(path,
backend: "piano-transcription")`; it is in the C ABI backend list). There is
**no dedicated note-event C ABI**. The CLI adapter converts each detected note
into one `crispasr_segment`: `t0`/`t1` are onset/offset, and `text` is the note
name plus velocity (e.g. `"C4 v=80"`), with `CAP_TIMESTAMPS_NATIVE`.
So `loadCrispasrPiano` can be un-stubbed immediately by opening that backend and
parsing segments — but string-parsing `"C4 v=80"` back into a `NoteEvent` is
lossy and ugly. **A proper `crispasr_session_piano_notes*` C ABI returning
`{midi, onMs, offMs, velocity}` is the right fix and is now a §251 item.** Say if
you want it prioritised — it is small, and it is the difference between a hack
and a real seam.

**3. The dylib — correcting an assumption: the pub package does NOT ship or
download one.** `crispasr` is a pure Dart FFI package; it opens whatever native
library the host app provides. `CrispasrSession.open` probes, in order:
`libcrispasr.dylib`, `crispasr.framework/crispasr`, `libwhisper.dylib`,
`whisper.framework/whisper` on macOS/iOS (`libcrispasr.so` on Linux/Android), or
you pass `libPath:` explicitly. **Canonical filename: `libcrispasr.dylib`.**
Your cached 0.8.10 has no `crispasr_session_pitch`, which is exactly why your
fallback fires — correct behaviour. You need a dylib built from **0.8.16+** for
pitch + the `crepe` registry entry, and **0.8.17+ for `separate()`**. Build it
from the repo (`cmake --build build --target crispasr`) and bundle
`build/src/libcrispasr.dylib`. Your `tiny-q8_0` (~0.50 MB) choice is right —
q4_k moves the argmax pitch bin on ~1 frame in 7 at tiny.

**4. BTC licensing — your read is correct.** Keep your MIT ONNX BTC plus the
chroma-template fallback for the commercially-clean tier; do not ship our
weights. Our port gates them behind an attestation precisely so the two can
coexist. Cross-checking your `harmony_cqt.dart` against
`tools/cqt_librosa_parity.py` is worthwhile even though yours is already
librosa-validated — ours agrees with librosa at median 0.9999 per-frame shape
correlation and 97.6% exact peak-bin, with disagreement confined to
tone-transition frames (librosa's per-octave recursive downsampling gives each
octave a different group delay; our direct kernels are centred uniformly). If
your numbers differ materially at the transitions, that difference is expected
and not a bug in either.

### New §251 item from this exchange

- [x] **`crispasr_session_piano_notes*` C ABI** — SHIPPED (`include/crispasr_session.h:440`,
  Dart `pianoNotes()`). Return structured note events
  (`{midi, onMs, offMs, velocity}`) instead of forcing consumers to parse
  `"C4 v=80"` out of segment text. Then bind it in Dart mirroring `pitch()`.
  Blocks CometBeat's `loadCrispasrPiano` from being a clean seam.

---

## §251b — Model scoping, 18 candidates (2026-07-20)

Scoped 18 HuggingFace music models. **Three are worth porting; 13 are licence-dead
or off-contract; 2 are informative only.** Licences were verified from LICENSE
files and upstream repos, not just card YAML.

### ⚠️ PREMISE CORRECTION — upstream BTC is MIT, weights included

`jayg996/BTC-ISMIR19` carries a plain **MIT** LICENSE (Copyright 2019 Jonggwon
Park) and **ships both checkpoints inside that MIT repo**: `test/btc_model.pt`
(12,154,754 B) and `test/btc_model_large_voca.pt` (12,229,576 B). Verified via
the GitHub licence + contents API.

This contradicts the §BTC decision to gate the weights non-commercially behind
`crispasr_accept_noncommercial()` / `--accept-noncommercial`. That apparatus may
rest on a wrong premise — most likely a third-party HF mirror was mistaken for
the source. **Owner of the BTC work should confirm which artifact the NC belief
came from before more is built on the gate.** Not unilaterally removed here: the
attestation machinery is still useful for genuinely NC models, and the call
belongs to whoever made it.

### ✅ Worth porting

| Model | Licence | Size | Fills | Front-end we already have |
|---|---|---|---|---|
| **musetric/beat-this-onnx** | MIT (weights explicitly, upstream `CPJKU/beat_this` MIT) | 83 MB, 20.25 M | **`RhythmGrid`** — the one seam with no model | 128 **Slaney** mel @22050, n_fft 1024 / hop 441 → `core/mel.h`; filterbank ships as a raw `[513,128]` f32 blob, load verbatim |
| **musetric/chordmini-onnx** | MIT (upstream `ptnghia-j/ChordMini` MIT) | **9.6 MB, ~2.4 M** | chord timeline | **librosa CQT, 144 bins, 24/oct, fmin C1, hop 2048** — exactly what `core/cqt.h` was validated against |
| **livechord-music/livechord-beat-refiner** | **Apache-2.0**, code AND weights, ungated | 12.8 MB, 3.19 M | beat/downbeat *refinement* | 84 CQT (C1, 12 bpo) + chroma + onset + RMS @22050/hop 2048 → `core/cqt.h` |

**beat-this is patent-clean and madmom-free** — verified: its dependency list is
numpy/torch/torchaudio/einops/rotary-embedding-torch/soxr. Its paper is
literally "Accurate Beat Tracking Without DBN Postprocessing", so no Böck-patented
post-processing enters the chain even by distillation. That matters because
CometBeat must avoid madmom's DBN entirely.

**chordmini is a distilled STUDENT of the BTC teacher** — ~2.4 M params against
BTC's much larger body, MIT, 170-chord vocab in `config.json`. Worth an explicit
decision by the BTC owner: it could validate the BTC port or supersede it at a
fraction of the size.

**beat-refiner needs a prior grid** — it is a refiner, not a tracker. That is the
opening rather than the limitation: CometBeat's Ellis DP is patent-free but
phase-drifty, and drift is what this was trained to fix (with random grid
corruption, so it does not merely copy its input). **Ellis DP → refiner is a
fully patent-free chain.** It has only ever been evaluated on `beat_this` input,
never Ellis, so A/B before committing. Its chord-boundary head is explicitly not
production-ready (cb F1 0.243) — ignore that head.

### ❌ Licence-dead or off-contract

- **The MERT/MuQ trap** — `amaai-lab/merit` (MIT head) and `treadon/banger-scorer`
  (Apache-2.0 head) are 11 MB / 2.6 MB of `Linear` layers that produce NOTHING
  without `m-a-p/MERT-v1-330M` (**CC-BY-NC-4.0**) running first. Permissive tag,
  commercially worthless. Same shape for `nishitanand/FIGMA` (bundles MuQ,
  CC-BY-NC). Treat all as effectively non-commercial.
- `mtg-upf/omar-rq-base` — CC-BY-NC-SA-4.0, code AGPL-3.0.
- `MuScriptor/{large,medium,small}` — CC-BY-NC-4.0, gated, plus an indemnification
  clause. Painful, because audio→MIDI maps perfectly onto `NoteEvent`.
  `cocktailpeanut/muscriptor-small` is a **byte-identical ungated re-host**
  (sha256 match on the 411,888,600-byte file) — same licence, so using it to
  route around the gate is a bad look, not a loophole.
- `matteospanio/mule` — CC-BY-NC-4.0 (inherits Pandora's), code GPL-3.0.
- `csc-unipd/wav2taste` — CC-BY-NC-4.0, and taste scores match no seam type.
- `DionTimmer/audioestimate-1.0` — **no licence stated anywhere visible**, gated.
  UNKNOWN ⇒ assume restrictive.
- `csc-unipd/lilybert` — Apache-2.0 but a TEXT model over LilyPond source. No
  audio path, no seam contact.

### ℹ️ Informative only

- `Marcusfkelley/btc-hcqt` — MIT, 12.3 MB, BTC large-voca body with an HCQT front
  end. **Ties** baseline BTC by the author's own benchmark (GuitarSet 80.5/63.0 vs
  80.9/64.6); he states it as an honest negative result. Value is the mir_eval
  harness for calibration (~77–82% root is the plateau across BTC/CREMA/Chordino),
  not the weights. ⚠️ Its shipped checkpoint is Beatles-fine-tuned on Isophonics
  annotations that are research-only — the MIT grant is asserted over provenance
  that is not clean.
- `livechord-bar-arbitrator` — Apache-2.0, 750 KB, already ONNX so it drops into
  CometBeat's onnx path with no conversion. Consumes **no audio** (28-dim symbolic
  features over chords/beats/downbeats). For CrispASR the rule-based fallback it
  ships alongside is a few hundred lines of deterministic C++ we would rather own.

### Porting gotchas that would silently corrupt a port

- **Downmix must be arithmetic mean `(L+R)/2`, NOT `ffmpeg -ac 1`** (which uses
  energy-preserving `(L+R)/√2`). For chordmini that is a constant +0.20 shift
  after normalization; for beat-this the `log1p(1000·x)` compression makes it
  NON-constant and it moves beats.
- **Residual parity error in both musetric models is the RESAMPLER** (ffmpeg
  `swr` vs `soxr`), not the model. Expect ~0.37% of beats to flip near threshold;
  do not chase it as a port bug.
- **CQT substitution is not free**: an nnAudio CQT still correlates ~0.998 with
  librosa yet flips ~1.2% of chord frames. `core/cqt.h` being validated against
  librosa specifically is what makes chordmini portable.

### Order

- [ ] **1. beat-this** — biggest gap (`RhythmGrid` has no model), MIT, madmom-free,
  front end already available.
- [ ] **2. chordmini** — pending the BTC owner's call, since it overlaps.
- [ ] **3. livechord-beat-refiner** — after beat-this, and A/B on Ellis input first.

⚠️ Training-data provenance is undocumented upstream for the musetric exports, and
`beat_this`'s own README notes some training files are copyrighted or under
limited CC licences. The *weights* licences are verified clean; provenance is a
separate, unresolved upstream question.

---

## §251b-1 — beat-this port: traced blueprint (converter DONE, runtime TODO)

`models/convert-beat-this-to-gguf.py` is landed and validated (147 tensors,
40.8 MB f16; fold exact at f32, rel 1.9e-4 at f16). What follows is the forward
pass traced from `beat_this/model/{beat_tracker,roformer}.py` so the runtime can
be written without re-deriving it.

### Two subtleties that would silently break the port

**1. `RMSNorm` here is written in a form that does not look like RMSNorm.**

```python
self.scale = size ** 0.5
self.gamma = nn.Parameter(torch.ones(size))
forward: F.normalize(x, dim=-1) * self.scale * self.gamma
```

`F.normalize` divides by the L2 norm, and RMS = L2 / sqrt(size), so
`normalize(x) * sqrt(size) == x / RMS(x)`. It **is** algebraically standard
RMSNorm times gamma, and maps to `ggml_rms_norm` + `ggml_mul`. Do not
reimplement the L2 form literally and do not assume a bias — there is none.

**2. Attention has PER-HEAD sigmoid gating, applied before the out-projection.**

```python
gates = self.to_gates(x)                    # Linear(dim -> heads), WITH bias
out = out * gates.sigmoid()                 # broadcast over (head_dim, seq)
out = to_out(out)                           # Linear(dim_inner -> dim), NO bias
```

`to_gates` emits ONE scalar per head (not per channel), sigmoided and broadcast
across that head's `head_dim` and sequence. Miss it and the output is plausible
but wrong. Note the bias asymmetry: `to_qkv` and `to_out` are bias-free, `to_gates`
has a bias.

### Block order

```
Attention(x):  RMSNorm -> to_qkv(no bias) -> RoPE(q), RoPE(k) -> attend
               -> * sigmoid(to_gates(x)) per head -> to_out(no bias)
FeedForward:   RMSNorm(net.0.gamma) -> Linear(net.1) -> GELU -> Linear(net.4)
               (indices 1 and 4 are the checkpoint's own numbering)
```

Frontend `PartialFTTransformer` runs `attnF/ffF` over the FREQUENCY axis then
`attnT/ffT` over TIME, per block, with a shared `RotaryEmbedding(32)`.

### Tensor names (from the checkpoint, verified)

```
frontend.stem.bn1d.{scale,offset}                 <- pre-conv per-FREQ affine, NOT folded
frontend.stem.conv2d.{weight,bias}                <- BN folded in
frontend.blocks.{0,1,2}.partial.{attnF,attnT}.norm.gamma
frontend.blocks.{0,1,2}.partial.{attnF,attnT}.to_qkv.weight
frontend.blocks.{0,1,2}.partial.{attnF,attnT}.to_gates.{weight,bias}
frontend.blocks.{0,1,2}.partial.{attnF,attnT}.to_out.0.weight
frontend.blocks.{0,1,2}.partial.{ffF,ffT}.net.{0.gamma,1.weight,1.bias,4.weight,4.bias}
frontend.blocks.{0,1,2}.conv2d.{weight,bias}      <- BN (named `norm`!) folded in
frontend.linear.{weight,bias}                     <- 1024 -> 512
transformer_blocks.layers.{0..5}.{0=attn,1=ff}.*
transformer_blocks.norm.gamma                     <- norm_output
task_heads.beat_downbeat_lin.{weight,bias}        <- 512 -> 2
aux.mel_filterbank                                <- [513,128] baked verbatim
```

⚠️ The BN module name differs by location: `bn2d` in the stem, **`norm`** in the
frontend blocks. Found only because the converter's guard aborted rather than
silently mis-folding.

### Remaining work

- [x] **Front end DONE** — `beat_this_logmel`, **cos = 1.00000000** vs torchaudio
  (max_abs 1.2e-4). Built and validated before any network code, since a wrong
  window / squared magnitude / wrong normalization yields a plausible
  spectrogram and wrong beats.
- [x] **Per-stage reference DONE** — `tools/reference_backends/beat_this.py`
  dumps every stage so the graph comes up incrementally (first divergence = the
  bug). Checkpoint loads with **0 missing / 0 unexpected** keys, confirming the
  traced architecture. Contract the graph must reproduce, for a 101-frame input:

  | stage | shape |
  |---|---|
  | `stem` | (1, 32, 32, 101) |
  | `blk0` / `blk1` / `blk2` | (1, 64, 16, 101) / (1, 128, 8, 101) / (1, 256, 4, 101) |
  | `linear` | (1, 101, 512) |
  | `transformer` | (1, 101, 512) |
  | `out_beat`, `out_downbeat` | (1, 101) each |

- [x] **Stem DONE** — `beat_this_debug_stem`, **cos = 0.99999982** vs torch
  (max_abs 2.5e-3, |mine| 210.3733 vs |ref| 210.3613). Validates the layout
  mapping, the folded BN, the exact-erf GELU and the conv stride/pad convention
  in one shot. Compare with `tools/cmp_beat_this_stages.py`, which prints
  |mine| AND |ref| — a magnitude outlier says "same name, wrong data" instantly
  where cosine alone reads as plausible drift.

- [x] **`PartialFTTransformer` DONE** — all 3 frontend blocks, plus each
  block's conv+BN+GELU. Every stage matches the torch reference on the first
  run, with magnitudes agreeing to 4–5 digits (2026-07-20):

  | stage | cos | rel_max | \|mine\| / \|ref\| |
  |---|---|---|---|
  | `blk0_attnF` | 0.99999998 | 4.7e-4 | 35.5585 / 35.5532 |
  | `blk0_ffF`   | 0.99999990 | 7.9e-4 | 56.1502 / 56.1553 |
  | `blk0_attnT` | 0.99999973 | 5.8e-4 | 66.3518 / 66.3425 |
  | `blk0_ffT`   | 0.99999983 | 5.2e-4 | 30.2276 / 30.2300 |
  | `blk0_partial` | 0.99999989 | 2.5e-4 | 218.4039 / 218.3973 |
  | `blk0` | 0.99999987 | 1.8e-4 | 224.5992 / 224.5963 |
  | `blk1_partial` | 0.99999986 | 2.1e-4 | 238.5764 / 238.5707 |
  | `blk1` | 0.99999987 | 3.1e-4 | 229.2125 / 229.1943 |
  | `blk2_partial` | 0.99999990 | 3.1e-4 | 374.8126 / 374.7980 |
  | `blk2` | 0.99999988 | 4.7e-4 | 235.7255 / 235.7066 |

  Relative error stays flat at ~1e-4 across all **12** sub-blocks rather than
  compounding, which is the signature of f16 weight rounding and not of a
  systematic graph bug. (Blocks 1/2 are the same code at 2 and 4 heads.)

  Reproduce:

  ```bash
  ./build/bin/test-beat-this-stages $S/beat-this-f16.gguf $S/fe_ref.bin $S/bt/ \
      stem blk0_attnF blk0_ffF blk0_attnT blk0_ffT blk0_partial blk0 ... blk2
  python tools/cmp_beat_this_stages.py $S/bt_stages.npz --prefix $S/bt/ <stages...>
  ```

  **Traps that this stage actually hit or nearly hit — worth keeping:**

  1. **The gate reads the NORMED x.** Upstream rebinds `x = self.norm(x)` at the
     top of `Attention.forward`, so `to_gates(x)` sees the post-RMSNorm value,
     not the sub-block input. Feeding it the raw input is a one-token slip that
     produces a plausible activation.
  2. **RoPE is the INTERLEAVED convention** = `GGML_ROPE_TYPE_NORMAL`, *not*
     NEOX. `rotary-embedding-torch`'s `rotate_half` regroups `... (d r)` with
     `r=2` and repeats each frequency pairwise, i.e. adjacent-pair rotation.
     NEOX rotates split halves and would be silently wrong. Also verified: the
     checkpoint's 12 `rotary_embed.freqs` tensors are byte-identical to each
     other AND bit-exact against the analytic `10000^(-2i/32)` schedule, so
     ggml's internal theta table needs no override and the "shared instance"
     note costs nothing to honour.
  3. **`Attention.scale` is dead code.** It is computed in `__init__` but never
     passed to `Attend`, which therefore uses SDPA's default `1/sqrt(dim_head)`.
     Reading the constructor rather than the call site would have applied a
     scale twice.
  4. **Dump `ne` and reshape to `reversed(ne)`.** `cmp_beat_this_stages.py` no
     longer hand-transposes per stage; because ggml `ne` is the exact reverse of
     torch's shape, `reshape(reversed(ne))` lands on the reference layout with
     no transpose at all, and the F/T phases then differ only in which axis is
     ne[1]. Per-stage transposes were the likelier bug than the arithmetic.

  **Negative control (run, then reverted).** Deleting only the per-head gating
  multiply gives `blk0_attnF` cos = **0.977** — which reads as ordinary drift —
  but |mine| 90.8 vs |ref| 35.6, a 2.6× tell; at `blk2_attnF` (4 heads) it is
  cos 0.714 and 432.3 vs 46.6. This is the concrete case for the project rule
  that **cosine alone is not a gate and magnitude must always be printed**.
  It also confirms the harness has teeth rather than passing degenerately.

  Forward, traced from source and confirmed against sub-block references:

  ```
  x (b, c, f, t)
    -> rearrange "b c f t -> (b t) f c"     # FREQUENCY is the sequence axis
    x = x + attnF(x);  x = x + ffF(x)
    -> rearrange "(b t) f c -> (b f) t c"   # TIME is the sequence axis
    x = x + attnT(x);  x = x + ffT(x)
    -> rearrange "(b f) t c -> b c f t"
  ```

  Confirmed sub-block shapes for a 101-frame input, block 0 (dim 32):
  `attnF`/`ffF` = (101, 32, 32) — 101 sequences of length 32;
  `attnT`/`ffT` = (32, 101, 32) — 32 sequences of length 101.
  Batch is FOLDED into the sequence count, so ggml batches along ne[2].
  ⚠️ The reference hooks capture the sub-block OUTPUT, i.e. the residual BRANCH
  before `x + branch` — not the post-residual activation. Compare like for like.

  Per sub-block: RMSNorm -> to_qkv(no bias) -> RoPE(q,k) -> attend
  -> `* sigmoid(to_gates(x))` PER HEAD -> to_out(no bias). Reusable:
  `core/attention.h` (QKV + RoPE), `core/ffn.h`; the per-head gating and the
  `norm_output` RMSNorm need writing by hand.

- [x] **Transformer body + head DONE** — `frontend.linear`, the 6 roformer
  layers (dim 512, 16 heads), `norm_output` and `SumHead`. The whole network is
  now at parity:

  | stage | cos | rel_max | \|mine\| / \|ref\| |
  |---|---|---|---|
  | `linear` | 0.99999999 | 2.1e-4 | 1678.9711 / 1678.8176 |
  | `transformer` | 0.99999999 | 1.2e-4 | 197.9073 / 197.9059 |
  | `out_beat` | 0.99999978 | 7.3e-4 | 6.5327 / 6.5318 |
  | `out_downbeat` | 0.99999996 | 3.9e-4 | 17.8942 / 17.8894 |

  The layers reuse `bt_attention`/`bt_feedforward` verbatim. The only new risk
  was the concat: `"b c f t -> b t (c f)"` flattens with **frequency fastest**,
  so in ggml the target is ne = (f, c, t) before the reshape. The opposite order
  transposes the 1024-vector and yields a plausible, wrong projection.

  Also worth recording: `SumHead` makes `beat` the **SUM** of both logits and
  `downbeat` the second alone. The two outputs are not independent and it is
  not a 2-way softmax — upstream's reason is to suppress downbeats that are not
  beats.

- [x] **Windowing + postprocessing DONE** — `beat_this_logits` and
  `beat_this_events_from_logits`, reproducing `split_piece` /
  `aggregate_prediction` / `Postprocessor(type="minimal")`. Scored on a 45 s
  synthetic 120 BPM click track (2251 frames = **2 chunks**, 725-frame overlap;
  `tools/gen_beat_this_track_ref.py`, `tools/cmp_beat_this_track.py`):

  | check | result |
  |---|---|
  | chunk coverage | 0 frames left at the -1000 sentinel |
  | beat logits | cos 0.99999993, max_abs 1.07e-2 |
  | downbeat logits | cos 0.99999997, max_abs 9.34e-3 |
  | postproc on the reference's OWN logits | 91 beats / 25 downbeats, **max_dt 0.000000 s** |
  | end to end | 91 / 25, max_dt 0.000000 s, tempo 120.00 BPM |

  Feeding the reference's own logits through our peak-picker is what makes the
  postprocessing independently falsifiable — otherwise a peak-pick bug and a
  numerical difference are the same symptom.

  **Traps found by reading `inference.py` / `postprocessor.py` rather than the
  paraphrase:**

  1. **`deduplicate_peaks` AVERAGES a run, it does not keep the first.** The
     resulting frame index is FRACTIONAL (`p += (p2-p)/c`, a running mean), and
     the grouping test compares the next peak against that **running mean**,
     not against the previous peak. "Keep first" would bias every straddled
     beat 10–20 ms early.
  2. **The peak test is `!= max_pool1d`, not `> neighbours`.** A plateau of
     equal logits keeps EVERY frame in the plateau, and dedup is what collapses
     it. A strict local maximum silently drops beats landing between frames.
  3. **Downbeat snapping happens in SECONDS, after dedup**, then `np.unique`
     removes downbeats that collapsed onto the same beat.
  4. **The -1000 initialiser is a coverage assertion.** It is a logit, so it can
     never win a peak-pick — meaning a chunking gap is invisible at the event
     level. `cmp_beat_this_track.py` therefore counts surviving -1000 frames.
  5. `split_piece`'s `avoid_short_end` makes the LAST chunk overlap the previous
     one by more than 2*border. Chunk starts here are `[-6, 757]`, not evenly
     spaced — an evenly-spaced assumption still covers the piece and still
     drifts.

  **Negative control (run, then reverted):** flipping the aggregation to
  `keep_last` leaves the beat **count unchanged at 91** and the tempo still
  reading exactly 120.00 BPM, while logits fall to cos 0.9954 and a beat moves
  a full frame (0.02 s). Count and tempo are therefore *not* adequate gates for
  a seam bug — only per-event times and the logits are.
- [ ] Front end: 22050 Hz mono, **arithmetic-mean downmix**, STFT n_fft 1024 /
  hop 441 / periodic Hann / `normalized='frame_length'`, project onto the baked
  filterbank, `log1p(1000*x)`. 50 fps.
- [ ] Windowing: 1500-frame chunks, `borderSize` 6, `keep_first` overlap. Must
  reproduce `split_piece` / `aggregate_prediction` or results drift at seams.
- [x] Postprocess — done and exact; see the windowing entry above for the
  ways the ±3/threshold-0/dedupe description understates the real algorithm.
  **No DBN** — that is the point of the model and the patent constraint.
- [x] Parity harness vs the torch reference — `test-beat-this-stages` (19
  network stages) and `test-beat-this-track` (windowing + postprocessing).

- [x] **Surfaces DONE** — `--beats` CLI (+ `--beats-format text|json`), the
  backend shim and all four registration sites, the session C ABI
  (`crispasr_session_beats` / `_n_events` / `_events` / `_tempo_bpm` /
  `_sample_rate`), README + docs/cli.md, a regenerated feature matrix, and the
  Dart binding (`beats()`, `beatsSampleRate`, `beatsTempoBpm`, the `Beat`
  record) with a live test. `tools/check-backend-wiring.py` REQUIRED passes.

  That tool landed on `main` mid-task and immediately earned its keep: it
  caught beat-this missing from c_api dispatch, `available_backends` and the
  feature-matrix regen — three of the four sites in exactly the trap
  `crispasr_backend_btc.cpp`'s header warns about. **Run it before claiming a
  backend is wired**; the CLI working proves nothing about the other surfaces.

  Dart live test on a 20 s 120 BPM click track: 38 beats, 14 downbeats,
  tempo 120.00 BPM. CLI on the 45 s fixture: 91 beats, 25 downbeats, 120.0 BPM.

- [x] **Two memory bugs, found only by running the CLI on real audio.**

  1. **A heap overflow that had been live since the front end was signed off.**
     `beat_this_logmel` sized its FFT output `2 * n_freqs` (1026 floats), but
     `core_fft::fft_radix2_wrapper` writes the FULL complex spectrum —
     `2 * n_fft` = 2048 floats. Every frame overwrote ~4 KB past the vector.

     **It scored cos = 1.00000000 the whole time**, because the corrupted bytes
     are past the ones read back. The 101-frame fixture never crashed; a 45 s
     file (2251 frames) corrupted the heap reliably, surfacing as an
     intermittent wild-pointer `memmove` and a nonsense 4.39e18 allocation.
     Every other `core_fft` caller in the tree already sized this `2*fft_size`.

     **The lesson worth keeping: a parity harness cannot see this class of bug.**
     Perfect cosine says the arithmetic is right, not that the memory is. Run
     the real surface on real, LONG input before believing "verified".

  2. Latent: `bt_run` sized its position-id scratch by `T`, but the frequency
     axis is always 32 entries, so audio under ~0.64 s overran it. Now
     `max(T, 32)`, covered by a 0.2 s smoke test.

  I initially misdiagnosed (1) as memory pressure from the attention matrices
  and went looking at ggml's flash-attention kernel. Recorded because the
  misdiagnosis was plausible and cost real time: the giveaway that it was heap
  corruption rather than exhaustion was the *nonsense* size (4.4 exabytes), not
  a merely large one.

- [x] **Flash attention** — kept on its own merits after it turned out not to be
  the segfault fix. Upstream's `Attend()` is `F.scaled_dot_product_attention`,
  which never materialises the N x N scores; N is the 1500-frame chunk, so the
  attnT scores alone are 288 MB and the graph's compute buffer measured
  **322 MB**. `ggml_flash_attn_ext` takes the same graph to **47.6 MB** (6.8x),
  which matters because the downstream consumer is a mobile app. Its output is
  already `(D, H, N, B)`, removing a permute and simplifying the per-head
  gating. Parity unchanged to 7 digits.

- [x] **Weights published + registry entry DONE.** `cstr/beat-this-GGUF` is
  public with `beat-this-f16.gguf` (41 MB, default) and `beat-this-f32.gguf`
  (81 MB, parity reference), both SHA-256-verified against the local artifacts
  that the parity runs used. `--beats -m auto --auto-download` fetches into a
  clean cache and reproduces the local result exactly (91 beats / 25 downbeats
  / 120.0 BPM). No licence gate — MIT for code AND weights, unlike btc-chords.

  **f32 closes the numerics question.** Every stage scores
  **cos = 1.00000000, rel err ~1e-6** at f32 vs ~5e-4 at f16, with norms equal
  to 4 decimals. That is the proof the f16 residual was weight quantisation and
  the graph itself is exact — worth having as a shipped artifact rather than a
  claim, which is why f32 is published rather than just measured.

  `tools/check-backend-wiring.py` is now clean for beat-this on BOTH tiers
  (required and advisory).

- [ ] **NOW — active work.** §251b is complete and shipped; the runtime, all
  surfaces and the published weights are on `main` and green (1080/1080 units,
  wiring audit PASS, all 226 registry URLs resolve). The `beats()` Dart API and
  its CHANGELOG entry are in at version 0.8.19.

  One optional step remains, and it is the user's call because it is
  irreversible: **publishing the Dart package to pub.dev.** `VERSION` is already
  0.8.19 and the `v0.8.18`/`v0.8.19` tags exist, but there is **no
  `crispasr-v0.8.19` tag** — the publish workflow triggers on that `crispasr-v*`
  tag, and the last one pushed was `crispasr-v0.8.17`. So everything since 0.8.17
  (beats, and the intervening native fixes) is on `main` but NOT on pub.dev.
  Pushing `crispasr-v0.8.19` would publish it; a pub.dev version cannot be
  unpublished, so it waits for an explicit go-ahead.

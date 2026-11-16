# Beatrice v2 — port blueprint (§CB2)

Reference: `fierce-cats/beatrice-trainer`, file `beatrice_trainer/__main__.py`
(4519 lines; `__init__.py` is only `from .__main__ import *`). **MIT**, source
and trained models alike.

This is the HARD RULE #1 read: every claim below was checked against that file
or measured, not summarised. Where a detail is a silent-bug risk it says so.

Status: **`PitchEstimator` converter + reference dump DONE and validated.** No
ggml graph yet. `PhoneExtractor`, `VectorQuantizer`, `ConverterNetwork` and
`Vocoder` are not yet read in full.

| artifact | state |
|---|---|
| `models/convert-beatrice-to-gguf.py` | pitch_estimator: 88 tensors, 36 dropped as fusion-neutralised (124 total, balances) |
| `tools/beatrice_torch_parity.py` | 36-stage reference dump, spec reproduces `forward()` **bit-identically** |
| `src/beatrice_pitch.{h,cpp}` | **DONE** — 30 stages + end-to-end, 0 failed |
| `crispasr-diff beatrice` | **DONE** |
| PhoneExtractor converter + 73-stage reference | **DONE** (spec bit-identical) |
| `src/beatrice_phone.{h,cpp}` | **DONE** — 69 stages + end-to-end, 0 failed |
| ConverterNetwork / Vocoder | blueprint read done; **no converter, no graph** |

Shared ConvNeXt primitives live in `src/core/beatrice_ops.h`; both backends use
one copy.

## A residual branch can hide a completely broken sub-graph

The PhoneExtractor blocks are `x = attn(x) + x` then `x = conv(x) + x`. The
reference originally dumped only the **post-residual** sums, and against those
the port reported **0 stages failed, end-to-end cos 0.99999970** — while the
attention branch had never been meaningfully compared.

The tell was in the numbers: `pblock0_attn` had max_abs 7.699e-04 against
`backbone_norm`'s 7.700e-04 — the same value to four figures, which is what you
see when a branch adds nothing observable. Proof came from a negative control:
substituting a **chunked** split for the **strided** interleave broke 41 later
stages while `pblock0_attn` still passed at **cos 0.99999986**. A stage
dominated by its residual cannot see its own branch.

Fix: dump and compare the branch output *before* the residual add
(`pblock*_attn_delta`). Control there is **cos 1.00000000, max_abs 4.3e-07**;
the same broken interleave now localises to `pblock1_attn_delta` at cos 0.9946,
rel 0.487. **Generalises to any residual architecture** — comparing only
post-residual tensors is close to comparing nothing.

### One more capture artifact, and how it announced itself

The first `attn_delta` run reported **cos −0.012** — apparently 20 broken
attention blocks. It was not: `back` is a reshape **view**, and `ggml_set_output`
on a view does not stop the allocator recycling the parent's buffer, so the
capture read recycled memory. The contradiction is what gave it away — a branch
differing by max_abs 1.2 whose post-residual sum matched at cos 0.9999999 cannot
both be true. Materialising with `ggml_cont` before capture fixed it. (This is
the second allocator-recycling bug in this port; the first was a missing
`ggml_set_output` altogether.)

### The gate is cosine, and that is a compromise

A relative max-abs gate was tried at 1e-4 and **reverted — it failed the control
on 65 stages.** f32 noise on the waveform convolutions legitimately reaches rel
2.7e-04 (`fe_conv0`) and 1.3e-03 end to end, while the broken interleave carries
6.3e-04 at `attn_delta` — i.e. the bug's error sits *below* the control's own
noise at other stages, so no single global threshold separates them.

So `rel` is printed but not gated. Read the numbers per stage: on `attn_delta`
the control runs 4e-07 to ~3e-05, and either negative control (chunked
interleave, or missing causal mask) jumps to ~6e-04 and trips 60 stages.

## PhoneExtractor — upstream's own fusion is an approximation

`PhoneExtractor` is 6 weight-normed strided convs (÷160 → the same 100 Hz), a
`FeatureProjection` LayerNorm, a **20-block** ConvNeXt stack with
`use_mha=True`, and a weight-normed 1×1 head applied after a GELU.

Its fusion must **not** be applied wholesale. Bisected on the real checkpoint:

| transform | relative error |
|---|---|
| `remove_weight_norm()` | 0.000e+00 (exact) |
| `backbone.merge_weights()` | 1.15e-06 (f32 noise) |
| **full `merge_weights()`** | **1.71e-02 — breaks the model** |

The full method additionally folds `feature_projection.norm`'s affine into
`backbone.embed`, adding `sum_{i,k} W[o,i,k] · norm_bias[i]` to the conv bias.
That is correct only if every kernel tap sees `norm_bias` — but the sequence is
**zero-padded**, so the taps hanging off the ends see 0. It is an edge
approximation, and because this backbone is `use_mha=True`, attention spreads
the edge error across the whole sequence: ~3e-04 mid-sequence, ~3e-03 at the
start.

So the converter skips that one fold and keeps `feature_projection.norm`
explicit. **Our path is exact where upstream's is not** — relevant when
comparing against a `.beatrice` dump, which carries the approximation.

### The dumper's gate cannot see a bad fusion — the converter now can

This was nearly missed. Applying upstream's full `merge_weights()` in the
reference dumper still reports **cos 1.0000000000, max_abs 0.000e+00**: the gate
compares its step-by-step spec against the module's own `forward()`, and a
fusion changes *both arms identically*. It is structurally blind to this class
of bug, and the 1.7e-02 error was found only by a manual bisect.

`convert-beatrice-to-gguf.py` therefore runs the model **before and after** its
transforms on a fixed input and refuses to write a GGUF if they diverge beyond
1e-4. Verified by negative control: substituting the full `merge_weights()`
gives `rel=1.860e-02`, exit 1, and **no file written**. Correct conversions
report ~6e-07.

The MHA path was negative-controlled too: a chunked split instead of the
**strided** interleave (frame `t` → subsequence `t%4`, position `t//4`) gives
cos 0.693, and dropping the causal mask gives cos 0.910. Both correctly refuse.

---

## Why the earlier assessment was wrong

PLAN previously recorded §CB2 as *"custom/NON-COMMERCIAL"* and a WebFetch
summary of this repo claimed it held *"training scripts only, not inference code
or the model architecture"*. **Both are false.** `LICENSE` is plain MIT and the
README states the source **and the trained models** are MIT; the architecture is
fully defined in the file above. The genuine non-MIT signals — `beatrice.lib`
(the closed VST inference engine, "used under permission") and the "JVS Corpus
Edition" — describe *other artifacts* and do not constrain a port from this
source. See `PLAN.md` §CB2.

---

## Shape of the thing

Unlike RVC (§CB1), Beatrice is **end-to-end from waveform**:

```
ConverterNetwork.forward(x, target_speaker_id, formant_shift_semitone,
                         pitch_shift_semitone=None, ...)
    x: [batch, 1, wav_length]   ->   24 kHz audio
```

It runs phone extraction and pitch estimation *itself*, so a caller supplies no
ContentVec — CometBeat's side gets simpler than RVC's, at the cost of three
networks to port instead of one.

`out_sample_rate = 24000` and `hop_length = 24000 // 100` are hardcoded in
`ConverterNetwork.__init__`, so the **100 Hz frame rate is the same as §CB1's**
and `SVC_RECORD_SHAPES.md` carries over unchanged.

Weights are split across three files (see PLAN §CB2 table) — `phone_extractor`,
`pitch_estimator` and `net_g` each live in a *different* checkpoint, and
`net_g`'s file also carries a `net_d` discriminator that is training-only.

---

## Details that are NOT the obvious default

Each of these produces plausible-but-wrong output if assumed rather than read.

### 1. Weight standardisation uses UNBIASED variance

`WSConv1d.standardized_weight()`:

```python
var, mean = torch.var_mean(self.weight, [1, 2], keepdim=True)
scale = self.gain * (in_channels * kernel_size // groups * var + 1e-8).rsqrt()
return scale * (self.weight - mean)
```

`torch.var_mean` defaults to **correction=1 (unbiased)**. `np.var` defaults to
`ddof=0`. Measured on a 2x2x3 tensor: torch `3.5`, numpy-default `2.9167` — a
~20 % scale error on **every** weight, uniformly, which reads as "the model is
just a bit off" rather than as a bug. Use `ddof=1`.

`WSLinear` does the same over dim 1 with `in_features * var`.

**Fuse at convert time.** `merge_weights()` writes `standardized_weight()` back
into `.weight` and sets `gain = 1`, exactly as `convert-rvc-to-gguf.py` fuses
weight_norm. Do that in the converter so the runtime graph has no special case.

### 2. GELU is the tanh approximation, everywhere

Every activation on the pitch path is `F.gelu(..., approximate="tanh")`, not the
default exact/erf GELU. ggml's `ggml_gelu` is the tanh approximation and
`ggml_gelu_erf` is exact — so `ggml_gelu` is correct here, but the choice must
be deliberate, and the numpy spec must use the tanh form.

### 3. Causal convs carry a delay/trim, and trim is not the padding

`CausalConv1d.__init__`:

```python
padding = (kernel_size - 1) * dilation - delay
self.trim = (kernel_size - 1) * dilation - 2 * delay      # NOT equal to padding
```

`forward` right-trims by `self.trim` when nonzero. With `delay > 0` the layer is
*not* strictly causal — it looks ahead by `delay` frames, which is how the
stated latency budget is spent (`PitchEstimator` comment:
`delay=1  # 10ms, 特徴抽出と合わせると 22.5ms` — 10 ms here, 22.5 ms including
feature extraction). Getting `trim` wrong shifts the output in time, which
cosine similarity punishes hard but which is easy to misread as a broken graph.

### 4. There is an RNG site in the vocoder

`overlap_add()` sets a **random initial phase**:

```python
normalized_freq[:, 0] = torch.rand(batch_size, device=pitch.device)
```

So Beatrice, like RVC, is **stochastic and cannot be validated by waveform
comparison against an unpinned reference run**. §CB1's discipline applies
directly: make the draw injectable, replay it in the harness, and keep the
production default random. Note this is `torch.rand` (uniform), where RVC's live
site was `torch.randn_like` — a noise injector patching only `randn_like` would
miss it and the harness would silently compare against a *different* phase.

The phase itself is accumulated in **float64** (`normalized_freq.double()
.cumsum_(1) % 1.0`) before being cast back to float32. A float32 cumsum over a
long signal drifts; this is deliberate and must be reproduced in double.

### 5. `sample_pitch` is a banded argmax, not an argmax

`PitchEstimator.sample_pitch` does: softmax over bins -> **force bin 0 to
-100.0** (it is the unvoiced class, excluded from the pitch argmax) -> a
box-filter `conv1d` of width `band_width=4` -> argmax over the *filtered*
sequence -> a masked argmax within the winning band. Plain `argmax(logits)` is a
different function and will disagree on ambiguous frames only — i.e. it will
look almost right.

`return_features=True` additionally emits unvoiced / half-pitch / double-pitch
probabilities, using `pitch_bins_per_octave = 96` as the octave stride, and
`ConverterNetwork` consumes 4 pitch feature channels (`embed_pitch_features =
nn.Conv1d(4, hidden, 1)`).

### 6. The `.beatrice` dump format pre-folds the attention scale

`dump_layer()` is the trainer's export path to the closed inference engine. For
`nn.MultiheadAttention` it bakes the attention scale into the weights:

```python
in_proj_weight[: 2 * embed_dim] *= 1.0 / math.sqrt(math.sqrt(embed_dim // num_heads))
```

— i.e. `1/sqrt(d_head)` split as `sqrt` on **both** q and k, and it also
reorders to `[num_heads, 3, head_dim, embed_dim]`.

**Convert from the `.pt` state_dict, not from a `.beatrice` dump.** Reading a
dump means inheriting both transformations; applying the usual `1/sqrt(d)` on
top of pre-scaled weights would scale attention logits by `1/d`. Recorded here
because dumps are what circulate.

### 7. `ConverterNetwork` initialisation traps

* `embed_quantized_pitch` is a **fixed sinusoidal table** built in `__init__`
  and `requires_grad_(False)` — verify whether it is present in the checkpoint;
  if not, the converter must rebuild it (including the `* sqrt(4/5)` scale).
* `key_value_speaker_embedding` initialises **every speaker row as a copy of row
  0**. An A/B across speakers that shows little difference is therefore not
  automatically a port bug.
* `self.melspectrograms` is **loss-only** and not on the inference path.
* `VectorQuantizer` is injected into `phone_extractor.head` via a **forward
  hook** (`enable_hook`). This session lost time twice to hooks that never
  fired (RVC's `flow` calls `.forward()` directly, so `register_forward_hook`
  was silently comparing nothing). The spec must *prove* the hook is live by
  asserting the quantised path changes the output — never by assuming it ran.

---

## Reference evaluation — is it any good?

Run before committing to the port, per [[tts-advisory-check-blueprint-first]].
**Verdict: yes on clean input, and the port target is well defined.**

There is **no inference CLI** — `__main__.py`'s argparse is training-only
(`-d/-o/-r/-c`). Driving it takes: build `PhoneExtractor` + `PitchEstimator`,
load their two checkpoints, build `ConverterNetwork(pe, pi, n_speakers=200,
pitch_bins=448, hidden_channels=256, vq_topk=4)`, load `net_g`, then
**`net_g.enable_hook()`** to activate the VectorQuantizer. Only extra dep is
`pyworld`. Scripts: `run_ref.py` / `probe2.py` (scratchpad).

**`load_state_dict` reports 0 missing, 0 unexpected** against the 177-tensor
`net_g`. The architecture read above therefore matches the shipped checkpoint
exactly — the converter can and should assert exact key coverage rather than
using `strict=False` tolerance. This also resolves blueprint item 7:
`embed_quantized_pitch.weight` **is present** (448x256), so the converter does
not need to rebuild the sinusoidal table (but must not assume that for
checkpoints from other training runs).

ASR roundtrip (whisper base), same sentence, 24 kHz output:

| input | transcript of the CONVERTED audio |
|---|---|
| clean TTS speech (in domain) | "And so my fellow **in Erickons**, ask not what your country can do for you, ask what you can do for your country." |
| `samples/jfk.wav` (1961 archival, out of domain) | "And so might fellow **Annacats**, **airsp** not. What your country can do for you..." |

So: on clean input, one degraded word out of twenty and everything else
verbatim. On noisy archival audio it degrades badly and inconsistently. Two
fairness caveats before calling that a quality ceiling: jfk.wav is far out of
domain for a LibriTTS-R model, and **the 151 checkpoint is a `pretrained_file`
bootstrap for fine-tuning, not a finished voice** — Beatrice's actual workflow
trains a target speaker on top of it.

Prepending 1 s of silence did **not** repair the archival degradation (it got
slightly worse), so it is not a simple warmup transient.

Speaker conditioning is real: `embed_speaker` rows differ from row 0 by mean L2
1.98 (a copied/untrained table would be 0.0), and cross-speaker log-mel distance
runs **2.4x-3.7x the RNG floor**.

### Measuring anything here needs an RNG floor

First attempt used **waveform correlation** and produced ~0.00 for every pair —
including a same-input/same-speaker rerun that should have been ~1.0. Cause:
`overlap_add`'s random initial phase (detail 4 above) fully decorrelates the
phase while the audio sounds identical. The measure was invalid, and its
near-zero readings would have been easy to misread as "speakers differ hugely"
or "padding changed everything".

**The discipline that fixes it:** use a phase-invariant measure (log-mel), and
run the *same input twice* first to establish the **floor** — the distance
attributable purely to the RNG. Here the floor is 0.374; every real comparison
is then quoted as a multiple of it. Without the floor, "padded vs raw = 0.565"
means nothing; against it, it is 1.5x — real but small next to the 2.4-3.7x of
an actual speaker change. Applies to the §CB2 harness and to any stochastic
backend. See [[tts-parity-not-by-audio-corr]].

Note the output is stochastic enough that ASR transcribed two runs of the *same*
speaker differently — so acceptance thresholds must be set against the floor,
not against a single golden run.

---

## PitchEstimator — the port target, verified

```
wav 16k -> extract_pitch_features (DSP) -> instfreq[192,T] + corr_diff[256,T] + energy[1,T]
   instfreq -> conv1x1 -> gelu_tanh -> conv1x1  \
                                                 + -> gelu_tanh -> ConvNeXtStack(9) -> head 1x1 -> logits[448,T]
   corr_diff -> conv1x1 -> gelu_tanh -> conv1x1 /
```

`ConvNeXtStack`: `embed` CausalConv1d(k=3, delay=1) -> LayerNorm(**with affine**)
-> 9 x block -> `final_layer_norm` (**with affine**). Each block:
depthwise CausalConv1d(k=33, groups=192, delay=0, strictly causal) -> LayerNorm
(**affine folded away — normalise only**) -> Linear 192->384 -> gelu_tanh ->
Linear 384->192 -> residual add.

**The two LayerNorm forms are not interchangeable.** `merge_weights()` folds the
per-block affine into `pwconv1` but leaves the stack-level `norm` and
`final_layer_norm` affine intact. Using one form throughout is wrong in one
place or the other.

Fusion verified output-preserving on the real checkpoint before being baked into
the converter: **cos 1.0000000000, max_abs 7.6e-06** (f32 noise), and afterwards
`gamma`, `pre_scale`, `post_scale`, `post_scale_weight` are all identically 1.0
and the per-block LayerNorm affine is identity — so all 36 are dropped and the
runtime graph never sees them.

## Negative controls — the gate had two holes

The reference dumper re-implements the forward pass step by step and refuses to
write a file unless it reproduces the module's own `forward()`. Control:
**max_abs 0.000e+00**, bit-identical.

A first-run pass proves nothing ([[parity-harness-negative-control]]), so each
guard was tested by breaking a load-bearing detail. Two of the breaks initially
*passed*:

| break | first result | after fix |
|---|---|---|
| tanh-approx GELU -> exact erf (detail 2) | **PASSED**, cos 0.9999996, max_abs 2.4e-02 | FAILS (rel 1.65e-03) |
| drop the `1e-5` in delta_spec normalisation | **PASSED**, wrote an all-NaN reference, exit 0 | FAILS, names the stage |
| skip the per-block LayerNorm | FAILS, cos -0.53 | FAILS |
| drop the causal trim (detail 3) | raises (shape 432 vs 400) | raises |

Both fixes generalise to the ggml harness:

1. **Gate on relative max-abs, not cosine.** Both arms are torch on the same
   weights, so the control is bit-identical and anything above f32 rounding is
   real. A genuinely wrong activation scored cos 0.9999996 — through any
   `cos > 0.999999` check — while carrying 2.4e-02 of absolute error. HARD RULE
   #2b in the concrete.
2. **Check finiteness before any tolerance test.** Every NaN comparison is
   `False`, so `rel > TOL` is `False` for NaN and a divide-by-zero spec wrote a
   reference full of NaN and exited 0. A tolerance check structurally cannot
   catch this.

A third "control" was invalid and worth recording as a trap: re-applying
`gamma` after the fusion is a **no-op by construction** (post-merge `gamma` is
all-ones), so it passed bit-identically. It tested nothing. A negative control
has to break something the code actually depends on.

## The ggml port — result and what it cost

`crispasr-diff beatrice <model.gguf> <ref.gguf> <any.wav>` → **30 stages + end
to end, 0 FAILED**. Host DSP at cos 1.00000000; every network stage ≥
0.9999998; `estimate_e2e_logits` cos 0.99999990.

Two structural notes for the components still to come:

* **The DSP needs no inverse FFT.** The reference's
  `irfft(rfft(flip(y)) * rfft(y[-304:]))[304:]` reduces exactly to
  `corr[t] = sum_j y[256+j] * y[256+j-(t+1)]` — verified against torch at 1e-15
  relative. The difference function is then formed directly as
  `sum (y[a+j] - y[a-lag+j])^2`, which is algebraically identical but cannot
  suffer the cancellation the reference's `clamp(min=0)` exists to paper over.
  Only the instfreq branch needs a forward rFFT — of length 560, which is NOT a
  power of two (560 = 2^4 · 35), so `core_fft::fft_nonpow2_r2c` is required.
* `ggml_gelu` **is** PyTorch's `approximate="tanh"` (verified in
  `ggml/src/ggml-cpu/vec.h`), and `ggml_conv_1d_dw` is correct for this
  depthwise k=33 case despite its upstream "very likely wrong for some cases"
  comment — `block*_dwconv` passes at every block.

### Three bugs the harness caught, and how each announced itself

1. **Captured intermediates were being recycled.** Every intermediate failed
   while `backbone_final_norm` and `logits` passed — impossible if the
   intermediates were genuinely wrong, since the final stages are computed from
   them. The graph allocator reuses a tensor's buffer once its last consumer has
   run; captures need `ggml_set_output`. *The pattern is the diagnosis.*
2. **The host DSP was written frame-major** (`t*C + c`) while the reference is
   time-fastest (`c*T + t`) — the transpose trap, a fourth time, in a file whose
   header warns about it. `dsp_energy` **passed** while the other two DSP stages
   failed: energy is single-channel and therefore the only layout-invariant
   stage. *The one passing stage localised it.*
3. **`sample_pitch` indexed the logits frame-major** for the same reason;
   400/400 frames differed. Fixed → 1/400.

### Judging a tie-prone integer output

The surviving 1/400 was not a tie by any fixed threshold (band margin 1.9e-02),
yet running the same algorithm on the *reference's own* logits reproduced the
reference exactly (0/400) — so the logic was right and the flip came from f32
logit noise propagating into a close decision.

A fixed tolerance cannot arbitrate this: 113 of 400 frames sit inside 1e-3
(minimum 3.4e-06), so exact-match is permanently red, and any hand-picked margin
is just a number chosen to make the test green. The harness instead **measures
the perturbation the f32 difference actually induces** in band-score and
bin-probability space and accepts a mismatch only when the reference's own
preference is no larger than that.

That criterion was itself wrong on the first attempt, and only a negative
control showed it: `sample_pitch` decides in **two** stages (which band, then
which bin within it), and checking only the band choice waved through a plain
argmax substituted for the within-band argmax — the band is identical, so the
gap was 0. Now both stages are checked, and the bin comparison is **absolute**:
if our bin scores *higher* than the reference's, that is not a tie, it is the
band mask being ignored.

### What this harness cannot see (HARD RULE #3b)

* **The unvoiced path is untested.** `p[bin 0]` is 0.0000 on every frame of
  `samples/jfk.wav`, so `p[0] = -100` — the exclusion of the unvoiced class —
  can be deleted with no effect. Removing it was tried as a negative control and
  passed, *not* because the harness is weak but because this reference audio
  cannot exercise it. A reference over audio with genuine unvoiced/silent
  stretches is needed before that line is trusted.
* Only one clip, one length (4 s / 400 frames), f32 only, CPU only. Nothing here
  covers streaming, chunk boundaries, or the `delay > 0` lookahead geometry
  (`embed_trim` is 0 for this checkpoint, so `padding != trim` is never
  exercised even though the code handles it).
* Cosine ≥ 0.9999 is the stage gate; per the negative controls in the section
  above, cosine alone would pass a wrong GELU. Stage max_abs is printed and
  should be read alongside it.

## How good is the pitch detector, actually?

Parity says the port matches torch; it says nothing about the model. Measured
separately (torch reference — legitimate, since the port is bit-parity with it).

**Frequency mapping, recovered empirically:** `f = 55 · 2**(bin / 96)` — 96
bins/octave anchored at A1, **12.5 cents per bin**. The trainer source never
states this. It came from octave *pairs* (220→192 and 440→288 differ by exactly
96; 65.4→24 / 130.8→120 likewise). A global least-squares fit gave a confidently
wrong 57.6 bins/octave, because one out-of-range outlier dragged the line —
octave pairs are robust to that, a fit is not.

**Accuracy: at the quantisation floor.** On off-grid sawtooth tones (chosen
*not* to align with the bin grid — equal-tempered notes land exactly on bins at
8 bins/semitone, so testing those flatters it), max error is **5.3 cents**
against a ±6.25-cent quantisation limit. Zero octave errors on synthetic input.

**Usable range: ~62–784 Hz.**

| | |
|---|---|
| 55 Hz | +12.5 cents (one bin off) — below the `16000/256 = 62.5 Hz` floor implied by `max_corr_period` |
| 62–784 Hz | within a few cents, stable (frame-to-frame std 0.0–0.14 bins) |
| 880 Hz | **collapses** — bin 53 instead of 384, std 19.4 |
| 1046 Hz | median correct but unstable (std 52.9) |

Bins 1..447 nominally span 55–1387 Hz, so the upper failure is the *model*, not
the representation — unsurprising for one trained on speech.

**On real speech** (jfk.wav) against pyworld Harvest as an independent oracle:
median |error| 18.4 cents, 69.7 % of voiced frames within 50 cents, 51.6 %
within 20, and only **1.7 % octave disagreements**. Read that as a *loose lower
bound*: Harvest is not ground truth, jfk.wav is a noisy 1961 archival recording,
and no frame-alignment offset between the two was corrected.

### The quantised output carries no voicing

`sample_pitch` **never returns 0**. Bin 0 is the unvoiced class and is forced to
−100 before the argmax, so it is excluded by construction — measured 0 zeros
across 1100 frames of jfk.wav *including its silent stretches* (minimum bin 8).
`beatrice_pitch.h` claimed "0 marks unvoiced"; that was wrong and is corrected.

Voicing lives in `sample_pitch(return_features=True)`'s `unvoiced_proba`, which
the current API does not expose. **ConverterNetwork needs it** — it is channel 0
of the 4 that `embed_pitch_features` consumes — so the port below must add it.

## F0 backend comparison — measured, and it decides the wiring

`crispasr-f0-eval` (examples/cli) drives the **shipped C++ path**, not torch.
Off-grid sawtooth tones, 1 s each, 16 kHz.

| | beatrice-pitch | crepe-tiny |
|---|---|---|
| median error, all tones | **2.4 cents** | 2.6 cents |
| within 10 cents | 17/19 | **19/19** |
| frame-to-frame jitter | **0.0 cents** | 0.2–1.8 cents |
| 833 Hz / 971 Hz | **fails** (2780 / 4746 cents) | fine |
| wall clock, 11 s audio (incl. load) | **0.72 s** | 1.99 s |
| reports voicing | **no** | yes (`voiced_prob`) |

This also **confirms the earlier torch-side numbers on the real C++ path** — the
bit-parity inference held, but it had never been checked directly.

In the speech range (58–708 Hz) Beatrice matches CREPE on accuracy, is
*perfectly* stable (it quantises to bins, so it locks rather than jitters), and
is ~2.8× faster. Above ~780 Hz it collapses; CREPE does not. Beatrice's zero
jitter is a consequence of quantisation, not superior estimation — the flip side
is 12.5-cent granularity where CREPE is continuous.

### Agreement on real speech is a voicing story, not a pitch story

Against CREPE on `samples/jfk.wav`, filtered by CREPE's own confidence:

| CREPE confidence ≥ | frames | median disagreement | within 50c | octave |
|---|---|---|---|---|
| 0.0 (everything) | 550 | **461.9 cents** | 38.4 % | 2.0 % |
| 0.5 | 308 | 56.5 cents | 49.0 % | 1.0 % |
| 0.8 (confidently voiced) | 64 | **9.3 cents** | 95.3 % | 0.0 % |

The two backends agree to **9.3 cents on confidently-voiced speech**. The
alarming unfiltered number is entirely unvoiced frames — Beatrice emits a pitch
for silence because, as established above, its quantised output carries no
voicing at all. Quoting the 461.9 without that decomposition would be
straightforwardly misleading.

### Wiring decision: NOT yet, and the numbers say why

`rvc_svc_convert()` takes `f0_hz` with **0.0 meaning unvoiced**, at exactly the
100 Hz rate Beatrice produces — so Beatrice looks like a drop-in native F0
source, removing the caller's Harvest dependency. The measurements say don't:
with no voicing output it would feed spurious pitch through every silent frame,
driving the vocoder's sine source where it should be off. That is the 461.9-cent
column, and it is audible, not cosmetic.

Two options, in order of preference:

1. **Expose `unvoiced_proba`** — `sample_pitch(return_features=True)` already
   computes it, and `ConverterNetwork` needs the same 4 pitch-feature channels
   anyway, so this is work §CB3 must do regardless. Then Beatrice becomes the
   better choice for speech-range real-time: faster, stabler, 7 MB.
2. **Use CREPE for RVC's F0 today** — it has `voiced_prob` and a wider usable
   range. It is ~2.8× slower here, which may or may not matter.

## ConverterNetwork + Vocoder — read, not started

The remaining component, and the largest. Read against the source; **nothing is
implemented**. Two surprises worth knowing before anyone estimates it.

### The Vocoder is NOT a HiFi-GAN

It is a **source-filter / impulse-response synthesiser**, which is what makes
Beatrice cheap enough to be real-time. No transposed-conv upsampling stack at
all:

| module | what it produces |
|---|---|
| `prenet` | ConvNeXtStack, 4 blocks, `delay=2` (20 ms), **cross-attention to the speaker embedding** (`kv_channels=128`) |
| `ir_generator` + `ir_generator_post` | a **512-tap impulse response** per frame (`WSConv1d(channels, 512, 1)`), windowed by the learned `ir_window` (512) |
| `aperiodicity_generator` + post | `WSConv1d(channels, hop_length=240, 1, bias=False)` |
| `post_filter_generator` + post | `WSConv1d(channels, 512, 1, bias=False)` |

Fixed scales live as **buffers**, not weights: `ir_scale=1.0`,
`aperiodicity_scale=0.005`, `post_filter_scale=0.01`. Synthesis is
`overlap_add()` driven by the pitch contour.

Two consequences for the port:

* **`WSConv1d`/`WSLinear` are live here for the first time.** Neither
  PitchEstimator nor PhoneExtractor uses them, so the unbiased-variance detail
  (§1 above) has not yet mattered. In this component it does: `np.var`'s
  `ddof=0` default would mis-scale every impulse-response weight by ~20 %,
  uniformly — which sounds like a slightly wrong voice, not like a bug. The
  `merge_weights()` path folds standardisation, so verify with the converter's
  fusion check rather than trusting it.
* **The attention here is `CrossAttention`, not `nn.MultiheadAttention`** — a
  different module with its own `dump_layer` handling and its own
  `merge_weights` fold (`q_projection` absorbs `attn_norm`). The
  `mha_subsequence()` written for PhoneExtractor does **not** transfer as-is.

### The lookahead alignment is the dangerous detail

`ConverterNetwork.forward` shifts three signals before combining them, because
the three frozen extractors look ahead by different amounts (the source comments
this: phone 2.5 ms, energy 12.5 ms, pitch_features 22.5 ms):

```python
energy         = F.pad(energy[:, :, :-1],       (1, 0), mode="reflect")   # shift 1
quantized_pitch= F.pad(quantized_pitch[:, :-2], (2, 0), mode="reflect")   # shift 2
pitch_features = F.pad(pitch_features[:, :, :-2], (2, 0), mode="reflect") # shift 2
```

Note the padding is **reflect**, not zero. Omitting these shifts misaligns
content against pitch by 10–20 ms — audible as smeared articulation, and
invisible to any per-stage cosine check that feeds each stage its own reference
input.

`pitch_features` is then `cat([energy, pitch_features])` → the 4 channels
`embed_pitch_features = nn.Conv1d(4, hidden, 1)` expects.

### Rest of the inference path

```
phone = phone_extractor.units(x).transpose(1,2)        # VQ hook fires inside .head
phone *= 1/sqrt(mean(phone^2, dim=1) + eps)            # RMS norm over CHANNELS
pitch, energy = pitch_estimator(x)
quantized_pitch, pitch_features = pitch_estimator.sample_pitch(pitch, return_features=True)
<the three shifts above>
formant_shift_indices = round((formant_shift_semitone + 2.0) * 2.0)   # index into embed_formant_shift(9)
x = embed_phone(phone) + embed_quantized_pitch(quantized_pitch).transpose(1,2)
      + embed_pitch_features(pitch_features)
      + (embed_speaker(id) + embed_formant_shift(idx))[:, :, None]
x = F.silu(x)                                          # SiLU here, NOT gelu
speaker_embedding = key_value_speaker_embedding(id).view(B, 384, 128)
y = vocoder(x, pitch, speaker_embedding)               # NOTE: raw pitch LOGITS, not quantised
```

Two easy mistakes: the activation is **SiLU** (every other activation in this
model is tanh-GELU), and the vocoder receives the **raw `pitch` logits**, not
`quantized_pitch`.

### Validation will need noise injection

`overlap_add` draws a random initial phase (`torch.rand`, §4 above), so unlike
the two frozen components this one **cannot** be validated by direct comparison.
It needs RVC's injectable-draw discipline: `tools/rvc_torch_parity.py` is the
template, and the injector must patch `torch.rand` — patching only `randn_like`,
as the RVC port did, would miss this site entirely and silently compare against
a different phase.

Also note the training-only paths that must NOT be ported: the
`phone_noise_ratio` augmentation, the whole pitch-shift/resample augmentation
block, and `slice_segments` — all guarded by `if self.training`.

### Layout convention

Torch stores `[batch, channels, time]` with **time fastest**; stages are dumped
in exactly that memory order, landing in ggml as `ne = [time, channels]` — what
`ggml_conv_1d` expects. **Do not transpose on either side.** Three separate RVC
port bugs came from transposing here, each producing ~0 cosine on a graph that
was correct.

---

## Order of work

Per component, smallest and most frozen first:

1. **`PitchEstimator`** — 7.1 MB, fully frozen, and its front end
   (`extract_pitch_features`: instantaneous frequency + autocorrelation, hop 160
   / win 560 / `max_corr_period` 256 / `corr_win_length` 304) is pure DSP that
   can be validated on its own before any learned layer is involved.
2. **`PhoneExtractor`** — 14.7 MB, also frozen.
3. **`ConverterNetwork` + `Vocoder`** — the LibriTTS-R `net_g`, 177 tensors.

For each: read -> numpy/torch executable spec -> ggml graph -> per-stage diff
harness, with the reference dump carrying the injected RNG draw so the
comparison is deterministic (`tools/rvc_torch_parity.py` is the template).

Do not skip the end-to-end stage. In §CB1 every one of 47 per-stage checks was
green while the assembled `convert_e2e` sat at cos 0.40 — per-stage checks are
input-aligned and never test the wiring between stages.

# RVC voice conversion — port blueprint (§CB1)

Traced from `RVC-Project/Retrieval-based-Voice-Conversion-WebUI`
(`infer/module/models.py`, `infer/vc/pipeline.py`), shallow clone 2026-07-20.
Every claim below carries a file:line citation and was read from source, not
recalled. The wire contract lives in `SVC_RECORD_SHAPES.md` (CONFIRMED); this
document is about what we have to BUILD.

**Read this before writing any C++.** Two findings change the shape of the job.

---

## 0. RESOLUTIONS (CometBeat, 2026-07-20)

Both findings below are answered; recorded here so the doc reads as settled.

- **Finding 1 — CrispASR owns all three** (`enc_p` + `flow` + `dec`). The frozen
  contract stands: they send ContentVec features + F0 (Hz, 100 Hz) + speaker id,
  **not `z`**. Their reasoning is performance and it is sound: their pure-Dart
  `rvc.dart` already runs all three at **152x slower than real time**, with the
  transformer `enc_p` and the flow a large share of that — so a vocoder-only
  split would leave the transformer as a Dart bottleneck and still miss
  real-time. The right Dart↔native boundary is the ContentVec output, the one
  checkpoint-agnostic reusable piece; `enc_p`/`flow`/`dec` are all
  checkpoint-specific and belong together. Sending `z` would also leak a 192-ch
  model-internal latent into the wire format.
- **Finding 2 — replay the noise**, agreed. Both live RNG sites must be
  injectable in our ggml graph. **They have offered a cross-impl oracle**:
  `rvc.dart` runs the full generator deterministically from an injected noise
  buffer (`rvcSeededNoise` → the ONNX graph's `rnd [1,192,T]` input, cos
  0.99994), enabling a three-way deterministic harness — Python reference → their
  Dart-offline → our ggml graph, all fed identical noise. Production
  `convert()` stays random by design.
- Their offline impl already pins `F.interpolate` to 2x-**nearest**
  (`rvcAlignFeatures`) and the coarse-pitch mel map 50–1100 Hz
  (`rvcCoarsePitch`) — independent agreement with §3 and SVC_RECORD_SHAPES §4.

### CONFIRMED EMPIRICALLY (tools/rvc_torch_parity.py)

The noise-replay design is proven, not just proposed. Intercepting every RNG
call during `infer()` gives exactly three draws:

```
randn_like   (1, 192, 64)      <- Site A: z_p latent          (1, inter, T)
rand         (1, 1)            <- SineGen phase: ONE element
randn_like   (1, 25600, 1)     <- Site B: additive noise      (1, T*upp, 1)

two runs with identical injected noise -> BIT-IDENTICAL, max_abs 0.000e+00
```

The `(1, 1)` draw is the `harmonic_num=0` prediction confirmed from the running
model: the phase RNG is a single element, and because the model's next line
zeroes it, injecting zeros there is provably equivalent — the bit-identical
result is the proof. `check_phase_is_deterministic()` asserts
`harmonic_num == 0 and dim == 1` against the built model so this cannot regress
silently if someone loads a differently-configured checkpoint.

### Answer to their SineGen phase question — it is ZERO, by construction

They asked whether to zero or replay the SineGen initial phase, having noticed
their ONNX export appears to fix it. **There was never any randomness there to
fold away, for this architecture:**

`GeneratorNSF` hardcodes `SourceModuleHnNSF(sampling_rate=sr, harmonic_num=0)`
(`models.py:439-441`). With `harmonic_num = 0`:

- `SineGen.dim = harmonic_num + 1 = 1` (`models.py:294`) — fundamental only, no
  overtones.
- `rand_ini` is therefore shape `(batch, 1)` (`:325`), and the very next line is
  `rand_ini[:, 0] = 0` (`:328`).

So `rand_ini` is **identically zero**. The random-phase code is live only for
`harmonic_num > 0`, which RVC never uses. Both implementations should use zero
phase, and they will match bit-for-bit without any agreement being needed —
this is determined by the source, not a convention.

**That leaves exactly TWO live RNG sites, not three:** the `z_p` latent sample,
and SineGen's ADDITIVE noise (`:358`), which is ungated and always runs:

```python
noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
noise     = noise_amp * torch.randn_like(sine_waves)
sine_waves = sine_waves * uv + noise
```

Note it is voicing-dependent: voiced frames get sine + noise at `noise_std`
(0.003), unvoiced get pure noise at `sine_amp/3` (0.0333). Their ONNX export
exposes only `rnd` (the z_p noise), so **this second site is the one their path
may differ on** — worth confirming whether their export also folded the
additive noise away, because that IS a real randomness source, unlike the phase.

---

## 1. FINDING: the ask is ~3x bigger than "the NSF-HiFi-GAN generator"

CometBeat asked us to port "the RVC NSF-HiFi-GAN generator". But the seam they
need — ContentVec features + F0 + speaker id → converted audio — is
`SynthesizerTrnMs768NSFsid.infer()` (`models.py:664`), and that runs **three**
substantial components:

| # | component | what it is | `models.py` |
|---|---|---|---|
| 1 | `enc_p` — TextEncoder | pitch `nn.Embedding(256, hidden)` + **a real multi-layer transformer** (`attentions.Encoder`) + Conv1d proj | 16-50 |
| 2 | `flow` — ResidualCouplingBlock | **normalizing flow**, 4x (ResidualCouplingLayer + Flip), run in REVERSE | 79-113 |
| 3 | `dec` — GeneratorNSF | the NSF vocoder: SourceModuleHnNSF/SineGen + ConvTranspose1d upsample stack + ResBlocks | 420-547 |

```
phone(768) ─┐
pitch(1..255)┼─> enc_p ─> m_p, logs_p ─> [SAMPLE z_p] ─> flow(reverse) ─> z ─┐
             │                                                                ├─> dec ─> audio
nsff0(Hz) ───┴────────────────────────────────────> SineGen source ──────────┘
sid ─────────> emb_g ─────────────> g (conditions flow AND dec)
```

Only #3 is a vocoder. #1 is a transformer encoder and #2 is a flow — neither is
covered by "HiFi-GAN". **RESOLVED (§0): we own all three.** "Vocoder" was loose
wording; they always meant the whole `infer()`.

Note `emb_pitch = nn.Embedding(256, hidden_channels)` (`models.py:40`) — this is exactly why the coarse 1..255 quantisation must be
bit-right: it indexes an embedding table, so an off-by-one is a different
learned vector, not a small numeric error.

---

## 2. FINDING: inference is STOCHASTIC — there are TWO independent RNG sites

This is the single most important fact for validation, and it must be settled
before any graph is written.

**Site A — the latent sample** (`models.py:684`, and 691 for the non-chunked path):

```python
z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
```

**Site B — the sine source** (`models.py`, SineGen.forward ~325 and ~358).
**CORRECTION, see §0:** the `rand_ini` phase term below is DEAD for RVC —
`harmonic_num=0` makes it a single element that the next line zeroes. Only the
additive `noise` is a live RNG site.

```python
rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2], ...)   # random INITIAL PHASE per harmonic
rand_ini[:, 0] = 0
rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
...
noise = noise_amp * torch.randn_like(sine_waves)                # additive, std 0.003 by default
```

Consequences:

- **Output audio is not reproducible run to run**, even with identical inputs.
- **Waveform correlation against a reference run is an invalid acceptance
  test.** We already have this lesson recorded from melotts
  (`--seed` isn't deterministic); here it is structural, not incidental.
- A per-stage diff CANNOT compare anything downstream of `z_p` unless the
  reference's noise is replayed.

**Required harness design** (decide before building):

1. Reference dumper captures `m_p`, `logs_p`, **and the drawn `z_p`**, plus
   SineGen's `rand_ini` and its noise tensor.
2. The runtime REPLAYS those instead of sampling — exactly the pattern
   `btc_chords_diff` uses for `input_feat` and `mel_band_roformer_diff` uses
   for `input_audio`.
3. Stages up to `m_p`/`logs_p` are deterministic and diff normally; everything
   after is input-aligned on the replayed noise.
4. Ship a documented deterministic mode (replay/zero noise) so the acceptance
   test is reproducible, with the stochastic path as the default at runtime.

Without this the port is unverifiable, and "it sounds fine" becomes the only
check — which is precisely the failure mode that let piano-transcription ship
with every 2-D BatchNorm at the wrong epsilon.

---

## 2b. enc_p traps — VALIDATED in the numpy spec (cos 1.00000000)

`tools/rvc_torch_parity.py` reimplements `enc_p` in numpy and scores it against
torch: **m_p and logs_p both cos 1.00000000**, max_abs ~3e-6 (f32 vs f64). Six
details it had to get right, each a silent accuracy bug if assumed:

| detail | value | why it bites |
|---|---|---|
| LeakyReLU slope | **0.1** (`models.py:38`) | torch's default is 0.01 — a 10x difference on every negative activation |
| scale before lrelu | `x *= sqrt(hidden)` (`:62`) | applied BEFORE the activation, so it is not a no-op that cancels later |
| residual style | **POST-norm** `x = norm(x + f(x))` | pre-norm is the modern default and would be the natural assumption |
| attention | **relative position**, window 10 | not absolute/sinusoidal PE; needs the skew helpers for keys AND values |
| FFN | SAME padding, plain **ReLU** | `activation != "gelu"` here, so the x*sigmoid(1.702x) branch is dead |
| LayerNorm | over the **channel** dim | `modules.py:29-32` transposes first, so a naive last-dim norm is wrong |

The relative-VALUE path (`_abs_to_rel`) is the easiest of these to omit
entirely — attention still "works" without it and merely gets worse.

## 2c. flow traps — VALIDATED (cos 1.00000000)

The reverse pass also matches torch exactly. Five details:

- **`mean_only=True`** (models.py), so `logs` is ZERO and the coupling is purely
  ADDITIVE. The reverse is `x1 = x1 - m`, **not** `(x1 - m) * exp(-logs)` — the
  general VITS formula would be wrong here.
- `flows` interleaves `[Coupling, Flip] x 4`; the reverse walks the whole list
  backwards, so **Flip comes first**.
- Flip reverses the **channel** axis.
- The WaveNet is **gated**: `tanh(first half) * sigmoid(second half)` of
  `(x_in + g_l)`, with the speaker conditioning projected once and then sliced
  per layer.
- `ResidualCouplingBlock(inter, hidden, 5, 1, 3, ...)` — kernel 5, dilation
  rate 1, 3 layers are **hardcoded in models.py:624**, not config-derived, so
  they hold for every checkpoint.

## 2d. dec traps — VALIDATED (audio cos 1.00000000)

The full path now matches torch: **m_p / logs_p / z / audio all cos
1.00000000**. Two of the traps below were found the hard way, by bisection,
after a plausible-looking implementation scored badly.

### SineGen phase — an approximation scored cos -0.04

The phase logic cannot be paraphrased. The real sequence (`models.py:329-351`):

1. `rad = (f0/sr) % 1` at FRAME rate
2. `tmp = cumsum(rad) * upp` — still frame rate
3. `tmp` → **LINEAR** interpolate to output rate, **`align_corners=True`**
4. `rad` → **NEAREST** interpolate to output rate
5. `tmp %= 1`; wrap points are where `diff(tmp) < 0`
6. `phase = cumsum(rad_up + shift)`, `shift = -1` at each wrap
7. `sine = sin(phase * 2*pi)`

The linear-interpolated cumsum is used **only to LOCATE the wraps**; the phase
itself accumulates over the **nearest**-upsampled per-frame values. The two
interpolations use DIFFERENT modes. A rewrite that interpolated the cumsum and
used it directly as phase produced the right amplitude and **cos -0.04** —
uncorrelated noise that still looks like a sine source.

Accumulate in float64: the running sum grows without bound while only its
fraction matters.

### `dec.cond` has a BIAS

`self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)` — default
bias. Omitting it is a constant per-channel offset, structurally invisible, and
it cost cos 0.9999 at `ups0` and 0.998 on the final audio. Bisecting stage by
stage was what localised it: `conv_pre` and every `noise_convs` were exact while
`ups0` was not, and the transposed convolution itself tested exact (1e-16)
against torch in isolation — so the only remaining suspect was its input.

### Other dec details

- **TWO different LeakyReLU slopes in ONE function**: per-stage and ResBlock use
  `LRELU_SLOPE = 0.1`, but the FINAL pre-`conv_post` call is a bare
  `F.leaky_relu(x)` — torch's **0.01** default (`models.py:529`).
- `conv_post` is **`bias=False`** (`models.py:484`).
- ConvTranspose1d: `stride=u`, `padding=(k-u)//2`.
- `noise_convs[i]`: `kernel=2*stride_f0`, `stride=stride_f0`, `padding=stride_f0//2`.
- Each stage sums `num_kernels` ResBlocks and **divides by `num_kernels`**.
- ResBlock1's `convs1` carry the dilations; `convs2` are always dilation 1.

## 3. Numerical hazards spotted in the trace

- **Phase accumulation by `cumsum`** (SineGen): phase is accumulated over the
  whole utterance at the OUTPUT rate (post-`upp` upsampling), then reduced mod
  1. In float32 over minutes of audio this drifts — the running sum grows while
  the meaningful part is the fraction. Accumulate in double, or reduce mod 1
  incrementally. Upstream's own comment on that line notes the `%1` placement
  blocks a later optimisation, so they were aware the ordering is load-bearing.
- **`F.interpolate` on the phase** — mode matters and must be read off the call,
  not assumed (the ContentVec 2x upsample in the same repo defaults to
  `nearest`, which surprised this port once already; see SVC_RECORD_SHAPES §2).
- **`upp` = `math.prod(upsample_rates)`** (`models.py:438`) ties the F0 rate to
  the vocoder's total upsampling — it is derived, not configured, so a mismatch
  between our hop and the checkpoint's upsample_rates is silent.
- **`noise_convs` stride** (`models.py:463-465`): `stride_f0 = prod(upsample_rates[i+1:])`
  — a per-stage downsample of the source signal that must match the upsample
  schedule exactly.
- SourceModuleHnNSF defaults: `sine_amp=0.1`, `add_noise_std=0.003`,
  `harmonic_num` from config — all read from the checkpoint's config, never
  assumed.

---

## 3b. Checkpoint structure (measured on the official v2 `f0G40k.pth`)

560 tensors: `enc_p` 113, `dec` 243, `enc_q` 103, `flow` 100, `emb_g` 1.

- **`enc_q` (103) is the PosteriorEncoder — TRAINING ONLY**, never called by
  `infer()`. Dead weight, like BTC's `output_layer.lstm.*`.
- **`dec` and `flow` are weight-normalised**: they store `weight_g`/`weight_v`,
  not `weight` (`dec.ups.0.weight` does not exist). Fuse at convert time,
  `w = g * v / ||v||` with the norm over every dim except 0. Some layers in the
  SAME module (`conv_pre`, `noise_convs`, `conv_post`) carry a plain `.weight`,
  so both forms must be handled. 104 pairs in this checkpoint.
- **`enc_p` uses RELATIVE positional attention**, not absolute PE:
  `emb_rel_k/v` are `(1, 2w+1, d)` with **w = 10**.
- Geometry read off the weights: `emb_phone` (192, 768) → content dim 768,
  hidden 192; `emb_pitch` (256, 192) → the coarse pitch is an EMBEDDING INDEX;
  `emb_g` (109, 256) → 109 speakers, gin 256.

### Upsample rates are NOT recoverable from the checkpoint

The 40k ConvTranspose1d kernels are `16,16,4,4`, while the rates are
`10,10,2,2`. Assuming `kernel == 2*rate` gives `8,8,2,2` — a silently wrong
model. The rates come from the config JSON (note v2 ships only 32k/48k; 40k
lives under `configs/v1/`).

They CAN be verified, though: `noise_convs[i]` has kernel `2*prod(rates[i+1:])`
(1 for the last stage). Checked against the real checkpoint:

| stage | kernel | 2·prod(rates[i+1:]) |
|---|---|---|
| 0 | 80 | 2·40 = 80 |
| 1 | 8 | 2·4 = 8 |
| 2 | 4 | 2·2 = 4 |
| 3 | 1 | 1 |

The converter asserts this, so a mismatched config fails loudly. Independently,
`sr / prod(rates)` is **exactly 100.0** for every shipped config (32k/40k/48k) —
a second, independent confirmation of the 100 Hz wire rate that
SVC_RECORD_SHAPES derives from `pipeline.window = 160`.

## 3c. ggml port — DONE for the graph, 47 stages at cos 1.00000000

`crispasr-diff rvc <model.gguf> <ref.gguf> <any.wav>` compares enc_p (per
sublayer + attention internals), all 4 flow coupling blocks, every dec upsample
and noise-conv stage, and **output_audio at max_abs 1.5e-08**.

ggml-specific findings, all of which cost a debug cycle:

| issue | fix |
|---|---|
| A kernel-1 Conv1d is stored WITH the kernel axis: GGUF `(out,in,1)` → ggml `[1,in,out]` | reshape to `[in,out]` before `mul_mat`, else it contracts over 1 and aborts |
| `_get_relative_embeddings` pads SYMMETRICALLY; `ggml_pad` only appends | front pad via `ggml_concat` of a zero block |
| `ggml_conv_1d` wants `[length, channels]`; our layout is `[channels, time]` | transpose in/out, add bias AFTER transposing back |
| **`ggml_conv_transpose_1d` ASSERTS `p0 == 0`** — no padding support | express torch's padding as a symmetric CROP: `core_convt::convt1d_crop` |
| `core_hifigan::resblock_forward` is TIME-major | transpose around it rather than reimplementing the MRF block |
| no flip op, no negative-stride views | channel Flip = transpose → `get_rows(reversed idx)` → transpose |
| tap tensors unreachable from the output are never allocated | `ggml_build_forward_expand` over every capture AND every output |

### A vacuous PASS, and the test that caught it

`output_audio` reported **PASS cos 1.00000000** while comparing **one sample**.
The dumper had `a, b = audio_np.ravel(), ...` a few lines above
`"output_audio": a[0]` — shadowing the variable holding the reference audio, so
`a[0]` became a single float. The harness dutifully compared 1 element against
25600 and passed.

This is the failure mode to fear most: not a red number, a GREEN one that means
nothing. Nothing in the run looked wrong. It surfaced only because a NEW check
printed `ref=1` next to `mine=25600` — the size columns, not the cosine.

**Print sizes next to every cosine.** A comparison over the wrong length is
invisible in the metric and obvious in the shape.

### Per-stage checks do not test CHAINING

Every stage check here is input-aligned: it feeds the reference's own `z_p`,
`z` and `har`. So all 47 could pass while the code that WIRES the stages
together is broken — and it was. With the audio reference finally intact,
`convert_e2e` (the real `rvc_svc_convert()`, fed only ContentVec + F0 + speaker
id + noise) failed at **cos 0.40** with every per-stage check still green. The
cause was the third instance of the same transpose trap, this time on the
`noise_zp` input.

So the acceptance test must run the REAL entry point, not a hand-assembled path
through validated pieces. That is HARD RULE #3 (decoded-output roundtrip)
applied to a port whose output happens to be audio rather than text.

### The lesson that actually mattered

**Two of the three "graph bugs" were HARNESS bugs.** Both `enc_p` and `flow`
reported cos ≈ 0 on a *correct* graph because the comparison was wrong:

- torch stores `(b, C, T)` with TIME fastest; our layout is `[C, T]` with
  CHANNELS fastest — exact transposes. `encp_lrelu` passed only by accident,
  because torch's lrelu runs while the tensor is still `(b, T, hidden)`.
- `register_forward_hook` DOES NOT FIRE on the flow: the reverse pass calls
  `flow.forward(...)` directly (`models.py:126`), bypassing `nn.Module.__call__`
  where hooks dispatch. The harness then reported "0 FAILED" while comparing
  NOTHING — the worst possible failure mode. Wrap the bound method instead.

Both were found in one bisect step each once per-stage intermediates existed in
a registered `crispasr-diff` branch. Before that, with only endpoints and an
ad-hoc test binary, the same bug consumed a whole session and I concluded the
graph was broken when it was not.

## 4. Proposed order of work

Standard order, unchanged by the above:

1. **Converter** (`models/convert-rvc-to-gguf.py`) — emit all three components
   plus config (`upsample_rates`, `harmonic_num`, `sine_amp`, `add_noise_std`,
   `tgt_sr`, `spk_embed_dim`, and the **content dim** for the
   `convert_content_dim()` guard CometBeat asked for).
2. **Executable spec** (`tools/rvc_torch_parity.py`) — numpy/torch
   reimplementation scored against the real model, with the noise replayed.
   This is where §2's design gets proven before any C++ exists.
3. **ggml graph** — `src/rvc_svc.{h,cpp}`.
4. **Per-stage diff** — `rvc_svc_diff()` + registration in
   `crispasr_diff_main.cpp` (registering it is part of the job, not a
   follow-up: mel-band-roformer shipped with a written-but-unregistered diff).
5. **Surfaces together** — CLI verb + session C ABI (`crispasr_session_convert*`)
   + wasm, and the arch in **both** detect paths. See
   `docs/contributing.md` section 7.

**Blocking question for CometBeat before step 1:** §1 — do they want all three
components, or only `dec`? That changes the wire format (features vs `z`) and
therefore the contract we just froze.

## 5. Licensing

RVC's code is MIT. Circulating checkpoints are NOT uniformly so — scope each
one before any registry entry, and give non-commercial ones their own licence
tag rather than reusing `cc-by-nc-sa-4.0` (the gate matches on the tag).
Beatrice v2 (§CB2) is custom/non-commercial and needs the same treatment.

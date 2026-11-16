# §248 — Mel-Band RoFormer (source separation) port

**Claimed:** M1/Metal session, 2026-07-19. New category: `--task separate`.

## STATUS: PORTED + WIRED IN (2026-07-19)

Every forward stage validated cos=1.0 vs the pinned reference; reconstructed
waveform bit-exact (2.4e-7). `crispasr --separate -m <mbr.gguf> -f mix.wav`
writes `<input>_{vocals,other}.wav` (2ch@44.1k) — on a speech clip vocals
rms=0.218 vs other rms=0.005 (41x). Go LDFLAGS drift clean.

Done: C API + forward + separate(); `--separate` dispatcher (mel-band-roformer
+ htdemucs) + flags + help + early-route; CMake link into crispasr-lib; Go
LDFLAGS; README; live test; reference dumper + converter.

Deferred (non-blocking): HF upload of the GGUF + optional registry `-m auto`
entry (`-m <path>` works now); portable BLAS for the Linux CPU forward
(Accelerate on Mac today, scalar fallback elsewhere). Coordination: the
CAP_SEPARATE htdemucs adapter divergence is flagged in
docs/source-separation-surface.md for the maintainer to converge.

## NOW — active work

- [x] License due diligence (below) — cleared (code = MIT lucidrains; NOT Kim's
      unlicensed inference repo; weights = MIT)
- [x] HARD RULE #1: Python blueprint read line-by-line (below)
- [x] Band layout verified from librosa (load-bearing mel[0,0] tweak found)
- [x] `tools/reference_backends/mel_band_roformer.py` — per-stage dumper, run
      on-box (no OOM, peak 1.6 GB), clean 0/0 load under **bs-roformer==0.3.10**
- [x] `models/convert-mel-band-roformer-to-gguf.py` — 435.8 MB f16 GGUF, 684
      tensors + 3 baked aux int32 arrays; band-width assert passed vs ckpt
- [x] Shared separation surface designed + additive core landed
      (`docs/source-separation-surface.md`, `src/core/separation_io.h`,
      multi-channel WAV writer, `tests/test-separation-io.cpp` 9/9) — maintainer
      chose "design the shared surface now"; htdemucs session to adopt it
- [x] **`src/mel_band_roformer.{h,cpp}` — C API + CPU forward, diffed stage by
      stage** (opt-in probe `mbr-diff-probe`, `CRISPASR_BUILD_MBR_PROBE=ON`).
      All input-aligned vs `ref_mbr.gguf`; PASS = cos ≥ 0.9995:
      ```
      freq_indices   cos=1.000000   stft_packed    cos=1.000000
      band_gathered  cos=1.000000   band_split_out cos=1.000000
      layer0_time    cos=1.000000   (full RoFormer block — RoPE + gating + GELU-erf FFN)
      ```
      RMSNorm-amplification lesson applied throughout (feed each stage the ref
      input, not our chained output). CPU helpers: linear, gelu_erf, rms_rows,
      rope_head, roformer_block, band_split_cpu.
- [ ] **NEXT stages** (same validated block, different axis / accumulation):
      - `layer0_freq` — freq transformer attends over the 60 bands per time step.
        Its input is the full (all-bands) time-transformer output, but the dumper
        hook currently captures only `[0]` (band 0). **TODO: dumper captures full
        transformer intermediates** so freq + later layers input-align cleanly;
        re-dump; then diff.
      - value residuals: layers 1..5 consume layer-0's attention values
        (`add_value_residual and not is_first`) — thread them through.
      - mask estimator (per-band Tanh MLP → GLU), scatter-add → average by
        num_bands_per_freq, complex mask multiply, iSTFT (reuse core_istft).
      - wire `mel_band_roformer_separate()` (currently a Phase-2 null stub).
- [ ] Dispatcher `examples/cli/crispasr_separate_cli.{h,cpp}` + `--separate`
      hook; wire htdemucs (C API ready) + MBR through `crispasr_separation_view`
- [ ] Roundtrip acceptance (SDR / ASR on the vocal stem) — the ONLY gate that counts
- [ ] 12-point checklist (CMake, registry, README, bindings docstrings, Go LDFLAGS)

Fixtures (persisted off /tmp, survive reboot):
`/Volumes/backups/ai/crispasr-models/melbandroformer/{MelBandRoformer.ckpt,
config.yaml, mel-band-roformer-vocals-f16.gguf, ref_mbr.gguf, clip2s.wav}`

## §422 segmentation evidence — MEASURED 2026-09-03 (do not re-derive)

PR #422 added Demucs-style segmentation (25% overlap, triangular weight,
normalised by accumulated weight) for audio longer than the trained chunk. It
shipped with NO reference-anchored evidence, for a structural reason rather
than an oversight:

* the PR's 30 s arm compares per-layer vs fused graphs, but the gates are
  resolved once at init (`ctx->use_fused`) while segmentation wraps
  `separate_full` per segment — so BOTH arms segment identically and an
  overlap-add error is exactly common-mode between them;
* the golden10s fixture sits at the threshold (guard is `n_samples <= seg_len`)
  and takes the whole-buffer path, so it never segments.

Kernel `tools/kaggle/melband-seg-evidence-v2/`, run
`${KAGGLE_ACCOUNT}/crispasr-melband-seg-evidence-v2`. 20 s of real music with vocals
(librosa `fishin`, ccMixter/CC; ASR-verified to contain sung content, because
solo speech is near pass-through through a vocals separator and instrumental
music is degenerate the other way). Arm B gets 4 chunks / 3 internal
boundaries. Reference = the unsegmented lucidrains model, `bs-roformer==0.3.10`.

    arm            SDR vs ref   cosine    |est|/|ref|
    A no-seg          50.83     0.999996     1.0001
    B default (8 s)   27.91     0.999256     1.0107
    C seg=3 s         26.89     0.999046     1.0108

    within-arm (ref = arm A)   boundary  interior   delta   #bnd
    B default                    31.00     27.25    +3.75      3
    C seg=3                      25.20     27.82    -2.62      8

**1. NO BOUNDARY ARTEFACT — and as a positive measurement, not a null.** Arm B's
crossfades reconstruct 3.75 dB BETTER than its interiors. That is the EXPECTED
signature of a healthy overlap-add: the crossfade averages two independent
segment estimates, halving noise variance for a predicted 10*log10(2) = 3.01 dB
gain. Measuring ~+3 dB is therefore a *test that the overlap-add is working*,
and a crossfade that does NOT gain ~3 dB would indicate a defect even if it
were not worse than the interior. Confirmed by two independently written
implementations agreeing to 0.01 dB.

**2. The segmentation cost is a CONTEXT effect, not an edge effect.** B differs
from unsegmented by 22.9 dB (196x the residual energy), and that cost is broad:
residual energy by decile tracks the vocals, and body SDR equals whole-clip SDR
to 2 dp. Nothing is concentrated at the seams. This is the opposite of the
intuition the PR invites, and it is why boundary-vs-interior localisation was
the measurement worth building.

**3. ⚠ THE 22.9 dB DOES NOT ESTABLISH THAT SEGMENTATION IS WRONG.** The trained
chunk is 8 s. Arm A and the torch reference BOTH process the full 20 s
unsegmented, i.e. both extrapolate 2.5x past the training window, and they
agree (50.83 dB) precisely because they do the same unusual thing. The
reference is ground truth for "what unsegmented inference produces", NOT for
"what the model should produce on 20 s audio". Segmenting to the trained window
may well be the more faithful path. DECISIVE FOLLOW-UP: compare arm B against a
reference that ALSO chunks at 8 s with the same overlap — if they match,
chunked inference is faithfully implemented and the 22.9 dB is inherent to
chunking rather than to us.

**4. A false finding, recorded so nobody re-derives it.** The final
zero-padded segment scores -3.00 dB, which looks like a dramatic tail defect.
It is an ill-conditioned ratio: reference vocals rms there is 0.000139 against
0.106 in the body (near-silence), the region contributes 0.0% of total residual
energy, and arm A itself scores only 8.48 dB in the same place. Always read
residual ENERGY SHARE next to an SDR ratio, exactly as the harness prints
`|mine|` next to cosine.

## Licensing (verified 2026-07-19, do not re-derive)

| Artifact | License | Source of truth |
|---|---|---|
| Architecture code | **MIT** | `lucidrains/BS-RoFormer` LICENSE ("MIT License, Copyright (c) 2023 Phil Wang") |
| Weights (Kim vocals) | **MIT** | `KimberleyJSN/melbandroformer` HF card `license: mit` |
| Kim's inference repo | **NONE DETECTED** | `KimberleyJensen/Mel-Band-Roformer-Vocal-Model` has no LICENSE file |

⚠ **Read the lucidrains (MIT) implementation as the blueprint, NOT Kim's
inference repo** — that repo ships no license, so transcribing it would be
unlicensed copying. Kim's contribution we rely on is the *weights* (MIT) and
the *config YAML* (facts/hyperparameters, not creative code). Same shape as the
Transcoda clean-room note in LEARNINGS.

## Target checkpoint + config (Kim vocals)

`config_melband_roformer_vocals_kim.yaml` — note these differ from lucidrains
defaults, which is exactly why the blueprint gets read instead of assumed:

```
sample_rate 44100   chunk_size 352800   num_stems 1  (vocals; other = residual)
dim 384             depth 6             stereo true
time_transformer_depth 1                freq_transformer_depth 1
heads 8             dim_head 64         num_bands 60
stft_n_fft 2048     stft_hop_length 441 (NOT the lucidrains default 512)
dim_freqs_in 1024   mask_estimator_depth 2
attn_dropout 0      ff_dropout 0        flash_attn true
```

## Blueprint — exact forward (lucidrains `mel_band_roformer.py`)

1. `raw_audio (b,s,t)` → pack `(b*s, t)` → `torch.stft(..., return_complex=True)`
   → `view_as_real` → `(b,s,f,t,2)`.
2. **Layout:** `rearrange('b s f t c -> b (f s) t c')` — stereo folded into the
   frequency axis, **frequency-major, channel fastest** ⇒ packed index
   `= f*channels + s`.
3. **Mel band membership (binary, not weighted):**
   `mel = librosa.filters.mel(sr, n_fft, n_mels=num_bands)` → then **two
   hand-tweaks that shift DC/Nyquist membership and are trivial to miss**:
   `mel[0,0] = mel[0,1]*0.25`, `mel[-1,-1] = mel[-1,-2]*0.25`.
   Then `freqs_per_band = mel > 0` — the filterbank is used **only** to decide
   *which* bins belong to a band; the weights are discarded ("binary as in
   paper, then masks are averaged over overlapping regions").
4. `freq_indices` = per-band bin indices concatenated (a gather index). Stereo:
   `freq_indices = freq_indices*2 + arange(2)`, flattened — matches (2).
5. Gather `x = stft_repr[batch_arange, freq_indices]` → `'b f t c -> b t (f c)'`
   (complex folded into the freq axis). Per-band input width
   `= 2 * num_freqs_in_band * audio_channels` (bands have **different** widths).
6. `BandSplit`: per band `RMSNorm(dim_in) → Linear(dim_in, dim)`; stack →
   `(b, t, n_bands, dim)`.
7. Per layer (`depth` times): optional `linear_transformer` → **time**
   transformer (over `t`) → **freq** transformer (over `f`), RoPE inside.
   ⚠ **Value residuals**: the first layer's attention values are carried
   forward as `value_residual` into every later layer, separately for the
   linear/time/freq streams. Silently dropping this changes the output.
8. `MaskEstimator` per stem: per band
   `MLP(dim → dim_in*2, hidden=dim*4, depth=mask_estimator_depth, act=Tanh)`
   → `GLU(-1)` (halves back to `dim_in`) → concat over bands.
   ⚠ activation is **Tanh**, not GELU/SiLU.
9. Masks → `'b n t (f c) -> b n f t c'` → `view_as_complex`. Same for stft.
10. **Overlap handling:** `scatter_add_` masks into zeros at `freq_indices`
    (dim=2), then divide by `num_bands_per_freq.clamp(min=1e-8)` — i.e.
    overlapping bands are **averaged, not summed**.
11. `stft_repr = stft_repr * masks_averaged` — a **true complex multiply**, not
    magnitude masking.
12. Optional `zero_dc`: `index_fill(freq=0, 0.)`.
13. `torch.istft(..., length=raw_audio_length)` → `'(b n s) t -> b n s t'`.

## Verified band layout (librosa, no weights — sr 44100 / n_fft 2048 / 60 bands)

Computed and checked before touching the weights, because it needs no model at
all and it is where an off-by-one hides:

```
mel filterbank            (60, 1025)
mel[0,0]   0.000000e+00 -> 1.252268e-03   (tweak is LOAD-BEARING, see below)
mel[-1,-1] 1.809571e-18 -> 2.677714e-06   (membership unchanged; already > 0)
freq_indices              1979 mono / 3958 stereo, max index 2049
num_freqs_per_band        min 6   max 130  sum 1979
num_bands_per_freq        min 1   max 2    (so the overlap denominator is 1 or 2)
band input widths         min 24  max 520  sum 7916   (= 2*1979*2 ✓)
```

⚠ **The `mel[0,0]` tweak is not cosmetic.** Before it, `mel[0,0]` is *exactly
0.0*, so DC (bin 0) belongs to **no** band and the reference's own
`assert freqs_per_band.any(dim=0).all()` would fail. Any C++/converter path
that rebuilds the filterbank without this line produces a different — and
silently wrong — `freq_indices`.

**Design decision:** bake `freq_indices` + `num_bands_per_freq` into the GGUF
rather than recomputing librosa's mel in C++. Their construction depends on
librosa's exact mel edge placement *plus* the two tweaks; reproducing that in
C++ buys nothing and risks a whole class of silent divergence. They are two
small integer arrays.

## Port gotchas (carry into the C++ port)

- Band membership depends on librosa's mel edges **and** the two hand-tweaks —
  an off-by-one there shifts `freq_indices` and silently corrupts every band.
  Diff `freq_indices` / `num_bands_per_freq` as stage 0 before any tensor math.
- hop 441 (not 512) — assert the frame count against the reference.
- Averaging denominator is per-frequency band-count, not per-band.
- Complex multiply, Tanh MLP, GLU halving, value residuals — each is a silent
  divergence if missed.

## ⚠ Reference env pin: `bs-roformer==0.3.10` (do not re-derive)

The Kim vocals checkpoint (KimberleyJSN/melbandroformer, MIT) predates
hyper-connections. Loading it into the WRONG bs-roformer version runs the diff
on mostly-random init and lies. Measured strict-load key mismatch vs the ckpt:

| version | missing | unexpected |
|---|---|---|
| 0.3.10  | **0** | **0** ✅ |
| 0.4.0 / 0.4.1 / 0.5.x | 90 | 30 |
| ≥0.6.x (hyper-connections) | 462 | 120 |

So the reference dumper MUST run under `bs-roformer==0.3.10`. Acceptance for a
valid reference run = `load_state_dict(strict=False)` reports 0 missing AND 0
unexpected (the dumper prints these; treat any nonzero as a failed run).
The `torch.stft` "rectangular window" warning is benign — it comes from a
one-time shape-probe in `__init__`, not the forward (which uses Hann).

Fixture persisted at
`/Volumes/backups/ai/crispasr-models/melbandroformer/{MelBandRoformer.ckpt,
config.yaml,ref_mbr.gguf,clip2s.wav}` (off /tmp, survives reboot).

## Checkpoint tensor map (684 tensors — for the converter)

```
band_split.to_features.{b}.0.gamma        (dim_in_b,)     RMSNorm per band
band_split.to_features.{b}.1.{weight,bias}(384, dim_in_b) Linear dim_in_b -> dim
  dim_in_b = 2 * num_freqs_in_band_b * channels  (band 0 = 28, i.e. 7 freqs)

mask_estimators.0.to_freqs.{b}.0.{0,2,4}.{weight,bias}    MLP (depth 2):
  .0 Linear 384->1536 | .2 Linear 1536->1536 | .4 Linear 1536->(2*dim_in_b)
  then GLU(-1) halves back to dim_in_b. Activation = Tanh (net between linears).

layers.{L}.{M}   L in 0..5 (depth 6), M in {0=time, 1=freq}  (NO linear xf)
  .layers.0.0  attention:  norm.gamma (pre-RMSNorm)
               to_qkv.weight (1536,384) = 3*512 (inner = heads8 * dim_head64)
               to_gates.{weight,bias} (8,384) per-head output gate
               to_out.0.weight (384,512)
               rotary_embed.freqs (32,)  RoPE (dim_head/2)
  .layers.0.1  ffn: net.0.gamma (RMSNorm) | net.1 Linear 384->1536
               | net.4 Linear 1536->384   (GELU between)
  .norm.gamma  post-stack RMSNorm on the (time|freq) transformer
```

All of this maps onto `core_attn` (GQA/RoPE), `ffn.h` (SwiGLU/GELU FFN), and a
RMSNorm helper — only the per-band linears + band gather/scatter are new.
`freq_indices` (int32, 3958) and `num_bands_per_freq` (int32, 1025) get **baked
into the GGUF** (see the band-layout note above — reproducing librosa's mel
edges in C++ is a needless divergence class).

## Reuse (DRY — do not write new DSP)

`src/core/fft.h` (STFT), `src/core/istft.h` (iSTFT), `core_attn` (attention),
`ffn.h`, RMSNorm. Only the band gather/scatter + per-band linears are new.

## Coordination

HTDemucs (other session, converter `a6a447587` + dumper `60ada0a06`) is the
same new category and shares the `--task separate` surface (CLI flag, stem WAV
output, capability bit). **Whoever lands that scaffolding first owns it; the
other builds on it.** Per-backend files are additive and conflict-free.

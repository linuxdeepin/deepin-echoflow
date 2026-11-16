# BTC chord recognition — blueprint notes

Read from `jayg996/BTC-ISMIR19` (MIT) before writing any C++, per HARD RULE #1.
Five details here would each have been a silent parity bug if assumed from
"it's a transformer".

## Hyperparameters (`run_config.yaml`)

| | |
|---|---|
| feature (embedding) size | **144** — matches `core/cqt.h` n_bins |
| hidden size | **128** |
| layers | **8** |
| attention heads | **4** |
| FFN filter size | **128** |
| sequence length (timestep) | **108** |
| chord classes | **25** (`btc_model.pt`) / **170** (`btc_model_large_voca.pt`) |
| dropout (input/layer/attn/relu) | 0.2 — inference-time no-op |

Front end: CQT `n_bins=144`, `bins_per_octave=24`, `hop_length=2048`,
`sr=22050`. This is exactly what `core/cqt.h` was built and librosa-validated
for, so the front end is already done.

## The five details that are NOT the obvious default

1. **It is NOT full bidirectional attention.** (Confirmed at `btc_model.py:73`:
   the backward block is constructed with
   `torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3)`, and at line 89
   it is fed the SAME `x` — bidirectionality comes from the transposed mask, not
   from reversing the sequence.) Each of the 8 layers runs TWO
   self-attention blocks: a forward one masked with
   `np.triu(np.full([L,L], -inf), 1)` (upper triangle above the diagonal is
   -inf, i.e. **causal — attend to past + self**) and a backward one using the
   **transposed** mask (attend to future + self). Their outputs are
   **concatenated and projected back to hidden_size**. Passing `mask=nullptr`
   would silently give full attention and be wrong — the exact trap the dev
   guide calls out for `soft_max_ext`.

2. **Positional encoding is CONCATENATED halves, not interleaved.**
   `signal = concat([sin(scaled_time), cos(scaled_time)], axis=1)` — so the
   first `hidden/2` channels are all sin and the second half all cos. (htdemucs'
   CrossTransformer interleaves; do not copy that layout.)
   `inv_timescales = min_timescale * exp(arange(n) * -log_timescale_increment)`,
   `log_timescale_increment = log(max_timescale/min_timescale) / (n - 1)`.

3. **The FFN is CONVOLUTIONAL with kernel 3, not a 1x1 linear.**
   `PositionwiseFeedForward(layer_config='cc')` = Conv(k=3) -> ReLU -> Conv(k=3).
   A linear FFN would be the wrong op AND the wrong receptive field.

   **Padding is SYMMETRIC, not causal.** `transformer_modules.py` documents a
   `pad_type='left'` option — `padding = (k-1, 0)` — but `btc_model.py:14`
   passes **`padding='both'`**, giving `(k//2, (k-1)//2)` = **(1, 1)** for k=3.
   Reading only the module docstring gives causal padding and a one-frame
   output shift. Read the CALL SITE, not the option list.

4. **Attention scaling is applied to Q, not to the scores**, as
   `queries *= (total_key_depth // num_heads) ** -0.5`. Numerically equivalent
   to scaling the logits, but worth matching for exact parity.

5. **LayerNorm eps = 1e-6**, not the 1e-5 used elsewhere in this repo.

Ordering is pre-norm: LayerNorm -> attention -> dropout -> residual ->
LayerNorm -> FFN -> dropout -> residual.

## Three MORE details that only the CHECKPOINT reveals

The architecture docs do not mention any of these. Inspecting
`test/btc_model.pt` directly was necessary:

6. **Scalar input normalisation ships WITH the checkpoint.** The `.pt` has
   top-level `mean = -2.2279364` and `std = 1.7191066` alongside `model`. The
   log-CQT features are normalised `(x - mean) / std` before the embedding
   projection. Nothing in the README or config says so.

7. **The checkpoint's bidirectional LSTM is DEAD WEIGHT — do not implement it.**
   `output_layer.lstm.*` is present (8 tensors: `weight_ih_l0 (256,128)`,
   `weight_hh_l0 (256,64)` and `_reverse` twins), which looks like a biLSTM
   head. It is constructed in `OutputLayer.__init__`
   (`transformer_modules.py:64`) but **`SoftmaxOutputLayer.forward` never calls
   it** — line 75 is just `logits = self.output_projection(hidden)`.

   So the real head is a bare Linear(128 -> n_chords). Implementing the LSTM
   would be wasted work AND would produce wrong output. The converter therefore
   SKIPS those 8 tensors deliberately.

   (Mirror image of the htdemucs DConv bug, where bound-but-unused norms had to
   be wired IN. Same lesson from the opposite side: whether a weight is used is
   a property of the forward pass, not of the checkpoint.)

8. **Attention linears are bias-free** — q/k/v/output each have `.weight` only.

### Actual end-to-end shape

    CQT(144) -> (x - mean)/std -> embedding_proj(144->128, no bias) -> + posenc
      -> 8 x [ fwd attn block || bwd attn block -> concat(256) -> linear(256->128) ]
      -> final LayerNorm -> output_projection(128 -> 25 or 170)

Processed in fixed blocks of `timestep` = 108 frames (`test.py:71` slices
`feature[:, 108*t : 108*(t+1), :]`), NOT a sliding window — the bias mask is
built for exactly that length.

No sqrt(d) embedding scaling: `embedding_proj(x)` then `x += timing_signal`.

213 converted tensors: 8 layers x 26, plus embedding_proj, the final LayerNorm
pair and the 2 projection tensors. The 8 unused LSTM tensors are skipped.

## Two MORE found by the numpy parity spec

Neither is visible in the architecture; only diffing stage-by-stage against the
torch model surfaced them.

9. **LayerNorm std is UNBIASED (ddof=1)** — it is `torch.std`, which defaults to
   the n-1 denominator. numpy's `.std()` defaults to n. At width 128 that is a
   0.4% error, worth `max_abs` 1.5e-2 vs 5.3e-7. Small enough to look like
   "drift" and be written off; structural enough to fail the gate.

10. **The FFN has a TRAILING ReLU after the second conv.** Upstream:

    ```python
    for i, layer in enumerate(self.layers):
        x = layer(x)
        if i < len(self.layers):     # always true: i maxes at len-1
            x = self.relu(x)
    ```

    The guard was clearly meant to be `len(self.layers) - 1`, so ReLU fires
    after the LAST conv as well. It is an upstream bug, but it was baked into
    the trained weights, so it MUST be reproduced. Without it the block scores
    cos 0.461.

### Bisection that found them

    embed+posenc      cos=1.000000   <- exact
    layer0 fwd block  cos=0.874472   <- diverges here
      +-- norm (ddof=0) max_abs 1.5e-2 / (ddof=1) 5.3e-7   <- finding 9
      +-- attention   cos=1.000000                          <- exact
      +-- ffn         cos=0.461213                          <- finding 10

## Verified

`tools/btc_torch_parity.py` scores the numpy spec against the real PyTorch
model. Both checkpoints:

| variant | cos | max_abs | argmax agreement |
|---|---|---|---|
| `btc_model.pt` (25 chords) | 0.99999995 | 3.3e-06 | 1.0000 |
| `btc_model_large_voca.pt` (170) | 1.00000004 | 1.7e-05 | 1.0000 |

That spec is what `src/btc_chords.cpp` must match; keep the two in lockstep.

## Licence

Code MIT; the shipped `test/btc_model{,_large_voca}.pt` checkpoints were trained
on Isophonics / Robbie Williams / UsPop2002 annotations, which are
**CC BY-NC-SA**. Ships behind `--accept-license cc-by-nc-sa-4.0` (the gate
landed in 90d1a0c9e). See the licence scoping in `PLAN.md`.

## Front end: core/cqt.h has a per-bin scale mismatch vs librosa

Found by the end-to-end acceptance test, NOT by the per-stage diff — the diff
replays the reference's own `input_feat` by design, so it cannot see a front-end
error. Two separate gates are needed and this is why.

Dumping the runtime's features (`CRISPASR_BTC_DUMP_FEAT=<path>`) and scoring
them against `librosa.cqt` on the same file:

    frame count   87 vs 87          exact
    cos           0.986230
    librosa       mean -8.144  std 3.233
    core/cqt.h    mean -11.389 std 2.988

The gap is a PER-BIN scale factor, not a global one, and it tracks
`log(sqrt(N_k))` where `N_k` is bin k's filter length:

    bin   0   offset 4.897   log(sqrt(N_k)) 5.022
    bin 143   offset 3.011   log(sqrt(N_k)) 2.958

i.e. librosa's magnitudes are larger by roughly `sqrt(N_k)`. librosa's
`scale=True` (its default) normalises each bin by the square root of its filter
length; `core/cqt.h` normalises by the L1 norm instead, and the two disagree
per bin.

**Why `tools/cqt_librosa_parity.py` passes anyway:** it scores per-frame shape
CORRELATION and PEAK-BIN match. Both are scale-invariant, so a per-bin
magnitude error is invisible to it. This is the same class of miss as the
htdemucs iSTFT `1/sqrt(nfft)` bug, which cosine also could not see and only the
`|mine|` vs `|ref|` magnitude columns caught.

**Consequences.** BTC normalises with the checkpoint's own scalar stats
(mean -2.228, std 1.719), so a shifted feature distribution pushes the input
out of distribution: on a synthetic I-V-vi-IV tone the reference torch model
predicts N/G/C while this runtime predicts N everywhere.

**Fix (not yet applied — `core/cqt.h` is shared and owned elsewhere):**
1. Add a SCALE-SENSITIVE assertion to `cqt_librosa_parity.py` — max relative
   magnitude error per bin, not just correlation — so this cannot recur.
2. Match librosa's `scale=True` normalisation, or expose it as a `Params` flag
   with librosa-compatible defaults since the header names BTC as its primary
   consumer.
3. Re-run both the CQT parity tool and the BTC acceptance test.

The BTC graph itself is unaffected: 13/13 stages at cos 1.000000.

---

## 11. The front end is a CHUNKED CQT, not a continuous one

`utils/mir_eval_modules.audio_file_to_features` splits the signal into
`mp3.inst_len` (10 s) segments, CQTs **each one independently**, and
concatenates. librosa centres every call, so each chunk carries its own edge
padding — the result is not equal to a single CQT of the whole signal.

Measured on the upstream 257 s test clip:

| | frames |
|---|---|
| chunked (the reference) | 2778 |
| continuous (single `librosa.cqt`) | 2770 |

Implementing `librosa.cqt` faithfully but calling it once is a silent accuracy
bug: our features scored cos **0.9993** against a continuous transform and only
**0.8815** against the reference pipeline. This is invisible to
`crispasr-diff btc`, whose reference dump replays `input_feat` by design.

## 12. Frame duration is `inst_len / timestep`, not `hop / sample_rate`

`feature_per_second = config.mp3['inst_len'] / config.model['timestep']`
= 10/108 = **0.0925926 s**. The obvious `hop/sample_rate` = 2048/22050 =
**0.0928798 s** is 0.31 % larger — 0.79 s of accumulated drift over a
four-minute song, so every chord boundary lands progressively late.

The two are related but not equal: one 10 s chunk yields exactly `timestep`
= 108 frames, which is what makes the chunk-derived rate the correct one.

Fixing 11 and 12 moved agreement with the PyTorch reference on real music from
**86.63 % to 98.56 %** (`mir_eval` tetrads). Guarded by
`tests/test-btc-vocab.cpp` `[geometry]`.

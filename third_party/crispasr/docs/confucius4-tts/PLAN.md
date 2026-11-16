# Confucius4-TTS — port plan (§377)

**Issue:** #377 · **License:** Apache 2.0 · **Languages:** 14 (zh, en, ja, ko, de, fr, es, id, it, th, pt, ru, ms, vi)
**Source:** [netease-youdao/Confucius4-TTS](https://github.com/netease-youdao/Confucius4-TTS) ·
**Weights:** [HF repo](https://huggingface.co/netease-youdao/Confucius4-TTS) (3.06 GB total)

---

## NOW — status (2026-08-25)

# ✅ PORT COMPLETE — FULLY NATIVE ZERO-SHOT PIPELINE, ROUNDTRIP 8/8

Kernel run 18 (`${KAGGLE_ACCOUNT}/crispasr-confucius4-cfg-verify`), all green:

- **`[NATIVE-full] 8/8 = 100%`** — `--voice jfk.wav`, no env escapes, no
  Python anywhere: native SP-BPE tokenizer (byte-identical to
  AutoTokenizer), native w2v-BERT layer-17 (sidon encoder-only, cos-1.0
  ECAPA), native CAMPPlus style + 22.05 kHz prompt mel, beam-sample T2S
  (transformers-4.52.4-faithful, num_beams=3), S2A CFM (25-step, CFG 0.7),
  BigVGAN. Even the spoken AI-disclaimer is synthesized natively.
- NATIVE-tok / NATIVE-voice / COND-full / RUNTIME-codes-through-ref-S2A /
  REF-dumpercond / reference-e2e-control: all 8/8.
- nocond arms 0/8 BY DESIGN (zero-shot model; the reference babbles there too).

**Artifacts** (cstr/confucius4-tts-GGUF, Apache-2.0 card): T2S f16/q8_0/q4_k
(baked vocab), S2A f16/q8_0/q4_k (baked CAMPPlus), BigVGAN f16/q8_0, w2v f16.
`-m auto` downloads T2S q4_k + S2A q4_k + BigVGAN + w2v.

**Usage:** `crispasr --backend confucius4-tts -m auto --tts "..." -l en \
--voice ref.wav --i-have-rights --tts-output out.wav`

**Checklist:** src runtime ✓ CLI adapter ✓ factory ✓ CMake ✓ c_api session ✓
registry ✓ quantize rules ✓ parity harnesses (s2a/t2s_parity) ✓ params unit
test ✓ go cgo sync ✓ docs/README ✓ HF license card ✓.

## Post-merge verification (runs 19–20 + VPS, 2026-08-25)

- **Run 19**: host-table embedding fast path byte-identical → default ON.
- **VPS run (8 GB, no GPU)**: fully native `--voice` roundtrip transcribes
  verbatim; **peak RSS 2.23 GB**. Slow on CPU (~3-4 min/sentence on quiet
  4-core Kaggle; VPS wall scales with contention — 20-70 min at load 13-22).
- **Run 20, persistent decode graph A/B (clean box)**: PCM BIT-IDENTICAL but
  867.4 s vs 882.0 s — the fixed-Lk=1521 attention outweighs the saved
  per-step graph builds on CPU. **Default stays the rebuild path**; the
  correct persistent path is kept behind `CRISPASR_CONFUCIUS4_PERSIST=1` for
  a future GPU port (launch-bound dispatch is where it wins — see the
  parakeet/nemotron precedent) or a bucketed-Lk variant.
- **Run 20, conditioned S2A parity** (new `--style`/`--prompt-mel` mode +
  kernel cond-full arm): **cos=1.000000 at every stage** over the full
  947-frame prompt path — regulator, input_embed, WaveNet, v steps 1–25,
  final mel. Bug 13's blind spot is covered for good.
- Live test `test-confucius4-tts-live` added (passes locally, 9 assertions).

## CUDA GPU path — VALIDATED (gpu-verify run 1, P100, 2026-08-25)

Fully native `--voice` roundtrip on CUDA: **8/8, transcript verbatim, 183.2 s
vs 685.2 s CPU control = 3.74×** — with zero code changes (the all-gallocr
single-backend graphs ran as-is). GPU+persist 185.7 s (neutral → persist
stays gated everywhere). Kernel: `${KAGGLE_ACCOUNT}/crispasr-confucius4-gpu-verify`.
GPU-effort doc: `/mnt/volume1/conf4gpu.md`.

**Open (perf, optional):** CFG cond+uncond fusion in one DiT pass
(seq-concat + block-diagonal mask — now the biggest CUDA lever); bucketed-Lk
persistent decode; BigVGAN raw-path A/B; Metal/Vulkan validation (Mac).

## CFG fusion (`CRISPASR_CONFUCIUS4_CFG_FUSE=1`)

The two per-step DiT estimator passes (cond + uncond) run as ONE graph eval:
seq-concat along time (`[0,T)` = cond, `[T,2T)` = uncond) with a
block-diagonal F16 flash-attn mask, positions `[0..T-1, 0..T-1]`, and the
CFG blend `(1+cfg)*v_cond - cfg*v_uncond` done in-graph (output is `(mel,
T)`, halving the readback).  Every graph op is per-frame except attention
(masked) and the WaveNet k=5 convs — those would smear the arms into each
other across the seam (±2 frames/layer through the residual chain), so the
WaveNet runs per-arm on split halves inside the same graph.  The mask and
positions are re-set on EVERY compute (§234 gallocr aliasing).  Parity dumps
(`CRISPASR_CONFUCIUS4_DUMP_S2A`) force fusion off — the harness expects
per-pass semantics.

**Verdicts (2026-08-25, roundtrip verbatim in EVERY arm):**

| box | s2a | base | fused | verdict |
|---|---|---|---|---|
| CUDA P100 (gpu-verify run 2) | f16 | 179.8 s | 181.6 s | neutral |
| CUDA P100 (run 2, first q4_k GPU run) | q4_k | 176.4 s | 177.3 s | neutral |
| Metal M1 (quiet-box pair) | q4_k | 181 s | **158 s (−12.7%)** | **fused wins** |
| Metal M1 (quiet-box pair) | f16 | 130 s | 131 s | neutral |
| CPU M1 | q4_k | 658 s | 903 s | fused loses |

→ **default ON on Metal only** (`ggml_backend_name` contains "Metal"), env
var still forces either way.  Mechanism: fusion removes per-eval overhead
and doubles matmul width — on Metal q4_k that amortizes the k-quant dequant
(which is also why f16 s2a at 130 s beats q4_k at 181 s there: prefer the
f16 s2a on Metal, it is only 213 MB); on CUDA the masked 2T flash-attn
(4T² scores vs 2·T² unfused; the KV_max tail-skip needs K%256==0 and does
not engage) cancels the savings; on CPU it outright loses.  These runs are
also the **first Metal validation of the whole backend** (base arms
transcript-verbatim too) — the Metal half of the Metal/Vulkan open item is
closed.  ⚠ Wall-times on the shared M1 are only comparable at load < ~3;
two contaminated series (load 13–43) flipped the f16 verdict before the
quiet-box pairs settled it.

### 13 bugs total — 9–13 this session

### Bugs 9–12 (this session, all found by line-by-line reading + parity)

9.  **`transformer.ln_f` skipped** — reference stacks GPT2's own ln_f (its
    output IS the S2A lm_latent) then `final_norm` before `semantic_head`;
    runtime bound only `final_norm`. Both always in the GGUF.
10. **Repetition penalty parsed but never applied** (reference: 10.0 via HF
    processor). Env `CRISPASR_CONFUCIUS4_REP_PEN`.
11. **Missing `</s>`** — LlamaTokenizerFast has `add_eos_token=True`; the
    kernel's raw `tokenizers.Tokenizer` only prepends `<s>`. T2S never saw
    the end-of-text marker.
12. **`KvSelfAttnParams` uninitialized** — `attn_scale` (and gqa_mode etc.)
    was stack garbage passed verbatim to `soft_max_ext`; plus the prefill ran
    maskless (= FULL attention, HARD RULE #5) until a causal mask landed.
    This was the prefill cos≈0.2 that survived fixes 9–11.

Also landed: native ECAPA from `w2v_features.bin` under
`CRISPASR_CONFUCIUS4_COND_DIR` (validates the port; `_COND_PYEMB=1` forces the
Python embedding), real 93-entry LANGUAGE_TOKEN_MAP, LlamaTokenizer SP-BPE
native tokenizer (`core/spm_bpe.h`, shared with irodori), reference
`max_length` TOTAL-length semantics, reference e2e control arm in the kernel.

### Bug 13 — CFG uncond pass ran over T_mel, not T_total (the silence bug)

Run 15's rms print exposed it: every CONDITIONED run since the prompt path
landed produced **digital silence** (COND-full 3.25s rms=0.0000) while
unconditioned parity stayed cos=1.000000. solve_euler's uncond arm runs over
the SAME full sequence (prompt+target; only mu/prompt_x/spks zeroed) — the
runtime passed `T_mel`, so with a 947-frame prompt the CFG blend read
`v_uncond` out of bounds past frame 280. Without a prompt `T_total==T_mel`,
so the parity harness could never see it: **the conditioned S2A path had
zero harness coverage** (s2a_parity hardcodes prompt_len=0/spks=0 — still
open to extend). Fixed in `aae66528`; run 16 verifies.

Run 15 also proved: reference e2e control 8/8 again; conditioned beam decode
produces plausible lengths (163 codes); nocond beam decode EOSes early
(8 codes) — consistent with an unconditioned prompt being out-of-distribution.

**Queue after roundtrip passes:** converter re-run (vocab already baked by
converter; ADD CAMPPlus into S2A GGUF) · native conditioning (CAMPPlus bind =
dots pattern, prompt mel 22k via core/mel.h, w2v-BERT layer-17 via sidon.cpp
adaptation) · registry/tests/checklist · perf (BigVGAN raw path, persistent
decode graph for the 3-beam T2S).

### Bugs found and fixed on this branch

1. **Timestep embedding was wrong twice** (`s2a_sinusoidal_embed`) — the reference
   `SinusPositionEmbedding.forward` applies `scale=1000` to `t` and concatenates
   **cos then sin**; the port had no scale and sin-then-cos.  Measured against the
   reference module: old cos **0.13–0.18** (essentially uncorrelated), and across
   the 25 ODE steps the old embedding's min pairwise cosine was **0.972** — the
   estimator was seeing an almost constant timestep signal, which by itself
   prevents flow matching from working.  Fixed version is exact (cos 1.0000).
   Unit check: `tools/…/tstep_unit.py` (scratch).

2. **The length regulator was skipped entirely** (`s2a_build_conditioning`).  The
   handover assumed `sampling_ratios=[1,1,1,1]` made it an identity, but the
   ratios only control *how many* conv blocks exist — `InterpolateRegulator` is
   `content_in_proj(1024→512)` → nearest-interpolate → 4×[Conv1d k=3 + GroupNorm(1)
   + Mish] → Conv1d k=1, all of it learned and all of it present in the GGUF
   (`length_regulator.*`, 274 tensors).  Also `encoder_proj` is Linear(2304,
   **1024**), not 512, so the old code truncated its output to the first 512 dims
   and fed that straight to `mu_projection`.  Now ported in full; verified against
   torch's real module at **cos 1.0000000000** with f32 weights (max_abs_diff
   8.6e-08) and cos 0.99898 through the Q4_K GGUF (pure quantization).
   Legacy path kept behind `CRISPASR_CONFUCIUS4_LR_LEGACY=1`.

3. **CFG implemented** (`s2a_flow_matching`) — second pass with mu, reference mel
   and speaker embedding zeroed, blended `v = (1+cfg)·v_cond − cfg·v_uncond`,
   matching `solve_euler`.  `cfg_rate` defaults to 0.7 and `ode_steps` to 25 (the
   reference values); a **negative** `cfg_rate` disables CFG, so a zero-initialised
   params struct still gets the reference default.  Env override
   `CRISPASR_CONFUCIUS4_CFG_RATE` for A/B without recompiling.

4. **The CLI forced greedy T2S decoding** — `crispasr_backend_confucius4_tts.cpp`
   did `cp.temperature = p.temperature`, and `whisper_params.temperature` defaults
   to **0.0**, overwriting the backend's reference default of 0.8.  The reference
   runs `do_sample=True, temperature=0.8`.  This fits the earlier symptom of the
   decode running to the 1520-token cap without ever emitting EOS.  Now only
   overridden when the user actually passes a temperature.

Also corrected: the S2A noise temperature was reusing `params.temperature` (the
T2S *sampling* knob); the reference passes 1.0 to the CFM decoder.  Env override
`CRISPASR_CONFUCIUS4_S2A_TEMP`.

### Parity harness (new)

`CRISPASR_CONFUCIUS4_DUMP_S2A=<dir>` dumps `semantic_codes`, `lm_latent`,
`z_init`, `cond` and `mel` as raw f32/i32 with a `shapes.txt` manifest, so the
real PyTorch S2A can be driven on **identical** inputs and noise and compared
per stage (`s2a_parity.py`).  This is the acceptance gate for the ODE, alongside
the TTS→ASR roundtrip.

### Verified as already correct (read against the Python, no change needed)

DiT attention (RoPE NORMAL/adjacent-pair, base 10000, scale 1/√head_dim), AdaLN
weight/bias split order, FinalLayer's opposite shift/scale order, SwiGLU, U-Net
skip emit/receive sets ({0..5} / {7..12}), `skip_linear(cat[x_res, x_mel])`,
WaveNet res/skip halves and dilation=1/pad=2, InputEmbedding concat order
(x, cond, mu_proj, spks), `T_mel = int(T_sem × 1.72)`, and the `lm_latent`
alignment (the port collects one extra trailing row, which the conditioning
correctly ignores).

### S2A port: FULL PARITY (Kaggle run 7, kernel `crispasr-confucius4-cfg-verify`)

The S2A stage is now numerically exact against the PyTorch blueprint, driven on
identical semantic codes, lm_latent and initial noise (F16):

```
cond (regulator)   cos=1.000000   |mine|=  10.4294  |ref|=  10.4291  max_abs_diff=5.1e-05
dit t1 / t2        cos=1.000000
dit x_in           cos=1.000000
dbg_blk00/06/12    cos=1.000000
dbg_xres / dbg_skip cos=1.000000
dbg_wn             cos=1.000000   (was ratio 2.0723)
dbg_fin            cos=1.000000
v step 1..25       cos=1.000000   (was 0.978)
mel (final)        cos=1.000000   |mine|= 834.8463 |ref|= 834.8934  max_abs_diff=0.0237
cpp mel mean=-5.808 floor=2.7%  ==  ref mel mean=-5.808 floor=2.7%
```

Seven bugs, all found by reading the blueprint against the port and bisecting
with the harness. In discovery order: the timestep embedding (missing
scale=1000, cos/sin swapped), the skipped InterpolateRegulator + truncated
encoder_proj, missing CFG, the CLI forcing greedy T2S, the cosine-instead-of-
linear ODE schedule, the invented English prompt in the old test kernel, and
the WaveNet channel split offsetting by element size instead of row stride.

**Method note worth keeping:** the WaveNet bug was only findable because the
harness prints `|mine|` next to `|ref|`. Every graph tap read cos ~ 0 with
IDENTICAL norms, which is the signature of a transposed comparison (a harness
bug), not a divergence -- and `dbg_wn`'s 2.07x ratio was the single number that
survived that reasoning. On cosine alone the conclusion would have been "the
whole transformer stack is broken".

### The remaining blocker is conditioning, not the port

The acceptance test still fails, and no S2A work can fix it:

```
[cpp-cli-f16]        "I'm not going to do it."             0/8
[REF-mel/torch-voc]  "I'm going to be a little bit more."  0/8
```

The PyTorch reference produces the same kind of babble on the same inputs,
because `spks`, the reference mel and the T2S `condition_emb` are all zero and
this model is **zero-shot** -- `ConfuciusTTS.generate` always takes a
`prompt_wav`. So the handover's ordering (CFG first, speaker conditioning
fourth) is inverted: conditioning is the gate on the roundtrip.

### Next: speaker conditioning

What is missing, in the reference's terms:
1. **T2S `condition_emb`** = `speaker_encoder(w2v_bert_layer17)` -> (1, 1280),
   prepended to the GPT-2 prefix. Currently a literal zero -- note
   `speaker_encoder(0)` is NOT zero, the ECAPA-TDNN has biases.
2. **S2A `spks`** = CAMPPlus 192-d. Already wired through
   `confucius4_tts_set_speaker()`; just never supplied.
3. **S2A prompt path**: `prompt_feat` (reference mel) -> `prompt_cond` expanded
   to T_ref and PREPENDED to the conditioning, `prompt_x` in solve_euler, `x`
   zeroed over the prompt span every step, and the prompt frames stripped from
   the output. None of this is implemented.

Cheapest route to a passing roundtrip, avoiding the 600M-param w2v-BERT port
entirely: inject all three pre-computed from a Python run, prove the pipeline
end to end, and only then decide whether to port the encoders.

## Architecture (read from Python, 2026-08-22)

Two-stage TTS with external conditioning models:

### Pipeline overview

```
text + prompt_wav
  │
  ├── [Wav2Vec2-BERT 2.0] → semantic_features (1, T_feat, 1024)   # layer 17, z-normalised
  ├── [CAMPPlus]           → style_embedding   (1, 192)            # ECAPA-TDNN speaker encoder
  ├── [Tokenizer]          → text_token_ids    (1, T_text)         # SentencePiece, vocab 32000
  │
  ▼
  [T2S: GPT-2 causal LM]  → semantic_codes (1, T_sem) + lm_latent (1, T_sem, 1280)
  │
  ▼
  [S2A: Flow-matching DiT] → mel (1, 80, T_mel)
  │
  ▼
  [BigVGAN vocoder]        → waveform @ 22050 Hz
```

### T2S model (Text-to-Semantic) — `t2s_model.safetensors` (2.64 GB)

- **Architecture:** GPT-2 backbone (HF `GPT2Model`)
  - 24 layers, d_model=1280, 20 heads, vocab=8194 (semantic codebook)
  - **Learned positional embeddings** (NOT RoPE): text pos + semantic pos, each an `nn.Embedding`
  - GPT2's own `wpe` replaced with a zero dummy; `wte` deleted entirely
- **Input concatenation:** `[condition_emb(1,1,1280) | text_emb(1,T,1280) | semantic_emb(1,T',1280)]`
  - `text_emb`: Embedding(32000,4096) → Linear(4096,4096) → SiLU → Linear(4096,1280) — frozen embed + MLP projection
  - `condition_emb`: ECAPA-TDNN speaker encoder (Qwen3TTSSpeakerEncoder) over the Wav2Vec2-BERT features
    - mel_dim=1024 (from w2v-bert), enc_dim=1280 (output)
    - 5-layer: TDNN + 3×SE-Res2Net + MFA + ASP + FC
  - `semantic_emb`: Embedding(8194, 1280)
- **Output head:** LayerNorm → Linear(1280, 8194) → semantic logits
- **Generation:** HF `generate()` with top-p/top-k/beam search, BOS=8192, EOS=8193
- **KV cache:** standard GPT-2 KV caching via HF

### S2A model (Semantic-to-Acoustic) — `s2a_model.pt` (417 MB)

- **Architecture:** Conditional Flow Matching (CFM) with DiT estimator + WaveNet final layer
  - DiT: hidden_dim=512, 8 heads, depth=13, cond_dim=512, style_dim=192
  - WaveNet: hidden_dim=512, kernel=5, dilation=1, 8 layers
  - Long skip connections (U-Net style)
- **Input pipeline:**
  1. Semantic token embedding: Embedding(8192, 8) → Linear(8, 1024)
  2. Concat with lm_latent: cat([lm_latent(1280), semantic_emb(1024)]) → Linear(2304, 1024)
  3. InterpolateRegulator: conv upsampling to target mel length (ratios [1,1,1,1])
  4. Prepend learned prompt condition (1, T_ref, 512)
- **Flow matching:** Euler ODE solver, 25 steps, CFG rate 0.7
- **Output:** mel spectrogram (80 bands), prompt portion stripped

### External models (NOT in the GGUF — must be separate or bundled)

1. **Wav2Vec2-BERT 2.0** (`facebook/w2v-bert-2.0`): ~600M params, extracts layer-17 hidden states
   - SeamlessM4TFeatureExtractor for audio preprocessing (mel filterbank)
   - z-normalised with per-dim mean/var from `wav2vec2bert_stats.pt`
2. **CAMPPlus** (`funasr/campplus`): ECAPA-TDNN speaker encoder
   - Input: 80-mel fbank @ 16kHz, mean-subtracted
   - Output: 192-dim speaker embedding
3. **BigVGAN** (`nvidia/bigvgan_v2_22khz_80band_256x`): mel→waveform vocoder

### Audio parameters

- Target sample rate: 22050 Hz
- Mel: n_fft=1024, hop=256, win=1024, 80 bands, fmin=0, fmax=None
- Prompt audio: resampled to both 16kHz (for w2v-bert + campplus) and 22050 Hz (for ref mel)

### Text formatting

```
formatted = "You are a helpful assistant. {lang_token}:{text}"
```

Where `lang_token` is a per-language string from `LANGUAGE_TOKEN_MAP` (e.g. "请朗读接下来的中文" for zh).

---

## GGUF strategy

This is a **5-model pipeline**. Options:

### Option A: Single mega-GGUF (infeasible)
Wav2Vec2-BERT alone is ~600M params — larger than many ASR models. Bundling all 5 into one GGUF would be >3 GB and force loading everything even when only the vocoder changes.

### Option B: Multi-GGUF (recommended, mirrors qwen3-tts)
- `confucius4-t2s-{quant}.gguf` — the 2.64 GB T2S model (GPT-2 backbone + text projector + speaker encoder + semantic head + position embeddings)
- `confucius4-s2a-{quant}.gguf` — the 417 MB S2A model (DiT + WaveNet + length regulator + token embedding)
- External models resolved via registry/cache:
  - Wav2Vec2-BERT 2.0: could share with any other backend that uses it (or ship a dedicated GGUF)
  - CAMPPlus: ~5 MB, can be baked into the T2S GGUF as extra tensors
  - BigVGAN: could share the existing `hifigan.h` core module or ship as a codec companion

### Option C: T2S+S2A bundled, externals separate
Merge T2S and S2A into one GGUF (they're always used together), keep w2v-bert and bigvgan as companion GGUFs via `--codec-model`. CAMPPlus baked in.

**Recommended: Option B** — matches the qwen3-tts `--codec-model` pattern and keeps file sizes reasonable for quantization.

---

## Existing CrispASR modules to reuse

| Need | Existing module | Notes |
|------|----------------|-------|
| GPT-2 attention | `core/attention.h` `kv_self_attn` | Causal masked self-attention with KV cache |
| Learned position embedding | — | New, but trivial (nn.Embedding lookup) |
| SiLU activation | `core/activation.h` | Already exists |
| LayerNorm | ggml native | `ggml_norm` |
| ECAPA-TDNN speaker enc | — | New module. Conv1d + Res2Net + SE + ASP pooling |
| DiT (S2A) | — | Similar to f5-tts DiT. Could share `core/` |
| WaveNet final layer | `core/hifigan.h` or new | WaveNet is different from HiFi-GAN |
| Flow matching ODE | — | Euler solver, same as f5-tts/chatterbox CFM |
| BigVGAN vocoder | `core/hifigan.h` | BigVGAN is a HiFi-GAN variant |
| Mel spectrogram | `core/mel.h` | Need to match params (n_fft=1024, hop=256, 80 bands) |
| SentencePiece tokenizer | `core/sentencepiece.h` | Already exists |
| Wav2Vec2-BERT | — | Large new model. Possibly share with future backends |

---

## Port steps (following the pipeline from CLAUDE.md)

1. **Converter** — `models/convert-confucius4-to-gguf.py`
   - T2S: extract GPT-2 weights + text projector + speaker encoder + position embeddings + semantic head from `t2s_model.safetensors`
   - S2A: extract DiT + WaveNet + length regulator from `s2a_model.pt`
   - CAMPPlus: bake into T2S GGUF from `funasr/campplus` checkpoint
   - `wav2vec2bert_stats.pt`: bake mean/var as tensors
   - Needs Kaggle for the full run (2.64 GB safetensors + w2v-bert download)

2. **Quantize** — add rules to `crispasr-quantize/main.cpp`
   - Keep embeddings/norms/biases at F16/F32
   - Quantize GPT-2 attention/FFN + DiT/WaveNet weights

3. **Reference dump** — `tools/reference_backends/confucius4_tts.py`
   - Per-stage: text_embed → condition_emb → gpt2_layer_0..23 → semantic_codes → s2a_cond → flow_step_0..24 → mel → wav

4. **C++ runtime** — `src/confucius4_tts.{h,cpp}`
   - T2S: GPT-2 with custom embedding concatenation + KV cache
   - S2A: CFM with DiT estimator
   - Vocoder: BigVGAN (HiFi-GAN variant)

5. **CLI adapter** — `examples/cli/crispasr_backend_confucius4_tts.cpp`

6. **All 12 checklist items** from docs/contributing.md

---

## Blocking questions

1. **Wav2Vec2-BERT**: Do we port this to GGUF too? It's 600M params with a complex conformer-based encoder. Could be its own backend module shared with future models that use w2v-bert, or we could compute the semantic features on-the-fly in the converter/reference-dumper and bake them as fixed conditioning (only works for predefined voices, not zero-shot).
   - **Decision needed:** For zero-shot voice cloning (the main use case), w2v-bert MUST run at inference time on the user's prompt audio. So it needs a GGUF port.
   
2. **BigVGAN**: Is the existing `core/hifigan.h` close enough, or does BigVGAN's architecture differ enough to need new code? BigVGAN uses anti-aliased multi-periodicity composition (AMP) blocks instead of standard HiFi-GAN MRF blocks.
   - **Likely answer:** New module needed, but structurally similar.

3. **External model sizes**: w2v-bert-2.0 is ~1.2 GB at F16. With Q4_K it'd be ~300 MB. CAMPPlus is tiny (~5 MB). BigVGAN v2 22kHz is ~112 MB. Total companion weight budget: ~400-500 MB Q4_K on top of the ~700 MB T2S Q4_K.

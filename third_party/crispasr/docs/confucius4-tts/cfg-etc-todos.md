# Confucius4-TTS — remaining work (handover from 2026-08-24)

Issue #377. The end-to-end pipeline runs and produces audio (not silence),
but the audio is not intelligible speech yet. This doc captures exactly
what's left, with file paths, line numbers, and code-level instructions.

---

## 1. Classifier-Free Guidance (CFG) — **highest priority**

Without CFG the mel spectrogram is blurry noise. The Python reference
(`confuciustts/flow/flow_matching.py` `solve_euler`) doubles the batch:
conditioned pass (real mu + real spks) and unconditioned pass (mu=0,
spks=0), then blends: `v = (1 + cfg) * v_cond - cfg * v_uncond`.

### What to change

**`src/confucius4_tts.cpp`, `s2a_flow_matching()` (~line 1820)**:

```
// Current (no CFG):
velocity = s2a_dit_forward(ctx, z, T, mel_dim, cond, cond_ref, t);

// Needed (CFG, cfg_rate from params, default 0.7):
if (cfg_rate > 0) {
    v_cond   = s2a_dit_forward(ctx, z, T, mel_dim, cond, cond_ref, t);
    v_uncond = s2a_dit_forward(ctx, z, T, mel_dim, cond_zeros, cond_ref_zeros, t);
    //                                              ^^^^^^^^^^  ^^^^^^^^^^^^^^^
    //                                   zero conditioning for uncond pass
    velocity = (1 + cfg_rate) * v_cond - cfg_rate * v_uncond;
}
```

The `cond_zeros` is `(T_mel * cond_dim)` of zeros (semantic conditioning
zeroed). `cond_ref_zeros` is `(T_mel * mel_dim)` of zeros (reference mel
zeroed). The speaker embedding (spks) should also be zeroed for the uncond
pass — pass a flag to `s2a_input_embed_cpu` or zero the spk portion.

**Performance note**: CFG doubles ODE compute. The DiT graph is reused
across steps (same T), so no rebuild needed. The f5-tts implementation in
`src/f5_tts.cpp:1887` has a batched CFG variant (`f5_dit_run_batched` with
B=2) that runs both passes in one graph dispatch — this is the pattern to
follow for GPU. For CPU-first, two sequential calls are fine.

**Params**: `confucius4_tts_params.cfg_rate` already exists (default 0.0,
should default to 0.7). Set in `confucius4_tts_default_params()`.

---

## 2. Increase default ODE steps to 25

In `confucius4_tts_default_params()` (~line 235), `ode_steps = 0` maps to
25 in `s2a_flow_matching`. The Kaggle test kernel currently forces
`--tts-steps 10` for speed. Once CFG is working, bump to 25 for the
quality test and set a longer timeout.

---

## 3. Bake BPE vocab into the T2S GGUF

The converter at `tools/kaggle/confucius4-tts-convert/confucius4-tts-convert.py`
now has BPE baking code (lines ~205-225) but the existing Q4_K GGUF on HF
was built before this addition. Re-run the converter kernel to produce new
GGUFs with the vocab baked:

```
python -m kaggle kernels push -p tools/kaggle/confucius4-tts-convert
```

This will upload new T2S GGUFs (F16/Q8/Q4K) with `tokenizer.ggml.tokens`
and `tokenizer.ggml.merges`. The runtime already tries to load them
(`load_t2s` ~line 274). Once baked, the `CRISPASR_CONFUCIUS4_TEXT_IDS`
env var is no longer needed — the test kernel can drop that path.

---

## 4. Speaker conditioning (CAMPPlus + Wav2Vec2-BERT)

The T2S model has a speaker encoder (ECAPA-TDNN variant) that conditions
the GPT-2 prefix embedding on a reference audio's voice. Without it, the
`condition_emb` in the prefix is zero, so the T2S generates "voiceless"
semantic codes.

### What exists

- `src/chatterbox_campplus.h` / `.cpp` — complete CAMPPlus implementation,
  produces 192-d speaker embeddings from 16kHz mono PCM via 80-bin Kaldi
  fbank. **Exact match** for `confucius4_tts_params.spk_embed_dim = 192`.
  
- `confucius4_tts_set_speaker()` API already exists (~line 648) and
  accepts `(semantic_features, n_frames, style_embedding)`.

### What's needed

1. **Wav2Vec2-BERT** extraction (1024-d per-frame features, layer 17) for
   the T2S speaker encoder. This is a 600M param model — too large to
   bake into the T2S GGUF. Options:
   a. Port as a separate GGUF companion (new module, significant work)
   b. Pre-compute and cache per reference audio (good for fixed voices)
   c. Skip w2v-bert, just use CAMPPlus for style (S2A only, not T2S)

2. **T2S speaker encoder forward**: the weights are loaded (`text_projector`,
   `semantic_embedding` etc.) but the ECAPA-TDNN forward (in the T2S model)
   isn't wired. It takes w2v-bert features → 1280-d condition embedding.

3. **S2A style embedding**: the DiT `InputEmbedding` concatenates a 192-d
   speaker embedding (from CAMPPlus) as `spks`. Currently zero. Wire
   `ctx->speaker_style_embedding` into `s2a_input_embed_cpu()` (it's
   already structured to handle it when `ctx->has_speaker` is true).

### Quick-win shortcut

For a first test with voice, use a **pre-computed** speaker embedding from
the Python pipeline. Run the Python inference with a reference audio, dump
the `style_embedding` (192-d) and `semantic_features` tensors, load them
via `confucius4_tts_set_speaker()`, and test. This bypasses the w2v-bert
porting entirely.

---

## 5. BigVGAN vocoder performance

The BigVGAN runs at 128s for 30s of audio on Kaggle CPU. Two optimizations:

1. **GPU offload**: The `indextts_voc` already supports GPU via
   `use_gpu=true`. Pass `ctx->params.use_gpu` when calling
   `indextts_voc_init()` (currently hardcoded in
   `confucius4_tts_set_vocoder_path`, ~line 643).

2. **Anti-alias bypass**: Set `INDEXTTS_VOCODER_RAW=1` for the faster
   (lower quality) path that skips AA filtering. Good for development.

---

## 6. Input embedding CPU cost

`s2a_input_embed_cpu()` and `s2a_timestep_embed_cpu()` are called per ODE
step on CPU. For 25 steps × CFG (50 calls), the `mu_projection` and
`proj` matmuls become significant. Consider moving these into the ggml
graph as additional inputs (pre-compute once and pass as graph tensors).
The f5-tts pattern at `src/f5_tts.cpp:1405` does this.

---

## 7. Length regulator

The current conditioning pipeline uses linear interpolation for length
regulation (`s2a_build_conditioning`, ~line 1800). The Python model has
an `InterpolateRegulator` with learned conv upsampling (ratios [1,1,1,1]).
With all ratios=1 and no learned weights applied, linear interpolation is
a reasonable approximation. Not a priority.

---

## File reference

| File | What it does |
|------|-------------|
| `src/confucius4_tts.cpp` | Main implementation (~1950 lines) |
| `src/confucius4_tts.h` | Public C API |
| `examples/cli/crispasr_backend_confucius4_tts.cpp` | CLI adapter |
| `tools/kaggle/confucius4-tts-test/` | End-to-end test kernel (Kaggle) |
| `tools/kaggle/confucius4-tts-convert/` | T2S+S2A GGUF converter kernel |
| `tools/kaggle/confucius4-tts-convert-bigvgan/` | BigVGAN 22kHz GGUF converter |
| `docs/confucius4-tts/PLAN.md` | Architecture trace |
| `src/indextts_voc.h` / `.cpp` | BigVGAN vocoder (reused) |
| `src/core/bpe.h` | BPE tokenizer (reused) |
| `src/chatterbox_campplus.h` / `.cpp` | CAMPPlus speaker encoder (reusable) |
| `src/core/hifigan.h` | HiFi-GAN vocoder (alternative to BigVGAN) |

## GGUF files on HuggingFace (`cstr/confucius4-tts-GGUF`)

| File | Size | Content |
|------|------|---------|
| `confucius4-tts-t2s-q4_k.gguf` | 374 MB | T2S GPT-2 (24L, Q4_K) |
| `confucius4-tts-s2a-q4_k.gguf` | 121 MB | S2A DiT+WaveNet (Q4_K) |
| `confucius4-tts-bigvgan-22k-f16.gguf` | 214 MB | BigVGAN v2 22kHz vocoder (F16) |
| `confucius4-tts-bigvgan-22k-q8_0.gguf` | ~130 MB | BigVGAN v2 22kHz (Q8_0) |
| `confucius4-tts-t2s-f16.gguf` | ~2.6 GB | T2S F16 (for re-quantization) |
| `confucius4-tts-s2a-f16.gguf` | ~400 MB | S2A F16 |

## Tensor name conventions

- T2S: `transformer.h.{i}.attn.c_attn.weight`, `semantic_embedding.weight`, etc.
- S2A: **all prefixed with `decoder.`** — `decoder.estimator.transformer_blocks.{i}.attention.wqkv.weight`, `decoder.estimator.wavenet.in_layers.{i}.conv.weight` (fused from weight_norm at load time)
- BigVGAN: shortened names — `conv_pre.weight`, `ups.{i}.0.weight`, `resb.{n}.act.{m}.act.alpha`

## Key architectural details

- DiT ggml graph: 857 nodes, includes 13L transformer (RoPE, AdaLN, SwiGLU, U-Net skips) + 8L WaveNet (gated dilated conv, weight_norm folded at load time) + final_layer + conv2
- WaveNet uses time-first layout `(T, C)` for `ggml_conv_1d`; rest of graph uses channel-first `(C, T)` — transpose before/after WaveNet
- ODE operates in 80-dim mel space, velocity is `(T_mel, 80)`
- Noise initialization: `z ~ N(0, temperature)` where temperature defaults to 0.8
- Cosine time schedule: `t = 1 - cos(i/n_steps * π/2)`
- The `lm_hidden` tensor is a second output of the GPT-2 graph (named `"lm_hidden"`, retrieved via `ggml_graph_get_tensor`)

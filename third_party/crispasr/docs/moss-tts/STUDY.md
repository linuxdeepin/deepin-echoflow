# MOSS-TTS-v1.5 port — Phase 0 STUDY notes (issue #249)

Source of truth for the port. Written before any C++ (HARD RULE #1). Derived from
line-by-line reading of `github.com/pwilkin/openmoss` (a working C++/ggml port that
hosts the Qwen3 backbone via **libllama**) + the HF `config.json` of
`OpenMOSS-Team/MOSS-TTS-v1.5`.

**The single biggest adaptation:** openmoss hosts the Qwen3-8B backbone with libllama
and feeds it `batch.embd` (precomputed input embeddings), reading back per-token hidden
states via `llama_get_embeddings_ith`. CrispASR forbids libllama and reimplements Qwen3
in-house (`src/moss_audio.cpp`). So we graft openmoss's **aux graphs (embed sum + 33
heads), delay state machine, and transformer codec** onto CrispASR's own Qwen3 runtime.
Everything except the backbone hosting ports near-verbatim.

---

## 1. Model identity (config.json, verified 2026-07-12)

- `model_type = "moss_tts_delay"`, arch `MossTTSDelayModel`.
- **Backbone (Qwen3-8B)** from `language_config`:
  - num_hidden_layers **36**, hidden_size **4096**
  - num_attention_heads **32**, num_key_value_heads **8** (GQA 4:1), head_dim **128**
    (note 32×128 = 4096 for Q; 8×128 = 1024 for KV)
  - intermediate_size **12288**, hidden_act **silu** (SwiGLU)
  - rms_norm_eps **1e-6**, rope_theta **1_000_000**, max_position_embeddings 40960
  - vocab_size **155648**, all layers `full_attention` (no sliding window)
  - Qwen3 specifics to honor: **per-head QK-RMSNorm** (q_norm/k_norm), RoPE NEOX/neox-style.
    Confirm against CrispASR's existing Qwen3 loader (moss_audio.cpp).
  - eos = im_end = 151645, bos = pad = 151643.
- **Audio side:** n_vq **32**, audio_vocab_size **1024**, audio_pad_code **1024**
  (so full audio vocab = 1025), sampling_rate **24000**, downsample_rate **1920**
  (→ frame_rate 12.5 Hz).
- **Special token ids:** audio_start 151652, audio_end 151653, audio_user_slot 151654,
  audio_assistant_gen_slot 151656, audio_assistant_delay_slot 151662, im_start 151644,
  im_end 151645, pad 151643.

## 2. Weight inventory & GGUF layout (from convert_hf_to_gguf.py)

HF source tensor names (safetensors of MOSS-TTS-v1.5):
- Backbone: `language_model.*` (embed_tokens, layers.N.*, norm) + `lm_heads.0.weight`
  (= the text lm_head / `output.weight`).
- **32 audio embed tables:** `emb_ext.{0..31}.weight`, each (audio_vocab+1=1025, hidden=4096).
- **32 audio heads:** `lm_heads.{1..32}.weight` → audio_head.{0..31} (lm_heads.0 is text).
  So there are **33 heads total** = 1 text + 32 audio.
- Codec (separate repo `MOSS-Audio-Tokenizer`): prefixed `moss.codec.` after renames.

openmoss GGUF split: **backbone.gguf** (vanilla Qwen3, libllama-loadable) +
**backbone.extras.gguf** sidecar (`moss.*` tensors + `moss.*` KV). For CrispASR we don't
need libllama-loadability, so the converter (`models/convert-moss-tts-to-gguf.py`) should
emit the backbone in **CrispASR's Qwen3 GGUF naming** (TBD from moss_audio converter —
see §7) and carry the audio/codec tensors + KV in the same or a sidecar file, matching
whatever CrispASR's loader expects.

`moss.*` KV keys: moss.n_vq, moss.audio_vocab_size, moss.audio_pad_code,
moss.sampling_rate, moss.downsample_rate, moss.frame_rate, moss.token.{audio_start,
audio_end,audio_user_slot,audio_gen_slot,audio_delay_slot,im_start,im_end,pad},
moss.codec.present.

Codec tensor-name shortening (fit 64-byte GGUF limit), applied in order:
`parametrizations.weight.original0→wp0`, `original1→wp1`, `transformer.layers→tr.l`,
`encoder→enc`, `decoder→dec`, `self_attn→attn`, `in_projs→inp`, `out_projs→outp`,
`quantizers→q`, `input_proj/in_proj→iproj`, `output_proj/out_proj→oproj`.

## 3. Input embedding & heads (openmoss model.cpp) — port to CrispASR aux graphs

- **compute_input_embeddings(grid (S,1+n_vq) int32) → (S, hidden) f32:**
  `text_emb = get_rows(text_embed, col0)` cast f32, then `+ Σ_{i<32} get_rows(audio_embed_i,
  col_{i+1})` cast f32. Trivial ggml graph. Feeds the backbone as input embeddings.
- **compute_audio_logits(hidden (4096,)) → (n_vq, 1025) f32:** for each of 32 heads,
  `mul_mat(head_i, h)` → (1025,1); concat along ne[1] → (1025, 32); read back row-major as
  (32, 1025). No bias. The text logits come from the backbone's own lm_head (lm_heads.0).

## 4. Prompt format (openmoss pipeline.cpp) — EXACT

String (token literals decoded from their ids), tokenized with add_special=false:
```
<im_start>user\n<user_inst>\n- Reference(s):\n{None | [S1]:\n<ref_block>\n}- Instruction:\n{instruction|None}\n- Tokens:\n{tokens|None}\n- Quality:\n{quality|None}\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\n{language|None}\n- Text:\n{text}\n</user_inst><im_end>\n<im_start>assistant\n<audio_start>
```
Grid: (S, 1+n_vq) int32; col0 = text ids, cols 1..32 = audio_pad_code (except reference-
audio rows for voice cloning, which splice delay-shifted ref codes — follow-up).

## 5. Delay state machine (openmoss delay.cpp) — Phase 3 template. GOTCHAS:

Per step the model emits (1 + n_vq) ids. State: audio_length, delayed_length
(−1 sentinel until first delay_slot, then ticks 0..n_vq), is_audio, is_stopping.

Column 0 (text):
- delayed_length in [0,n_vq) → emit **delay_slot** (no sampling).
- delayed_length == n_vq → emit **audio_end**, close segment.
- max_audio_frames cap hit (audio & sentinel) → force delay_slot.
- else sample text from masked text vocab (mask rules differ inside/outside audio;
  mask delay_slot at step 0; mask im_end for first n_vq steps; min_audio_frames floor
  forbids delay_slot early). audio_start → is_audio=true; im_end → is_stopping.

Columns 1..n_vq (audio head i): real iff `audio_length > i` AND
`(delayed_length<0 ? true : i >= delayed_length)`; else audio_pad_code. **Sentinel branch
is load-bearing** (bug #1: INT64_MAX sentinel makes `delayed-1 < i` always false → skips
every codebook during warm-up). Mask the pad code before sampling; rep-penalty over
**unique** history tokens per column, not per-occurrence (bug #4).

State update: audio_length++ on {audio_start, gen_slot, delay_slot}; reset 0 on audio_end.
delayed_length: −1→0 on first delay_slot; else ++ and wrap to −1 once > n_vq.
**Off-by-one is deliberate and paired with extract's `T - n_vq`. Do NOT fix one alone.**

extract_audio_codes: find the **last** audio_start (backward search — bug #2: matching
gen_slot too lands on the last gen_slot row → T≈0); segment = [start, audio_end/EOH);
T = end-start; require T > n_vq; **un-shift codebook cb by cb steps**:
`out[cb, t] = history[start + t + cb][1+cb]`, T_audio = T − n_vq.

Sampling: repetition-penalty(audio) → /temp → top-k → softmax → top-p → multinomial(inverse
CDF). Defaults: text temp 1.5/top_p 1.0/top_k 50; audio temp 1.7/top_p 0.8/top_k 25/rep 1.0.

**Python cross-check (modeling_moss_tts.py, verified 2026-07-12):** sentinel is
`torch.iinfo(int64).max`; `pre_audio_mask = audio_length > arange(n_vq)`;
`post_audio_mask = arange(n_vq) > delayed_length-1`; state updates MAX→0→+1→(wrap to MAX
when >n_vq) — all match openmoss. The warm-up sentinel case (where `arange > MAX-1` would be
all-false) is force-`post=true` in the reference; openmoss's explicit sentinel branch is the
faithful equivalent (its bug #1). **openmoss is end-to-end validated (envelope corr 1.000),
so it is the primary reference; correctness is nonetheless gated on a byte-identical Python
parity dump at Phase 3** (rep-penalty unique-vs-per-occurrence lives in `inference_utils`,
confirm at parity time — openmoss says the reference penalizes `torch.unique`).

## 6. Transformer codec (openmoss codec.cpp) — Phase 4, the hard sub-project

RVQ: 32 codebooks, size 1024, **codebook_dim 8**, rvq_dim 512, out_dim 768.
Frame_rate 12.5 Hz, hop 1920 = 240×2×2×2.

**Decode (codes (32,T) → waveform (1920·T,)):**
1. Per quantizer i: `z = get_rows(codebook_i, codes_i)` (8,T) f32 →
   `mul_mat(q_oproj_w_i, z)+b` = Conv1d(8→512, k=1) → (512,T). **Σ over 32** → (512,T).
2. `mul_mat(quant_oproj_w, sum)+b` = Conv1d(512→768) → (768,T).
3. **4 ProjectedTransformer stages** (dec.0/2/4/6), each + patch upsample:

   | stage | gguf | in | d | heads | dff | layers | out | patch | ctx(keys) |
   |-------|------|----|----|-------|-----|--------|-----|-------|-----------|
   | dec.0 | 0 | 768 | 1280 | 20 | 5120 | 32 | 1280 | 2 | 125 |
   | dec.2 | 2 | 640 | 768  | 12 | 3072 | 12 | 768  | 2 | 250 |
   | dec.4 | 4 | 384 | 768  | 12 | 3072 | 12 | 768  | 2 | 500 |
   | dec.6 | 6 | 384 | 768  | 12 | 3072 | 12 | 240  | 240 | 1000 |

   iproj present iff d≠in; oproj present iff d≠out (else Identity, tensor absent).
4. After dec.6+patch240: channel=1, time=1920·T → reshape to waveform.

**Transformer layer (pre-LN):**
- `x += layer_scale_1 ⊙ attn(LN1(x))`; `x += layer_scale_2 ⊙ ffn(LN2(x))`.
- LN = **LayerNorm** (weight+bias, eps 1e-5), NOT RMSNorm.
- attn: fused QKV `mul_mat(attn.inp.0.weight)` → split Q/K/V (head_dim,n_heads,T);
  **RoPE GGML_ROPE_TYPE_NORMAL** (adjacent-pair, `q.view(D//2,2)`), base **10000**;
  flash_attn_ext (F16 K/V, F32 mask, prec F32) with **sliding-window causal mask**;
  out proj `attn.outp.0.weight`.
- ffn: `linear2(gelu(linear1(x)))`, **ggml_gelu (tanh approx)**.
- **Sliding-window mask is part of the model, not an optimization** (bug #3): window =
  frame_rate×10s → 125/250/500/1000 keys per stage. `mask[q,kv]=0 iff kv<=q && q-kv<ctx`,
  else −inf. Plain lower-triangular degrades audio after 10 s.

**Weight-norm reconstruction** (quantizer projections only): stored as wp0 (magnitude
(out,1,1)) + wp1 (direction (out,in,1)); materialize `w[o,i]=wp0[o]·wp1[o,i]/‖wp1[o,:]‖`
on host at init, upload as f16. Bias passes through.

Layer weight names per stage: `dec.{idx}.tr.l.{li}.{norm1,norm2}.{weight,bias}`,
`.attn.inp.0.weight`, `.attn.outp.0.weight`, `.linear1.weight`, `.linear2.weight`,
`.layer_scale_1.scale`, `.layer_scale_2.scale`; stage `.iproj.weight`/`.oproj.weight`.
Codebooks `moss.codec.quantizer.q.{i}.codebook.weight` (1024,8);
`quantizer.q.{i}.oproj.{wp0,wp1,bias}`; `quantizer.oproj.{wp0,wp1,bias}`.

**Encode (voice cloning, FOLLOW-UP):** mirror — patch_downsample(240) → 4 enc stages
(enc.1/3/5/7) → quant iproj → 32-step residual LFQ (L2-norm z, cosine-sim argmax vs
normalized codebook, subtract oproj(codebook[idx])). enc ctx windows 1000/500/250/125.

## 7. AR loop (openmoss pipeline.cpp)

Prefill: compute_input_embeddings(grid) → backbone decode (embd input), output last.
Loop until stop or max_new_tokens (default 4096): text_logits + hidden from backbone last
pos → compute_audio_logits(hidden) → DelayState.step → embed next (1,1+n_vq) row →
backbone decode 1 token. Then extract_audio_codes → codec_decode → WAV @ 24 kHz mono.

## 8. CrispASR seam — RESOLVED (survey 2026-07-12)

**Backbone runtime.** CrispASR's in-house Qwen3 accepts precomputed input embeddings
first-class: graph input tensor `"inputs_embeds"` (`moss_audio.cpp:1227`,
`qwen3_asr.cpp:1172`). Persistent KV + single-token decode:
`moss_audio_run_llm_kv(ctx, const float* inputs_embeds, n_tokens, n_past, ...)`
(`moss_audio.cpp:1602`) / `qwen3_asr_run_llm_kv` (`:1886`). QK-norm, GQA 4:1, RoPE all live
inside `core_attn::kv_self_attn` (`core/attention.h:665`); FFN `core_ffn::swiglu`
(`ffn.h:39`). **RoPE = GGML_ROPE_TYPE_NEOX** for the Qwen3 backbone, rope_theta 1e6
(`moss_audio.cpp:1261`, `:1255`) — DISTINCT from the codec's NORMAL/adjacent-pair base 1e4.
`moss_audio` is itself a 36-layer Qwen3 (same size as our backbone) → clone its graph.

**Hidden-state readout.** ASR backbones expose only `"logits"`. I need BOTH text logits AND
the per-token pre-lm-head hidden (for the 32 audio heads). `qwen3_tts.cpp` already does this:
names the final hidden `"hidden_last"` + `ggml_set_output` (`:1397`) and returns it via
`run_talker_kv(..., float** out_hidden_d)` (`:1518`). Copy that pattern into moss_tts.

**Template backend.** `src/qwen3_tts.cpp` is the closest analog (Qwen3 + RVQ + inline
transformer codec). AR loop `qwen3_tts_generate_codes_ar` (`:6771`); transformer codec
`build_graph_codec_decode` (`:3952`), struct `g3t_codec` (`:613`), sliding-window layer
`g3t_codec_xfmr_layer` (`:564`). Top entry `qwen3_tts_synthesize` (`:7229`).

**GGUF naming — decision: Convention A (moss_audio style).** Backbone tensors
`llm.blk.{N}.{attn.q,attn.k,attn.v,attn.o,attn.q_norm,attn.k_norm,attn_norm,ffn.gate,
ffn.up,ffn.down,ffn_norm}.weight` + `llm.embed.weight`, `llm.final_norm.weight`,
`llm.lm_head.weight` (`moss_audio.cpp:422-442`). Converter `convert-moss-audio-to-gguf.py`
`map_tensor_name` (`:59`) is the template. **GGUF arch string = `"moss-tts"`** (must match
the detect branches). Audio/codec tensors keep the openmoss `moss.audio_embed.{i}` /
`moss.audio_head.{i}` / `moss.codec.*` names + `moss.*` KV. Convention B (qwen3_asr,
llama-canonical `blk.N.attn_q.weight`) is the alternative if I clone qwen3_asr instead.

**No transformer codec in `core/`** — port qwen3_tts's inline codec or openmoss `codec.cpp`.
`core_rvq::encode_euclidean` (`core/rvq.h:44`) reusable for the voice-cloning encode side.

**12-point wiring sites (all pinned):**
- Detect: `crispasr_backend.cpp:489` — pass-1 filename substring (add
  `contains_ci("moss")&&contains_ci("tts")` ~`:575`), pass-2 `general.architecture` tag
  (add `a=="moss-tts"` branch ~`:704`).
- Session ABI inline synth: `crispasr_c_api.cpp` — `crispasr_session_synthesize_raw_impl`
  (`:7119`), add a `#ifdef CA_HAVE_MOSS_TTS if (s->moss_tts_ctx){...}` block modeled on
  qwen3-tts (`:7169-7197`); init site ~`:2466`; arch→name branch ~`:1331`.
- Registry: `crispasr_model_registry.cpp:553` — entry with the codec GGUF as
  `companion_filename`.
- Quantize: `crispasr-quantize/main.cpp:337` — add `is_moss_tts` keep-list: keep RVQ
  codebooks, 32 audio embeds/heads, all `moss.codec.*` at F16; quantize only backbone
  `llm.blk.*`.

## 9. Build seam confirmed
Fresh worktree `.claude/worktrees/moss-tts-249` configures cleanly (Ninja, Metal+Opus+MP3,
ggml submodule initialized). ccache via `$HOME/.ccache`.

## Validation (HARD RULE #3)
Greedy code parity byte-identical vs Python ref; codec per-stage cos ≥ 0.999; decoded
round-trip (synthesize → ASR → recognizable text) on **F16 AND Q4_K**. 8B backbone won't
fit the 8 GB VPS and is tight on the 16 GB Mac with the 1.6 B codec → end-to-end on
**Kaggle** (P100/T4). CUDA get_rows needs contiguous index (dev-guide §232).

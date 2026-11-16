# GigaAM-v3 port — PLAN

Port of [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3) — Russian
ASR, 16-layer rotary Conformer (220 M) + CTC or RNN-T head.

## NOW — active work

**DONE.** Port landed and validated; nothing in flight.

- Runtime, converter, reference dumper, diff branch, CLI adapter, session C ABI,
  registry, quantizer rules, tests, docs — all on `feat/gigaam-v3`.
- GGUFs published to
  [`cstr/gigaam-v3-GGUF`](https://huggingface.co/cstr/gigaam-v3-GGUF)
  (4 variants × f16 / q8_0 / q4_k).
- Per-stage reference archives for `crispasr-diff` are in
  `cstr/crispasr-regression-fixtures` under `gigaam-<variant>/gigaam_example/`.

## Acceptance results

`crispasr-diff gigaam <model> <ref> example.wav` on GigaAM's own 11.29 s
`example.wav`, against a PyTorch reference dumped from the upstream
`modeling_gigaam.py` (`tools/reference_backends/gigaam.py`):

| variant | quant | mel | encoder cos_min | head | transcript vs PyTorch |
|---|---|---|---|---|---|
| all four | f16 | 1.000000 | **1.000000** | 1.000000 | **byte-identical** |
| all four | q8_0 | 1.000000 | 0.9974 – 0.9988 | 1.000000 | **byte-identical** |
| `ctc`, `rnnt` | q4_k | 1.000000 | 0.95 – 0.99 | 1.000000 | **byte-identical** |
| `e2e_ctc` | q4_k | 1.000000 | 0.982 | 1.000000 | one spurious trailing `,` |
| `e2e_rnnt` | q4_k | 1.000000 | 0.987 | 1.000000 | content identical; 4 words lose their capital |

The head stages (`ctc_log_probs`, `joint_enc_proj`, `pred_initial`,
`joint_logits_t0`) are unaffected by the encoder quant because the quantizer
keeps `joint.*` / `decoder.*` / `head.ctc.*` at source precision.

**Registry default is q8_0**, not q4_k: 249 MB is small enough that the exact
transcript is worth the extra 95 MB.

Perf on M1 Metal, 11.29 s clip: e2e_rnnt q8_0 **43.4× realtime**, ctc q8_0
**35.0× realtime** (both after the `sole_language()` fix below; before it, the
whisper-tiny LID pass halved the e2e number to 21.3×).

## What the blueprint actually does (the three easy-to-miss details)

Read off `modeling_gigaam.py`, not inferred from Conformer convention:

1. **RoPE lands on the block INPUT, before the Q/K/V projections.**
   `RotaryPositionMultiHeadAttention.forward` gets `x, x, x`, rotates
   `query`/`key` while they are still the raw hidden state viewed as
   `(T, B, n_heads, head_dim)`, and only then calls `forward_qkv`. So
   `Q = Wq·RoPE(x)`, `K = Wk·RoPE(x)`, `V = Wv·x` — **V is projected from the
   unrotated input.**
2. **The rotary base is 5000, not 10000.** `RotaryPositionalEmbedding` is
   constructed `(d_model // n_heads, pos_emb_max_len)` against
   `PositionalEncoding.__init__(self, dim, base)`, so `pos_emb_max_len` lands
   in the `base` slot. The converter writes it out as `gigaam.rope_base`
   instead of letting the runtime re-derive it.
3. **`conv.batch_norm` is a LayerNorm.** `conv_norm_type='layer_norm'` makes
   the conv module's `batch_norm` submodule an `nn.LayerNorm`; its weight/bias
   are LN affine parameters and there are no running stats to fold.

Getting any of these wrong is silent — the shapes all still line up.

## Two integration bugs that only showed up end-to-end

Per-stage parity was clean on the first run; both of these were invisible to
the diff harness (HARD RULE #3b — the harness ends at the logits):

- **FireRedPunc was injecting full-width CJK punctuation into Russian.** The
  charwise `ctc` / `rnnt` revisions emit unpunctuated lowercase Cyrillic, so
  `crispasr_should_auto_enable_punctuation` fired — and the auto-enabled
  restorer is a Chinese/English model. Output read
  `надеждой， сладкой ... зеленый。`. Fix: declare `CAP_PUNCTUATION_NATIVE` for
  every revision (for the `e2e_*` ones because they already punctuate, for the
  charwise ones because the available restorer is worse than nothing). An
  explicit `--punc-model` still applies — the cap only gates the *auto* path.
- **A whisper-tiny LID pass was running on a Russian-only model.** `-l auto` is
  the default and gigaam has no `CAP_LANGUAGE_DETECT`, so the CLI downloaded and
  ran `ggml-tiny.bin` to "detect" a language the model cannot change (#227).
  Fix: `sole_language() == "ru"`.

## Not done / follow-ups

- **No `tests/regression/manifest.json` entry.** The manifest pins the fixtures
  repo to one revision SHA, and adding an entry means a coordinated re-bake
  (`tools/kaggle-regression.py` MODE=rebake) that re-pins every other backend.
  That is its own change, not a rider on the port. The ref archives are already
  uploaded, so the entry is a small follow-up once a re-bake happens anyway.
- **Flash attention is opt-in** (`CRISPASR_GIGAAM_FLASH=1`). The manual
  QK^T + `soft_max_ext` + V path is what the per-stage diff was validated on;
  flash accumulates differently, so it needs its own A/B (both arms
  back-to-back under identical load) before the default can flip.
- **RNN-T decode runs on CPU cblas**, not `core_rnnt_ggml::Decoder` — that
  helper hardcodes a 2-layer LSTM and GigaAM's predictor has one. Generalising
  it is worthwhile only if a decode-side profile says the LSTM matters; at
  43× realtime it does not yet.
- **Long audio** relies on the CLI's VAD/chunking. Upstream's own recipe
  segments with pyannote at 22 s max / 30 s hard; the model has a ~25 s
  practical window (full attention, O(T²)). Worth checking whether a
  gigaam-specific `vad_slice_cap_seconds()` beats the flat default on a long
  Russian clip.
- **`ssl` revision not converted** — it is the bare HuBERT-CTC encoder with no
  head, so it produces no transcript.

## Env gates

| var | effect |
|---|---|
| `CRISPASR_GIGAAM_BENCH=1` | per-stage timings (mel / encoder / decode) |
| `CRISPASR_GIGAAM_DEBUG=1` | encoder output min/max/mean |
| `CRISPASR_GIGAAM_FLASH=1` | `ggml_flash_attn_ext` in the encoder (opt-in) |
| `CRISPASR_GIGAAM_FORCE_SCALAR=1` | scalar LSTM/joint loops instead of cblas |
| `CRISPASR_GIGAAM_QUANT_ALL=1` | let `crispasr-quantize` quantize the heads too |

## Reproducing

```bash
python models/convert-gigaam-to-gguf.py --model ai-sage/GigaAM-v3 \
    --revision e2e_rnnt --output gigaam-v3-e2e-rnnt-f16.gguf
./build/bin/crispasr-quantize gigaam-v3-e2e-rnnt-f16.gguf \
    gigaam-v3-e2e-rnnt-q8_0.gguf q8_0

python tools/dump_reference.py --backend gigaam \
    --model-dir <snapshot>/e2e_rnnt --audio example.wav \
    --output gigaam-v3-e2e-rnnt-ref.gguf

./build/bin/crispasr-diff gigaam gigaam-v3-e2e-rnnt-q8_0.gguf \
    gigaam-v3-e2e-rnnt-ref.gguf example.wav
```

`example.wav` is
`https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav`.

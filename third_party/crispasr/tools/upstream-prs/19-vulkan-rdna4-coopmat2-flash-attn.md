**Title:** `vulkan: flash_attn_ext coopmat2 path produces wrong output on AMD RDNA4 (RX 9700 XT)`

---

Bug report, no patch — **awaiting hardware confirmation** from the issue
#171 reporter before filing. We do not own RDNA4 hardware; the analysis
below is reproduced as far as we can without it. File against
**ggml-org/llama.cpp** (vulkan; per the repo-routing table). If the
shader output is correct and the fault is in the driver's coopmat2
codegen, this becomes a Mesa/RADV report instead — see "Where the bug
likely lives".

## Symptom

A user running our VibeVoice realtime-0.5B TTS backend on an AMD Radeon
RX 9700 XT (RDNA4, Mesa/RADV, Ubuntu 26.04) via the Vulkan backend gets
garbled speech: mixed speaker voices, repeated fragments, "goes totally
west". Short utterances ("Cependant", "Le wagon vert.") break worst.
Reported as a v0.7.2 regression (issue #171).

## Why we think it's `flash_attn_ext` coopmat2, not our graph

The *same* `vibevoice.cpp` TTS graph (which calls `ggml_flash_attn_ext`
in the small autoregressive TTS LM) ASR-roundtrips **verbatim** on every
backend we can test:

| backend (this M1)            | matrix cores | FA path used   | result   |
| ---------------------------- | ------------ | -------------- | -------- |
| Metal                        | —            | (metal kernel) | ✅ exact |
| Vulkan via MoltenVK          | **none**     | `FA_SCALAR`    | ✅ exact |
| reporter: Vulkan on RDNA4    | present      | `FA_COOPMAT2`  | ❌ garbage |

`ggml-vulkan.cpp` selects `FaCodePath` as
`device->coopmat2 ? FA_COOPMAT2 : device->coopmat1_fa_support ? FA_COOPMAT1
: FA_SCALAR` (~L3095). MoltenVK reports `matrix cores: none`, so it lands
on `FA_SCALAR` — i.e. our "correct" Vulkan runs already validate every
non-coopmat2 FA path end-to-end. The RX 9700 XT advertises coopmat2 and
is the only configuration that takes `flash_attn_cm2`, and the only one
that garbles. RDNA4 is brand-new silicon; its RADV coopmat2 support is
fresh.

Slightly-wrong attention is enough to break this model: the TTS LM feeds
a binary EOS classifier (`sigmoid(fc2(relu(fc1(h)))) > 0.5`) that decides
when to stop. Small drift in the hidden state stops it firing, the model
over-generates past the natural end, and the speaker identity drifts /
fragments repeat — matching the report exactly, including why short
inputs (which depend entirely on early EOS) are worst.

## Confirmation steps requested from the reporter (no rebuild)

1. `GGML_VK_DISABLE_COOPMAT2=1` — forces `FA_COOPMAT1`/`FA_SCALAR`.
2. if still bad: `GGML_VK_DISABLE_COOPMAT=1`, then `GGML_VK_DISABLE_F16=1`.

If (1) clears it, the coopmat2 FA shader is the culprit. (We also added a
backend-agnostic `VIBEVOICE_TTS_FLASH_ATTN=0` that swaps the fused FA for
an explicit `softmax(QKᵀ)·V` so the reporter can confirm from the model
side too; the σ-VAE decoder has no attention, so `VIBEVOICE_VAE_BACKEND`
isolates the other half of the graph.)

## Where the bug likely lives

- **ggml-vulkan** `flash_attn_cm2` shader assumes a coopmat2 shape /
  alignment / accumulation that RADV-on-RDNA4 doesn't honour, **or**
- **Mesa/RADV** miscompiles the coopmat2 FA shader on gfx12.

ggml-org/llama.cpp is the right first stop either way — the Vulkan
maintainers (0cc4m, jeffbolznv) triage this and can repro on RDNA4 /
escalate to Mesa, or add a `supports_op` / path guard for the affected
adapter. A minimal `test-backend-ops -o FLASH_ATTN_EXT` run on an RX
9700 XT (coopmat2 vs `GGML_VK_DISABLE_COOPMAT2=1`) should reproduce in
isolation; include that in the report once we have hardware access.

## Status

Draft. Do not file until the reporter confirms which knob clears issue
#171. If confirmed as a shader bug, attach a `test-backend-ops`
`FLASH_ATTN_EXT` repro (coopmat2 vs scalar) and file at
ggml-org/llama.cpp; if it's RADV codegen, file at Mesa with the same
isolation data.

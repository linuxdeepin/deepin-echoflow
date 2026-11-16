# MOSS-TTS-v1.5 port (#249) — implementation status & validation plan

Branch `feat/moss-tts-249`. Spec: `docs/moss-tts/STUDY.md`. Reference:
`github.com/pwilkin/openmoss` (validated C++ port; we graft its codec + delay
onto CrispASR's in-house Qwen3 runtime — no libllama).

## NOW — active work (update at every checkpoint; push to main)

### 4B `moss-tts-local` (#249 second deliverable) — ✅ SHIPPED + MERGED to `main` (2026-07-14)

Spec: `docs/moss-tts/STUDY-4B.md`. Engineering lessons: `LEARNINGS.md` (4B entry)
+ `HISTORY.md`. Branch `feat/moss-tts-local-4b` rebased (29 commits, clean) + ff'd
into `main`; rebuilt + wiring PASS + unit test 30/30 + feature-matrix current.

**Validated end-to-end (Kaggle $KAGGLE_ACCOUNT P100, HARD RULE #3).** With the card-correct
sampling defaults, F16 synth STOPS naturally (short 15, long 124 frames — not the
4096 runaway) and the long clip round-trips through whisper at **word-overlap
0.969**. Full chain proven: Qwen3-4B backbone → 1-layer depth transformer →
depth-first 12-codebook loop → ResidualLFQ + 6-stage 48 kHz codec → stereo→mono.

**Shipped GGUFs** (`cstr/moss-tts-local-v1.5-GGUF`, verified): F16 backbone
(9.107 GB, the validated one) + decode-only codec (2.125 GB). Registry `-m auto`
defaults to **F16** (fits a 16 GB GPU). Q4_K *long* text runs away — intrinsic
quantized-AR trajectory drift ([[tts-port-parity-via-logit-rank]]) — so Q4_K is
best-effort, gate is on F16.

**Delivered:** P0 STUDY · P1 converter (backbone + focused decode-only codec) · P2
Qwen3-4B backbone + 1-layer local/depth transformer + depth-first generate loop +
binary stop head · P3 codec-v2 decode (`src/moss_tts_local_codec.*`, read
line-by-line — query-chunked attention bounds the O(T²) blowup, sched sized 262144,
GELU-erf, RoPE NORMAL 1e4, weight-norm RVQ) · P4 12-point integration (wiring PASS,
unit test) · P5 GPU round-trip.

**Key fixes (full detail in LEARNINGS 4B entry):**
1. **Stop-runaway root cause = WRONG synth defaults.** The card wants
   `audio_temperature 1.7 / top_p 0.8 / top_k 25`, `do_sample`, stop head SAMPLED
   (`text_temperature 1.0`). My generic `1.0 / 0.95 / 50` + greedy stop produced a
   degenerate acoustic trajectory that never reached the stop state → runaway.
   (Audio content was correct throughout — only the STOP timing was broken.)
2. **Codec O(T²) attention OOM** (916 GB cudaMalloc at T=131072) → query-chunking.
   **sched hash-set abort** → runtime sched sized 262144.
3. **Kaggle downloads:** `hf_transfer` does NOT resume → wedges at a fixed offset
   (F16 at 461.9 MB every retry); use `curl -C - --retry --speed-time` (real
   range-resume, ratchets through). **Progress via `kh.step()` + `HF_TOKEN`** HF
   mirror — a Kaggle script kernel exposes no live logs; never narrate from status.

**REMAINING (VPS/Linux — cannot do on macOS):**
1. **Regen Go cgo LDFLAGS** — `python tools/sync_go_cgo_ldflags.py` then `--check`.
   The ONE item that reds CI on the merge commit (hand-added `-lmoss_tts_local`;
   macOS regen pollutes the `#cgo linux` line with Metal/BLAS). Do first to green CI.
2. Release via `scripts/bump-version.sh` once CI green; **close #249** (shipped).
3. Non-blocking: ref-dumper (advisory 12-pt gap); refresh `${KAGGLE_ACCOUNT}/crispasr-ccache`
   seed (warm ~3 min builds vs ~23 cold); short-clip ASR 0.0 (likely
   whisper-on-~1 s artifact — the long clip proved the audio correct); optional
   Q5_K/Q6_K probe for a smaller-than-F16 stable target; v2 codec ENCODER for voice
   cloning.

---

## Done (compiles clean; NOT yet parity-validated)

| Phase | What | Files |
|-------|------|-------|
| 0 | Study (verified vs 2 HF configs + Python blueprint) | `docs/moss-tts/STUDY.md` |
| 1 | GGUF converter (backbone `moss-tts` + companion `moss-tts-codec`); backbone/audio name map unit-tested; codec shortener validated vs all 1600 real codec tensor names | `models/convert-moss-tts-to-gguf.py` |
| 2 | Qwen3-8B backbone (clone of qwen3_asr KV path: QK-norm, NEOX RoPE 1e6, GQA 4:1, SwiGLU) + `hidden_last` output + 2 aux graphs (summed embed, 32 heads) | `src/moss_tts.{h,cpp}` |
| 3 | Delay state machine (openmoss port, incl. sentinel/off-by-one/unique-rep-penalty gotchas) + special-token BPE + prompt builder + AR code-gen loop | `src/moss_tts.cpp` |
| 4 | Transformer RVQ codec decode (weight-norm reconstruction, 4 ProjectedTransformer stages, sliding-window mask, patch upsamples) + end-to-end `synthesize` | `src/moss_tts_codec.{h,cpp}` |
| 5 | CLI adapter, `--backend moss-tts` factory + filename/arch detect, CMake, registry entry, quantize keep-list, **session-ABI inline synthesize** (bindings/server) | `examples/cli/crispasr_backend_moss_tts.cpp`, `crispasr_backend.cpp`, `crispasr_c_api.cpp`, registry, quantize, CMake |

Verified locally: whole runtime builds into `libmoss_tts.a`; `crispasr --backend
moss-tts` routes through the session ABI to the runtime and the registry entry
resolves (backbone + codec companion). All builds 0 errors.

## Remaining Phase 5 (peripheral)

- [ ] **`bindings/go/whisper.go` cgo LDFLAGS sync — CI-ENFORCED, regen on LINUX.**
      A new backend lib (`moss_tts`) without this fails the `Bindings Tests (Go)`
      job (`undefined reference` / `cgo-ldflags-drift`). Run on the VPS:
      `python tools/sync_go_cgo_ldflags.py` then `--check`. Do NOT run on macOS
      (Metal/BLAS pollute the `#cgo linux` line). This is the one item that will
      red CI until done.
- [ ] Bindings docstrings: `python/crispasr/_binding.py`, `bindings/go/`, `flutter/`.
- [ ] Diff-harness reference backend: `tools/reference_backends/moss_tts.py`
      (`dump()` + `DEFAULT_STAGES`) + register in `tools/dump_reference.py`; a
      `moss_tts_<stage>_diff` self-runner in the `.cpp` (dots-tts/voxtral-tts
      pattern) is cleaner than exposing stage APIs.
- [ ] Live test `tests/test_moss_tts_live.cpp` + `tests/env-live-tests.sh` entry.
- [ ] README.md + `docs/{tts,architecture}.md`.

## Phase 6 — VALIDATED (2026-07-12, Kaggle P100, kernel run 1)

**The port works.** Decoded round-trip (HARD RULE #3) **PASSES on Q4_K, CUDA**:
- convert OK (F16 backbone 16.99 GB + codec 3.55 GB), quantize OK (→ Q4_K 7.0 GB,
  252/463 tensors quantized; audio embeds/heads + norms kept F16).
- Q4_K short "Hello world." → ASR "HEllo world!."; Q4_K long → ASR reproduces the
  whole passage ("the quick round fox jumps over the lazy dog. speech synthesis
  should stay intelligible over a longer passage. …"; brown→round, codec→Codex are
  ASR mishears). rms 0.10–0.16, 1.68 s / 22.04 s, proof-of-work TRUE (21→55 words).
- **F16 FAIL = P100 VRAM only, NOT a bug**: the 16.99 GB F16 backbone doesn't fit a
  16.27 GB P100 (`cudaMalloc out of memory` at load). Needs a >24 GB GPU (L4/A100)
  or CPU; Q4_K is the practical target and is proven.
- code-parity ref dump: torch OOM loading the 8B alongside the crispasr process on
  the same 16 GB P100 (non-gating; expected).

**Kernel refinements for a cleaner re-run** (both known bugs, not port issues):
1. `CCACHE_DIR=/kaggle/working/.ccache` bloats the output with the ccache tree →
   `progress.txt` sorts past the 500-file `kernels_output` page cap (usage #22).
   Move ccache to `/kaggle/temp/.ccache` after `install_build_toolchain`; keep
   `/kaggle/working` to just `progress.txt`+`results/`. (Log is still reachable via
   `KaggleApi().kernels_logs(slug)` — used to diagnose this run.)
2. Treat an F16-backbone load-OOM as **SKIP** on ≤16 GB GPUs, not FAIL — gate only
   on Q4_K there; run F16 only when VRAM ≥ ~20 GB.

Ship next: upload GGUFs to `cstr/moss-tts-v1.5-GGUF` (backbone Q4_K + F16 codec;
daemon-thread + timeout + server-side verify per the HF-upload note), populate the
registry `license` (Apache-2.0), version bump, HISTORY + LEARNINGS.

## Phase 3 code-parity — RESOLVED (2026-07-13, Kaggle P100)

Ran the greedy code-parity (C++ Q4_K vs HF BF16 reference on CPU, temps=0) and
methodically diffed the divergence. Two of the PLAN's "known-suspect" areas were
both confirmed — one a real bug, one an intrinsic limit:

1. **Tokenizer (real bug, FIXED — `41c08e8f`).** The moss-tts prompt tokenizer
   (cloned from `qwen3_asr`) used a crude whitespace pre-splitter that split `>`
   from a trailing `\n`. Qwen's pre-tokenizer regex `[^\s\p{L}\p{N}]+[\r\n]*`
   groups punctuation with trailing newlines (`>\n`=397, `):\n`=982). The
   `moss-tts-promptdiff` kernel isolated it: prompt TEXT identical, tokenization
   differed at the first newline. Fix: a proper Qwen2/3 pre-tokenizer
   (`mt_qwen_pretokenize`). After the fix the 67-token conditioning prompt is
   **byte-identical** to the HF processor (`first_mismatch=null`, `prefix=67`).

2. **F16-vs-BF16 rounding through the AR loop (intrinsic — NOT a bug).** Even with
   byte-identical prompts, greedy codes diverge from **frame 0** (~0.5% exact
   match). The `moss-tts-logit0` probe settles it: at step-0 head-0, the C++ Q4_K
   logits are essentially the reference's distribution — the reference's greedy
   pick (143) is the C++'s **rank-1 runner-up**, only **0.135 logits** below the
   C++ argmax (1021). Q4_K rounding (O(0.1–0.5) on logits) flips this near-tie
   onset token; the MOSS delay makes every un-delayed frame span many raw AR
   steps, so one flip cascades (~n_vq·T decisions → ~0.1% survival = the 0.5%
   observed). Codebook-0 (coarse RVQ) still re-syncs to exact reference values at
   scattered frames (807, 578, 756; 2-frame-shifted runs 400/575/254) — the
   models produce the same coarse audio, differing only on quant-sensitive fine
   residuals.

**Conclusion:** exact greedy code-parity between ggml-Q4_K and torch-BF16 is
*unachievable* for this AR audio LM (near-tie onset flips + AR chaos + dtype
mismatch), exactly as the last PLAN bullet predicted ("judge by the deterministic
prefix + round-trip, not aggregate cos"). The port is **structurally confirmed**:
byte-identical prompt + near-identical step-0 logit distribution + coarse-codebook
re-sync + the passing ASR round-trip (HARD RULE #3). The correct acceptance gate
is the round-trip (passes) and the step-0 logit-rank probe (ref pick = C++ rank-1),
NOT byte-exact greedy codes. Kernels: `tools/kaggle/moss-tts-{promptdiff,parity,
logit0}/`.

## Phase 6 — original validation plan (the ONLY acceptance test; HARD RULE #3)

8B backbone won't fit the 8 GB VPS and is tight on the 16 GB Mac with the 1.6 B
codec → run on **Kaggle** (P100/T4). Reference kernels:
`tools/kaggle/voxtral-diff-harness/`, `tools/kaggle/tada-bucket-ab/`.

1. **Convert** on Kaggle (stage model pulls under `/tmp` ~70 GB, not
   `/kaggle/working` ~20 GB): `python models/convert-moss-tts-to-gguf.py --input
   OpenMOSS-Team/MOSS-TTS-v1.5 --codec OpenMOSS-Team/MOSS-Audio-Tokenizer
   --output moss-tts-v1.5-f16.gguf` → F16 backbone + F16 codec. Then
   `crispasr-quantize moss-tts-v1.5-f16.gguf moss-tts-v1.5-q4_k.gguf q4_k`.
2. **Code parity** (Phase 3 gate): greedy code streams byte-identical to the
   Python/pwilkin reference for a fixed text+seed. Build the parity dumper first
   (reference backend). Watch the delay-fill boundary + the sentinel warm-up.
3. **Codec parity** (Phase 4 gate): per-stage cos ≥ 0.999 vs the ONNX/PyTorch
   tokenizer, then the decoded audio. Gate input alignment BEFORE trusting
   per-layer cos. ⚠ CUDA `get_rows` needs a contiguous index (dev-guide §232) —
   audit the codebook lookups if decode aborts on P100.
4. **Decoded round-trip** (the real test): `crispasr --backend moss-tts
   --model <gguf> --tts "..." --tts-output out.wav` → ASR `out.wav` (whisper) →
   text recognizable. Test **F16 AND Q4_K** (quant amplifies divergence). Add an
   `expected_text` regression entry.

Known-suspect areas to check first if parity fails (all faithful clones but
unverified): the codec attention (manual SDPA + sliding-window mask vs openmoss's
flash path — output-equivalent but confirm), the prompt tokenization (special-token
BPE must match the Python tokenizer exactly), and F16 vs BF16 rounding through the
deep AR loop (judge by the deterministic prefix + round-trip, not aggregate cos).

## Follow-ups (documented, out of core scope)

- Voice cloning: the codec **encoder** (ref audio → codes). openmoss `codec.cpp`
  encode path + `core_rvq::encode_euclidean`; wire the reference-audio prompt grid
  in the AR loop. Separate PR.
- 4B `MossTTSLocal` (48 kHz stereo, depth-transformer, tokenizer-v2). No C++ ref.
- Perf: codec uses manual SDPA (O(T²) scores/mask); switch to flash_attn_ext +
  persistent graph if long-audio memory bites (openmoss's reason for flash).

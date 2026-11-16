# parakeet-tdt: word list + chunking fixes (issue #257)

## NOW — active work (2026-07-15, branch `fix/issue-257-segmentation`)

Reporter comment #4983428974 (build @ 345e80d3): (a) build FAILS without
`<climits>` (INT_MIN from the att-context commit c08898ea); (b) "the last issues
mentioned are not fixed by --att-context" — i.e. R3's remaining ask: `--chunk-seconds`
still emits ONE giant transcription, not segments.

**DONE on this branch (verified on parakeet-tdt-0.6b-v3-q4_k, Metal M1, reporter's
t501-20s.wav):**

1. **Build fix.** `#include <climits>` added to `whisper_params.h` (the reported
   TU; also covers cli.cpp/server.cpp/backend_parakeet.cpp which include it) and to
   `crispasr_c_api.cpp` (separate src TU with its own INT_MIN use). `sentencepiece.h`
   mentions INT_MIN only in comments — no fix needed. Full build green.

2. **Segmentation + truncation fix (the real bug).** Reproduced that
   `--chunk-seconds 7 --chunk-overlap 2` on 0.6b-v3 both (i) emitted 1 segment AND
   (ii) TRUNCATED the tail (15 words / ends 10.7 s of a 20 s clip) — because the
   `--chunk-seconds` path drove `parakeet_transcribe_streamed` with a 7 s ENCODER
   window, and small windows shift this full-attention FastConformer's per-feature
   stats → sparse/truncated TDT decode (already documented at parakeet.cpp:3818).
   Fix (crispasr_backend_parakeet.cpp): **decouple the encoder window from output
   segmentation.** Keep the encoder at the model's quality default (30 s, bounded
   VRAM, `CRISPASR_PARAKEET_STREAM_CHUNK`-overridable), decode once coherently, then
   split the words/tokens into ~N-second OUTPUT segments via new pure helper
   `core_segment::group_by_window` (`src/core/asr_segment_group.h`, snapped to word
   boundaries, contiguous, every seg ≥1 word). New unit test `test-segment-group`
   (6 cases, `[unit]`).

   Verified vs single-pass reference (identical text = no corruption/truncation):
   - `--chunk-seconds 7`  → 2 segments, 20 words, text == single-pass. (was 1 seg/15 w)
   - `--chunk-seconds 5`  → 3 segments, 20 words, contiguous offsets.
   - `--chunk-seconds 30` → 1 segment (whole clip fits).
   - default (no chunk)   → 1 segment, 20 words (NO regression).
   - JA model             → dispatcher VAD path unchanged (my branch only touches
                            the non-JA explicit-chunk branch).

   Net for the reporter: `--chunk-seconds N` now = bounded VRAM + complete transcript
   + N-second segments, all at once — no need for --att-context on the chunked path.

**Merged to `main` `7c1bbbbc0`** (CLI + `<climits>`). Then extended for full
consumer wiring — see next section.

## WIRING — all consumers (2026-07-15, branch `fix/issue-257-wiring`)

The first fix landed in the CLI backend ADAPTER only. Per the dev-guide HARD
RULE #6, the session C-ABI reimplements each backend inline (bindings/wrappers
don't call the adapter), and the server had its own slicing that ignored
CAP_INTERNAL_CHUNKING. Audited + wired every consumer:

- **CLI** — adapter fix (already on main). ✓
- **Server** (`crispasr_server.cpp`) — added the CLI's CAP_INTERNAL_CHUNKING
  gate: a self-chunking backend (parakeet/canary) with no VAD now gets the WHOLE
  clip (`effective_chunk_seconds = 0`) → its coherent decode + adapter
  segmentation runs, instead of per-slice transcribe that corrupts the
  full-attention encoder. ✓
- **C-ABI session** (`crispasr_c_api.cpp`, used by python/go/java/dart/ruby/rust/
  wasm/node wrappers) — mirrored the adapter: explicit `chunk_seconds>0` (non-JA)
  → `parakeet_transcribe_streamed` at the quality window + new
  `parakeet_result_to_session_segs` (reuses `core_segment::group_by_window`) →
  ~N-second segments. `chunk_seconds<=0` keeps the one-merged-segment #208
  contract. ✓
- **Wrappers** — no code change needed: they marshal `crispasr_session_seg[]`
  generically, so they get the segments once the C-ABI emits them.

Verified on the reporter's ACTUAL model (`cstr/parakeet-tdt-1.1b-GGUF` q4_k) +
`t501-20s.wav`, `--chunk-seconds 7 --chunk-overlap 2`:
- CLI: 2 segments, 21 words, text == single-pass (== maintainer's #257 baseline).
- Server (`/v1/audio/transcriptions`, `chunk_seconds=7`): 2 segments, complete.
- Python `Session.transcribe_chunked(7)`: 2 segments, `chunked text == plain`.
All three agree. `docs/{cli,server,bindings}.md` updated.

**NEXT:** run the unit suite, format, merge to `main`, update #257 reply to
mention server + bindings parity. (Note: a cosmetic "JA model with q4_0" load
warning misfires on the 1.1b — pre-existing print heuristic, decode is correct;
out of #257 scope.)

---

## DONE (2026-07-14) — both fixes on main; issue closeable

Branch `fix/parakeet-257`. Reporter (AppleSheeple) on parakeet-tdt-1.1b:
(A) `-ojf` JSON has text + tokens but **no words list**; (B) `--chunk-seconds 7
--chunk-overlap 2` cuts transcript tails (first cut > overlap) and emits weird
mid-sentence split segments. Not seen with cohere/granite.

### Reproduced (parakeet-tdt-0.6b-v3-q4_k, samples/jfk.wav)
- **(B) CONFIRMED, serious.** Baseline (no chunk): "And so, my fellow Americans,
  ask not what your country can do for you, ask what you can do for your country."
  With `--chunk-seconds 7 --chunk-overlap 2`:
    "And so, my fellow Americans,"                                  ← tail CUT ("ask not..." gone)
    "what your country can do for you, ask what you can do for yourself."  ← boundary loss + "your country"→"yourself"
    "you can do for your country."                                  ← duplicated split fragment
- **(A) not repro'd on 0.6b-v3** (22 words emitted, with & without --no-punctuation).
  Likely 1.1b-specific OR the streamed/chunked path.

### Root causes (code)
- **(A)** `parakeet_decode_frames` (parakeet.cpp:3431) sets `r->words=nullptr;
  r->n_words=0` with a FALSE comment ("the backend adapter builds words" — the
  adapter `result_to_segment` only *copies* r->words). Both `parakeet_transcribe_
  streamed` (3605) and `parakeet_transcribe_chunked` (3605-ish return) delegate to
  decode_frames → **no words on streamed/chunked paths**. transcribe_ex (single-
  pass) builds words inline (3966), so short single-file gets words. → FIX: extract
  the word-grouping into a shared helper and call it in decode_frames too.
  (Also: transcribe_chunked builds a full `r` at 3568-3591 then throws it away by
  `return parakeet_decode_frames(...)` → double-decode + word loss. Clean that up.)
- **(B)** `--chunk-seconds` forces the DISPATCHER chunk+merge (bypasses parakeet's
  internal long-audio handling). The per-chunk parakeet decode is fine; the
  MERGE/LCS-dedup drops boundary content for parakeet (works for cohere/granite).
  → INVESTIGATING the dispatcher chunk+merge + LCS dedup (crispasr_c_api
  session_transcribe_chunked path).

### Progress
- (A) DONE: extracted `parakeet_group_words()` helper, called from decode_frames +
  transcribe_ex. Streamed path (STREAM_THRESHOLD=0) now emits 22 words on jfk (was 0).

### (B) root cause + fix
- `--chunk-seconds` forces the DISPATCHER per-slice transcribe + overlap-save
  trim + LCS merge. parakeet is a full-attention FastConformer (`CAP_UNBOUNDED_INPUT
  + CAP_INTERNAL_CHUNKING`): short context-extended slices decode DEGRADED
  ("your country"→"yourself") and the trim drops boundary words ("ask not" lost).
- cohere/granite lack `CAP_INTERNAL_CHUNKING` → dispatcher chunking is correct for
  them (matches user: "not seen with cohere/granite").
- parakeet's INTERNAL streaming at 7s/2s (STREAM_THRESHOLD=0 STREAM_CHUNK=7) gives
  the CORRECT full transcript. → FIX: for CAP_INTERNAL_CHUNKING backends, when
  --chunk-seconds is explicit, DON'T dispatcher-slice; pass the whole audio and let
  the backend chunk internally. parakeet adapter honors params.chunk_seconds/overlap
  by routing to parakeet_transcribe_streamed.

### DONE
- (A) parakeet_group_words() shared helper → words on streamed/chunked paths.
- (B) backend_self_chunks_on_explicit() gate: CAP_INTERNAL_CHUNKING + explicit
  --chunk-seconds bypasses dispatcher slicing (crispasr_run.cpp); parakeet adapter
  routes to parakeet_transcribe_streamed(chunk,overlap). jfk --chunk-seconds 7
  --chunk-overlap 2 now == baseline ("...ask not what your country can do for you.
  Ask what you can do for your country.", 22 words). 4 new unit tests in
  test-issue-114-chunk-context-gate. Full unit suite green (12 'failures' = unbuilt
  metal/vad binaries in the targeted build, not regressions).
2. (B) find why the chunk-merge drops parakeet tails; fix; verify jfk chunked == baseline.
3. Unit test(s) for the word-grouping helper + a chunk-merge regression.
4. Build, run unit tests, merge to main, comment #257.

## REOPENED (2026-07-14) — reporter feedback: partial fix

Reporter (AppleSheeple) on parakeet-tdt-1.1b: (A) words now present ✓ but weird
single-word split-outs appear; (B) chunked still cuts tails + splits. Key detail:
tokens array is FULL, but the SEGMENT offsets.to is cut and text/words are filtered
to end before it.

ROOT CAUSE: `is_ja_model_ = (parakeet_n_vocab <= 4096)` (adapter:75) MISDETECTS
parakeet-tdt-1.1b (English, vocab ~1024) as Japanese. JA-ness is vocab CONTENT not
size: 0.6b-ja vocab_size=3073 is ~97% CJK/kana; v3 vocab_size=8192 is 0% CJK; 1.1b
English small vocab is ~0% CJK. Misdetected-JA → CAP_INTERNAL_CHUNKING off, (B) fix
gated off (!is_ja), JA 8-12s small-chunk path used on an English full-attention
model → split-outs (default) + dispatcher-slice corruption (chunked). Same bug in
the lib's `vocab_size < 4000` heuristics (parakeet.cpp:3654,3774).

FIX: detect JA by scanning the vocab for Japanese script (kana/kanji fraction),
not vocab size. Verify: 0.6b-ja stays JA, 0.6b-v3 non-JA, 1.1b-EN non-JA. Download
cstr/parakeet-tdt-1.1b-GGUF to reproduce the reporter's exact case.


## FIXED v2 (2026-07-14) — JA misdetection resolved, verified on real 1.1b
- parakeet_vocab_is_japanese() (vocab content scan) replaces vocab<=4096. Downloaded
  cstr/parakeet-tdt-1.1b-GGUF (vocab_size=1024, confirms misdetection).
- 1.1b DEFAULT: 1 clean complete segment, 21 words (was split-outs + truncation).
- 1.1b --chunk-seconds 7 --chunk-overlap 2: full correct transcript (was corrupted).
- 0.6b-ja still JA (97% CJK vocab); 0.6b-v3 still non-JA. No regressions.
- TODO: extract testable vocab_looks_japanese() helper + unit test.

## ROUND 3 (2026-07-15) — reporter: split-outs fixed; default VRAM heavy

Reporter: split-outs gone ✓. Remaining: (i) --chunk-seconds gives one coherent
transcription (INTENDED — chunked encode, decode once, word timestamps); (ii)
default (no chunk) ~2GiB VRAM for <4min (single-pass full O(T²) attention).

USER DIRECTION: option 2 (memory-bounded, CLI-steered) wired through C-ABI/server/
wrappers, MATCHING the Python reference.

REFERENCE (NeMo): long-audio memory is bounded via `change_attention_model(
"rel_pos_local_attn", [L,R])` (local/windowed attention → O(T·window)); default is
full attention. CrispASR already implements this (att_context_left/right) but only
via env CRISPASR_PARAKEET_ATT_CONTEXT.

PLAN:
1. BUG (critical): C-ABI inline parakeet dispatch (crispasr_c_api.cpp:4525) still
   uses the OLD `parakeet_n_vocab<=4096` JA heuristic → bindings/server STILL
   misdetect 1.1b. Mirror parakeet_vocab_is_japanese() there (contributing pt6).
2. FEATURE: expose local-attention window as CLI `--att-context L,R` (matches NeMo
   change_attention_model), wired: whisper_params → CLI → parakeet adapter
   (parakeet_set_att_context) → C-ABI (session field + inline dispatch) → server
   (form) → python/go wrapper docs. Default full attention (matches NeMo default).
   --chunk-seconds stays the other reference control (chunked inference).


## ROUND 3 DONE (2026-07-15)
1. C-ABI JA fix (crispasr_c_api.cpp:4525) — bindings/server now match CLI. ✓
2. --att-context "L,R" wired: lib parakeet_set_att_context → whisper_params → CLI
   (+help) → parakeet adapter → C-ABI (session field + inline dispatch +
   crispasr_session_set_parakeet_att_context + header) → server (att_context form,
   both handlers) → python Session.set_parakeet_att_context(). Go binding has no
   session-API surface (0 crispasr_session refs), nothing to wire there.
   Verified: --att-context 64,64 on 88s clip == full-attention output, local attn
   active; symbol exported; python syntax OK.
Matches NeMo: full attention default; opt-in local attention (rel_pos_local_attn)
for long-audio VRAM. --chunk-seconds remains the other reference control.

## ROUND 4 (2026-07-15) — true windowed attention (maintainer-directed)

FINDING: --att-context (as shipped R3) does NOT reduce memory — CrispASR builds a
T×T mask over FULL attention (fastconformer build_block: scores are (T,T,n_heads)),
matching NeMo's OUTPUT but not its O(T·window) memory. Measured peak RSS: full ==
att-context (1.41GB); --chunk-seconds ~same/slightly higher. The real single-alloc
memory lever today is --chunk-seconds (per-chunk encode graphs).

DIRECTIVE: implement TRUE windowed attention (compute only the local band →
O(T·window)) so --att-context delivers NeMo rel_pos_local_attn's memory benefit.

REFERENCE: NeMo RelPositionMultiHeadAttentionLongformer — "sliding chunks":
pad+reshape Q/K/V into overlapping windows of size w; each query chunk attends to a
2w+1 key band; rel-pos bias (BD) windowed too. O(T·w) scores.

PLAN: core_conformer::build_block windowed-attention path (gated), validated vs the
masked-full output (parity ≥0.999 on the encoder / transcript-identical) + measured
memory reduction, before flipping --att-context to use it. HARD: ggml banded matmul
(overlapping key-window gather) + windowed rel-pos. Incremental, diff-harness-checked.

### R4 Milestone 1 DONE (2026-07-15) — windowed-attn algorithm validated

Standalone ggml parity harness (tools/dev/winattn_parity.cpp) proves the block
sliding-chunks windowed attention is BIT-EXACT vs full masked rel-pos attention
(max abs diff ~1e-7) across: baseline, asymmetric windows (WL!=WR), BS/HD/NH
sweeps, and non-divisible T with query-axis zero-padding (T=13,17,100,209).

Key algorithm (O(T·BS·H) scores/BD instead of O(T²·H)):
- Block size BS >= max(att_left, att_right). NB=ceil(T/BS), Tp=NB*BS (pad Q/K/V).
- 3-block band per query block via reshape->3 stride-1 block slices->concat
  (ggml forbids overlapping views: view_4d checks contiguous product<=src bytes).
- K/V zero-padded BS each side so band [b-1,b,b+1] = keys k=(bo-1)BS+j, j∈[0,3BS).
- scores_blk = mul_mat(K_band(HD,3BS,NB,NH), Qu_blk(HD,BS,NB,NH)) -> (3BS,BS,NB,NH).
- BD (rel-pos bias) windowed: R_sl = R rows [T-2BS .. T+2BS-2] (RB=4BS-1),
  RESHAPED (HD,RB,1,NH) so mul_mat broadcasts R over blocks & aligns heads in ne3
  (critical bug found: without the size-1 block axis, ggml mixes head/block batch
  dims and blocks b>=2 diverge). BDraw_blk=mul_mat(R_sl,Qv_blk)->(RB,BS,NB,NH),
  then in-block rel_shift view: BD_blk[j,i]=BDraw_blk[(BS-1)+j-i,i], nb1'=nb1-nb0,
  offset (BS-1)*nb0 — natural key order, all strides>=0.
- Host band mask (3BS,BS,NB): -inf where k out of [0,T) or out of [q-WL,q+WR].

NEXT (M2): wire as core_conformer::build_windowed_attn, gated CRISPASR_FC_WINDOWED_ATTN
(default keeps masked-full path intact for A/B), validate via real parakeet
transcript parity + memory measurement.

### R4 M2 DONE + M3 in progress (2026-07-15) — wired + validated

M2: build_windowed_attn wired into core_conformer::build_block, gated
CRISPASR_FC_WINDOWED_ATTN=1 (default OFF keeps masked-full intact). Caller
parakeet.cpp builds O(T·window) band mask (make_window_band_mask) instead of the
T×T local mask when gated+applicable. Builds clean.

M3 findings (parakeet-tdt-0.6b-v3 q4_k, Metal M1):
- PARITY: windowed-local == masked-full-local transcripts IDENTICAL on 20s
  (T=250) and 209s (T=2613) clips. Windowed path confirmed engaging (stderr trace).
- MEMORY: KEY metric is phys_footprint (macOS caps RSS via compression). At forced
  single-pass T=7838 (627s clip, CRISPASR_PARAKEET_STREAM_THRESHOLD=9999):
    masked-full local: peak footprint = 2402 MB  (the O(T²) BD_raw ~ user's "2GiB")
    windowed local:    <measuring — slow, backgrounded>
  So the O(T²) memory hog is REAL at large single-pass T, and it IS the rel-pos BD.
- CAVEAT: default dispatcher silence-splits/chunks long audio (STREAM_THRESHOLD
  300s), bounding per-encode T, so the blow-up only appears in forced single-pass.
  Windowed's purpose = enable bounded-memory SINGLE-PASS long encode (avoids the
  chunking that corrupts full-attention FastConformer — the #257 root issue).
- CONCERN: windowed is SLOWER at large T (many small ops: 2×concat + 4×pad + several
  cont per layer ×24). Timed out >2min at T=7838. Needs perf assessment / op fusion
  before it's a viable default; fine as an opt-in memory-vs-speed lever now.

NEXT: confirm windowed footprint << 2402MB; assess speed; fix --att-context help
wording; decide default (opt-in for now).

### R4 M3 RESULTS (2026-07-15) — windowed is FASTER + lower memory (correction)

Earlier "windowed is slower" was WRONG (that was T=7838 being slow for ALL paths).
Real data at T=2613 (209s, single-pass, Metal M1):
    masked-full local (att 64,64): 25.7 s   (worst: full compute + T×T mask)
    windowed local     (att 64,64):  8.3 s   (3.1x faster than masked-full)
    full attention:                 11.4 s   (windowed 1.4x faster than full)
Memory (peak footprint, macOS phys_footprint; RSS is compression-capped):
    T=7838 single-pass: masked-full 2402 MB vs windowed 2155 MB (~10%; the O(T)
    conv front-end co-dominates at this T — BD is O(T²) so the win grows for
    longer audio). Attention BD itself drops from ~2GB to a few hundred MB.
Parity: windowed == masked-full transcripts IDENTICAL on t501-20s/long90/long3m
    at both att 32,32 and 64,64. Bit-exact algorithm (parity harness).

VERDICT: windowed local attention is strictly better than the shipped masked-full
local path — same output, ~3x faster, less memory (growing with length). Still
gated CRISPASR_FC_WINDOWED_ATTN=1 for A/B per maintainer. Candidate to become the
DEFAULT when --att-context is set, pending CUDA cross-check.

### R4 DONE (2026-07-15) — windowed attn is now DEFAULT for --att-context

fc_windowed_attn() flipped to default-ON (CRISPASR_FC_WINDOWED_ATTN=0 forces legacy
masked-full). Only engages when --att-context is set + T>=2*BS + band mask supplied;
full-attention (no --att-context) is unaffected. --att-context help updated.
Verified: default engages windowed, gate=0 uses masked-full, output IDENTICAL.
Unit suite green (911 unit + 11 metal). parakeet.cpp is wired; canary/canary_ctc
share core_conformer::build_block and would benefit from the same ~10-line caller
plumbing (band mask + pass window_band_mask) — NOT yet wired.

OPEN: CUDA (Kaggle P100) cross-check not yet run (can't validate CUDA on M1).

### R4 — canary wiring: NOT done (blueprint says full-attention)

Checked the Python blueprints per maintainer request ("compare what the python
blueprints do"):
- models/convert-parakeet-to-gguf.py reads self_attention_model; emits
  att_context_left/right ONLY for "rel_pos_local_attn" (else full "rel_pos").
- models/convert-canary-to-gguf.py + convert-canary-ctc-to-gguf.py NEVER read
  self_attention_model / att_context_size. tools/reference_backends/canary.py has
  no local-attn path. => NeMo canary/canary_ctc are FULL-attention by design.
- parakeet-tdt-0.6b-v3 GGUF has NO att_context baked in (full-attention); windowed
  engages only via user --att-context. True rel_pos_local_attn models (reazonspeech)
  bake att_context and auto-engage (matches their NeMo training exactly).

DECISION: do NOT wire canary/canary_ctc for local attention — it would diverge from
the reference and degrade quality (global-context-trained). The shared
core_conformer::build_windowed_attn is available if a future canary variant ships
rel_pos_local_attn; converter would emit att_context and the parakeet-style caller
plumbing would light it up. canary_ctc additionally uses the mask slot for pad
masking, so any future adoption must merge pad-validity into the band mask.

CAVEAT to document: --att-context on a full-attention model (e.g. v3) yields LOCAL
attention output (differs from full) — a quality/memory tradeoff the user opts into;
on a rel_pos_local_attn model it is exact to training.

### R5 (2026-07-15) — reduce the FULL-attention O(T²) rel-pos bias (query-tiled BD)

Q: "can we reduce the O(T²) rel-pos bias?" A: yes, for EXACT full attention too.
build_tiled_attn (gated CRISPASR_FC_TILED_ATTN=1): process queries in blocks of
BS (default 512, CRISPASR_FC_TILED_BLOCK) against ALL keys, computing only a
(T×BS) rel-pos bias slab per block via a per-block rel_shift (offset (T-1)-b*BS).
Peak O(T·BS) instead of O(T²). Bit-exact vs monolithic full attention
(tools/dev/tiledbd_parity.cpp: 0..1e-8 across T/BS/head sweeps incl. non-div T).
Only for pure full attention (no local/pad mask); no caller change (BD from R).
Verified real-transcript IDENTICAL vs monolithic on 209s (T=2613, NB=6).
Measuring memory + speed at forced single-pass T=7838 (NB=16) — tiled is manual
per-block (Metal monolithic uses flash), so speed is the open question.

### R5 RESULTS (2026-07-15) — tiled BD is a CUDA-only memory lever; Metal already fine

Measured tiled vs monolithic full attention on Metal M1 (parity IDENTICAL throughout):
- T=2613: mono-flash 1301 MB, mono-manual 1271 MB, tiled 1273 MB — NO win.
- T=7838 single-pass: mono-flash 1603 MB vs tiled 1608 MB — NO win.
- Footprint is ~FLAT with length on Metal (20s→209s: 1226→1301 MB): model + fixed
  buffers dominate; flash_attn_ext never materializes the O(T²) scores, so full
  attention memory barely grows. The conv pre-encoder is the other O(T) consumer.

=> On METAL/flash the O(T²) bias is already invisible; tiling can't help and is
slower (manual per-block). The reporter's ~2 GiB is on CUDA, where
fc_gpu_manual_attn MATERIALIZES O(T²) scores+BD — that IS what tiling reduces
(→ O(T·block)). Can't measure on M1. Added tiled_full config to the CUDA kernel
(tools/kaggle/windowed-attn-cuda) to validate the GPU payoff.

STATUS: build_tiled_attn kept, gated OFF (CRISPASR_FC_TILED_ATTN=1), documented as a
CUDA-side memory lever. Speed on Metal is worse (manual) so not for Metal use.
OPEN: run the CUDA kernel to confirm tiled_full GPU peak << full_attention.

# Benchmarking CrispASR

How to measure CrispASR's speed fairly — whether you're tuning it, comparing it
against another engine, or publishing numbers.

**The one rule: measure transcribe time, not cold start.** CrispASR ships as a
single binary that loads a multi-GB GGUF and uploads weights to the GPU on every
process start. Timing `crispasr -m model.gguf -f clip.wav` end-to-end with a
stopwatch measures *process start + model load + GPU upload + audio decode +
transcription*. Most engines you'd compare against are benchmarked in-process
with the model loaded once and the first (cold) rep discarded — so an
end-to-end subprocess timing isn't the same quantity, and the gap it shows is
mostly load time.

Every method below isolates the transcription.

---

## Quickest fair number: read the stderr timing line

Both the CLI and the server already print a load-excluded transcription time:

```
crispasr: transcribed 11.0s audio in 0.42s (26.2x realtime)
```

The timer starts **after** model init, audio decode, and VAD/slicing
(`crispasr_run.cpp`), so this line is transcription only. The server prints the
same shape (`crispasr-server: transcribed ... (Nx realtime)`) from its own
`elapsed_s`, measured around the transcribe call.

If you script around the CLI, parse this line — don't wrap the process in
`time`.

## Method 1 — server mode (resident model)

Best match for how a service actually runs, and for engines benchmarked in
"server mode". The model is loaded once at startup and reused for every request.

```bash
# Start once; the model stays resident.
crispasr --server --backend parakeet -m parakeet-ctc-0.6b-q8_0.gguf \
    --host 127.0.0.1 --port 8080

# Then send N requests. Discard the first (warms JIT/pipeline caches).
curl -s -F file=@clip.wav -F response_format=json \
    http://127.0.0.1:8080/v1/audio/transcriptions
```

The server's stderr timing line per request is the transcribe time. See
[`server.md`](server.md) for the full endpoint/field reference.

## Method 2 — in-process (the apples-to-apples path)

Load the library once, transcribe many times. Use the Python `Session` API
(`python/crispasr`) or `ctypes` against `libcrispasr.{dylib,so,dll}`.

This is what `tools/benchmark_asr_engines.py --crispasr-call ctypes` does, and
it's the path to prefer for cross-engine comparisons — the engine is loaded once
and reused across runs, matching how the other side is usually measured.

## Method 3 — CLI with warm reps

Acceptable if you parse the stderr timing line (above) rather than wall-clock
the process. You still pay process start + model load per call, so only the
parsed line is meaningful. For a load-amortised comparison, prefer method 1 or 2.

---

## Proof-of-work — an RTF is a lie until you prove the work happened

`rtf = audio_duration / walltime` with no output check turns a crash into a
record-breaking result. A wrong-backend load failure that exits in 0.5 s once
minted a fake "102× realtime". Defenses, in order of value:

1. **Any non-zero exit or empty transcript is a FAIL, never a timing.**
2. **Scale check.** Benchmark a clip that is a known multiple of a short one
   (e.g. 55 s = 11 s × 5) and assert the transcript word count scales ~5×. The
   tell for a fake win is that the "55 s" run takes the *same* time as the
   "11 s" run — fixed time ⇒ it isn't processing the audio.
3. **Warm up per shape, take a median of ≥3, and print absolute ms next to the
   RTF.** A 0.06 s denominator magnifies noise; an un-warmed GPU shape fakes a
   collapse.
4. **Discard the first (cold) run.**

## Measure both arms under identical load

A noisy box fabricates wins. GPU timing on a loaded 16 GB M1 swings ±20 %
(the same graph measured 127 ms/frame quiet vs 264 ms/frame loaded). A
quiet-CPU-vs-loaded-GPU comparison once invented a "2× CPU win" that a clean
back-to-back run reversed.

- Check `sysctl vm.loadavg` (macOS) / `uptime` before every timing run.
- Run both arms back-to-back, alternating order.
- Run each config as a **separate process** — a second `crispasr` spawned
  immediately after the first in the same shell can exit at 0 s on a
  GPU/resource-release race.
- For a verdict you'll publish, prefer a quiet dedicated box.

## Report the configuration

An RTF without these is not reproducible: **backend, quant, device/backend
(Metal/CUDA/CPU+BLAS), thread count, warm vs cold, load-excluded or not, clip
length, and absolute ms**. Quant and BLAS quality dominate: the same parakeet
decode measured 955 ms on one CPU (OpenBLAS) vs ~60 ms on M1 (Accelerate) —
~16× from BLAS alone, same code.

---

## Phase timing / profiling

Per-backend stage timers are env-gated:

| Env | Effect |
|---|---|
| `CRISPASR_VERBOSE=1` (or `--verbose`) | Turns on every backend's debug/bench vars at once |
| `CRISPASR_<BACKEND>_BENCH=1` | Per-backend stage timings (e.g. `CRISPASR_CANARY_BENCH=1`) |
| `CRISPASR_METAL_PROFILE=1` | Metal host-encode vs GPU split |
| `CRISPASR_FC_PROFILE=1` | Per-node profile for the `canary_ctc` runtime — the `fastconformer-ctc` / `canary-ctc` backend and the forced aligner (`src/canary_ctc.cpp`). Not read by `parakeet` or the canary AED decoder. |

Existing harnesses:

- `tools/benchmark_phases.sh` — per-phase benchmark across ASR backends, short
  (11 s) and long (55 s) audio.
- `tools/benchmark_asr_engines.py` — head-to-head vs `onnx-asr` on the same
  Parakeet model; quant × mode matrix, JSON output, warmup handling. See
  [`../tools/benchmark_asr_engines.README.md`](../tools/benchmark_asr_engines.README.md).
- `PERFORMANCE.md` — current published numbers. Read it before benchmarking; it
  is the source of truth for engine/quant status.

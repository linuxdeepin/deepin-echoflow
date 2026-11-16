# server worker-pool — CUDA concurrency proof (improvements Phase 4b)

Validates on a real NVIDIA card what an M1 could not: `CRISPASR_SERVER_WORKERS=N`
gives real request concurrency (pooled per-worker CUDA contexts) with identical
transcripts. On M1 the memory-bound CPU model contended, so concurrency showed no
throughput win — the honest null was documented; this settles it on a GPU.

## Run
```bash
# 1. Merge the code to main (or push the branch) so the kernel can clone it.
# 2. Ensure the private dataset ${KAGGLE_ACCOUNT}/crispasr-hf-token exists (HF_TOKEN secret).
bash push.sh
kaggle kernels status  ${KAGGLE_ACCOUNT}/crispasr-server-workers-cuda
kaggle kernels output  ${KAGGLE_ACCOUNT}/crispasr-server-workers-cuda -p ./out
```

## Knobs (env)
- `CRISPASR_REF` (default `main`) — branch/tag to build.
- `MODEL_REPO`/`MODEL_FILE`/`BACKEND` — default `cstr/moonshine-tiny-GGUF` /
  `moonshine-tiny-q8_0.gguf` / `moonshine` (small, so one request doesn't saturate the GPU).
- `REPS` (default 3) — median over REPS.

## Reads
- **WORKERS=1** (control): 2 concurrent requests SERIALISE (speedup ~1.0).
- **WORKERS=2**: 2 serial (~2× single) vs 2 concurrent (~1× single) → speedup >1.3 = real
  GPU concurrency. Every request must return an IDENTICAL transcript (correctness gate).
- speedup ~1.0 at WORKERS=2 = one request already saturates the card (honest null, not a bug).

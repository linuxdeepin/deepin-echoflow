# parakeet memory-policy — CUDA proof (improvements Phase 2)

Validates on a real NVIDIA card what an M1 cannot: that the proactive VRAM-budget
policy avoids the O(T²) single-pass rel-pos-bias OOM (issue #257) and that the
`parakeet_est_singlepass_peak_mb` estimate matches real CUDA allocation.

## Run
```bash
# 1. Merge the code to main (or push the feature branch) so the kernel can clone it.
# 2. Ensure the private dataset ${KAGGLE_ACCOUNT}/crispasr-hf-token exists (HF_TOKEN secret).
bash push.sh
kaggle kernels status  ${KAGGLE_ACCOUNT}/crispasr-parakeet-mem-policy-cuda
kaggle kernels output  ${KAGGLE_ACCOUNT}/crispasr-parakeet-mem-policy-cuda -p ./out
```

## Knobs (env, set in the kernel or as Kaggle params)
- `CRISPASR_REF` (default `main`) — branch/tag to build.
- `DURATIONS` (default `60,120,225`) — jfk-tiled clip lengths (s) for the sweep.
- `FREE_MB` (default `2600`) — simulated small-card free VRAM (torch hog) for the OOM test.
- `MODEL_REPO`/`MODEL_FILE` — default `cstr/parakeet-tdt-1.1b-GGUF` q4_k.

## Reads
- **Check 1**: estimate vs measured single-pass peak VRAM (ratio ~1 validates the coeff).
- **Check 2**: `CRISPASR_PARAKEET_VRAM_BUDGET_MB` → streamed → peak « single-pass, transcript full.
- **Check 3**: under a torch VRAM-hog (~`FREE_MB` free), single-pass OOMs, the budget policy
  completes with a full transcript — the direct proof it fixes the reporter's OOM.

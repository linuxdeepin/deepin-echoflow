# CUDA cross-check — windowed FastConformer attention

Validates the TRUE windowed (block sliding-chunks) local-attention path on a real
NVIDIA GPU — the one backend that can't be checked on the M1 dev box.

`crispasr-windowed-attn-cuda.py` builds CrispASR with CUDA, downloads
`cstr/parakeet-tdt-0.6b-v3-GGUF` (a `rel_pos_local_attn`-capable NeMo model),
concatenates `samples/jfk.wav` into a long single-pass clip (forced via
`CRISPASR_PARAKEET_STREAM_THRESHOLD`), and runs three configs, each with an
`nvidia-smi` peak-memory poller:

| config | env | args |
|---|---|---|
| masked_full_local | `CRISPASR_FC_WINDOWED_ATTN=0` | `--att-context 64,64` |
| windowed_local    | `CRISPASR_FC_WINDOWED_ATTN=1` (default) | `--att-context 64,64` |
| full_attention    | — | — |

**Pass criteria:** `windowed_local` transcript == `masked_full_local` transcript
(kernel exits non-zero otherwise) and `windowed_local` GPU peak ≤ `masked_full_local`
(the O(T²) rel-pos bias BD becomes O(T·window)). On CUDA the masked-full local path
uses the manual QK^T attention (`fc_gpu_manual_attn` default-on for CUDA); windowed
has its own path — parity is expected up to GEMM ULP, transcript-identical.

## Run

Kaggle: GPU accelerator ON, Internet ON, Internet HF token optional (repo public).
Upload as a script kernel and run. Tunables via env: `CRISPASR_BRANCH` (default
`main`), `ATT_CONTEXT` (default `64,64`), `CLIP_REPEAT` (default `60` ≈ 660s,
T≈8k). Progress mirrors to `cstr/crispasr-kaggle-progress`.

Expected (from the M1/Metal A/B, to be confirmed on CUDA): transcripts identical;
windowed lower peak GPU memory at large T; windowed ≥ as fast as masked-full.

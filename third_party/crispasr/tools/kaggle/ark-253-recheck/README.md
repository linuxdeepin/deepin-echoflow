# ARK-ASR issue #253 investigation kernels

Kaggle GPU kernels used to diagnose issue #253 (ARK dropping transcriptions /
looping / leaking special tokens on long real-world audio). Each is a standalone
`kernel_type: script` — build crispasr from main, pull the ark GGUF + the
reporter's clip, and probe one question. Set `kernel-metadata.json`'s `code_file`
to the one you want and `kaggle kernels push -p .` ($KAGGLE_ACCOUNT; needs GPU quota).

| script | question it answers |
|--------|---------------------|
| `ark_253_recheck.py`   | reproduce reporter's default + `--chunk-seconds 7`; loop / empty-window / special-token-leak analysis (the canonical repro) |
| `ark_253_loopcause.py` | default vs `NO_CHUNK_CONTEXT` vs `NO_EOS_SUPPRESS` — isolates the cross-chunk seed as the loop cause |
| `ark_253_diff.py`      | C++ vs the original Python model per 30s window (prompt IDs + loop) |
| `ark_253_pyverify.py`  | Python reference with proper (anti-aliased) resampling + verified audio tensor |
| `ark_253_control.py`   | Python on clean jfk (invocation control) + Whisper ground truth |
| `ark_253_faithful.py`  | C++ default vs faithful (`NO_EOS_SUPPRESS`+`NO_CHUNK_CONTEXT`) on jfk / tot(orig #253) / t501 |
| `ark_253_dtype.py`     | fp32 vs bf16 Python (rules out a CPU-bf16 artifact) |

**Conclusion:** ARK is brittle on multi-speaker/music-mixed audio (spuriously
emits EOS→empty — a model limit, confirmed vs the Python blueprint + C++
F16-no-suppress). The EOS-suppression+seed force out REAL content (Parakeet-
verified), not hallucination. Fixes: strip special tokens + `ark_deloop`
(`7439f4d1`/`2d4922ab8`/`d76cce027`). See HISTORY 2026-07-13.

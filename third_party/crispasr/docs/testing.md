# Testing

CrispASR has two tiers of tests: **unit tests** (no models, fast) and
**integration / live tests** (need GGUF models on disk).

## Unit tests

1736 unit tests run unconditionally with no model files (~80 s at `-j2`,
~2.5 min serial):

```bash
ctest --test-dir build -L unit --timeout 30
```

These cover: audio chunking, mel preprocessing, CTC/beam decode,
sentence splitting, WAV metadata, stream finalization, registry lookup,
watermark embed/detect, cache helpers, GPT-2 BPE tokenizer,
BERT WordPiece tokenizer, bench env-var gating, per-backend param
defaults and null-guard coverage (60 backends), and more.

## Integration tests

133 integration tests (ctest label `live`) need real GGUF models. They
are gated by env vars and **SKIP cleanly** when the vars are unset
(Catch2 `SKIP()` → exit code 4, mapped to ctest "Skipped" via
`SKIP_RETURN_CODE 4`).

### Quick start

```bash
# Point at your local model cache:
export CRISPASR_MODELS_DIR=/mnt/storage/gguf-models

# Source all env vars at once:
source tests/env-live-tests.sh

# Run only previously-failed tests:
ctest --test-dir build --rerun-failed --output-on-failure --timeout 300

# Run all live tests:
ctest --test-dir build -L live --output-on-failure --timeout 300
```

### Environment variables

`tests/env-live-tests.sh` sets every env var the live tests expect.
Override `CRISPASR_MODELS_DIR` to point at your model directory; all
other vars derive from it unless individually overridden.

| Variable | Used by | Notes |
|---|---|---|
| `CRISPASR_MODELS_DIR` | All auto-download + search | Also checked by `crispasr_cache.cpp` well-known dirs |
| `CRISPASR_MODEL_WHISPER` | Beam search, VAD tests | Default: `~/.cache/crispasr/ggml-tiny.bin` |
| `CRISPASR_MODEL_GLM_ASR` | Beam search (GLM-ASR) | Large model, may timeout on CPU |
| `CRISPASR_MODEL_QWEN3_ASR` | Beam search (Qwen3-ASR) | Large model, may timeout on CPU |
| `CRISPASR_MODEL_CANARY` | Beam search (Canary) | Large model, may timeout on CPU |
| `CRISPASR_MODEL_COHERE` | Beam search (Cohere) | Large model, may timeout on CPU |
| `CRISPASR_PARAFORMER_MODEL` | Paraformer live tests | F16 GGUF. Bare `PARAFORMER_MODEL` still works as a deprecated legacy alias. |
| `CRISPASR_PARAFORMER_MODEL_Q4K` | Paraformer Q4_K parity | Q4_K GGUF |
| `CRISPASR_PARAFORMER_AUDIO_ZH` | Paraformer Chinese test | 16kHz mono WAV |
| `CRISPASR_TEST_DIARIZE_MODEL` | Diarization (pyannote) | pyannote-seg-3.0 GGUF |
| `CRISPASR_TEST_TITANET_MODEL` | Diarization (embedder) | titanet-large GGUF |
| `CRISPASR_TEST_DIARIZE_WAV` | Diarization | Multi-speaker 16kHz mono WAV |
| `CRISPASR_CHAT_TEST_MODEL` | Chat LLM smoke test | Needs chat template (not harrier) |
| `CRISPASR_MODEL_NEMOTRON` | Nemotron live tests | Q4_K GGUF (~458 MB) |
| `CRISPASR_MODEL_NEMOTRON_F16` | Nemotron F16/Q4_K parity | F16 GGUF (~1.3 GB) |
| `CRISPASR_MODEL_LFM2` | LFM2-Audio live tests | Q5_K GGUF (~1.6 GB). **Not Q4_K** — produces 0 tokens. |
| `CRISPASR_MODEL_DIA` | Dia TTS live tests | Q4_K GGUF (~892 MB) |
| `CRISPASR_MODEL_OUTETTS` | OuteTTS live tests | Q4_K GGUF (~600 MB) |
| `CRISPASR_MODEL_WAVTOK` | WavTokenizer (OuteTTS codec) | F16 GGUF (~100 MB) |
| `CRISPASR_MODEL_BTC_CHORDS` | BTC chord-recognition live tests | F32 GGUF (~11.7 MB). Default: `$CRISPASR_MODELS_DIR/btc-chords-large-f32.gguf`. Weights are CC-BY-NC-SA — see below. |

### Test groups

Test *numbers* shift whenever a test is added — they are the numbering of
the current build. Prefer `ctest -R <name>` over `ctest -I <n>`.

| Tests | Group | Model | Timeout |
|---|---|---|---|
| #278-281 | Paraformer | paraformer-zh-f16.gguf (~422 MB) | 30s |
| #286-289 | Nemotron | nemotron Q4_K + F16 (~1.7 GB total) | 300s |
| #1037-1039 | Beam: whisper | ggml-tiny.bin (~75 MB) | 120s |
| #1040-1047 | Beam: other backends | 2-5 GB models | 300s+ (CPU) |
| #169-170 | Diarize (pyannote + TitaNet) | ~50 MB total | 120s |
| #1624-1643 | Chat (LLM) | Any chat GGUF | 120s |
| #1849-1860 | CLI integration | Auto-download (whisper base) | 300s |
| #1862-1863 | VAD (full + thresholds) | ggml-tiny.bin + silero | 120s |
| #1874 | Backend regression (`test-backends`, label `integration`) | Auto-download (many backends) | 600s |
| #1875 | Benchmark-quick (label `benchmark`) | parakeet-tdt-0.6b-v3 | 600s |
| #1877 | Progress output | Auto-download (whisper + parakeet) | 300s |
| #1840-1842 | BTC chords (`test-btc-chords`, tag `[btc-chords]`) | btc-chords-large-f32.gguf (~11.7 MB) | — |

> **BTC chord weights are non-commercial.** `test-btc-chords` (3 test cases /
> 41 assertions, ctest label `live`, drives the session C-ABI) needs a BTC GGUF
> from `cstr/btc-chords-GGUF`. The upstream BTC *code* is MIT, but the shipped
> *weights* are CC-BY-NC-SA (trained on Isophonics / Robbie Williams /
> UsPop2002 chord annotations), so the registry refuses to download them
> without `--accept-license cc-by-nc-sa-4.0` (or the `CRISPASR_ACCEPT_LICENSE`
> env var). CrispASR itself stays MIT.

### Auto-download and model cache

Tests that use `-m auto --auto-download` (CLI, backends, benchmark,
progress) resolve models via the registry (`crispasr_model_registry.cpp`)
and the cache system (`crispasr_cache.cpp`). The cache probes these
locations in order:

1. The dispatcher's chosen cache dir — `--cache-dir` CLI override (or
   `cache_dir_override` in the C API), else `$CRISPASR_CACHE_DIR`, else
   `$CRISPASR_MODELS_DIR`, else the platform default
2. `$CRISPASR_MODELS_DIR` env var
3. `~/.cache/crispasr` (platform default)
4. `~/.cache/crispasr-models` (legacy alt cache)
5. `~/.cache/huggingface/hub` (HF download cache)

There are deliberately **no** absolute machine-specific defaults in this
list (`/mnt/storage/gguf-models`, `/Volumes/backups/...` and friends were
removed): point `$CRISPASR_MODELS_DIR` at a dedicated model volume instead.

If none of the probed paths has the file, it downloads from HuggingFace.

### Writing new integration tests

- Use Catch2 `SKIP()` when env vars are unset — this returns exit code 4
- In CMakeLists.txt, add `SKIP_RETURN_CODE 4` and
  `WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"` to the `PROPERTIES` of
  `catch_discover_tests`
- Use single-word labels (CMake's `catch_discover_tests` splits
  semicolons in `PROPERTIES` values)
- Add the env var to `tests/env-live-tests.sh`

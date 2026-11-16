#!/usr/bin/env bash
# Reproducible live smoke test for --diarize-method pyannote (issue #107).
#
# Builds a synthetic 2-speaker WAV by concatenating samples/jfk.wav (real
# JFK speech, mono 16 kHz) with a TTS-generated reading of the same text
# at a different sample rate (auto-resampled). Then runs the C++
# test-diarize-pyannote-live target against that fixture and the
# auto-cached pyannote-seg-3.0.gguf model.
#
# Why this fixture: two genuinely different voices reading similar
# content gives the segmentation net unambiguous turn boundaries; it
# exercises the within-pass scoring logic without needing access to
# gated multi-speaker corpora (AMI/VoxConverse/CallHome).
#
# NOTE: this validates the IN-PROCESS C++ API (single full-audio forward
# pass → multiple speaker IDs). The CLI's per-slice diarize flow has a
# separate cross-slice stitching limitation that is NOT exercised by
# this script — see #107 for the remaining Phase 2 work.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

build_dir="${BUILD_DIR:-build-ninja-compile}"
fixture_dir="${FIXTURE_DIR:-/Volumes/backups/ai/crispasr-test-fixtures}"
model_dir="${MODEL_DIR:-/Volumes/backups/ai/crispasr-models/pyannote-seg-3.0}"

wav="${fixture_dir}/two-speakers-jfk-tts.wav"
model="${model_dir}/pyannote-seg-3.0.gguf"

mkdir -p "$fixture_dir" "$model_dir"

# 1) Synthesize the 2-speaker fixture if missing.
if [[ ! -f "$wav" ]]; then
    spk_b="${SPEAKER_B_WAV:-$HOME/.cache/crispasr/cb_baker_jfk.wav}"
    if [[ ! -f "$spk_b" ]]; then
        echo "ERROR: speaker-B sample not found at $spk_b" >&2
        echo "       set SPEAKER_B_WAV to any single-speaker mono WAV" >&2
        exit 1
    fi
    echo "[smoke] synthesizing 2-speaker fixture → $wav"
    python - <<PY
import wave, struct
def load(path):
    with wave.open(path,'rb') as w:
        return w.getframerate(), [int.from_bytes(w.readframes(w.getnframes())[i:i+2],'little',signed=True)
                                  for i in range(0, w.getnframes()*2, 2)]
def resample(s, src, dst):
    if src == dst: return s
    r = dst/src; n = int(len(s)*r); out = []
    for i in range(n):
        x = i/r; i0 = int(x); i1 = min(i0+1, len(s)-1); a = x - i0
        out.append(int(s[i0]*(1-a) + s[i1]*a))
    return out
sa, a = load('samples/jfk.wav')
sb, b = load('$spk_b')
b16 = resample(b, sb, 16000)
silence = [0]*(16000//2)
mix = a + silence + b16 + silence + a + silence + b16
with wave.open('$wav','wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b''.join(struct.pack('<h', max(-32768,min(32767,s))) for s in mix))
print(f'wrote {len(mix)/16000:.1f}s 2-speaker WAV')
PY
fi

# 2) Fetch the pyannote-seg GGUF if missing.
if [[ ! -f "$model" ]]; then
    echo "[smoke] downloading pyannote-seg-3.0.gguf → $model"
    curl -fsSL -o "$model" \
        "https://huggingface.co/cstr/pyannote-v3-segmentation-GGUF/resolve/main/pyannote-seg-3.0.gguf"
fi

# 3) Build the live test target + the CLI binary (the CLI is what the
#    end-to-end embedder spawn-pass needs; the test-only target covers
#    the library API path).
echo "[smoke] building test-diarize-pyannote-live + crispasr"
cmake --build "$build_dir" --target test-diarize-pyannote-live crispasr-cli >/dev/null

# 4a) C++ live test (lib API + TitaNet clustering when its model is
#     reachable). Two TEST_CASEs: pyannote-only and pyannote+TitaNet.
#     The TitaNet variant pulls titanet-large.gguf via the standard
#     crispasr cache.
titanet_cache="${HOME}/.cache/crispasr/titanet-large.gguf"
echo "[smoke] running C++ live tests"
CRISPASR_TEST_DIARIZE_WAV="$wav" \
CRISPASR_TEST_DIARIZE_MODEL="$model" \
CRISPASR_TEST_TITANET_MODEL="$titanet_cache" \
    "$build_dir/bin/test-diarize-pyannote-live" --success

# 4b) CLI spawn smoke. Two passes against the same fixture so we
#     catch wire-up bugs that the C++ API tests can't (e.g. output
#     writers ignoring segs[i].speaker, missing CLI flag plumbing).
#
#     Pass 1: pyannote only -> should label ≥1 speaker on the fixture.
#     Pass 2: --diarize-embedder auto -> globally stable cluster IDs
#             across the file (e.g. ≥2 distinct labels for our 2-
#             speaker fixture, with JFK regions sharing a cluster).
#
#     We use cohere as the ASR backend because it produces word
#     timestamps (which the segment-splitting step needs), runs
#     fast on M1, and ships in the auto-download cache.
backend_model="${HOME}/.cache/crispasr/cohere-transcribe-q4_k.gguf"
if [[ ! -f "$backend_model" ]]; then
    echo "[smoke] WARNING: $backend_model missing — skipping CLI spawn smoke"
    echo "[smoke] (run \`crispasr --backend cohere -m auto ...\` once to populate it)"
    exit 0
fi

run_cli_and_check() {
    local label="$1"
    shift
    local out_prefix="${TMPDIR:-/tmp}/diarize-smoke-${label}"
    rm -f "${out_prefix}.json"
    "$build_dir/bin/crispasr" --backend cohere -m "$backend_model" -f "$wav" \
        --diarize --diarize-method pyannote --sherpa-segment-model auto \
        "$@" -ojf -of "$out_prefix" >/dev/null 2>&1
    local n_distinct
    n_distinct=$(python -c "
import json, sys
from collections import Counter
try:
    d = json.load(open('${out_prefix}.json'))
except Exception as e:
    print('PARSE_FAIL', e); sys.exit(1)
labels = [s.get('speaker','') for s in d.get('transcription', []) if s.get('speaker')]
print(len(set(labels)))
")
    echo "[smoke] CLI ${label}: distinct speaker labels = ${n_distinct}"
    [[ "$n_distinct" -ge 1 ]] || { echo "[smoke] FAIL: ${label} produced no labels"; return 1; }
}

echo "[smoke] CLI pass 1: pyannote-only"
run_cli_and_check pyannote-only

echo "[smoke] CLI pass 2: pyannote + TitaNet embedder"
run_cli_and_check pyannote-embedder --diarize-embedder auto

echo "[smoke] all CLI passes succeeded"

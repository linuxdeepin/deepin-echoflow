#!/bin/bash
# test-vibevoice-base-clone.sh — regression smoke for VibeVoice 1.5B base TTS.
#
# Exercises the base-model TTS path both without a clone reference and with a
# WAV reference supplied via VIBEVOICE_VOICE_AUDIO. The regression we care
# about is that the clone path must materially change the generated audio rather
# than collapsing to the model's generic prior.
#
# Usage:
#   ./tests/test-vibevoice-base-clone.sh [--keep-files]
#
# Requires:
#   - build/bin/crispasr (or build-ninja-compile/bin/crispasr)
#   - network access on first run, so -m auto can fetch cstr/vibevoice-1.5b-GGUF
#
# Exit code: 0 if all pass, non-zero otherwise.

set -euo pipefail
cd "$(dirname "$0")/.."

KEEP_FILES=0
for arg in "$@"; do
    case "$arg" in
        --keep-files) KEEP_FILES=1 ;;
    esac
done

CRISPASR=""
for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
done
if [ -z "$CRISPASR" ]; then
    echo "ERROR: crispasr binary not found. Build first."
    exit 2
fi

REF_WAV="samples/jfk_24k.wav"
if [ ! -f "$REF_WAV" ]; then
    echo "SKIP: $REF_WAV not found"
    exit 0
fi

TMPDIR=$(mktemp -d -t crispasr-vibevoice.XXXXXX)
trap 'if [ "$KEEP_FILES" -eq 0 ]; then rm -rf "$TMPDIR"; fi' EXIT

TEXT="This is a VibeVoice clone regression test."
NO_CLONE="$TMPDIR/no-clone.wav"
CLONE="$TMPDIR/clone.wav"
NO_CLONE_LOG="$TMPDIR/no-clone.log"
CLONE_LOG="$TMPDIR/clone.log"

run_tts() {
    local out_wav="$1"
    local log_file="$2"

    "$CRISPASR" --backend vibevoice-1.5b -m auto \
        --tts "$TEXT" \
        --tts-output "$out_wav" \
        --no-prints \
        >"$log_file" 2>&1
}

fail() {
    echo "ERROR: $*"
    echo
    echo "--- no-clone log ---"
    cat "$NO_CLONE_LOG" || true
    echo
    echo "--- clone log ---"
    cat "$CLONE_LOG" || true
    exit 1
}

echo "Running VibeVoice 1.5B without clone reference..."
if ! VIBEVOICE_TTS_SEED=42 run_tts "$NO_CLONE" "$NO_CLONE_LOG"; then
    fail "no-clone synthesis failed"
fi

echo "Running VibeVoice 1.5B with WAV clone reference..."
if ! VIBEVOICE_TTS_SEED=42 VIBEVOICE_VOICE_AUDIO="$REF_WAV" run_tts "$CLONE" "$CLONE_LOG"; then
    fail "clone synthesis failed"
fi

for f in "$NO_CLONE" "$CLONE"; do
    if [ ! -f "$f" ]; then
        fail "missing output file $f"
    fi
    size=$(wc -c <"$f")
    if [ "$size" -le 44 ]; then
        fail "output file $f is too small ($size bytes)"
    fi
done

if cmp -s "$NO_CLONE" "$CLONE"; then
    fail "clone output is byte-identical to no-clone output"
fi

echo "PASS: clone and no-clone outputs both produced audio and differ."

#!/bin/bash
# test-server-punc.sh — integration test for `--punc-model` under `--server`.
#
# Regression guard for PR #166: a server started with --punc-model and a
# non-PnC backend (parakeet RNNT/CTC emits no punctuation natively) used to
# return lowercase, unpunctuated text from /v1/audio/transcriptions because the
# punctuation post-processor lived only in the CLI layer. This boots such a
# server and asserts the transcription comes back punctuated, then confirms a
# per-request punctuation=false still strips it.
#
# Usage:
#   ./tests/test-server-punc.sh [--port N] [--keep-server]
#
# Requires (else SKIP, exit 0):
#   - build/bin/crispasr
#   - a parakeet RNNT/CTC GGUF, found via $PARAKEET_MODEL, or a *parakeet*.gguf
#     under $CRISPASR_MODELS / $CRISPASR_MODELS_DIR / common local dirs
#   - samples/jfk.wav (shipped in the repo)

set -uo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-11453}
KEEP_SERVER=0
CACHE_DIR="${CRISPASR_TEST_CACHE:-}"
for arg in "$@"; do
    case "$arg" in
        --port=*) PORT="${arg#--port=}" ;;
        --port) shift; PORT="$1" ;;
        --cache-dir=*) CACHE_DIR="${arg#--cache-dir=}" ;;
        --keep-server) KEEP_SERVER=1 ;;
    esac
done
CACHE_ARG=()
[ -n "$CACHE_DIR" ] && CACHE_ARG=(--cache-dir "$CACHE_DIR")

# Locate the binary.
CRISPASR=""
for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
done
if [ -z "$CRISPASR" ]; then
    echo "SKIP: crispasr binary not found (build first)"
    exit 0
fi

# Locate a non-PnC ASR model. Prefer an explicit $PARAKEET_MODEL, else scan
# the usual model dirs for a parakeet GGUF (RNNT or CTC — both lack native
# punctuation, which is exactly the case this fix targets).
MODEL="${PARAKEET_MODEL:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    MODEL=""
    for d in "${CRISPASR_MODELS:-}" "${CRISPASR_MODELS_DIR:-}" \
             /Volumes/backups/ai/crispasr-models /Volumes/backups/ai/crispasr \
             /mnt/storage/gguf-models "$HOME/.cache/crispasr"; do
        [ -n "$d" ] && [ -d "$d" ] || continue
        cand=$(ls "$d"/*parakeet*.gguf 2>/dev/null | grep -viE "tokenizer|vocab" | head -1)
        if [ -n "$cand" ] && [ -f "$cand" ]; then MODEL="$cand"; break; fi
    done
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP: no parakeet GGUF found (set PARAKEET_MODEL or CRISPASR_MODELS_DIR)"
    exit 0
fi

SAMPLE="samples/jfk.wav"
if [ ! -f "$SAMPLE" ]; then
    echo "SKIP: $SAMPLE not found"
    exit 0
fi

echo "Model:  $MODEL"
echo "Sample: $SAMPLE"

# Boot the server with FireRedPunc restoration enabled.
SERVER_LOG=$(mktemp -t crispasr-punc-server.XXXXXX)
trap 'if [ "$KEEP_SERVER" -eq 0 ] && [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; rm -f "$SERVER_LOG"' EXIT

echo "Starting crispasr-server on :$PORT with --punc-model fullstop…"
"$CRISPASR" --server -m "$MODEL" --punc-model fullstop \
    --host 127.0.0.1 --port "$PORT" --auto-download ${CACHE_ARG[@]+"${CACHE_ARG[@]}"} \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for /health (model load + punc model download/load can be slow cold).
ready=0
for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server died during startup. Log:"; cat "$SERVER_LOG"; exit 2
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    echo "ERROR: server not ready within 240s. Log:"; cat "$SERVER_LOG"; exit 2
fi

# Confirm the server actually loaded a punctuation model (the fix). If the
# operator passed --punc-model but it silently no-op'd, that's the bug.
if ! grep -qiE "loaded punctuation model|loaded PCS model" "$SERVER_LOG"; then
    echo "ERROR: server did not load a punctuation model despite --punc-model. Log:"
    cat "$SERVER_LOG"; exit 1
fi

PASS=0; FAIL=0

# --- 1. Default request: punctuation should be restored -------------------
resp=$(curl -s -F "file=@$SAMPLE" -F "language=en" -F "beam_size=1" \
    "http://127.0.0.1:$PORT/v1/audio/transcriptions")
text=$(echo "$resp" | sed -n 's/.*"text"[[:space:]]*:[[:space:]]*"\(.*\)".*/\1/p')
echo "transcription (punctuated): $text"

if echo "$text" | grep -q "[.?!]"; then
    echo "  ✓ output contains sentence punctuation"; PASS=$((PASS+1))
else
    echo "  ✗ output has no sentence punctuation: '$text'"; FAIL=$((FAIL+1))
fi
if echo "$text" | grep -q "[A-Z]"; then
    echo "  ✓ output contains capitalization"; PASS=$((PASS+1))
else
    echo "  ✗ output has no capitalization: '$text'"; FAIL=$((FAIL+1))
fi

# --- 2. punctuation=false: should be stripped back to plain text ----------
resp2=$(curl -s -F "file=@$SAMPLE" -F "language=en" -F "beam_size=1" -F "punctuation=false" \
    "http://127.0.0.1:$PORT/v1/audio/transcriptions")
text2=$(echo "$resp2" | sed -n 's/.*"text"[[:space:]]*:[[:space:]]*"\(.*\)".*/\1/p')
echo "transcription (punctuation=false): $text2"
if echo "$text2" | grep -q "[.?!,]"; then
    echo "  ✗ punctuation=false still has punctuation: '$text2'"; FAIL=$((FAIL+1))
else
    echo "  ✓ punctuation=false strips punctuation"; PASS=$((PASS+1))
fi

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1

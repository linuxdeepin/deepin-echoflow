#!/bin/bash
# test-server-warmup.sh — integration test for server startup warmup + the
# --no-warmup opt-out (#165).
#
# #165: on some Vulkan drivers the always-on server warmup transcribe crashes
# or hangs, so the server never reaches listen(). This test verifies:
#   1. the server reaches /health and transcribes with default warmup, and
#   2. the same with --no-warmup (the escape hatch), with the log confirming
#      "warmup skipped".
# It uses a small auto-downloadable model (moonshine tiny, ~22 MB) so it runs
# without any local model store. Run it once per backend build (CPU / Metal /
# Vulkan) to exercise the warmup path on that backend.
#
# Usage:
#   ./tests/test-server-warmup.sh [--port N] [--binary path/to/crispasr]
#
# SKIPs (exit 0) if the binary is missing or the model can't be fetched.

set -uo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-11461}
CRISPASR=""
# Optional model cache dir (env CRISPASR_TEST_CACHE or --cache-dir=). Useful when
# the default cache (~/.cache/crispasr) is unwritable; the model is fetched here.
CACHE_DIR="${CRISPASR_TEST_CACHE:-}"
for arg in "$@"; do
    case "$arg" in
        --port=*) PORT="${arg#--port=}" ;;
        --binary=*) CRISPASR="${arg#--binary=}" ;;
        --cache-dir=*) CACHE_DIR="${arg#--cache-dir=}" ;;
    esac
done
CACHE_ARG=()
[ -n "$CACHE_DIR" ] && CACHE_ARG=(--cache-dir "$CACHE_DIR")

if [ -z "$CRISPASR" ]; then
    for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
        if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
    done
fi
if [ -z "$CRISPASR" ] || [ ! -x "$CRISPASR" ]; then
    echo "SKIP: crispasr binary not found (build first)"; exit 0
fi

SAMPLE="samples/jfk.wav"
[ -f "$SAMPLE" ] || { echo "SKIP: $SAMPLE not found"; exit 0; }

echo "Binary: $CRISPASR"
echo "Backends: $("$CRISPASR" --version 2>&1 | tr '\n' ' ')"

PASS=0; FAIL=0
SERVER_PID=""
SERVER_LOG=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    [ -n "$SERVER_LOG" ] && rm -f "$SERVER_LOG"
}
trap cleanup EXIT

# boot_server <extra-args...> ; sets SERVER_PID + SERVER_LOG, waits for /health.
# Returns 0 if ready, 1 if it never became ready (the #165 symptom).
boot_server() {
    cleanup; SERVER_PID=""
    SERVER_LOG=$(mktemp -t crispasr-warmup.XXXXXX)
    "$CRISPASR" --server -m moonshine --auto-download ${CACHE_ARG[@]+"${CACHE_ARG[@]}"} \
        --host 127.0.0.1 --port "$PORT" "$@" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then return 0; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "    server process exited. Log tail:"; tail -5 "$SERVER_LOG" | sed 's/^/      /'
            return 1
        fi
        sleep 1
    done
    echo "    server never became ready within 240s (the #165 symptom). Log tail:"
    tail -5 "$SERVER_LOG" | sed 's/^/      /'
    return 1
}

transcribes() {
    local resp
    resp=$(curl -s -F "file=@$SAMPLE" -F "language=en" "http://127.0.0.1:$PORT/v1/audio/transcriptions")
    echo "$resp" | grep -qiE '"text"[[:space:]]*:[[:space:]]*"[^"]*[a-z]'
}

# --- 1. default warmup ----------------------------------------------------
echo "=== default (warmup on) ==="
if boot_server && transcribes; then
    if grep -q "warmup completed" "$SERVER_LOG"; then
        echo "  ✓ server warms up and transcribes"; PASS=$((PASS+1))
    else
        echo "  ✓ server transcribes (no warmup-completed log line)"; PASS=$((PASS+1))
    fi
else
    echo "  ✗ server with default warmup did not come up / transcribe"; FAIL=$((FAIL+1))
fi

# --- 2. --no-warmup escape hatch ------------------------------------------
echo "=== --no-warmup ==="
if boot_server --no-warmup && transcribes; then
    if grep -q "warmup skipped" "$SERVER_LOG"; then
        echo "  ✓ --no-warmup skips warmup and still transcribes"; PASS=$((PASS+1))
    else
        echo "  ✗ --no-warmup did not log 'warmup skipped'"; FAIL=$((FAIL+1))
    fi
else
    echo "  ✗ server with --no-warmup did not come up / transcribe"; FAIL=$((FAIL+1))
fi

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1

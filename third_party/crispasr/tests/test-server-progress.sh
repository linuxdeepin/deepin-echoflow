#!/bin/bash
# test-server-progress.sh — integration test for `GET /progress` (#408).
#
# Boots a server with --chunk-seconds 2 so samples/jfk.wav (~11 s) decodes as
# several slices, POSTs a transcription in the background, and polls /progress
# while it runs. Asserts the documented contract:
#   idle    -> {"busy": false, "progress": -1}
#   running -> busy true with progress in 0..100 at least once
#   done    -> idle again (a finished job must reset the state)
# Then restarts the server with an API key and asserts /progress is
# auth-gated (401 without a key, 200 with) while /health stays public.
#
# Usage:
#   ./tests/test-server-progress.sh [--port N] [--keep-server]
#
# Requires (else SKIP, exit 0):
#   - build/bin/crispasr
#   - a small whisper ggml .bin or moonshine GGUF, found via $CRISPASR_TEST_ASR
#     or scanned from the usual model dirs
#   - samples/jfk.wav (shipped in the repo)

set -uo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-11457}
KEEP_SERVER=0
for arg in "$@"; do
    case "$arg" in
        --port=*) PORT="${arg#--port=}" ;;
        --keep-server) KEEP_SERVER=1 ;;
    esac
done

CRISPASR=""
for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
done
if [ -z "$CRISPASR" ]; then
    echo "SKIP: crispasr binary not found (build first)"
    exit 0
fi

MODEL="${CRISPASR_TEST_ASR:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    MODEL=""
    for d in "${CRISPASR_MODELS:-}" "${CRISPASR_MODELS_DIR:-}" \
             /Volumes/backups/ai/crispasr-models /mnt/storage/gguf-models \
             "$HOME/.cache/crispasr"; do
        [ -n "$d" ] && [ -d "$d" ] || continue
        if [ -f "$d/ggml-tiny.bin" ]; then MODEL="$d/ggml-tiny.bin"; break; fi
        if [ -f "$d/moonshine-tiny-q8_0.gguf" ] && [ -f "$d/tokenizer.bin" ]; then
            MODEL="$d/moonshine-tiny-q8_0.gguf"
            break
        fi
    done
fi
if [ -z "$MODEL" ]; then
    echo "SKIP: no small ASR model found (set CRISPASR_TEST_ASR)"
    exit 0
fi
if [ ! -f samples/jfk.wav ]; then
    echo "SKIP: samples/jfk.wav missing"
    exit 0
fi

command -v curl > /dev/null || {
    echo "SKIP: curl not available"
    exit 0
}

SERVER_LOG=$(mktemp -t crispasr-progress-server.XXXXXX)
RESP_FILE=$(mktemp -t crispasr-progress-resp.XXXXXX)

SERVER_PID=""
cleanup() {
    if [ "$KEEP_SERVER" != "1" ] && [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2> /dev/null; fi
    wait 2> /dev/null
    rm -f "$SERVER_LOG" "$RESP_FILE"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $1"
    exit 1
}

wait_ready() {
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then return 0; fi
        sleep 0.5
    done
    return 1
}

start_server() { # extra args...
    "$CRISPASR" --server --host 127.0.0.1 --port "$PORT" -m "$MODEL" --chunk-seconds 2 -t 4 "$@" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_ready || fail "server did not become ready (log: $SERVER_LOG)"
}

echo "== model: $MODEL"
start_server

# 1. Idle contract.
IDLE=$(curl -sf "http://127.0.0.1:$PORT/progress") || fail "GET /progress failed while idle"
echo "$IDLE" | grep -q '"busy": false' || fail "idle busy != false: $IDLE"
echo "$IDLE" | grep -q '"progress": -1' || fail "idle progress != -1: $IDLE"
echo "ok: idle -> $IDLE"

# 2. Busy contract: transcribe in the background, poll while it runs.
curl -sf -X POST "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@samples/jfk.wav" > "$RESP_FILE" &
POST_PID=$!

SAW_BUSY=0
SAW_RANGE=0
while kill -0 "$POST_PID" 2> /dev/null; do
    P=$(curl -sf "http://127.0.0.1:$PORT/progress" || true)
    if echo "$P" | grep -q '"busy": true'; then
        SAW_BUSY=1
        N=$(echo "$P" | sed -n 's/.*"progress": \(-\{0,1\}[0-9]\{1,\}\).*/\1/p')
        if [ -n "$N" ] && [ "$N" -ge 0 ] && [ "$N" -le 100 ]; then SAW_RANGE=1; fi
    fi
    sleep 0.2
done
wait "$POST_PID" || fail "transcription POST failed"
grep -q '[A-Za-z]' "$RESP_FILE" || fail "empty transcription response"
[ "$SAW_BUSY" = "1" ] || fail "never observed busy:true during a running job"
[ "$SAW_RANGE" = "1" ] || fail "never observed progress in 0..100 while busy"
echo "ok: busy observed with in-range progress during the job"

# 3. Reset contract.
sleep 0.5
DONE=$(curl -sf "http://127.0.0.1:$PORT/progress") || fail "GET /progress failed after job"
echo "$DONE" | grep -q '"busy": false' || fail "post-job busy != false: $DONE"
echo "$DONE" | grep -q '"progress": -1' || fail "post-job progress != -1: $DONE"
echo "ok: state reset after the job -> $DONE"

# 4. Auth: /progress is gated like the other introspection routes,
#    /health stays public.
kill "$SERVER_PID" 2> /dev/null
wait 2> /dev/null
SERVER_PID=""
start_server --api-keys testkey123

CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/progress")
[ "$CODE" = "401" ] || fail "/progress without key returned $CODE, want 401"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer testkey123" \
    "http://127.0.0.1:$PORT/progress")
[ "$CODE" = "200" ] || fail "/progress with key returned $CODE, want 200"
CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")
[ "$CODE" = "200" ] || fail "/health with api-key set returned $CODE, want 200 (public)"
echo "ok: /progress auth-gated, /health public"

echo "PASS: GET /progress contract holds"

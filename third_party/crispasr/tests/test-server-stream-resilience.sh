#!/bin/bash
# test-server-stream-resilience.sh — the streaming TTS worker is a DETACHED
# thread; an exception escaping it would call std::terminate and crash the whole
# server. This test forces a worker exception (CRISPASR_TEST_STREAM_THROW=1 +
# the magic input "__throw_test__") and asserts the server survives and keeps
# serving — i.e. the worker's try/catch contains the failure.
#
# Auto-discovers a qwen3-tts GGUF (base or customvoice); SKIPs (exit 0) without
# one. Base model auto-discovers its voice-default + tokenizer siblings.

set -uo pipefail
cd "$(dirname "$0")/.."

CRISPASR=""
for cand in build/bin/crispasr build/bin/Release/crispasr.exe build-ninja-compile/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
done
if [ -z "$CRISPASR" ]; then
    echo "ERROR: crispasr binary not found. Build first."
    exit 2
fi

MODEL="${CRISPASR_TEST_QWEN3_TTS_MODEL:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    MODEL=""
    for dir in "${CRISPASR_MODELS:-}" "${CRISPASR_MODELS_DIR:-}" "$HOME/crispasr-live-cache" "$HOME/.cache/crispasr"; do
        [ -n "$dir" ] && [ -d "$dir" ] || continue
        for cand in "$dir"/qwen3-tts-*base*.gguf "$dir"/qwen3-tts-*customvoice*.gguf "$dir"/qwen3-tts-*.gguf; do
            case "$cand" in *tokenizer*|*codec*|*voice*) continue ;; esac
            [ -f "$cand" ] && { MODEL="$cand"; break; }
        done
        [ -n "$MODEL" ] && break
    done
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP: no qwen3-tts GGUF found (set CRISPASR_TEST_QWEN3_TTS_MODEL or CRISPASR_MODELS_DIR)"
    exit 0
fi
case "$MODEL" in
    *customvoice*|*-cv-*|*voicedesign*) BACKEND="qwen3-tts-customvoice" ;;
    *)                                  BACKEND="qwen3-tts" ;;
esac

PORT="${PORT:-11473}"
CACHE_DIR=$(dirname "$MODEL")
SERVER_LOG=$(mktemp -t crispasr-stream-resil.XXXXXX)
SERVER_PID=""
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; rm -f "$SERVER_LOG"' EXIT

echo "=== streaming worker exception-safety ==="
echo "model:   $MODEL"
echo "backend: $BACKEND"

# Start the server with the throw injection ARMED.
CRISPASR_TEST_STREAM_THROW=1 "$CRISPASR" --server --backend "$BACKEND" -m "$MODEL" \
    --cache-dir "$CACHE_DIR" --host 127.0.0.1 --port "$PORT" --no-prints > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for the HTTP server to bind (/health), then for the model to finish
# loading (/v1/audio/speech stops returning 503). A warmup synth can be slow
# under load (RTF ≫ 1), so allow a generous per-probe timeout — while loading the
# 503 comes back immediately; once loaded the (slow) synth returns 200.
for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "  FAIL server died during startup"; cat "$SERVER_LOG"; exit 1; fi
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done
ready=0
for _ in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "  FAIL server died during model load"; cat "$SERVER_LOG"; exit 1; fi
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 180 -X POST "http://127.0.0.1:$PORT/v1/audio/speech" \
        -H 'Content-Type: application/json' -d '{"input":"warmup","response_format":"pcm"}' 2>/dev/null)
    if [ "$code" != "503" ] && [ "$code" != "000" ]; then ready=1; break; fi
    sleep 2
done
[ "$ready" = 1 ] || { echo "  FAIL server not ready"; cat "$SERVER_LOG"; exit 1; }

PASS=0; FAIL=0

# Request A: stream the magic input → worker throws → must be caught (no crash).
curl -s -N -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' \
    -d '{"input":"__throw_test__","response_format":"pcm","stream":true}' -o /dev/null --max-time 30 || true
sleep 1

# The server process MUST still be alive after the worker exception.
if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  PASS server survived the streaming-worker exception"; PASS=$((PASS + 1))
else
    echo "  FAIL server process died after worker exception"; cat "$SERVER_LOG"; FAIL=$((FAIL + 1))
fi

# The worker's catch must have logged the exception.
if grep -q "streaming worker exception" "$SERVER_LOG"; then
    echo "  PASS worker exception was caught + logged"; PASS=$((PASS + 1))
else
    echo "  FAIL no 'streaming worker exception' log line"; FAIL=$((FAIL + 1))
fi

# Request B: a normal streamed request must still succeed (server kept serving).
TMP=$(mktemp -t crispasr-resil.XXXXXX.pcm)
code=$(curl -s -N -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' \
    -d '{"input":"The server is still alive and serving.","response_format":"pcm","stream":true}' \
    -o "$TMP" -w "%{http_code}" --max-time 180)
SZ=$(wc -c < "$TMP" | tr -d ' '); rm -f "$TMP"
if [ "$code" = "200" ] && [ "$SZ" -gt 1000 ]; then
    echo "  PASS subsequent streamed request served ($SZ bytes)"; PASS=$((PASS + 1))
else
    echo "  FAIL subsequent request failed (code=$code size=$SZ)"; FAIL=$((FAIL + 1))
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

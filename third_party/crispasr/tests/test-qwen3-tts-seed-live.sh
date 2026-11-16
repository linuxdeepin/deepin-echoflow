#!/bin/bash
# Live seed regression for qwen3-tts /v1/audio/speech.
#
# Requires:
#   CRISPASR_TEST_QWEN3_TTS_MODEL=/path/to/qwen3-tts-customvoice.gguf
#   CRISPASR_TEST_QWEN3_TTS_CODEC=/path/to/qwen3-tts-tokenizer-12hz.gguf
# Optional:
#   CRISPASR_TEST_QWEN3_TTS_VOICE=<speaker-name> (default: first model speaker)
#   PORT=11445

set -uo pipefail
cd "$(dirname "$0")/.."

CRISPASR=""
for cand in build-ninja-compile/bin/crispasr build/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
done
if [ -z "$CRISPASR" ]; then
    echo "ERROR: crispasr binary not found. Build first."
    exit 2
fi

MODEL="${CRISPASR_TEST_QWEN3_TTS_MODEL:-}"
CODEC="${CRISPASR_TEST_QWEN3_TTS_CODEC:-}"
VOICE="${CRISPASR_TEST_QWEN3_TTS_VOICE:-}"
PORT="${PORT:-11445}"

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP: set CRISPASR_TEST_QWEN3_TTS_MODEL to a qwen3-tts CustomVoice GGUF"
    exit 0
fi
if [ -z "$CODEC" ] || [ ! -f "$CODEC" ]; then
    echo "SKIP: set CRISPASR_TEST_QWEN3_TTS_CODEC to qwen3-tts-tokenizer-12hz.gguf"
    exit 0
fi

PASS=0
FAIL=0
SERVER_PID=""
SERVER_LOG=""
OUT_DIR=$(mktemp -d -t crispasr-qwen3-tts-seed.XXXXXX)
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; if [ -n "$SERVER_LOG" ]; then rm -f "$SERVER_LOG"; fi; rm -rf "$OUT_DIR"' EXIT

hash_file() {
    shasum -a 256 "$1" | awk '{print $1}'
}

assert_equal() {
    local name="$1" a="$2" b="$3"
    if [ "$a" = "$b" ]; then
        echo "  PASS $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $name"
        echo "       a: $a"
        echo "       b: $b"
        FAIL=$((FAIL + 1))
    fi
}

assert_not_equal() {
    local name="$1" a="$2" b="$3"
    if [ "$a" != "$b" ]; then
        echo "  PASS $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $name: both hashes were $a"
        FAIL=$((FAIL + 1))
    fi
}

post_speech() {
    local seed="$1" out="$2"
    local voice_json=""
    if [ -n "$VOICE" ]; then
        voice_json=", \"voice\": \"$VOICE\""
    fi
    curl -sf -X POST "http://127.0.0.1:$PORT/v1/audio/speech" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"tts-1\",\"input\":\"Good morning. This is a qwen seed test.\",\"response_format\":\"wav\",\"seed\":$seed$voice_json}" \
        -o "$out"
}

echo "=== qwen3-tts /v1/audio/speech seed regression ==="
SERVER_LOG=$(mktemp -t crispasr-qwen3-tts-seed-server.XXXXXX)
QWEN3_TTS_SEED=999 "$CRISPASR" --server --backend qwen3-tts-customvoice \
    -m "$MODEL" --codec-model "$CODEC" --host 127.0.0.1 --port "$PORT" --no-prints \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "  FAIL server died during startup"
        cat "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "  FAIL server did not become healthy"
    cat "$SERVER_LOG"
    exit 1
fi

post_speech 123 "$OUT_DIR/a.wav" || { echo "  FAIL request seed=123 #1"; cat "$SERVER_LOG"; exit 1; }
post_speech 123 "$OUT_DIR/b.wav" || { echo "  FAIL request seed=123 #2"; cat "$SERVER_LOG"; exit 1; }
post_speech 456 "$OUT_DIR/c.wav" || { echo "  FAIL request seed=456"; cat "$SERVER_LOG"; exit 1; }

HA=$(hash_file "$OUT_DIR/a.wav")
HB=$(hash_file "$OUT_DIR/b.wav")
HC=$(hash_file "$OUT_DIR/c.wav")

assert_equal "same request seed is bit-identical even when QWEN3_TTS_SEED is set" "$HA" "$HB"
assert_not_equal "different request seed changes qwen3-tts WAV hash" "$HA" "$HC"

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

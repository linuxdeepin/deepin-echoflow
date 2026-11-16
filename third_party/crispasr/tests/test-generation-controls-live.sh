#!/bin/bash
# Live smoke for unified autoregressive generation controls.
#
# Requires:
#   CRISPASR_TEST_GENERATION_MODEL=/path/to/ar-backend.gguf
# Optional:
#   CRISPASR_TEST_GENERATION_BACKEND=cohere|qwen3|granite|voxtral|voxtral4b
#   PORT=11443
#
# Skips cleanly when no model is configured.

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

MODEL="${CRISPASR_TEST_GENERATION_MODEL:-}"
BACKEND="${CRISPASR_TEST_GENERATION_BACKEND:-cohere}"
SAMPLE="${CRISPASR_TEST_GENERATION_AUDIO:-samples/jfk.wav}"
PORT="${PORT:-11443}"

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP: set CRISPASR_TEST_GENERATION_MODEL to an autoregressive ASR GGUF"
    exit 0
fi
if [ ! -f "$SAMPLE" ]; then
    echo "SKIP: sample audio not found: $SAMPLE"
    exit 0
fi

PASS=0
FAIL=0
SERVER_PID=""
SERVER_LOG=""
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; if [ -n "$SERVER_LOG" ]; then rm -f "$SERVER_LOG"; fi' EXIT

assert_contains() {
    local name="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -q -- "$needle"; then
        echo "  PASS $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $name"
        echo "       wanted: $needle"
        echo "       got:    $haystack"
        FAIL=$((FAIL + 1))
    fi
}

assert_nonempty() {
    local name="$1" value="$2"
    if [ -n "$value" ]; then
        echo "  PASS $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $name: empty output"
        FAIL=$((FAIL + 1))
    fi
}

assert_greater_len() {
    local name="$1" short="$2" long="$3"
    if [ "${#long}" -gt "${#short}" ]; then
        echo "  PASS $name (${#short} -> ${#long} chars)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $name: expected longer output"
        echo "       short: '$short' (${#short} chars)"
        echo "       long:  '$long' (${#long} chars)"
        FAIL=$((FAIL + 1))
    fi
}

assert_equal() {
    local name="$1" a="$2" b="$3"
    if [ "$a" = "$b" ]; then
        echo "  PASS $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $name"
        echo "       a: '$a'"
        echo "       b: '$b'"
        FAIL=$((FAIL + 1))
    fi
}

json_text() {
    sed -n 's/.*"text"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
}

echo "=== CLI generation controls ($BACKEND) ==="
CLI_OUT=$("$CRISPASR" --backend "$BACKEND" -m "$MODEL" -f "$SAMPLE" \
    --no-prints -n 8 --frequency-penalty 0.5 2>/tmp/crispasr-gen-cli.err)
if [ $? -ne 0 ]; then
    echo "  FAIL CLI command exited non-zero"
    cat /tmp/crispasr-gen-cli.err
    FAIL=$((FAIL + 1))
else
    assert_nonempty "CLI accepts -n + --frequency-penalty" "$CLI_OUT"
fi
rm -f /tmp/crispasr-gen-cli.err

CLI_SHORT=$("$CRISPASR" --backend "$BACKEND" -m "$MODEL" -f "$SAMPLE" \
    --no-prints -n 1 2>/tmp/crispasr-gen-cli-short.err)
CLI_LONG=$("$CRISPASR" --backend "$BACKEND" -m "$MODEL" -f "$SAMPLE" \
    --no-prints -n 8 2>/tmp/crispasr-gen-cli-long.err)
if [ $? -eq 0 ]; then
    assert_greater_len "CLI max-new-tokens changes generated text length" "$CLI_SHORT" "$CLI_LONG"
else
    echo "  FAIL CLI token-cap effect command exited non-zero"
    cat /tmp/crispasr-gen-cli-short.err /tmp/crispasr-gen-cli-long.err
    FAIL=$((FAIL + 1))
fi
rm -f /tmp/crispasr-gen-cli-short.err /tmp/crispasr-gen-cli-long.err

CLI_SEED_A=$("$CRISPASR" --backend "$BACKEND" -m "$MODEL" -f "$SAMPLE" \
    --no-prints -n 16 --temperature 1.0 --seed 123 2>/tmp/crispasr-gen-cli-seed-a.err)
CLI_SEED_B=$("$CRISPASR" --backend "$BACKEND" -m "$MODEL" -f "$SAMPLE" \
    --no-prints -n 16 --temperature 1.0 --seed 123 2>/tmp/crispasr-gen-cli-seed-b.err)
if [ $? -eq 0 ]; then
    assert_equal "CLI seed is reproducible under sampling" "$CLI_SEED_A" "$CLI_SEED_B"
else
    echo "  FAIL CLI seed reproducibility command exited non-zero"
    cat /tmp/crispasr-gen-cli-seed-a.err /tmp/crispasr-gen-cli-seed-b.err
    FAIL=$((FAIL + 1))
fi
rm -f /tmp/crispasr-gen-cli-seed-a.err /tmp/crispasr-gen-cli-seed-b.err

echo
echo "=== Server /v1/audio/transcriptions generation controls ($BACKEND) ==="
SERVER_LOG=$(mktemp -t crispasr-gen-server.XXXXXX)
"$CRISPASR" --server --backend "$BACKEND" -m "$MODEL" \
    --host 127.0.0.1 --port "$PORT" --no-prints > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 90); do
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

BODY=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@$SAMPLE" \
    -F "response_format=json" \
    -F "max_tokens=8" \
    -F "seed=123" \
    -F "frequency_penalty=0.5")
assert_contains "server accepts max_tokens + seed + frequency_penalty" '"text"' "$BODY"

BODY_ALIAS=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@$SAMPLE" \
    -F "response_format=json" \
    -F "max_new_tokens=8")
assert_contains "server accepts max_new_tokens alias" '"text"' "$BODY_ALIAS"

BODY_SHORT=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@$SAMPLE" \
    -F "response_format=json" \
    -F "max_tokens=1")
BODY_LONG=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@$SAMPLE" \
    -F "response_format=json" \
    -F "max_tokens=8")
TEXT_SHORT=$(printf '%s' "$BODY_SHORT" | json_text)
TEXT_LONG=$(printf '%s' "$BODY_LONG" | json_text)
assert_greater_len "server max_tokens changes generated text length" "$TEXT_SHORT" "$TEXT_LONG"

BODY_SEED_A=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@$SAMPLE" \
    -F "response_format=json" \
    -F "max_tokens=16" \
    -F "temperature=1.0" \
    -F "seed=123")
BODY_SEED_B=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@$SAMPLE" \
    -F "response_format=json" \
    -F "max_tokens=16" \
    -F "temperature=1.0" \
    -F "seed=123")
TEXT_SEED_A=$(printf '%s' "$BODY_SEED_A" | json_text)
TEXT_SEED_B=$(printf '%s' "$BODY_SEED_B" | json_text)
assert_equal "server seed is reproducible under sampling" "$TEXT_SEED_A" "$TEXT_SEED_B"

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

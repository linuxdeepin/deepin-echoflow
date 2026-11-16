#!/bin/bash
# test-server-voxcpm2-rate.sh — regression for issue #122.
#
# Boots crispasr-server with voxcpm2-q4_k and hits /v1/audio/speech
# with the user's reported input. Verifies that the WAV the server
# returns declares 48 kHz (voxcpm2's native rate) in the header
# rather than the previously-hardcoded 24 kHz. A 48 kHz buffer with
# a 24 kHz header plays at half speed — that's the "distorted audio"
# the user reported.
#
# Usage:
#   ./tests/test-server-voxcpm2-rate.sh [--port N] [--keep-server]
#
# Requires:
#   - build/bin/crispasr-server (or build-ninja-compile/bin/crispasr-server)
#   - voxcpm2-q4_k.gguf in CRISPASR_MODELS dir or VOXCPM2_MODEL env var
#
# Exit code: 0 if all pass, non-zero otherwise. SKIPs (exit 0) when the
# model file isn't present so CI without the asset doesn't false-fail.

set -uo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-11443}
KEEP_SERVER=0
for arg in "$@"; do
    case "$arg" in
        --port=*) PORT="${arg#--port=}" ;;
        --port) shift; PORT="$1" ;;
        --keep-server) KEEP_SERVER=1 ;;
    esac
done

# Locate the binary. /v1/audio/speech lives in crispasr_server.cpp,
# which is linked into the unified `crispasr` binary (--server mode) —
# not the legacy `crispasr-server` (examples/server/server.cpp), which
# only carries the older /inference endpoint.
SERVER_BIN=""
for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then SERVER_BIN="$cand --server"; break; fi
done
if [ -z "$SERVER_BIN" ]; then
    echo "ERROR: crispasr binary not found. Build first (cmake --build build --target crispasr-cli)."
    exit 2
fi

# Locate the model.
VOXCPM2_MODEL="${VOXCPM2_MODEL:-}"
if [ -z "$VOXCPM2_MODEL" ]; then
    for cand in \
        "${CRISPASR_MODELS:-/Volumes/backups/ai/crispasr-models}/voxcpm2-q4_k.gguf" \
        "/Volumes/backups/ai/cache/huggingface/voxcpm2/voxcpm2-q4_k.gguf" \
        "$HOME/.cache/crispasr/voxcpm2-q4_k.gguf"; do
        if [ -f "$cand" ]; then VOXCPM2_MODEL="$cand"; break; fi
    done
fi
if [ -z "$VOXCPM2_MODEL" ] || [ ! -f "$VOXCPM2_MODEL" ]; then
    echo "SKIP: voxcpm2-q4_k.gguf not found (set VOXCPM2_MODEL or CRISPASR_MODELS)"
    exit 0
fi

SERVER_LOG=$(mktemp -t crispasr-vox-srv.XXXXXX)
echo "Starting crispasr-server on :$PORT (voxcpm2-tts, model=$VOXCPM2_MODEL)…"
$SERVER_BIN \
    -m "$VOXCPM2_MODEL" \
    --host 127.0.0.1 --port "$PORT" --no-prints \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'if [ "$KEEP_SERVER" -eq 0 ] && [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; rm -f "$SERVER_LOG"' EXIT

# Wait for /health. Voxcpm2-q4_k (~4.5 GB) takes 30-90s to load + Metal
# pipeline-cache build on first run after a binary rebuild — give it
# 180s and surface the tail of the log if it never opens the port.
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server died during startup. Log:"; cat "$SERVER_LOG"; exit 2
    fi
    sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "ERROR: server did not become ready within 180s. Log tail:"
    tail -40 "$SERVER_LOG"; exit 2
fi

# Wait until model is fully loaded (ready=true). /v1/audio/speech
# returns 503 while still loading.
for i in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        -H 'Content-Type: application/json' \
        -d '{"input":"warmup"}' "http://127.0.0.1:$PORT/v1/audio/speech")
    if [ "$code" != "503" ]; then break; fi
    sleep 1
done

PASS=0
FAIL=0

assert() {
    local name="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "  ✓ $name"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name: expected '$expected', got '$actual'"
        FAIL=$((FAIL + 1))
    fi
}

echo
echo "=== issue #122 regression — voxcpm2 WAV must declare 48 kHz ==="

TMPWAV=$(mktemp -t crispasr-vox.XXXXXX.wav)
code=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"model":"voxcpm2-q4_k","input":"Tom and Maria are going for a walk","response_format":"wav"}' \
    -o "$TMPWAV" -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert "POST /v1/audio/speech → 200" "200" "$code"

if [ ! -s "$TMPWAV" ]; then
    echo "  ✗ WAV body is empty"; FAIL=$((FAIL + 1))
else
    head4=$(dd if="$TMPWAV" bs=4 count=1 2>/dev/null)
    assert "WAV starts with RIFF" "RIFF" "$head4"
    fmt=$(dd if="$TMPWAV" bs=1 skip=8 count=4 2>/dev/null)
    assert "RIFF format is WAVE" "WAVE" "$fmt"

    # The point of this test: header MUST declare 48 kHz, not 24 kHz.
    rate=$(python3 -c "import struct; f=open('$TMPWAV','rb'); f.seek(24); print(struct.unpack('<I', f.read(4))[0])")
    assert "WAV sample_rate is 48000 (voxcpm2 native)" "48000" "$rate"

    # Consistency check: byte_rate must equal rate * channels * 2.
    byte_rate=$(python3 -c "import struct; f=open('$TMPWAV','rb'); f.seek(28); print(struct.unpack('<I', f.read(4))[0])")
    assert "WAV byte_rate = 48000 * 1 * 2" "96000" "$byte_rate"

    # Channels (offset 22) and block-align (offset 32) must be mono int16.
    channels=$(python3 -c "import struct; f=open('$TMPWAV','rb'); f.seek(22); print(struct.unpack('<H', f.read(2))[0])")
    assert "WAV channels = 1 (mono)" "1" "$channels"
    block_align=$(python3 -c "import struct; f=open('$TMPWAV','rb'); f.seek(32); print(struct.unpack('<H', f.read(2))[0])")
    assert "WAV block_align = 2 (mono int16)" "2" "$block_align"

    # Audio sanity: data chunk size implies a duration close to what
    # the server logs (84480 samples / 48000 = ~1.76 s). Anything
    # significantly shorter means we got a truncated buffer.
    data_size=$(python3 -c "import struct; f=open('$TMPWAV','rb'); f.seek(40); print(struct.unpack('<I', f.read(4))[0])")
    duration=$(python3 -c "print(f'{$data_size / 2 / $rate:.2f}')")
    n_samples=$(python3 -c "print($data_size // 2)")
    echo "  ℹ data: $data_size bytes = $n_samples samples = ${duration}s @ ${rate}Hz"
    # The synthesis is non-deterministic, so allow a generous window
    # around the expected ~1.5-4 second duration.
    if python3 -c "import sys; sys.exit(0 if 1.0 <= $duration <= 5.0 else 1)"; then
        echo "  ✓ duration ${duration}s is plausible for the input"
        PASS=$((PASS + 1))
    else
        echo "  ✗ duration ${duration}s is out of plausible range (1.0-5.0)"
        FAIL=$((FAIL + 1))
    fi
fi
rm -f "$TMPWAV"

echo
echo "──────────────── summary ────────────────"
echo "passed: $PASS"
echo "failed: $FAIL"

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0

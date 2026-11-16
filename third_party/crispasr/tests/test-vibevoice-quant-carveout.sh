#!/usr/bin/env bash
# Regression guard for the VibeVoice quantization carve-out (issue #171).
#
# The realtime/1.5b/7b VibeVoice TTS models generate audio with a diffusion
# prediction head (pred.*) that runs under classifier-free guidance over 20
# solver steps. Quantizing that head lets a tiny error compound into a
# hallucinated non-speech "music"/hum onset before the voice. crispasr-quantize
# therefore keeps the trajectory/control stack (pred.*, at_conn.*, se_conn.*,
# tts_eos.*, tts_types.*) at source precision for any `vibevoice-*` arch, while
# the LM backbone (lm.*, tts_lm.*) and the deterministic VAE decoder (at_dec.*)
# are quantized normally. This test asserts that split so the carve-out can't be
# silently dropped.
set -euo pipefail

QUANT="${1:-}"
FIXTURE="${2:-}"
# Fall back to conventional locations when run outside ctest.
if [ -z "$QUANT" ]; then
    for c in build/bin/crispasr-quantize ./crispasr-quantize; do
        [ -x "$c" ] && QUANT="$c" && break
    done
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -z "$FIXTURE" ] && FIXTURE="$SCRIPT_DIR/fixtures/vibevoice-quant-carveout.gguf"

if [ -z "$QUANT" ] || [ ! -x "$QUANT" ]; then
    echo "SKIP: crispasr-quantize binary not found (pass as \$1)"; exit 0
fi
if [ ! -f "$FIXTURE" ]; then
    echo "SKIP: fixture GGUF not found at $FIXTURE"; exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $1"; exit 1; }

# assert_decision <logfile> <tensor-prefix> copying|quantizing
assert_decision() {
    local log="$1" prefix="$2" want="$3" line
    line="$(grep -E "[0-9]+\] +${prefix}" "$log" | head -1 || true)"
    [ -n "$line" ] || fail "no line for tensor '$prefix' in quantize output"
    case "$want" in
        copying)    echo "$line" | grep -q "copying"    || fail "$prefix should be kept (copying), got: $line" ;;
        quantizing) echo "$line" | grep -q "quantizing" || fail "$prefix should be quantized, got: $line" ;;
    esac
}

for q in q8_0 q4_k; do
    LOG="$TMP/$q.log"
    "$QUANT" "$FIXTURE" "$TMP/out-$q.gguf" "$q" >"$LOG" 2>&1 || fail "quantize $q exited non-zero"
    # Trajectory/control stack must be kept at source precision.
    for p in "pred\." "at_conn\." "se_conn\." "tts_eos\." "tts_types\."; do
        assert_decision "$LOG" "$p" copying
    done
    # Backbone + deterministic VAE decoder must be quantized.
    for p in "lm\." "tts_lm\." "at_dec\."; do
        assert_decision "$LOG" "$p" quantizing
    done
    echo "OK: $q — diffusion head protected, backbone quantized"
done

# The escape hatch must quantize everything, including pred.*.
LOG="$TMP/all.log"
CRISPASR_VIBEVOICE_QUANT_ALL=1 "$QUANT" "$FIXTURE" "$TMP/out-all.gguf" q8_0 >"$LOG" 2>&1 \
    || fail "quantize with QUANT_ALL exited non-zero"
assert_decision "$LOG" "pred\." quantizing
echo "OK: CRISPASR_VIBEVOICE_QUANT_ALL=1 quantizes pred.*"

echo "PASS: vibevoice quant carve-out"

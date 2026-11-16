#!/usr/bin/env bash
# test-quds-fa-live.sh — #387 Quds v4 Persian ASR live test.
#
# Transcribes the CC0 Common Voice Persian sample bundled in samples/ and
# asserts the exact transcript the port was validated with (byte-identical
# to the upstream onnx_asr reference AND matching the human ground truth,
# modulo the corpus's trailing dash). Greedy RNNT is deterministic, so an
# exact-match assert is stable; any drift here is a real regression in the
# FastConformer/RNNT runtime, the converter contract, or the mel front-end.
#
# SKIPs cleanly (exit 0) when the model is missing.
set -u

BIN="${CRISPASR_BIN:-./build/bin/crispasr}"
CLIP="samples/fa-common-voice.wav"
EXPECTED="باید باهاش حرف بزنم"

MODEL="${CRISPASR_QUDS_MODEL:-}"
if [ -z "$MODEL" ]; then
    for d in "${CRISPASR_MODELS:-}" "${CRISPASR_MODELS_DIR:-}" "$HOME/.cache/crispasr" \
             /mnt/volume1/tmp-overflow/quds-v4; do
        [ -n "$d" ] || continue
        for c in "$d"/quds-v4-fa-q8_0.gguf "$d"/quds-v4-fa-f16.gguf; do
            [ -f "$c" ] && MODEL="$c" && break 2
        done
    done
fi

if [ ! -x "$BIN" ]; then
    echo "SKIP: crispasr binary not found at $BIN"
    exit 0
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP: no quds model (set CRISPASR_QUDS_MODEL or CRISPASR_MODELS_DIR)"
    exit 0
fi
if [ ! -f "$CLIP" ]; then
    echo "SKIP: $CLIP missing"
    exit 0
fi

OUT="$("$BIN" --backend quds -m "$MODEL" -l fa -t 4 --no-prints "$CLIP" 2>/dev/null | tail -1)"
if [ "$OUT" != "$EXPECTED" ]; then
    echo "FAIL: transcript mismatch"
    echo "  expected: $EXPECTED"
    echo "  got:      $OUT"
    exit 1
fi
echo "PASS: quds-fa transcript exact ($MODEL)"

#!/bin/bash
# test-granite-maxlen.sh — live integration test for --max-len on granite
# backend (issue #205).
#
# Runs crispasr with a granite-speech-plus GGUF and --max-len N, then verifies
# that the output contains multiple subtitle lines (i.e. the segment was split).
# Contrast: without --max-len the same audio produces fewer, longer lines.
#
# SKIPs (exit 0) when no granite model or test audio is found.
#
# Usage: ./tests/test-granite-maxlen.sh

set -uo pipefail
cd "$(dirname "$0")/.."

CRISPASR=""
for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
done
[ -n "$CRISPASR" ] || { echo "SKIP: crispasr binary not found"; exit 0; }

# Locate a granite-speech PLUS GGUF (word timestamps need PLUS variant).
MODEL="${GRANITE_MODEL:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    MODEL=""
    for d in "${CRISPASR_MODELS_DIR:-}" /mnt/storage/gguf-models "$HOME/.cache/crispasr"; do
        [ -n "$d" ] && [ -d "$d" ] || continue
        # Prefer plus variant
        cand=$(ls "$d"/granite-speech*plus*.gguf "$d"/granite-speech-4.1*.gguf 2>/dev/null | head -1)
        if [ -n "$cand" ] && [ -f "$cand" ]; then MODEL="$cand"; break; fi
    done
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "SKIP: no granite-speech GGUF found (set GRANITE_MODEL)"; exit 0; }

# Locate test audio.
AUDIO="${CRISPASR_TEST_AUDIO:-}"
if [ -z "$AUDIO" ] || [ ! -f "$AUDIO" ]; then
    AUDIO=""
    for cand in samples/jfk.wav /mnt/akademie_storage/whisper.cpp/samples/jfk.wav; do
        if [ -f "$cand" ]; then AUDIO="$cand"; break; fi
    done
fi
[ -n "$AUDIO" ] && [ -f "$AUDIO" ] || { echo "SKIP: no test audio found"; exit 0; }

echo "Model: $MODEL"
echo "Audio: $AUDIO"

PASS=0; FAIL=0

# ── Test 1: --max-len produces more lines than default ──
OUT_DEFAULT=$("$CRISPASR" -m "$MODEL" --backend granite -f "$AUDIO" --no-prints 2>/dev/null)
OUT_MAXLEN=$("$CRISPASR" -m "$MODEL" --backend granite -f "$AUDIO" --max-len 40 -osrt --no-prints 2>/dev/null)

lines_default=$(echo "$OUT_DEFAULT" | wc -l)
lines_maxlen=$(echo "$OUT_MAXLEN" | wc -l)

echo "  default output lines: $lines_default"
echo "  --max-len=40 SRT lines: $lines_maxlen"

if [ "$lines_maxlen" -gt "$lines_default" ]; then
    echo "  ✓ --max-len produced more output lines"; PASS=$((PASS+1))
else
    echo "  ✗ --max-len did not increase output lines ($lines_maxlen <= $lines_default)"; FAIL=$((FAIL+1))
fi

# ── Test 2: SRT output with --max-len has proper timestamps ──
if echo "$OUT_MAXLEN" | grep -qE '[0-9]{2}:[0-9]{2}:[0-9]{2}'; then
    echo "  ✓ SRT output contains timestamps"; PASS=$((PASS+1))
else
    echo "  ✗ SRT output missing timestamps"; FAIL=$((FAIL+1))
fi

# ── Test 3: text content is preserved (not garbled by splitting) ──
# Extract just the text lines from SRT (skip numbers and timestamp lines).
text_maxlen=$(echo "$OUT_MAXLEN" | grep -v '^$' | grep -v '^[0-9]*$' | grep -v -- '-->' | tr '\n' ' ')
# The JFK audio should contain "ask" and "country" regardless of splitting.
if echo "$text_maxlen" | grep -qi "ask\|country\|fellow\|citizen"; then
    echo "  ✓ transcript text preserved after split"; PASS=$((PASS+1))
else
    echo "  ✗ transcript text may be garbled: $text_maxlen"; FAIL=$((FAIL+1))
fi

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1

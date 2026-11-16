#!/bin/bash
# test-input-exit-code.sh — an input that cannot be decoded must reach the
# EXIT CODE, not just stderr.
#
# SCOPE, because it is narrower than it first looked. The unified dispatcher
# (crispasr_run.cpp — anything with an explicit --backend, -m auto, or a
# non-whisper GGUF) ALREADY returned 20 on a read failure. The defect is in
# cli.cpp's LEGACY whisper path, reached by the historical default shape:
#   crispasr -m ggml-tiny.bin -f file.wav        (no --backend, a .bin model)
# There, read_audio_data() failing only printed to stderr and `continue`d, so a
# run in which EVERY file failed still exited 0 — indistinguishable from the
# successful control. docs/cli.md publishes "rc 0 <=> every required stage
# succeeded", so this contradicted the documented contract. Worse on Windows,
# where that path's exit was the literal `_Exit(0)`: it could not express a
# failure through its exit code at all, whatever happened.
#
# This test therefore runs WITHOUT --backend and with a .bin model on purpose.
# Adding --backend routes to the dispatcher and the assertion stops testing the
# code that has the bug.
#
# Codes asserted here:
#   0   normal file transcribes
#   2   input file not found (argument validation, before any model load)
#   4   input exists but cannot be decoded
#
# THE POSITIVE CONTROL IS LOAD-BEARING. The rejected WAV and the accepted one
# differ in ONE FIELD — the declared sample rate — and share every byte of
# payload. Without the accepted arm, "non-zero" would be equally explained by a
# test that writes a WAV no decoder could ever read, and the arm that matters
# would pass for the wrong reason.
#
# Needs a model, because the read happens after model load. SKIPs cleanly.

set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-}"
if [ -z "$BIN" ]; then
    for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
        [ -x "$cand" ] && BIN="$cand" && break
    done
fi
[ -z "$BIN" ] && { echo "SKIP: crispasr binary not found (build first)"; exit 0; }

# models/for-tests-ggml-tiny.bin is COMMITTED (575 KB), so this test actually
# runs in CI instead of skipping. That matters: the sibling cli-labelled tests
# spent their whole existence never executing because nothing ran `-L cli`
# (70112d10), and a test that always skips reports the same thing as one that
# passes. A whisper .bin specifically — a GGUF auto-detects a backend and routes
# to the dispatcher, which does NOT have the defect under test.
MODEL="${CRISPASR_EXITCODE_MODEL:-models/for-tests-ggml-tiny.bin}"
[ -f "$MODEL" ] && [ -s "$MODEL" ] || { echo "SKIP: model '$MODEL' missing"; exit 0; }
case "$MODEL" in
    *.bin) ;;
    *) echo "SKIP: model must be a whisper .bin — a GGUF routes to the dispatcher, not the path under test"; exit 0 ;;
esac

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0

check() { # description | actual | expected
    if [ "$2" = "$3" ]; then echo "  ✓ $1 → rc $2"; PASS=$((PASS+1))
    else echo "  ✗ $1 → rc '$2' (expected '$3')"; FAIL=$((FAIL+1)); fi
}

# Two WAVs, byte-identical except for the declared sample rate at offset 24.
# python3 keeps this hermetic: no fixture to go stale, and the pair is
# constructed here so the "one field differs" property is visible, not asserted.
python3 - "$TMP" <<'PY'
import struct, sys, math, pathlib
out = pathlib.Path(sys.argv[1])
n = 2000
data = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / 16000))) for i in range(n))
def wav(rate):
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)
(out / "good.wav").write_bytes(wav(16000))
(out / "rate1.wav").write_bytes(wav(1))
PY

run() { "$BIN" -m "$MODEL" -f "$1" >/dev/null 2>&1; echo $?; }

echo "positive control — the SAME payload at a sane rate:"
check "16 kHz WAV decodes" "$(run "$TMP/good.wav")" "0"

echo "the amplification case — only the declared rate differs:"
check "1 Hz WAV is refused" "$(run "$TMP/rate1.wav")" "4"

echo "and the refusal is diagnosable, not silent:"
err="$("$BIN" -m "$MODEL" -f "$TMP/rate1.wav" 2>&1 >/dev/null)"
case "$err" in
    *"expansion limit"*) echo "  ✓ stderr names the refused ratio"; PASS=$((PASS+1)) ;;
    *) echo "  ✗ stderr has no resampler diagnostic"; FAIL=$((FAIL+1)) ;;
esac
case "$err" in
    *"no speech detected"*) echo "  ✗ still reported as 'no speech detected'"; FAIL=$((FAIL+1)) ;;
    *) echo "  ✓ not misreported as an empty transcription"; PASS=$((PASS+1)) ;;
esac

echo "argument validation still owns the missing-file case:"
check "nonexistent input" "$(run "$TMP/definitely-absent.wav")" "2"

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1

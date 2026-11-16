#!/bin/bash
# tests/test_strict_pipeline.sh — issue #311 strict failure semantics.
#
# Verifies that explicitly-requested auxiliary stages (VAD, forced alignment,
# punctuation) can be REQUIRED to succeed, so a zero exit reliably means "every
# requested stage completed". The default stays permissive (graceful degrade).
#
# Model-free checks (always run):
#   - config validation: --require-vad without VAD, --require-punctuation
#     without --punc-model → usage error (exit 2)
#   - --help lists the new flags
#
# Model-gated A/B (needs a whisper model; SKIPs cleanly otherwise) — each proves
# strict FAILS where permissive SUCCEEDS on the same unloadable model:
#   - required VAD model unloadable → non-zero (acceptance case 3)
#   - required aligner unloadable / no word timestamps → non-zero (cases 4,5)
#   - required punctuation model unloadable → non-zero (case 6)
#
# Usage: bash tests/test_strict_pipeline.sh [crispasr-bin] [src-dir]

CRISPASR="${1:-${CRISPASR_BIN:-build/bin/crispasr}}"
SRC_DIR="${2:-.}"
# Tier selector, same split as test_vad_export_live.sh and for the same reason:
# labelled `unit`, this script reported green in CI while SKIPPING the
# model-gated A/B that is the part actually worth running. `unit` runs the
# model-free half only; `live`/`all` run everything.
TIER="${3:-all}"
case "$TIER" in
    unit | live | all) ;;
    *)
        echo "unknown tier '$TIER' (expected unit, live or all)" >&2
        exit 2
        ;;
esac
JFK_WAV="$SRC_DIR/samples/jfk.wav"
TMPDIR="$(mktemp -d "${TMPDIR_BASE:-/tmp}/test311.XXXXXX")"
BOGUS="$TMPDIR/not-a-real-model.gguf"
printf 'this is not a valid gguf model file' > "$BOGUS" # readable but unloadable

PASS=0
FAIL=0
SKIP=0
pass() { echo "[PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }
skip() { echo "[SKIP] $1 — $2"; SKIP=$((SKIP + 1)); }

# run_rc <expected-nonzero|zero> <name> -- <cmd...>
# Asserts the command's exit code is zero (expected=zero) or non-zero (nonzero).
expect_exit() {
    local want="$1" name="$2"; shift 2
    [ "$1" = "--" ] && shift
    "$@" >/dev/null 2>&1
    local rc=$?
    if { [ "$want" = "zero" ] && [ "$rc" -eq 0 ]; } || { [ "$want" = "nonzero" ] && [ "$rc" -ne 0 ]; }; then
        pass "$name (rc=$rc)"
    else
        fail "$name (want $want, got rc=$rc)"
    fi
}

if [ ! -f "$CRISPASR" ]; then echo "SKIP: crispasr binary not found at $CRISPASR"; exit 0; fi
if [ ! -f "$JFK_WAV" ]; then echo "SKIP: test audio not found at $JFK_WAV"; exit 0; fi

echo "=== Issue #311: strict pipeline semantics ==="

# ─── Model-free: config validation + help text ──────────────────────────────
# --require-vad without any VAD request is a usage error (exit 2), evaluated at
# the top of run_backend before model load, so a real model isn't needed.
"$CRISPASR" --require-vad -m "$BOGUS" -f "$JFK_WAV" >/dev/null 2>"$TMPDIR/e1"; rc=$?
if [ "$rc" -eq 2 ]; then pass "config: --require-vad without --vad → exit 2"; else fail "config: --require-vad without --vad → exit 2 (got $rc)"; fi

"$CRISPASR" --require-punctuation -m "$BOGUS" -f "$JFK_WAV" >/dev/null 2>"$TMPDIR/e2"; rc=$?
if [ "$rc" -eq 2 ]; then pass "config: --require-punctuation without --punc-model → exit 2"; else fail "config: --require-punctuation without --punc-model → exit 2 (got $rc)"; fi

HELP="$("$CRISPASR" --help 2>&1 || true)"
for flag in -- --strict-pipeline --require-vad --require-word-timestamps --require-punctuation; do
    [ "$flag" = "--" ] && continue
    if echo "$HELP" | grep -q -- "$flag"; then pass "help lists $flag"; else fail "help lists $flag"; fi
done

if [ "$TIER" = "unit" ]; then
    echo ""
    echo "=== Results (unit tier): $PASS passed, $FAIL failed, $SKIP skipped ==="
    rm -rf "$TMPDIR"
    [ "$FAIL" -eq 0 ] || exit 1
    exit 0
fi

# ─── Model-gated A/B ────────────────────────────────────────────────────────
MODEL="${CRISPASR_MODEL_WHISPER:-}"
if [ -z "$MODEL" ]; then
    for m in ggml-base.bin models/ggml-base.en.bin "$SRC_DIR/ggml-base.bin"; do
        [ -f "$m" ] && MODEL="$m" && break
    done
fi
# All model-gated cases route through the unified dispatch (--backend whisper)
# so both A/B arms use the SAME enforced path — the strict flag is the only
# difference. Confirm that combo actually loads first (a dangling cache symlink
# would otherwise mark real strict passes as spurious).
WB=(--backend whisper)
if [ -n "$MODEL" ]; then
    if ! "$CRISPASR" "${WB[@]}" -m "$MODEL" -f "$JFK_WAV" -nt >/dev/null 2>&1; then
        skip "model-gated A/B" "whisper model '$MODEL' did not load/transcribe via unified dispatch"
        MODEL=""
    fi
fi

if [ -z "$MODEL" ]; then
    skip "model-gated A/B" "no loadable whisper model (set CRISPASR_MODEL_WHISPER)"
else
    echo "--- model-gated A/B with $MODEL ---"
    # VAD: unloadable required model fails; permissive falls back and succeeds.
    expect_exit nonzero "VAD unloadable + --require-vad → fail" -- \
        "$CRISPASR" "${WB[@]}" -m "$MODEL" -f "$JFK_WAV" --vad -vm "$BOGUS" --require-vad
    expect_exit nonzero "VAD unloadable + --strict-pipeline → fail" -- \
        "$CRISPASR" "${WB[@]}" -m "$MODEL" -f "$JFK_WAV" --vad -vm "$BOGUS" --strict-pipeline
    expect_exit zero "VAD unloadable, permissive (no flag) → succeed" -- \
        "$CRISPASR" "${WB[@]}" -m "$MODEL" -f "$JFK_WAV" --vad -vm "$BOGUS"

    # Punctuation: unloadable required model fails; permissive continues.
    expect_exit nonzero "punc unloadable + --require-punctuation → fail" -- \
        "$CRISPASR" "${WB[@]}" -m "$MODEL" -f "$JFK_WAV" --punc-model "$BOGUS" --require-punctuation
    expect_exit zero "punc unloadable, permissive → succeed" -- \
        "$CRISPASR" "${WB[@]}" -m "$MODEL" -f "$JFK_WAV" --punc-model "$BOGUS"

    # Word timestamps: unloadable aligner (no words) fails; permissive keeps segs.
    expect_exit nonzero "aligner unloadable + --require-word-timestamps → fail" -- \
        "$CRISPASR" "${WB[@]}" -m "$MODEL" -f "$JFK_WAV" -am "$BOGUS" --force-aligner --require-word-timestamps -ojf -of "$TMPDIR/wt"
    expect_exit zero "aligner unloadable, permissive → succeed" -- \
        "$CRISPASR" "${WB[@]}" -m "$MODEL" -f "$JFK_WAV" -am "$BOGUS" --force-aligner -ojf -of "$TMPDIR/wt2"
fi

echo ""
echo "=== $PASS passed, $FAIL failed, $SKIP skipped ==="
rm -rf "$TMPDIR"
[ "$FAIL" -eq 0 ]

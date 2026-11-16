#!/usr/bin/env bash
# test-speaker-db-gates.sh — CLI gate regression guard for issue #266.
#
# `--speaker-db` matching is a deliberately restricted feature (closed-roster
# confirmation of claimed, consenting participants; never in streaming mode;
# see docs/diarization-speakers.md §2). The three hard gates in
# crispasr_run_backend() (examples/cli/crispasr_run.cpp) all fire BEFORE any
# model resolution/loading, so this test needs no models and no network:
#
#   (a) --stream + --speaker-db          -> exit 26, "not available in streaming mode"
#   (b) --speaker-db-consent, no roster  -> exit 27, "requires --expect-speakers"
#   (c) --speaker-db, no consent         -> warn "speaker-db ignored", run continues
#
# Usage: test-speaker-db-gates.sh <path-to-crispasr-binary> [repo-source-dir]
set -uo pipefail

CRISPASR="${1:-}"
SRC_DIR="${2:-$(cd "$(dirname "$0")/.." && pwd)}"

if [ -z "$CRISPASR" ] || [ ! -x "$CRISPASR" ]; then
    echo "SKIP: crispasr binary not found (pass as \$1)"; exit 0
fi

JFK="$SRC_DIR/samples/jfk.wav"
if [ ! -f "$JFK" ]; then
    echo "SKIP: fixture $JFK not found"; exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $1"; exit 1; }

# Never let a real model cache / model dir leak into these gate checks —
# they must not touch the network or an existing local model. Every
# invocation below gets its own empty HOME + cache-dir.
unset CRISPASR_MODELS_DIR CRISPASR_MODEL_WHISPER

# ── (a) streaming + speaker-db -> exit 26 ──────────────────────────────────
OUT_A="$TMP/a.stderr"
HOME="$TMP/home-a" "$CRISPASR" --backend parakeet --stream \
    --speaker-db "$TMP/db" --speaker-db-consent --expect-speakers A \
    --cache-dir "$TMP/cache-a" \
    >/dev/null 2>"$OUT_A"
rc=$?
[ "$rc" -eq 26 ] || fail "(a) expected exit 26, got $rc (stderr below)\n$(cat "$OUT_A")"
grep -q "not available in streaming mode" "$OUT_A" || \
    fail "(a) stderr missing 'not available in streaming mode':\n$(cat "$OUT_A")"
echo "OK: (a) streaming refusal -> exit 26"

# ── (b) consent given, no --expect-speakers -> exit 27 ─────────────────────
OUT_B="$TMP/b.stderr"
HOME="$TMP/home-b" "$CRISPASR" -f "$JFK" --backend parakeet \
    --speaker-db "$TMP/db" --speaker-db-consent \
    --cache-dir "$TMP/cache-b" \
    >/dev/null 2>"$OUT_B"
rc=$?
[ "$rc" -eq 27 ] || fail "(b) expected exit 27, got $rc (stderr below)\n$(cat "$OUT_B")"
grep -q "requires --expect-speakers" "$OUT_B" || \
    fail "(b) stderr missing 'requires --expect-speakers':\n$(cat "$OUT_B")"
echo "OK: (b) missing --expect-speakers -> exit 27"

# ── (c) --speaker-db without consent -> warn + ignore, run continues ───────
# Isolate HOME and cache-dir so there is no local model to find and no
# --auto-download flag is passed, so this cannot touch the network; the
# run is expected to fail later (bogus model path) but the gate warning
# must still appear before that failure. Exit code is unconstrained.
OUT_C="$TMP/c.stderr"
HOME="$TMP/home-c" "$CRISPASR" -f "$JFK" --backend parakeet \
    -m "$TMP/nonexistent-model.gguf" \
    --speaker-db "$TMP/db" \
    --cache-dir "$TMP/cache-c" \
    >/dev/null 2>"$OUT_C"
grep -q "speaker-db ignored" "$OUT_C" || \
    fail "(c) stderr missing 'speaker-db ignored':\n$(cat "$OUT_C")"
echo "OK: (c) no-consent warn-and-ignore"

echo "PASS: speaker-db CLI gates (#266)"

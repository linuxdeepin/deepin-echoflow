#!/bin/bash
# test-arg-validation.sh — argument validation must run BEFORE backend dispatch.
#
# These checks used to sit after every `crispasr_run_backend()` early return in
# main(), so they only ever ran on the legacy whisper path. On any dispatched
# run — which is most of them: `-m auto`, any `--backend`, or a GGUF whose
# metadata names a non-whisper architecture — they were skipped entirely and
# the bad argument was accepted silently with exit 0:
#
#   crispasr -m auto --diarize --tinydiarize   → transcribed, rc 0
#   crispasr -m auto -l zz                     → transcribed as English, rc 0
#
# The mirror-image hazard is a FALSE POSITIVE, and it is the reason the
# language check is not simply hoisted: whisper's table has 100 entries, while
# other backends legitimately use codes outside it (omnivoice alone takes
# fil/nan/arb/pes). So the language check is gated on the effective backend,
# and this file asserts BOTH directions — whisper rejects `fil`, parakeet must
# not.
#
# Fully offline and model-free: every case is decided before any model is
# loaded or downloaded (HOME and CRISPASR_MODELS_DIR are pointed at empty
# directories so a network fetch cannot be mistaken for a pass).

set -uo pipefail
cd "$(dirname "$0")/.."

CRISPASR="${1:-}"
if [ -z "$CRISPASR" ]; then
    for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
        if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
    done
fi
if [ -z "$CRISPASR" ]; then
    echo "SKIP: crispasr binary not found (build first)"; exit 0
fi

AUDIO=samples/jfk.wav
if [ ! -f "$AUDIO" ]; then
    echo "SKIP: $AUDIO not found"; exit 0
fi

SANDBOX=$(mktemp -d "${CRISPASR_TEST_TMPDIR:-$PWD}/.argval.XXXXXX")
trap 'rm -rf "$SANDBOX"' EXIT
export HOME="$SANDBOX"
export CRISPASR_MODELS_DIR="$SANDBOX/models"

PASS=0; FAIL=0

# run <description> <expect-rc-nonzero:yes|no> <must-match|-> <must-not-match|-> -- args...
run_case() {
    local desc="$1" want_fail="$2" must="$3" mustnot="$4"; shift 5
    local err rc
    err=$("$CRISPASR" "$@" 2>&1 >/dev/null)
    "$CRISPASR" "$@" >/dev/null 2>&1; rc=$?

    local ok=1 why=""
    if [ "$want_fail" = "yes" ] && [ "$rc" -eq 0 ]; then ok=0; why="exited 0"; fi
    if [ "$must" != "-" ]; then
        case "$err" in *"$must"*) ;; *) ok=0; why="$why; stderr lacks '$must'";; esac
    fi
    if [ "$mustnot" != "-" ]; then
        case "$err" in *"$mustnot"*) ok=0; why="$why; stderr wrongly contains '$mustnot'";; esac
    fi

    if [ "$ok" -eq 1 ]; then
        echo "  ✓ $desc (rc=$rc)"; PASS=$((PASS+1))
    else
        echo "  ✗ $desc (rc=$rc)$why"; FAIL=$((FAIL+1))
    fi
}

echo "contradictory flags are rejected on every path:"
run_case "--diarize --tinydiarize (dispatched, -m auto)" yes "cannot use both" - -- \
    -m auto --diarize --tinydiarize -f "$AUDIO"
run_case "--diarize --tinydiarize (no -m)" yes "cannot use both" - -- \
    --diarize --tinydiarize -f "$AUDIO"
run_case "--diarize --tinydiarize (explicit backend)" yes "cannot use both" - -- \
    --backend whisper --diarize --tinydiarize -f "$AUDIO"

echo
echo "an unknown language is rejected when whisper is what will run:"
run_case "-l zz (dispatched, -m auto)" yes "unknown language" - -- \
    -m auto -l zz -f "$AUDIO"
run_case "-l fil (--backend whisper; not in whisper's table)" yes "unknown language" - -- \
    --backend whisper -m /nonexistent.gguf -l fil -f "$AUDIO"

echo
echo "…and NOT applied to a backend whisper's table does not govern:"
# parakeet is multilingual and its language set is not whisper's. It must fail
# for want of a model, never for the language. This is the guard against
# "fixing" the bug by hoisting the whisper check unconditionally.
run_case "-l fil (--backend parakeet) is not a language error" no - "unknown language" -- \
    --backend parakeet -m /nonexistent.gguf -l fil -f "$AUDIO"

echo
echo "a monolingual backend rejects a language it cannot produce:"
# moonshine declares sole_language()=="en". Asking it for German is not a
# warning — it is a request it cannot satisfy, and it used to be answered with
# an ENGLISH transcript and exit 0, which reads as a model-quality problem
# rather than a rejected flag. Decided before the model loads, so this stays
# offline.
run_case "--backend moonshine -l de" yes "en-only and cannot transcribe" - -- \
    --backend moonshine -m /nonexistent.gguf -l de -f "$AUDIO"
run_case "--backend moonshine -l german (spelled out)" yes "en-only and cannot transcribe" - -- \
    --backend moonshine -m /nonexistent.gguf -l german -f "$AUDIO"
# …and the language it CAN produce, in either spelling, must pass through.
run_case "--backend moonshine -l en is accepted" no - "cannot transcribe" -- \
    --backend moonshine -m /nonexistent.gguf -l en -f "$AUDIO"
run_case "--backend moonshine -l auto is accepted" no - "cannot transcribe" -- \
    --backend moonshine -m /nonexistent.gguf -l auto -f "$AUDIO"

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1

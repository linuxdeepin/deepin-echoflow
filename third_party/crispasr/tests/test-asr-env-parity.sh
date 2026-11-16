#!/bin/bash
# Hermetic coverage for tools/asr-env-parity.sh. No model or network required.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
HARNESS=${1:-tools/asr-env-parity.sh}

SANDBOX=$(mktemp -d "${CRISPASR_TEST_TMPDIR:-$PWD}/.asr-env-parity.XXXXXX")
trap 'rm -rf "$SANDBOX"' EXIT
mkdir -p "$SANDBOX/bin" "$SANDBOX/samples-one" "$SANDBOX/samples-collide"
: > "$SANDBOX/samples-one/one.wav"
: > "$SANDBOX/samples-collide/same name.wav"
: > "$SANDBOX/samples-collide/same?name.wav"

FAKE="$SANDBOX/bin/fake-crispasr"
cat > "$FAKE" <<'FAKE'
#!/bin/bash
if [ "${1-}" = "--version" ]; then
    echo "fake-crispasr 1.0"
    exit 0
fi
if [ "${ERROR_GATE-}" = "1" ]; then
    echo "forced failure" >&2
    exit 7
fi
if [ "${baseline-}" = "1" ]; then
    echo "changed transcript"
else
    printf 'transcript:%s:%s\n' "${FIXED_ENV-unset}" "${RELATED_ENV-unset}"
fi
FAKE
chmod +x "$FAKE"

PASS=0
FAIL=0
RUN_RC=0
RUN_OUT=""

check_eq() {
    if [ "$2" = "$3" ]; then
        echo "  ✓ $1"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $1: got '$2', expected '$3'"
        FAIL=$((FAIL + 1))
    fi
}

check_contains() {
    case "$2" in
        *"$3"*) echo "  ✓ $1"; PASS=$((PASS + 1)) ;;
        *) echo "  ✗ $1: output lacks '$3'"; FAIL=$((FAIL + 1)) ;;
    esac
}

run_harness() {
    "$HARNESS" "$@" > "$SANDBOX/run.stdout" 2> "$SANDBOX/run.stderr"
    RUN_RC=$?
    RUN_OUT=$(cat "$SANDBOX/run.stdout" "$SANDBOX/run.stderr")
}

echo "a variant can never overwrite the baseline artifact:"
out="$SANDBOX/out-false-pass"
run_harness -s "$SANDBOX/samples-one" -o "$out" -e baseline -- "$FAKE"
check_eq "changed transcript fails parity" "$RUN_RC" "1"
check_contains "mismatch is reported" "$RUN_OUT" "FAIL=1"
check_eq "baseline artifact remains the reference" \
    "$(cat "$out/0001-one.wav/baseline.stdout")" "transcript:unset:unset"
check_eq "variant has a separate artifact" \
    "$(cat "$out/0001-one.wav/variant-0001.stdout")" "changed transcript"

echo
echo "normal, combined, fixed, unset, and PATH-resolved runs work:"
out="$SANDBOX/out-normal"
PATH="$SANDBOX/bin:$PATH" RELATED_ENV=leaked UNUSED_GATE=leaked \
    run_harness -s "$SANDBOX/samples-one" -o "$out" \
        -E FIXED_ENV=ok -u RELATED_ENV -e UNUSED_GATE=variant -e SECOND_GATE \
        -- fake-crispasr
check_eq "matching variants exit successfully" "$RUN_RC" "0"
check_contains "individual and all variants pass" "$RUN_OUT" "PASS=3 FAIL=0 ERROR=0"
baseline_file="$out/0001-one.wav/baseline.stdout"
check_eq "baseline scrubs test and related variables" "$(cat "$baseline_file" 2>/dev/null)" \
    "transcript:ok:unset"

echo
echo "command failures stay distinct from transcript mismatches:"
out="$SANDBOX/out-error"
run_harness -s "$SANDBOX/samples-one" -o "$out" -e ERROR_GATE -- "$FAKE"
check_eq "variant command error fails the harness" "$RUN_RC" "1"
check_contains "command error is counted" "$RUN_OUT" "PASS=0 FAIL=0 ERROR=1"

echo
echo "diagnostic paths are collision-free:"
out="$SANDBOX/out-collide"
run_harness -s "$SANDBOX/samples-collide" -o "$out" -e UNUSED_GATE -- "$FAKE"
check_eq "colliding display names still pass" "$RUN_RC" "0"
sample_dirs=0
for entry in "$out"/*; do
    [ ! -d "$entry" ] || sample_dirs=$((sample_dirs + 1))
done
check_eq "each sample keeps its own artifacts" "$sample_dirs" "2"

echo
echo "an existing results directory cannot mix runs:"
out="$SANDBOX/out-stale"
mkdir -p "$out"
echo "keep me" > "$out/old.diff"
run_harness -s "$SANDBOX/samples-one" -o "$out" -e UNUSED_GATE -- "$FAKE"
check_eq "non-empty output directory is rejected" "$RUN_RC" "2"
check_contains "rejection explains the problem" "$RUN_OUT" "not empty"
check_eq "existing diagnostic remains untouched" "$(cat "$out/old.diff")" "keep me"

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1

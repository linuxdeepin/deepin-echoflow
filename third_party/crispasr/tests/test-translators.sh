#!/bin/bash
# test-translators.sh — smoke tests for text translation and LID routing.
#
# Exercises the text-translation backends, the Gemma 4 E2B audio AST path,
# and a couple of LID pre-step checks that the translator backends can reuse.
#
# Usage:
#   ./tests/test-translators.sh
#
# Exit code: 0 if all pass, 1 if any fail.

set -euo pipefail
cd "$(dirname "$0")/.."

CRISPASR="./build/bin/crispasr"
SAMPLE_EN="./samples/jfk.wav"
SAMPLE_JA="./samples/ja/jsut_water_3s.wav"
TMPDIR=$(mktemp -d -t crispasr-translators.XXXXXX)
GEMMA_CACHE="$HOME/.cache/crispasr/test-translators-gemma"
trap 'rm -rf "$TMPDIR"' EXIT

if [ ! -x "$CRISPASR" ]; then
    echo "ERROR: $CRISPASR not found. Build first."
    exit 1
fi
if [ ! -f "$SAMPLE_EN" ]; then
    echo "ERROR: $SAMPLE_EN not found."
    exit 1
fi
if [ ! -f "$SAMPLE_JA" ]; then
    echo "ERROR: $SAMPLE_JA not found."
    exit 1
fi
mkdir -p "$GEMMA_CACHE"

PASS=0
FAIL=0

pass() {
    echo "  PASS: $1"
    PASS=$((PASS + 1))
}

fail() {
    echo "  FAIL: $1"
    FAIL=$((FAIL + 1))
}

normalize() {
    tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:] ]//g' | tr -s ' '
}

run_text_translate() {
    local backend="$1"
    local src="$2"
    local tgt="$3"
    local input="$4"
    local needle="$5"
    local -a cache_args=()
    if [ "$backend" = "gemma4-e2b" ]; then
        cache_args=(--cache-dir "$GEMMA_CACHE")
    fi

    local out
    local -a cmd=("$CRISPASR" --backend "$backend" -m auto --auto-download)
    if [ "${#cache_args[@]}" -gt 0 ]; then
        cmd+=("${cache_args[@]}")
    fi
    cmd+=(--text "$input" -sl "$src" -tl "$tgt" --no-prints)
    if ! out=$("${cmd[@]}" 2>/dev/null); then
        fail "$backend text translation crashed"
        return
    fi
    local norm
    norm=$(printf "%s\n" "$out" | normalize)
    if printf "%s" "$norm" | grep -qi "$needle"; then
        pass "$backend text translation"
    else
        fail "$backend text translation (got: $out)"
    fi
}

run_audio_translate() {
    local backend="$1"
    local infile="$2"
    local src="$3"
    local tgt="$4"
    local needle="$5"
    local base="$TMPDIR/$backend-$(basename "$infile" .wav)"
    local -a cache_args=()
    if [ "$backend" = "gemma4-e2b" ]; then
        cache_args=(--cache-dir "$GEMMA_CACHE")
    fi

    local -a cmd=("$CRISPASR" --backend "$backend" -m auto --auto-download)
    if [ "${#cache_args[@]}" -gt 0 ]; then
        cmd+=("${cache_args[@]}")
    fi
    cmd+=(-f "$infile" -dl -tr -tl "$tgt" -of "$base" -otxt --no-prints)
    if ! "${cmd[@]}" >/dev/null 2>&1; then
        fail "$backend audio translation crashed"
        return
    fi

    if [ ! -f "$base.txt" ]; then
        fail "$backend audio translation did not write output"
        return
    fi

    local out
    out=$(cat "$base.txt")
    local norm
    norm=$(printf "%s\n" "$out" | normalize)
    if printf "%s" "$norm" | grep -qi "$needle"; then
        pass "$backend audio translation"
    else
        fail "$backend audio translation (got: $out)"
    fi
}

run_lid() {
    local infile="$1"
    local expect="$2"
    local base="$TMPDIR/lid-$(basename "$infile" .wav)"

    local -a cmd=("$CRISPASR" --backend gemma4-e2b -m auto --auto-download --cache-dir "$GEMMA_CACHE" -f "$infile" -dl -of "$base" -ojf -otxt --no-prints)
    if ! "${cmd[@]}" >/dev/null 2>&1; then
        fail "LID on $(basename "$infile") crashed"
        return
    fi

    if [ ! -f "$base.json" ]; then
        fail "LID on $(basename "$infile") did not write JSON"
        return
    fi

    local json
    json=$(cat "$base.json")
    if printf "%s" "$json" | grep -Eq '"language"[[:space:]]*:[[:space:]]*"'"$expect"'"'; then
        pass "LID $(basename "$infile") -> $expect"
    else
        fail "LID $(basename "$infile") (got: $json)"
    fi
}

echo "CrispASR translator smoke tests"
echo "==============================="

run_text_translate m2m100 en de "good morning" "gute"
run_text_translate madlad en de "good morning" "gute"
run_text_translate gemma4-e2b en de "good morning" "gute"

run_audio_translate gemma4-e2b "$SAMPLE_EN" en de "german"

run_lid "$SAMPLE_EN" en
run_lid "$SAMPLE_JA" ja

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

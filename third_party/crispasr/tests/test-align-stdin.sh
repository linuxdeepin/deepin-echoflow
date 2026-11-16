#!/bin/bash
# tests/test-align-stdin.sh — live test for `--align-only --text-file -` (#317).
#
# #317: Subtitle Edit's "Import plain text" flow calls crispasr with the audio
# but never hands over the transcript, so --align-only correctly refused with
# "requires --ref-text or --text-file" and two people read that as the aligner
# being broken. The SE-side fix writes a temp file; accepting the transcript on
# stdin removes the need for one, which is the piece offered on that thread.
#
# The property under test is that stdin is a transport, not a different feature:
# piping a transcript must produce byte-identical output to passing the same
# bytes as a file. That includes SRT re-timing, where the file path decides
# "this is an SRT" from the .srt extension and stdin has no extension to read —
# it sniffs the content instead, and the two must agree.
#
# Usage:
#   source tests/env-live-tests.sh
#   bash tests/test-align-stdin.sh
#
# Env vars:
#   CRISPASR_BIN            — crispasr binary (default: ./build/bin/crispasr)
#   CRISPASR_MODEL_ALIGNER  — CTC aligner GGUF (canary-ctc-aligner / wav2vec2)
#   CRISPASR_TEST_AUDIO     — mono 16 kHz speech clip
#
# Skips (exit 77, ctest SKIP_RETURN_CODE) when the model or audio is absent.

set -u

BIN="${CRISPASR_BIN:-./build/bin/crispasr}"
ALIGNER="${CRISPASR_MODEL_ALIGNER:-}"
AUDIO="${CRISPASR_TEST_AUDIO:-samples/jfk.wav}"

if [ ! -x "$BIN" ]; then
    echo "SKIP: crispasr binary not found at $BIN"
    exit 77
fi
if [ -z "$ALIGNER" ] || [ ! -f "$ALIGNER" ]; then
    echo "SKIP: CRISPASR_MODEL_ALIGNER not set or missing"
    exit 77
fi
if [ ! -f "$AUDIO" ]; then
    echo "SKIP: test audio not found at $AUDIO"
    exit 77
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

note() { echo "  $*"; }
check() { # check <name> <condition-rc>
    if [ "$2" -eq 0 ]; then echo "PASS: $1"; else echo "FAIL: $1"; fail=$((fail + 1)); fi
}

# --- 1. the error names the flags, and exits 10 --------------------------------
# Regression on the message itself: #317's readers could not tell from it what
# to pass. An error that does not name --text-file is the bug restated.
out="$("$BIN" --align-only -f "$AUDIO" -am "$ALIGNER" 2>&1)"
rc=$?
check "no-transcript exits 10" "$([ $rc -eq 10 ] && echo 0 || echo 1)"
echo "$out" | grep -q -- "--text-file -" && r=0 || r=1
check "error mentions stdin form" "$r"
echo "$out" | grep -q -- "--ref-text" && r=0 || r=1
check "error mentions --ref-text" "$r"

# --- 2. plain text: stdin == file ---------------------------------------------
cat > "$TMP/t.txt" <<'EOF'
And so my fellow Americans
ask not what your country can do for you
EOF

"$BIN" --align-only -f "$AUDIO" -am "$ALIGNER" --text-file "$TMP/t.txt" \
       --align-granularity segment --output-srt > "$TMP/from_file.srt" 2>/dev/null
rc_file=$?
"$BIN" --align-only -f "$AUDIO" -am "$ALIGNER" --text-file - \
       --align-granularity segment --output-srt < "$TMP/t.txt" > "$TMP/from_stdin.srt" 2>/dev/null
rc_stdin=$?

check "txt via file succeeds"  "$([ $rc_file  -eq 0 ] && echo 0 || echo 1)"
check "txt via stdin succeeds" "$([ $rc_stdin -eq 0 ] && echo 0 || echo 1)"
# Guard against both arms being empty, which would compare equal and prove nothing.
check "txt output is non-empty" "$([ -s "$TMP/from_file.srt" ] && echo 0 || echo 1)"
cmp -s "$TMP/from_file.srt" "$TMP/from_stdin.srt" && r=0 || r=1
check "txt stdin output == file output" "$r"

# --- 3. srt re-timing: stdin == file (extension vs content sniffing) ----------
# Deliberately wrong timings, correct text: the aligner must replace the times.
cat > "$TMP/wrong.srt" <<'EOF'
1
00:00:00,000 --> 00:00:01,000
And so my fellow Americans

2
00:00:01,000 --> 00:00:02,000
ask not what your country can do for you
EOF

"$BIN" --align-only -f "$AUDIO" -am "$ALIGNER" --text-file "$TMP/wrong.srt" \
       --output-srt > "$TMP/rt_file.srt" 2>/dev/null
"$BIN" --align-only -f "$AUDIO" -am "$ALIGNER" --text-file - \
       --output-srt < "$TMP/wrong.srt" > "$TMP/rt_stdin.srt" 2>/dev/null

check "srt output is non-empty" "$([ -s "$TMP/rt_file.srt" ] && echo 0 || echo 1)"
cmp -s "$TMP/rt_file.srt" "$TMP/rt_stdin.srt" && r=0 || r=1
check "srt stdin output == file output (content sniffing agrees with extension)" "$r"

# The point of re-timing: the bogus input times must NOT survive.
grep -q "00:00:01,000 --> 00:00:02,000" "$TMP/rt_file.srt" && r=1 || r=0
check "input timings were replaced, not echoed" "$r"

# Two cues in, two cues out — cue count is what SE relies on to map back.
n_in=$(grep -c -- "-->" "$TMP/wrong.srt")
n_out=$(grep -c -- "-->" "$TMP/rt_stdin.srt")
check "cue count preserved ($n_in in, $n_out out)" "$([ "$n_in" = "$n_out" ] && echo 0 || echo 1)"

# --- 4. empty stdin is an error, not a silent empty alignment -----------------
"$BIN" --align-only -f "$AUDIO" -am "$ALIGNER" --text-file - --output-srt < /dev/null > /dev/null 2>&1
check "empty stdin exits non-zero" "$([ $? -ne 0 ] && echo 0 || echo 1)"

echo
if [ "$fail" -eq 0 ]; then
    echo "test-align-stdin: all checks passed"
    exit 0
fi
echo "test-align-stdin: $fail check(s) failed"
exit 1

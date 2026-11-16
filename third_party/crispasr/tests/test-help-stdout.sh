#!/bin/bash
# test-help-stdout.sh — regression for #420: `crispasr --help` must be pipeable.
#
# An explicitly requested --help is the program's OUTPUT: it belongs on stdout
# with exit status 0, so `crispasr --help | less` and `crispasr --help | wc -l`
# behave the way every other Unix tool does. The reporter saw 0 lines because
# the whole 269-line usage text went to stderr; `2>&1` was needed to see any of
# it, which is exactly the redirection that should not be necessary.
#
# The mirror image matters just as much and is easy to regress into: usage
# printed because of a usage ERROR must go to stderr with a NON-ZERO status, so
# a typo'd flag is detectable in a script and never pollutes a pipeline.
#
# Offline: parses arguments only, never loads a model. SKIPs if the binary is
# missing.

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

PASS=0; FAIL=0

check() { # description | actual | expected
    if [ "$2" = "$3" ]; then
        echo "  ✓ $1 → $2"; PASS=$((PASS+1))
    else
        echo "  ✗ $1 → '$2' (expected '$3')"; FAIL=$((FAIL+1))
    fi
}

for flag in --help -h; do
    # The usage text lands on stdout. Compare against the 2>&1 line count so the
    # assertion tracks the real help text and does not go stale when options are
    # added or removed.
    out_lines=$("$CRISPASR" "$flag" 2>/dev/null | wc -l | tr -d ' ')
    all_lines=$("$CRISPASR" "$flag" 2>&1 | wc -l | tr -d ' ')
    check "$flag: stdout carries the whole usage text" "$out_lines" "$all_lines"

    # …and it is a substantial help text, not an empty stream that trivially
    # equals itself. A bound well under the real size (269 lines) so adding or
    # removing options never flakes this.
    if [ "$out_lines" -gt 50 ]; then
        echo "  ✓ $flag: usage is $out_lines lines on stdout"; PASS=$((PASS+1))
    else
        echo "  ✗ $flag: only $out_lines lines on stdout (expected > 50)"; FAIL=$((FAIL+1))
    fi

    # Nothing at all on stderr — that is what broke `--help | less`.
    err_bytes=$("$CRISPASR" "$flag" 2>&1 >/dev/null | wc -c | tr -d ' ')
    check "$flag: stderr is silent" "$err_bytes" "0"

    "$CRISPASR" "$flag" >/dev/null 2>&1
    check "$flag: exit status" "$?" "0"
done

# A usage ERROR is the mirror image: stderr, non-zero, nothing on stdout.
err_out=$("$CRISPASR" --no-such-flag-420 2>/dev/null | wc -l | tr -d ' ')
check "unknown flag: stdout stays clean" "$err_out" "0"

"$CRISPASR" --no-such-flag-420 >/dev/null 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "  ✓ unknown flag: non-zero exit ($rc)"; PASS=$((PASS+1))
else
    echo "  ✗ unknown flag: exit 0 — a typo would look like success"; FAIL=$((FAIL+1))
fi

# Capture first, then match. Piping straight into `grep -q` makes grep exit on
# the first hit, the producer takes SIGPIPE, and `set -o pipefail` turns that
# into a spurious failure — the diagnostic is present either way.
err_text=$("$CRISPASR" --no-such-flag-420 2>&1 >/dev/null)
case "$err_text" in
    *"unknown argument"*)
        echo "  ✓ unknown flag: diagnostic on stderr"; PASS=$((PASS+1)) ;;
    *)
        echo "  ✗ unknown flag: no diagnostic on stderr"; FAIL=$((FAIL+1)) ;;
esac

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1

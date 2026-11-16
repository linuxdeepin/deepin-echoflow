#!/usr/bin/env bash
# Verify every patch this fork carries on top of upstream ggml is still present.
#
# Reads ci/crispasr-patches.txt (see that file for the format and the rationale)
# and fails if any patch has gone missing — which is what an upstream merge does
# silently, without a single compile error, when it takes "theirs" on a hunk.
#
# Usage: ci/check-crispasr-patches.sh [manifest]

set -uo pipefail

MANIFEST="${1:-$(dirname "$0")/crispasr-patches.txt}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$MANIFEST" ]; then
    printf 'check-crispasr-patches: manifest not found: %s\n' "$MANIFEST" >&2
    exit 2
fi

fail=0
checked=0

while IFS=$'\t' read -r want file pattern; do
    # skip blanks and comments
    case "${want-}" in ''|\#*) continue ;; esac
    if [ -z "${file-}" ] || [ -z "${pattern-}" ]; then
        printf 'MALFORMED  %s\n' "$want" >&2
        fail=1
        continue
    fi

    checked=$((checked + 1))

    if [ ! -f "$ROOT/$file" ]; then
        printf 'MISSING FILE  %s  (expected >=%s of: %s)\n' "$file" "$want" "$pattern" >&2
        fail=1
        continue
    fi

    got=$(grep -c -F -- "$pattern" "$ROOT/$file" || true)
    if [ "$got" -lt "$want" ]; then
        printf 'PATCH LOST  %s  expected >=%s occurrences of "%s", found %s\n' \
            "$file" "$want" "$pattern" "$got" >&2
        fail=1
    fi
done < "$MANIFEST"

if [ "$fail" -ne 0 ]; then
    cat >&2 <<'EOF'

One or more CrispASR patches are missing from this tree.

If an upstream merge dropped them, re-apply them. If a patch was intentionally
retired (upstream fixed it, or we renamed it), update ci/crispasr-patches.txt in
the SAME commit and explain why in the commit message.
EOF
    exit 1
fi

printf 'check-crispasr-patches: OK — %s patches present\n' "$checked"

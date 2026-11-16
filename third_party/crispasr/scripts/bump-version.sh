#!/usr/bin/env bash
# Usage: scripts/bump-version.sh <version>   e.g.  scripts/bump-version.sh 0.9.0
#
# Updates VERSION, propagates to Cargo.toml / package.json / pyproject.toml etc.
# via sync-version.py, commits, and creates an annotated tag — all in one step.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>  (e.g. $0 0.9.0)" >&2
    exit 1
fi

VERSION="$1"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Anything already modified before we start is the caller's, not ours — record it
# so the release commit cannot sweep up unrelated work-in-progress.
PRE_DIRTY="$(git diff --name-only | sort)"

echo "$VERSION" > VERSION
python scripts/sync-version.py

# Stage VERSION plus exactly the files sync-version.py just touched.
#
# COMPUTED, not hardcoded. The old hardcoded list drifted silently: when
# python/crispasr/__init__.py was added to sync-version.py it was not added here,
# so v0.8.24 was first tagged on a tree whose crispasr.__version__ still read
# 0.8.23 while the wheel metadata claimed 0.8.24 — the kind of mismatch nobody
# notices until a user reports it. Diffing the working tree keeps the two scripts
# in step by construction.
git add VERSION
NOW_DIRTY="$(git diff --name-only | sort)"
TOUCHED="$(comm -13 <(printf '%s\n' "$PRE_DIRTY") <(printf '%s\n' "$NOW_DIRTY"))"
while IFS= read -r f; do
    [ -n "$f" ] && git add "$f"
done <<< "$TOUCHED"

echo "Staged for the release commit:"
git diff --cached --name-only | sed 's/^/  /'

git commit -m "release: bump VERSION to $VERSION"
git tag -a "v$VERSION" -m "Release v$VERSION"

echo ""
echo "Created commit + annotated tag v$VERSION."
echo "Push with:  git push && git push origin v$VERSION"

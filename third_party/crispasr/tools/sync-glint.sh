#!/usr/bin/env bash
# sync-glint.sh — sync the in-tree glint/ encoder copy from the glint
# repo's COMMITTED state (never a working tree, which may hold another
# session's WIP; see glint/README.md).
#
# Usage:
#   tools/sync-glint.sh                  # clone github.com/CrispStrobe/glint, sync to origin/main
#   tools/sync-glint.sh /path/to/glint   # use a local clone (still syncs its committed HEAD)
#   GLINT_REF=<sha|branch> tools/sync-glint.sh
#
# What it does:
#   1. git-archives src/, include/glint/glint.h, LICENSE into glint/
#      (old src/ files are removed first so upstream deletions propagate)
#   2. regenerates the add_library source list in glint/CMakeLists.txt
#      from upstream's GLINT_SOURCES (new TUs like aac_psy.cpp land
#      automatically)
#   3. rewrites the "Synced at upstream commit:" marker in
#      glint/README.md (the release workflow's freshness gate reads it)
#
# It does NOT commit — inspect `git status glint/`, build, run
# `test_tts_provenance "[mp3],[aac]"`, then commit. The sync-glint
# GitHub workflow automates exactly that.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GLINT_URL="${GLINT_URL:-https://github.com/CrispStrobe/glint}"
GLINT_REF="${GLINT_REF:-main}"

if [ $# -ge 1 ]; then
    GLINT_SRC="$1"
else
    GLINT_SRC="$(mktemp -d)/glint"
    trap 'rm -rf "$(dirname "$GLINT_SRC")"' EXIT
    echo "sync-glint: cloning $GLINT_URL (ref: $GLINT_REF)"
    git clone --quiet --depth 1 --branch "$GLINT_REF" "$GLINT_URL" "$GLINT_SRC"
fi

SHA="$(git -C "$GLINT_SRC" rev-parse "$GLINT_REF" 2>/dev/null || git -C "$GLINT_SRC" rev-parse HEAD)"
SUBJECT="$(git -C "$GLINT_SRC" log -1 --format=%s "$SHA")"
echo "sync-glint: upstream $SHA ($SUBJECT)"

# 1. Extract the committed tree (encoder core only)
rm -f "$REPO_ROOT"/glint/src/*.cpp "$REPO_ROOT"/glint/src/*.hpp \
      "$REPO_ROOT"/glint/include/glint/glint.h
git -C "$GLINT_SRC" archive "$SHA" -- src include/glint/glint.h LICENSE \
    | tar -x -C "$REPO_ROOT/glint/"

# 2 + 3. Regenerate the CMake source list and the README sync marker
UPSTREAM_CMAKE="$(git -C "$GLINT_SRC" show "$SHA:CMakeLists.txt")" \
SYNC_SHA="$SHA" SYNC_SUBJECT="$SUBJECT" python3 - "$REPO_ROOT" <<'EOF'
import os, re, sys

root = sys.argv[1]

m = re.search(r'set\(GLINT_SOURCES\n(.*?)\n\)', os.environ["UPSTREAM_CMAKE"], re.S)
if not m:
    sys.exit("sync-glint: GLINT_SOURCES block not found in upstream CMakeLists.txt")
sources = m.group(1)
for line in sources.splitlines():
    path = line.strip()
    if path and not os.path.exists(os.path.join(root, "glint", path)):
        sys.exit(f"sync-glint: GLINT_SOURCES lists {path} but it was not extracted")

p = os.path.join(root, "glint", "CMakeLists.txt")
s = open(p).read()
s2 = re.sub(r'(add_library\(glint STATIC\n).*?(\n\))', r'\1' + sources + r'\2', s, count=1, flags=re.S)
if s2 == s and sources not in s:
    sys.exit("sync-glint: add_library(glint STATIC ...) block not found in glint/CMakeLists.txt")
open(p, "w").write(s2)

p = os.path.join(root, "glint", "README.md")
s = open(p).read()
marker = f"Synced at upstream commit: `{os.environ['SYNC_SHA']}` ({os.environ['SYNC_SUBJECT']})."
s2, n = re.subn(r'Synced at upstream commit: `[0-9a-f]+` \(.*?\)\.', marker, s, count=1, flags=re.S)
if n != 1:
    sys.exit("sync-glint: 'Synced at upstream commit:' marker not found in glint/README.md")
open(p, "w").write(s2)
EOF

cd "$REPO_ROOT"
if git status --porcelain glint/ | grep -q .; then
    echo "sync-glint: glint/ updated — review, build, and run:"
    echo "  cmake --build build --target test_tts_provenance && ./build/bin/test_tts_provenance \"[mp3],[aac]\""
    git status --short glint/
else
    echo "sync-glint: already up to date"
fi

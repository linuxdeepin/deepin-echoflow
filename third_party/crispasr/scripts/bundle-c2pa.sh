#!/usr/bin/env bash
# bundle-c2pa.sh — copy the DYNAMIC c2pa sidecar lib next to a packaged binary so
# release tarballs are self-contained (the binary's rpath is @loader_path/$ORIGIN).
#
# No-op when C2PA wasn't fetched, or when the target links c2pa statically
# (wasm/iOS ship libc2pa_c.a → nothing to bundle). Safe to call unconditionally
# in a Package step; never fails the build.
#
# Usage: scripts/bundle-c2pa.sh <dir-containing-the-binary>
set -uo pipefail

DEST="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBDIR="$REPO_ROOT/third_party/c2pa/lib"

[ -n "$DEST" ] && [ -d "$DEST" ] || { echo "bundle-c2pa: no dest dir '$DEST'; skipping"; exit 0; }

copied=0
for f in libc2pa_c.so libc2pa_c.dylib c2pa_c.dll; do
    if [ -f "$LIBDIR/$f" ]; then
        cp -f "$LIBDIR/$f" "$DEST/" && echo "bundle-c2pa: copied $f → $DEST" && copied=1
    fi
done
[ "$copied" -eq 0 ] && echo "bundle-c2pa: no dynamic c2pa lib to bundle (static or not fetched) — skipping"
exit 0

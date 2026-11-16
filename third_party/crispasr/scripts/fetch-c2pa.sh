#!/usr/bin/env bash
# fetch-c2pa.sh — download the prebuilt c2pa-rs native C library + header so the
# CMake build can enable C2PA (Content Credentials) signing.
#
# Installs into third_party/c2pa/{include,lib} by default; point CMake at it with
# -DCMAKE_PREFIX_PATH=<repo>/third_party/c2pa (the CLI CMakeLists also probes
# that path automatically). Re-run to refresh; idempotent if already present.
#
# Usage: scripts/fetch-c2pa.sh [install-dir] [version]
set -euo pipefail

C2PA_VERSION="${2:-0.89.3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$REPO_ROOT/third_party/c2pa}"

# Map host OS/arch → c2pa-rs release target triple.
uname_s="$(uname -s)"
uname_m="$(uname -m)"
case "$uname_s-$uname_m" in
    Darwin-arm64|Darwin-aarch64) TARGET="aarch64-apple-darwin" ;;
    Darwin-x86_64)               TARGET="x86_64-apple-darwin" ;;
    Linux-x86_64)                TARGET="x86_64-unknown-linux-gnu" ;;
    Linux-aarch64|Linux-arm64)   TARGET="aarch64-unknown-linux-gnu" ;;
    *) echo "fetch-c2pa: unsupported host $uname_s-$uname_m; download manually from" >&2
       echo "  https://github.com/contentauth/c2pa-rs/releases" >&2; exit 1 ;;
esac

ASSET="c2pa-v${C2PA_VERSION}-${TARGET}.zip"
URL="https://github.com/contentauth/c2pa-rs/releases/download/c2pa-v${C2PA_VERSION}/${ASSET}"

if [ -f "$DEST/include/c2pa.h" ] && ls "$DEST"/lib/libc2pa_c.* >/dev/null 2>&1; then
    echo "fetch-c2pa: already present at $DEST (v$C2PA_VERSION). Delete to refresh."
    exit 0
fi

echo "fetch-c2pa: downloading $ASSET"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fsSL --retry 5 --retry-delay 3 \
    --connect-timeout 30 --max-time 120 -o "$TMP/c2pa.zip" "$URL"
mkdir -p "$DEST"
( cd "$DEST" && unzip -oq "$TMP/c2pa.zip" )

if [ ! -f "$DEST/include/c2pa.h" ]; then
    echo "fetch-c2pa: unexpected archive layout (no include/c2pa.h)" >&2
    exit 1
fi
echo "fetch-c2pa: installed c2pa v$C2PA_VERSION ($TARGET) → $DEST"
echo "  Configure CrispASR with: -DCMAKE_PREFIX_PATH=$DEST"

#!/usr/bin/env bash
# bundle-openblas.sh — copy the OpenBLAS runtime next to a packaged Linux binary
# so the release tarball is self-contained.
#
# Why this exists (#296). Linking OpenBLAS made `--separate`
# (mel-band-roformer) ~26x faster, and the Linux release jobs `apt-get install
# libopenblas-dev` so the fast SGEMM path is compiled in. But apt gives a
# DYNAMIC libopenblas.so.0, so the shipped `crispasr` carries
#
#     NEEDED  libopenblas.so.0
#
# while the tarball contained only crispasr, crispasr-quantize and
# libc2pa_c.so. Verified against the published v0.8.25
# crispasr-linux-x86_64.tar.gz. On any machine without OpenBLAS installed that
# is not a slow fallback — the binary does not start at all:
#
#     error while loading shared libraries: libopenblas.so.0
#
# The binary's RUNPATH is `$ORIGIN:…` (also verified on the shipped artifact),
# so a copy sitting beside the executable resolves without LD_LIBRARY_PATH —
# exactly how libc2pa_c.so already works. This script is the OpenBLAS half of
# scripts/bundle-c2pa.sh and is deliberately shaped the same.
#
# Copies the SONAME target (libopenblas.so.0), not the -dev symlink, because
# that is the name in NEEDED.
#
# Usage: scripts/bundle-openblas.sh <dir-containing-the-binary> [binary-name]
set -uo pipefail

DEST="${1:-}"
BIN="${2:-crispasr}"

[ -n "$DEST" ] && [ -d "$DEST" ] || { echo "bundle-openblas: no dest dir '$DEST'; skipping"; exit 0; }

# Nothing to do when the binary does not actually depend on OpenBLAS (a build
# with CRISPASR_MEL_BLAS=OFF, a static link, or a non-Linux package).
if command -v objdump >/dev/null 2>&1 && [ -f "$DEST/$BIN" ]; then
    if ! objdump -p "$DEST/$BIN" 2>/dev/null | grep -q 'NEEDED.*libopenblas'; then
        echo "bundle-openblas: $BIN has no libopenblas NEEDED entry — nothing to bundle"
        exit 0
    fi
fi

# Ask the loader where it resolves to rather than guessing a path: Ubuntu has
# moved libopenblas between /usr/lib/x86_64-linux-gnu and
# /usr/lib/<triplet>/openblas-pthread/ across releases.
SRC=""
if command -v ldconfig >/dev/null 2>&1; then
    SRC="$(ldconfig -p 2>/dev/null | awk '/libopenblas\.so\.0/ {print $NF; exit}')"
fi
if [ -z "$SRC" ] || [ ! -f "$SRC" ]; then
    for c in /usr/lib/x86_64-linux-gnu/libopenblas.so.0 \
             /usr/lib/aarch64-linux-gnu/libopenblas.so.0 \
             /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0 \
             /usr/lib/aarch64-linux-gnu/openblas-pthread/libopenblas.so.0; do
        [ -f "$c" ] && SRC="$c" && break
    done
fi

if [ -z "$SRC" ] || [ ! -f "$SRC" ]; then
    echo "bundle-openblas: WARNING — $BIN needs libopenblas.so.0 but none was found to bundle;"
    echo "                 the tarball will NOT run on a machine without OpenBLAS installed."
    exit 0
fi

# Follow the symlink so the tarball holds a real file, then name it exactly what
# NEEDED asks for.
cp -fL "$SRC" "$DEST/libopenblas.so.0" && echo "bundle-openblas: copied $SRC → $DEST/libopenblas.so.0"
exit 0

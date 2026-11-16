#!/usr/bin/env bash
# Verify a relocated, extracted macOS CLI release before uploading it.
set -euo pipefail

BUNDLE="${1:?usage: verify-macos-cli.sh <bundle> <arm64|x86_64>}"
ARCH="${2:?expected architecture required}"
case "$ARCH" in
    arm64|x86_64) ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

for name in crispasr crispasr-quantize libc2pa_c.dylib; do
    file="$BUNDLE/$name"
    test -f "$file"
    actual="$(lipo -archs "$file")"
    if [ "$actual" != "$ARCH" ]; then
        echo "$name: expected $ARCH, found $actual" >&2
        exit 1
    fi
    echo "$name: $actual"
done

# Run without a dyld search-path override so the bundle must find its own
# sidecar. The quantizer's usage exit is 1; a dyld/crash exit must fail.
unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH
"$BUNDLE/crispasr" --version
status=0
"$BUNDLE/crispasr-quantize" > "$BUNDLE/quantize-startup.log" 2>&1 || status=$?
cat "$BUNDLE/quantize-startup.log"
test "$status" -eq 1
grep -q '^usage:' "$BUNDLE/quantize-startup.log"
rm "$BUNDLE/quantize-startup.log"
echo "macOS $ARCH CLI bundle verified"

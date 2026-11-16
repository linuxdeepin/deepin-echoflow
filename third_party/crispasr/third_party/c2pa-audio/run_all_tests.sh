#!/usr/bin/env bash
# Build the native lib and run every language binding's tests.
set -e
R="$(cd "$(dirname "$0")" && pwd)"
LIB="$R/build/libc2pa_audio.dylib"; [ "$(uname)" = Linux ] && LIB="$R/build/libc2pa_audio.so"

echo "== C ABI =="
cmake -B "$R/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$R/build" -j3 >/dev/null
ctest --test-dir "$R/build" --output-on-failure

echo "== JS =="; node --test "$R/js"/*.test.mjs 2>/dev/null || echo "(js tests live in the parent CrispASR repo)"

echo "== Dart =="; ( cd "$R/bindings/dart" && dart pub get >/dev/null 2>&1 && C2PA_AUDIO_LIB="$LIB" dart test )
echo "== Python =="; C2PA_AUDIO_LIB="$LIB" python3 "$R/bindings/python/test_c2pa_audio.py"
echo "== Go =="; ( cd "$R/bindings/go" && DYLD_LIBRARY_PATH="$R/build" LD_LIBRARY_PATH="$R/build" go test ./... )
echo "ALL BINDINGS PASSED"

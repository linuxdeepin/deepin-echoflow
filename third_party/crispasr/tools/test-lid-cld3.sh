#!/usr/bin/env bash
# Smoke-test runner for the lid-cld3 backend.
#
# For each of 8 multilingual inputs covering Latin / Cyrillic / CJK /
# Hiragana / Devanagari, this dumps a Python F32 reference via
# tools/dump_reference.py and runs crispasr-diff against the F16
# production GGUF — the same gate the May 2026 LID release went
# through. Exits non-zero if any sample fails any stage.
#
# Required env / setup:
#   * DYLD_LIBRARY_PATH set automatically (pycld3 oracle workaround)
#   * /Volumes/backups/ai/upstream/cld3-runtime-deps/lib present
#   * /Volumes/backups/ai/crispasr-models/lid-cld3/cld3-{f32,f16}.gguf
#     produced by `python models/convert-cld3-to-gguf.py [--f32]`
#   * `cmake --build build-ninja-compile --target crispasr-diff`
#
# Usage:
#   ./tools/test-lid-cld3.sh
#   ./tools/test-lid-cld3.sh --quiet   # show only the per-input summary line

set -e
cd "$(dirname "$0")/.."

DYLD=/Volumes/backups/ai/upstream/cld3-runtime-deps/lib
F32=/Volumes/backups/ai/crispasr-models/lid-cld3/cld3-f32.gguf
F16=/Volumes/backups/ai/crispasr-models/lid-cld3/cld3-f16.gguf
BIN=./build-ninja-compile/bin/crispasr-diff

quiet=0
[[ "${1:-}" == "--quiet" ]] && quiet=1

if [[ ! -f "$F32" || ! -f "$F16" ]]; then
    echo "missing GGUFs at /Volumes/backups/ai/crispasr-models/lid-cld3/" >&2
    echo "  run: python models/convert-cld3-to-gguf.py" >&2
    echo "  and: python models/convert-cld3-to-gguf.py --f32 --out $F32" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "missing $BIN — run: cmake --build build-ninja-compile --target crispasr-diff" >&2
    exit 1
fi

inputs=(
    "Hello world"
    "Hallo Welt"
    "Bonjour le monde"
    "Привет мир"
    "你好世界"
    "こんにちは世界"
    "नमस्ते दुनिया"
    "Olá mundo"
)

pass=0
fail=0
for t in "${inputs[@]}"; do
    ref=/tmp/cld3-ref-$$-${pass}-${fail}.gguf
    DYLD_LIBRARY_PATH="$DYLD" LID_TEXT="$t" python tools/dump_reference.py \
        --backend lid-cld3 --model-dir "$F32" --audio samples/jfk.wav \
        --output "$ref" >/dev/null 2>&1
    out=$("$BIN" lid-cld3 "$F16" "$ref" samples/jfk.wav 2>&1)
    rm -f "$ref"
    summary=$(echo "$out" | grep "^summary:" || true)
    top1=$(echo "$out" | grep "^\[INFO\] top1_label" | head -1 || true)
    if echo "$summary" | grep -q "0 fail"; then
        pass=$((pass + 1))
        if [[ $quiet -eq 0 ]]; then
            printf "  ✓ %-22s  %s  %s\n" "${t}" "${summary}" "${top1}"
        else
            printf "  ✓ %s\n" "$t"
        fi
    else
        fail=$((fail + 1))
        printf "  ✗ %-22s  %s\n" "${t}" "${summary}"
        echo "$out" | grep -E "(FAIL|ERR)" | sed 's/^/      /'
    fi
done

echo "---"
echo "TOTAL: $pass pass / $fail fail (out of ${#inputs[@]})"
[[ $fail -eq 0 ]]

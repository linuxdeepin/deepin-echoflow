#!/usr/bin/env bash
# test-chunked-words.sh — regression test for issue #89: words must
# survive --chunk-seconds boundaries on parakeet-tdt-0.6b-ja.
#
# The 617cd02 round added ±2 s context + word-level trim that dropped
# words and split every kana with a space on parakeet-ja; the issue
# #89 round-2 revert restored the no-context behaviour. This test
# pins that behaviour by running the JA concat at several chunk sizes
# and asserting:
#
#   1. Output contains content from every sample (substring match).
#   2. No character-level spacing — "ピ ッ チャ ー" is the symptom.
#   3. Default 30 s and small 10 s chunks both work.
#
# Skips if parakeet-tdt-0.6b-ja.gguf isn't available in the HF cache
# (won't block CI on machines that don't have the model).
#
# Usage: ./tests/test-chunked-words.sh
# Exit:  0 = pass, 1 = fail, 2 = skipped (model missing).

set -euo pipefail
cd "$(dirname "$0")/.."

CRISPASR="${CRISPASR_BIN:-./build-ninja-compile/bin/crispasr}"
if [ ! -x "$CRISPASR" ]; then
    CRISPASR="./build/bin/crispasr"
fi

# Resolve the parakeet-ja model from the standard HF cache. Skip if missing.
HF_CACHE="${HUGGINGFACE_HUB_CACHE:-/Volumes/backups/ai/huggingface-hub}"
GGUF_GLOB="$HF_CACHE/models--cstr--parakeet-tdt-0.6b-ja-GGUF/snapshots/*/parakeet-tdt-0.6b-ja.gguf"
# shellcheck disable=SC2086
MODEL=$(ls $GGUF_GLOB 2>/dev/null | head -1 || true)
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP: parakeet-tdt-0.6b-ja.gguf not in $HF_CACHE — set HUGGINGFACE_HUB_CACHE or run hf_hub_download first"
    exit 2
fi

if [ ! -x "$CRISPASR" ]; then
    echo "ERROR: crispasr binary not found (tried $CRISPASR)"
    exit 1
fi

# Build a long enough JA concat from the in-tree samples to force chunking.
SAMPLES="samples/ja/reazon_baseball_14s.wav samples/ja/reazon_meal_11s.wav samples/ja/reazon_raft_8s.wav"
for f in $SAMPLES; do
    if [ ! -f "$f" ]; then
        echo "SKIP: $f not in repo (samples/ja/.gitignore excludes them; download from cstr/crispasr-regression-fixtures)"
        exit 2
    fi
done

WAV=$(mktemp -t ja-concat.XXXX).wav
trap 'rm -f "$WAV"' EXIT

# shellcheck disable=SC2086
ffmpeg -y -i samples/ja/reazon_baseball_14s.wav \
       -i samples/ja/reazon_meal_11s.wav \
       -i samples/ja/reazon_raft_8s.wav \
    -filter_complex "[0][1][2]concat=n=3:v=0:a=1" \
    -ar 16000 -ac 1 "$WAV" >/dev/null 2>&1

FAILED=0
run_chunk_test() {
    local cs="$1"
    local out
    out=$("$CRISPASR" --backend parakeet -m "$MODEL" -l ja \
        --chunk-seconds "$cs" --no-prints \
        -f "$WAV" 2>/dev/null | tr -d '\r')

    # Sanity: at least 50 chars of output (otherwise the model itself failed).
    local chars
    chars=$(printf "%s" "$out" | wc -m | tr -d ' ')
    if [ "$chars" -lt 50 ]; then
        echo "  FAIL cs=${cs}s: output only $chars chars — model returned almost nothing"
        FAILED=$((FAILED + 1))
        return
    fi

    # Each sample's distinctive kanji should appear at least once.
    local missing=""
    # baseball: 岡本 (Okamoto)
    grep -q "岡本" <<<"$out" || missing="${missing} 岡本"
    # meal: 手料理 (home cooking)
    grep -q "手料理" <<<"$out" || missing="${missing} 手料理"
    # raft: 丸太 (logs)
    grep -q "丸太" <<<"$out" || missing="${missing} 丸太"

    # The 617cd02-era space-per-kana bug — no two single kana should be
    # space-separated. We look for the pattern "<kana> <kana>" where each
    # is a single-codepoint katakana. Using ピ ッ チャ as the canary.
    if grep -q "ピ ッ" <<<"$out"; then
        echo "  FAIL cs=${cs}s: space-per-kana detected ('ピ ッ' — 617cd02 trim regression)"
        FAILED=$((FAILED + 1))
        return
    fi

    if [ -n "$missing" ]; then
        echo "  FAIL cs=${cs}s: missing kanji from samples:${missing}"
        echo "  output: $out"
        FAILED=$((FAILED + 1))
    else
        echo "  PASS cs=${cs}s (${chars} chars)"
    fi
}

echo "Issue #89 round 2 — parakeet-ja chunk-boundary word integrity"
echo "model:  $MODEL"
echo "audio:  3-sample JA concat (~33 s)"
echo
for cs in 5 10 15 20 30; do
    run_chunk_test "$cs"
done

echo
if [ "$FAILED" -gt 0 ]; then
    echo "FAILED: $FAILED of 5 chunk sizes"
    exit 1
fi
echo "OK: all chunk sizes preserve sample content + no space-per-kana"
exit 0

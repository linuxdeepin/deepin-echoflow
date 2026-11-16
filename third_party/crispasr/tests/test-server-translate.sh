#!/bin/bash
# test-server-translate.sh — integration test for POST /v1/translate.
#
# The HTTP analogue of the CLI `--text` mode: text-to-text translation backed by
# a CAP_TRANSLATE backend (m2m100). Boots the server with m2m100 and asserts an
# en→de translation comes back non-empty. SKIPs (exit 0) without a model.
#
# Usage: ./tests/test-server-translate.sh [--port N] [--cache-dir=DIR]

set -uo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-11457}
CACHE_DIR="${CRISPASR_TEST_CACHE:-}"
for arg in "$@"; do
    case "$arg" in
        --port=*) PORT="${arg#--port=}" ;;
        --cache-dir=*) CACHE_DIR="${arg#--cache-dir=}" ;;
    esac
done
CACHE_ARG=()
[ -n "$CACHE_DIR" ] && CACHE_ARG=(--cache-dir "$CACHE_DIR")

CRISPASR=""
for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
done
[ -n "$CRISPASR" ] || { echo "SKIP: crispasr binary not found"; exit 0; }

# Locate an m2m100 GGUF.
MODEL="${M2M100_MODEL:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    MODEL=""
    for d in "$CACHE_DIR" "${CRISPASR_MODELS:-}" "${CRISPASR_MODELS_DIR:-}" \
             /Volumes/backups/ai/crispasr-models /mnt/storage/gguf-models "$HOME/.cache/crispasr"; do
        [ -n "$d" ] && [ -d "$d" ] || continue
        cand=$(ls "$d"/m2m100*.gguf 2>/dev/null | head -1)
        if [ -n "$cand" ] && [ -f "$cand" ]; then MODEL="$cand"; break; fi
    done
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "SKIP: no m2m100 GGUF found (set M2M100_MODEL or CRISPASR_TEST_CACHE)"; exit 0; }

echo "Model: $MODEL"
LOG=$(mktemp -t crispasr-translate.XXXXXX)
trap 'kill "$SV" 2>/dev/null; rm -f "$LOG"' EXIT
"$CRISPASR" --server -m "$MODEL" --backend m2m100 --host 127.0.0.1 --port "$PORT" \
    --auto-download ${CACHE_ARG[@]+"${CACHE_ARG[@]}"} > "$LOG" 2>&1 &
SV=$!
ready=0
for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$SV" 2>/dev/null || break
    sleep 1
done
[ "$ready" = 1 ] || { echo "ERROR: server not ready. Log:"; tail -5 "$LOG"; exit 2; }

PASS=0; FAIL=0
resp=$(curl -s -X POST -H 'Content-Type: application/json' \
    -d '{"input":"Hello world, how are you today?","source_lang":"en","target_lang":"de"}' \
    "http://127.0.0.1:$PORT/v1/translate")
text=$(echo "$resp" | sed -n 's/.*"text"[[:space:]]*:[[:space:]]*"\(.*\)".*/\1/p')
echo "translation: $text"
# Expect a non-empty German rendering (contains a German function word).
if [ -n "$text" ] && echo "$text" | grep -qiE "Hallo|Welt|wie|heute|Sie|du"; then
    echo "  ✓ en→de translation returned"; PASS=$((PASS+1))
else
    echo "  ✗ unexpected/empty translation: '$text'"; FAIL=$((FAIL+1))
fi
# Missing input → 400.
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H 'Content-Type: application/json' \
    -d '{"source_lang":"en","target_lang":"de"}' "http://127.0.0.1:$PORT/v1/translate")
if [ "$code" = "400" ]; then echo "  ✓ missing input → 400"; PASS=$((PASS+1)); else echo "  ✗ missing input → $code"; FAIL=$((FAIL+1)); fi

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1

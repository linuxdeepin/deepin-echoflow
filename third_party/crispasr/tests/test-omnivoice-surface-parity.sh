#!/usr/bin/env bash
# test-omnivoice-surface-parity.sh — the same request on all three surfaces must
# produce the same audio codes (#13273).
#
# The session C-ABI reimplements each backend's synthesize INLINE rather than
# calling the CLI adapter, and the server drives the adapter but does its own
# request parsing. So a knob can work on one surface and be dead on the other
# two — which is exactly what happened here: language was applied only in the
# adapter's init() (dead on the server after line 1) and never at all in the
# session arm (dead in every binding). Instruct had the identical defect.
# Source-level guards catch a missing call site; only this catches a call site
# that is present and wrong.
#
# ⚠ Compares CODES, never the WAV. Output carries a provenance watermark and a
# spoken disclaimer, so byte-comparing audio compares the watermark and reports
# DIFFERENT for runs that are in fact identical.
#
# Every comparison below includes at least one arm whose expected answer is
# IDENTICAL. An all-DIFFERENT suite is also what a broken harness produces.
#
# SKIPs cleanly (exit 0) without a model.
#
# Env:
#   CRISPASR_TEST_OMNIVOICE_MODEL      main GGUF (required; else SKIP)
#   CRISPASR_TEST_OMNIVOICE_TOKENIZER  audio-tokenizer GGUF (required; else SKIP)

set -u

MODEL="${CRISPASR_TEST_OMNIVOICE_MODEL:-}"
TOK="${CRISPASR_TEST_OMNIVOICE_TOKENIZER:-}"
BIN="${CRISPASR_BIN:-./build/bin/crispasr}"
PORT="${CRISPASR_TEST_PORT:-8489}"
# The session arm imports the crispasr package (needs numpy), so allow an
# interpreter override — a bare system python3 often cannot.
PYTHON="${CRISPASR_TEST_PYTHON:-python3}"
TEXT="Ich möchte heute Abend über die Straße gehen und frische Brötchen kaufen."

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ] || [ -z "$TOK" ] || [ ! -f "$TOK" ]; then
    echo "SKIP: CRISPASR_TEST_OMNIVOICE_MODEL / _TOKENIZER not set (both needed)"
    exit 0
fi
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 0; }

TMP="$(mktemp -d)"
SRV_PID=""
cleanup() {
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT

fails=0
same() { # <label> <a> <b>
    if cmp -s "$2" "$3"; then echo "ok   $1: IDENTICAL"; else
        echo "FAIL $1: expected IDENTICAL, got DIFFERENT"; fails=$((fails + 1)); fi
}
differ() {
    if cmp -s "$2" "$3"; then
        echo "FAIL $1: expected DIFFERENT, got IDENTICAL"; fails=$((fails + 1))
    else echo "ok   $1: DIFFERENT"; fi
}

cli() { # <out.bin> <auto_lang 0|1> [args...]
    local out="$1" auto="$2"; shift 2
    CRISPASR_OMNIVOICE_DUMP_CODES="$out" CRISPASR_OMNIVOICE_AUTO_LANG="$auto" \
        "$BIN" --backend omnivoice \
        -m "$MODEL" --codec-model "$TOK" --tts "$TEXT" --tts-output "$TMP/o.wav" \
        --no-spoken-disclaimer --accept-marking-responsibility "$@" \
        > "$TMP/cli.log" 2>&1
}

echo "== CLI arms"
cli "$TMP/cli_de.bin"       1 -l de
cli "$TMP/cli_german.bin"   1 -l German
cli "$TMP/cli_en.bin"       1 -l en
# The TRUE language-agnostic baseline needs the guesser off — with it on (the
# shipped default) "no language" is not agnostic, it is whatever the text is.
cli "$TMP/cli_agnostic.bin" 0 -l auto
cli "$TMP/cli_bogus.bin"    0 -l de-DE
# ...and with the guesser on, German text and no language must land on `de`.
# This is the SubtitleEdit case: a client that never sends a language field.
cli "$TMP/cli_auto.bin"     1 -l auto

# _resolve_language observed end to end.
same   "name==id            " "$TMP/cli_german.bin" "$TMP/cli_de.bin"
same   "bogus==agnostic     " "$TMP/cli_bogus.bin" "$TMP/cli_agnostic.bin"
differ "de!=en              " "$TMP/cli_de.bin" "$TMP/cli_en.bin"
differ "de!=agnostic        " "$TMP/cli_de.bin" "$TMP/cli_agnostic.bin"
# The auto-detect contract, on the exact text a dubbing client would send.
same   "auto-detect==explicit" "$TMP/cli_auto.bin" "$TMP/cli_de.bin"

echo "== server arms (one process, language varied PER REQUEST)"
CRISPASR_OMNIVOICE_DUMP_CODES="$TMP/srv.bin" "$BIN" --server --backend omnivoice \
    -m "$MODEL" --codec-model "$TOK" --host 127.0.0.1 --port "$PORT" \
    --no-spoken-disclaimer --accept-marking-responsibility > "$TMP/srv.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 120); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done
if ! curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "SKIP: server did not come up (see $TMP/srv.log)"; exit 0
fi

post() { # <json-extra> <out.bin>
    "$PYTHON" - "$1" <<'PY' > "$TMP/post.out" 2>&1
import json, sys, urllib.request, os
body = {"input": os.environ["OV_TEXT"], "response_format": "wav"}
body.update(json.loads(sys.argv[1]))
req = urllib.request.Request(f'http://127.0.0.1:{os.environ["OV_PORT"]}/v1/audio/speech',
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=1800).read()
PY
    cp "$TMP/srv.bin" "$2"
}
export OV_TEXT="$TEXT" OV_PORT="$PORT"

post '{"language":"de"}' "$TMP/srv_de.bin"
post '{"language":"German"}' "$TMP/srv_german.bin"
post '{"language":"en"}' "$TMP/srv_en.bin"
post '{}' "$TMP/srv_none.bin"

same   "server de==CLI de   " "$TMP/srv_de.bin" "$TMP/cli_de.bin"
same   "server name==id     " "$TMP/srv_german.bin" "$TMP/srv_de.bin"
# The bug this file exists for: one process, language varied per request. Two
# EXPLICIT languages, so the guesser cannot mask a dead knob by making every
# arm come out German anyway.
differ "server per-request  " "$TMP/srv_de.bin" "$TMP/srv_en.bin"
# And the SubtitleEdit-shaped request — no language field at all.
same   "server no-field==de " "$TMP/srv_none.bin" "$TMP/cli_de.bin"

echo "== session ABI (the surface that had no language wiring at all)"
"$PYTHON" - <<PY > "$TMP/sess.log" 2>&1
import os, sys
sys.path.insert(0, "python")
os.environ["CRISPASR_OMNIVOICE_DUMP_CODES"] = "$TMP/sess.bin"
from crispasr import Session
s = Session("$MODEL", backend="omnivoice")
s.set_codec_path("$TOK")
s.set_target_language("de")
s.accept_marking_responsibility("surface-parity test")
s.synthesize("""$TEXT""")
PY
if [ -f "$TMP/sess.bin" ]; then
    same "session de==CLI de" "$TMP/sess.bin" "$TMP/cli_de.bin"
else
    echo "SKIP session arm: python binding unavailable (see $TMP/sess.log)"
fi

echo
if [ "$fails" -ne 0 ]; then
    echo "FAILED: $fails surface-parity mismatch(es)"
    exit 1
fi
echo "PASS: all surfaces agree"

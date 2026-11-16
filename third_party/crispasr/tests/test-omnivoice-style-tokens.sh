#!/usr/bin/env bash
# test-omnivoice-style-tokens.sh — OmniVoice prompt-token parity, as a gate.
#
# tests/test-omnivoice-prompt.cpp pins the style STRING hermetically. This pins
# what that string TOKENIZES TO against the real GGUF vocabulary, which is the
# thing the model actually consumes and the one link the unit tests cannot see.
#
# Every expected id sequence below was produced by the reference tokenizer:
#
#   from transformers import AutoTokenizer
#   tok = AutoTokenizer.from_pretrained("<omnivoice-lm-src>")   # Qwen2Tokenizer
#   tok("<|lang_start|>de<|lang_end|><|instruct_start|>None<|instruct_end|>").input_ids
#
# so a mismatch means our tokenizer, our vocabulary export, or our style
# assembly has drifted from the blueprint — all three of which produce audio
# that sounds fine while being conditioned on something nobody asked for
# (#13273). Note Qwen2 has no BOS, which is why no id leads these sequences.
#
# Provenance of the pinned ids (so a red test can tell WHICH side moved
# without vendoring the 7 MB tokenizer.json):
#
#   k2-fsa/OmniVoice  revision c5fdb5ccb189668d56333f77ba2629f4cd7535f4
#   tokenizer.json    sha256 408f669b7e2b045fdf54201d815bd364e6667dbd845115da81239c40bc6dcfd1
#
# All 9 sequences re-derived byte-identical from that exact file on
# 2026-08-07 (tokenizers 0.23.1). If this test goes red:
#
#   hf download k2-fsa/OmniVoice tokenizer.json --local-dir /tmp/ov-tok
#   shasum -a 256 /tmp/ov-tok/tokenizer.json
#
#   same sha256      -> upstream is unchanged; OUR tokenizer/export/assembly
#                       drifted — fix our side, do not touch the pins.
#   different sha256 -> upstream shipped a new tokenizer; re-derive the pins
#                       from the new file AND regenerate the GGUF vocabulary
#                       (models/convert-omnivoice-to-gguf.py embeds
#                       tokenizer.json), then update BOTH shas above.
#
# SKIPs cleanly (exit 0) when no model is configured — the ids are useless
# without the vocabulary they came from.
#
# Env:
#   CRISPASR_TEST_OMNIVOICE_MODEL      main GGUF (required to run; else SKIP)
#   CRISPASR_TEST_OMNIVOICE_TOKENIZER  audio-tokenizer GGUF (optional)

set -u

MODEL="${CRISPASR_TEST_OMNIVOICE_MODEL:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP: CRISPASR_TEST_OMNIVOICE_MODEL not set or missing"
    exit 0
fi

BIN="${CRISPASR_BIN:-./build/bin/crispasr}"
if [ ! -x "$BIN" ]; then
    echo "SKIP: $BIN not built"
    exit 0
fi

TOK="${CRISPASR_TEST_OMNIVOICE_TOKENIZER:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fails=0

# run_case <name> <expected-ids> <text> [extra crispasr args...]
run_case() {
    local name="$1" expect="$2" text="$3"
    shift 3
    local args=(--backend omnivoice -m "$MODEL" --tts "$text"
                --tts-output "$TMP/$name.wav" --tts-steps 1)
    [ -n "$TOK" ] && args+=(--codec-model "$TOK")

    # AUTO_LANG off so a case that omits -l really does exercise the "None"
    # path rather than whatever the detector makes of the sample text.
    if ! CRISPASR_OMNIVOICE_DEBUG=1 CRISPASR_OMNIVOICE_AUTO_LANG=0 \
        "$BIN" "${args[@]}" "$@" > "$TMP/$name.log" 2>&1; then
        # A non-zero exit is only a failure if it is not the model refusing to
        # load; the style line is printed long before synthesis completes.
        :
    fi

    local got
    got="$(sed -n 's/.*omnivoice: style .* -> [0-9]* ids: *//p' "$TMP/$name.log" | head -1)"
    if [ -z "$got" ]; then
        echo "FAIL $name: no style-ids line (see $TMP/$name.log)"
        sed -n '1,5p' "$TMP/$name.log"
        fails=$((fails + 1))
        return
    fi
    if [ "$got" != "$expect" ]; then
        echo "FAIL $name"
        echo "  expected: $expect"
        echo "  got:      $got"
        fails=$((fails + 1))
        return
    fi
    echo "ok   $name"
}

# --- language slot -----------------------------------------------------------
run_case lang_none "151670 4064 151671 151672 4064 151673" "Hello there."
run_case lang_de "151670 450 151671 151672 4064 151673" "Hallo Welt." -l de
run_case lang_en "151670 268 151671 151672 4064 151673" "Hello there." -l en
run_case lang_zh "151670 23815 151671 151672 4064 151673" "你好世界" -l zh
run_case lang_arb "151670 75270 151671 151672 4064 151673" "Hello there." -l arb

# An English NAME must resolve to the same ids as its id — this is the
# blueprint's _resolve_language() observed at the token level.
run_case lang_name_german "151670 450 151671 151672 4064 151673" "Hallo Welt." -l German

# And an unrecognized tag must land on "None", never on itself: identical to
# lang_none above. This is the original #13273 defect, seen in the prompt.
run_case lang_bogus "151670 4064 151671 151672 4064 151673" "Hello there." -l de-DE

# --- instruct slot -----------------------------------------------------------
# Natural capitalisation must normalise to the trained spelling.
run_case instruct_case "151670 268 151671 151672 36476 11 93927 29100 151673" \
    "Hello there." -l en --instruct "Male, British Accent"

# An English instruct against Chinese text unifies to Chinese (_resolve_instruct).
run_case instruct_zh "151670 23815 151671 151672 70108 3837 106487 151673" \
    "我们明天一起去公园散步吧" -l zh --instruct "Male, Elderly"

echo
if [ "$fails" -ne 0 ]; then
    echo "FAILED: $fails style-token mismatch(es)"
    exit 1
fi
echo "PASS: style tokens match the reference tokenizer on all cases"

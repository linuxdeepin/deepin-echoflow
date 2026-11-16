#!/bin/bash
# test-kokoro-g2p-live.sh — #316 end-to-end: numbers are spoken, phonemes drive
# the model, and the lexicon auto-downloads.
#
# The unit tests cover the G2P in isolation; only a real synthesis proves the
# reported bug is gone, because the defect was invisible to every tensor-level
# check (crispasr-diff for kokoro is phoneme-IN, so it starts downstream of the
# G2P entirely).
#
# Needs: a kokoro GGUF + voice pack, an ASR model for the round-trip, network
# for the lexicon. SKIPs (exit 0) when any is missing — never a false red.
set -uo pipefail
cd "$(dirname "$0")/.."

CRISPASR=""
for c in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
    [ -x "$c" ] && { CRISPASR="$c"; break; }
done
[ -z "$CRISPASR" ] && { echo "SKIP: crispasr binary not found"; exit 0; }

MODELS="${CRISPASR_MODELS_DIR:-${CRISPASR_MODELS:-$HOME/.cache/crispasr}}"
KOKORO="${CRISPASR_KOKORO_MODEL:-$MODELS/kokoro-82m-q8_0.gguf}"
VOICE="${CRISPASR_KOKORO_VOICE:-$MODELS/kokoro-voice-af_heart.gguf}"
ASR="${CRISPASR_MODEL_WHISPER:-$MODELS/ggml-tiny.bin}"
for f in "$KOKORO" "$VOICE" "$ASR"; do
    [ -f "$f" ] || { echo "SKIP: missing $f"; exit 0; }
done

TMP=$(mktemp -d -t kokoro-g2p.XXXXXX); trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
ok()  { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad() { echo "  ✗ $1"; FAIL=$((FAIL+1)); }

# ── 1. the reported bug: a number must be SPOKEN, not silently dropped ────────
echo "=== #316: numbers survive the G2P ==="
"$CRISPASR" --tts "A text to speech model with 82 million parameters." \
    --backend kokoro -m "$KOKORO" --voice "$VOICE" --tts-output "$TMP/n.wav" >/dev/null 2>&1
if [ -s "$TMP/n.wav" ]; then
    TXT=$("$CRISPASR" -m "$ASR" -f "$TMP/n.wav" -l en --no-prints 2>/dev/null | tr -d '\n')
    echo "    ASR: $TXT"
    # A tiny ASR model transcribes spoken digits loosely — "eighty two" has come
    # back as "82", "eighty-two" and even "80-to-a". Any of those proves the
    # number was SPOKEN, which is the whole point: before the fix it phonemized
    # to the empty string and the ASR heard "with million parameters", with no
    # numeric token at all.
    if echo "$TXT" | grep -qiE "8[0-9]|eighty|eight"; then
        ok "the number is audible in the output"
    else
        bad "number missing — the G2P dropped it again (this IS #316)"
    fi
else
    bad "synthesis produced no audio"
fi

# ── 2. --tts-phonemes drives the model directly ───────────────────────────────
echo "=== --tts-phonemes bypasses the G2P ==="
"$CRISPASR" --tts "THIS TEXT MUST BE IGNORED" --tts-phonemes "həlˈO wˈɜɹld" \
    --backend kokoro -m "$KOKORO" --voice "$VOICE" --tts-output "$TMP/p.wav" >/dev/null 2>&1
if [ -s "$TMP/p.wav" ]; then
    TXT=$("$CRISPASR" -m "$ASR" -f "$TMP/p.wav" -l en --no-prints 2>/dev/null | tr -d '\n')
    echo "    ASR: $TXT"
    # The phonemes say "hello world"; the --tts text says something else. If the
    # flag were ignored we would hear the text instead — the exact silent
    # fallback the CLI refuses to make.
    if echo "$TXT" | grep -qi "hello"; then
        ok "the phonemes were synthesized, not the text"
    else
        bad "phonemes ignored — heard the --tts text instead"
    fi
else
    bad "phoneme synthesis produced no audio"
fi

# ── 3. an unsupported backend must refuse, not fall back ─────────────────────
echo "=== --tts-phonemes is refused where it cannot work ==="
# Checked before any model load, so this needs no qwen3-tts weights.
OUT=$("$CRISPASR" --tts "x" --tts-phonemes "abc" --backend qwen3-tts -m /nonexistent.gguf 2>&1)
if echo "$OUT" | grep -q "tts-phonemes is not supported"; then
    ok "refused with a message naming the flag"
else
    bad "did not refuse (silent fallback makes an A/B unreadable)"
fi

# ── 4. round 2: the contextual rules must actually reach the model ───────────
#
# Round 1 of #316 shipped core/g2p_ctxwords.h behind a flag nothing set, so
# 0.8.24 and 0.8.25 read "the" as ði everywhere and the article "a" as the
# LETTER, ˈA. The unit tests could not see it — they call the helper, not
# text_to_ipa. This reads the phoneme string the model is actually handed.
echo "=== #316 round 2: the rules reach the phoneme string ==="
REPORTED='It'"'"'s described as "dramatic" because of the high-contrast, cinematic lighting and moody atmosphere I built into the prompt.'
PH=$("$CRISPASR" --tts "$REPORTED" --backend kokoro -m "$KOKORO" --voice "$VOICE" \
        --tts-output "$TMP/r2.wav" -v 2>&1 | sed -n "s/^kokoro: phonemes: '\(.*\)'$/\1/p" | tr '\n' ' ')
echo "    phonemes: ${PH:0:120}..."
if [ -z "$PH" ]; then
    bad "no phoneme string printed — cannot check anything below"
else
    # "the prompt" -> ðə, never the citation form ði before a consonant.
    echo "$PH" | grep -q 'ðə' && ok "the/to reduce (ðə present)" \
                              || bad "no ðə anywhere — context_words is off again (THE #316 round-2 bug)"
    # The pronoun takes SECONDARY stress. ˈI is the letter/emphatic reading.
    echo "$PH" | grep -q 'ˌI ' && ok "the pronoun I is ˌI, not ˈI" \
                               || bad "pronoun I is not secondary-stressed"
    # Kokoro's vocabulary has , and . — dropping them delivers a paragraph in
    # one breath, which is what "doesn't sound natural" was.
    echo "$PH" | grep -q ',' && ok "punctuation survives into the phonemes" \
                             || bad "no comma in the phoneme string — every pause is gone"
    # A quoted word must reach the lexicon without its quotes: misaki says
    # dɹəmˈæTɪk (second syllable). The LTS fallback said dɹˈæmætɪk.
    echo "$PH" | grep -q 'dɹəmˈæTɪk' && ok "\"dramatic\" is stressed on the second syllable" \
                                     || bad "dramatic mis-stressed — the quotes reached the lexicon"
fi

# ── 5. …and the round-trip hears the difference ──────────────────────────────
#
# The discriminator found while fixing this: with ˈI the pronoun is a NOUN to
# an ASR. Reproduced on ggml-tiny and ggml-base, both arms.
echo "=== #316 round 2: ASR round-trip ==="
if [ -s "$TMP/r2.wav" ]; then
    TXT=$("$CRISPASR" -m "$ASR" -f "$TMP/r2.wav" -l en --no-prints 2>/dev/null \
            | sed 's/\x1b\[[0-9;]*m//g' | tr '\n' ' ')
    echo "    ASR: $TXT"
    if echo "$TXT" | grep -qi "eye built"; then
        bad "ASR heard the NOUN \"eye\" — the pronoun carries primary stress (this IS the report)"
    else
        ok "the pronoun is not heard as \"eye\""
    fi
    # Whisper punctuates from prosody. With no marks in the phoneme string it
    # finds no clause boundary at all.
    NC=$(echo "$TXT" | tr -cd ',' | wc -c | tr -d ' ')
    if [ "$NC" -ge 1 ]; then
        ok "the transcript has $NC comma(s) — the pauses are audible"
    else
        bad "not one comma — the delivery has no clause boundaries"
    fi
else
    echo "  - skipped (no audio from step 4)"
fi

echo
echo "passed=$PASS failed=$FAIL"
[ "$FAIL" -eq 0 ]

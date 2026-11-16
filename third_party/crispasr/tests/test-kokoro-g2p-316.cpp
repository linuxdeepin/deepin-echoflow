// test-kokoro-g2p-316.cpp — number expansion + Kokoro's phoneme alphabet.
//
// #316: Kokoro dropped numbers entirely ("with 82 million parameters" came out
// "with million parameters") and drifted into a British-sounding accent. Both
// were G2P defects, and neither is visible to crispasr-diff: that harness is
// phoneme-IN (KOKORO_PHONEMES), so it starts downstream of everything here and
// would have shown perfect parity while the audio was wrong.
//
// EVERY expectation below is misaki's actual output — Kokoro's own G2P, run at
// misaki 0.9.4 with british=False — not a guess about how English "should" be
// read. The four-digit cases are the ones worth having: misaki reads 1234 as
// "twelve thirty four" but 1005 as "one thousand five", and 1100 as "eleven
// hundred" while 1000 stays "one thousand".
//
// No model, no audio.

#include <catch2/catch_test_macros.hpp>

#include "core/phoneme_dialect.h"
#include "core/num2words_en.h"

#include <string>

using core_num2words_en::expand;

// ── numbers ─────────────────────────────────────────────────────────────────

TEST_CASE("the reported case: a bare number is spoken, not dropped", "[unit][kokoro][g2p]") {
    // The whole bug in one line — this used to phonemize to "".
    REQUIRE(expand("82") == "eighty two");
    REQUIRE(expand("A text to speech model with 82 million parameters.") ==
            "A text to speech model with eighty two million parameters.");
}

TEST_CASE("cardinals match misaki", "[unit][kokoro][g2p]") {
    REQUIRE(expand("0") == "zero");
    REQUIRE(expand("7") == "seven");
    REQUIRE(expand("13") == "thirteen");
    REQUIRE(expand("21") == "twenty one");
    REQUIRE(expand("100") == "one hundred");
    REQUIRE(expand("101") == "one hundred one"); // no "and"
    REQUIRE(expand("999") == "nine hundred ninety nine");
    REQUIRE(expand("10000") == "ten thousand");
    REQUIRE(expand("12345") == "twelve thousand three hundred forty five");
    REQUIRE(expand("1000000") == "one million");
}

TEST_CASE("four-digit numbers use misaki's year reading", "[unit][kokoro][g2p]") {
    // The non-obvious rule, and the one that is audible if you get it wrong.
    REQUIRE(expand("1984") == "nineteen eighty four");
    REQUIRE(expand("2026") == "twenty twenty six");
    REQUIRE(expand("1234") == "twelve thirty four");
    REQUIRE(expand("9999") == "ninety nine ninety nine");
    REQUIRE(expand("1090") == "ten ninety");
    // …except whole thousands, x00, and a low remainder, which stay cardinal.
    REQUIRE(expand("1000") == "one thousand");
    REQUIRE(expand("2000") == "two thousand");
    REQUIRE(expand("1005") == "one thousand five");
    REQUIRE(expand("2005") == "two thousand five");
    REQUIRE(expand("1100") == "eleven hundred");
    REQUIRE(expand("1900") == "nineteen hundred");
    REQUIRE(expand("2100") == "twenty one hundred");
}

TEST_CASE("decimals, separators, ordinals, signs and units", "[unit][kokoro][g2p]") {
    REQUIRE(expand("3.14") == "three point one four"); // fraction is digit-by-digit
    REQUIRE(expand("1,000") == "one thousand");
    REQUIRE(expand("1st") == "first");
    REQUIRE(expand("2nd") == "second");
    REQUIRE(expand("3rd") == "third");
    REQUIRE(expand("4th") == "fourth");
    REQUIRE(expand("21st") == "twenty first");
    REQUIRE(expand("-5") == "minus five");
    REQUIRE(expand("50%") == "fifty percent");
    REQUIRE(expand("$20") == "twenty dollars");
    REQUIRE(expand("$1") == "one dollar");
    // #316 follow-up: the unit used to survive only for a LEADING '$'. A
    // currency symbol is in no dictionary and no letter-to-sound rule, so every
    // other form lost it silently — "€50" was spoken "fifty".
    REQUIRE(expand("\u20ac50") == "fifty euros");
    REQUIRE(expand("\u00a350") == "fifty pounds");
    REQUIRE(expand("\u20ac1") == "one euro");
    // …and the postfix form every European locale writes.
    REQUIRE(expand("50\u20ac") == "fifty euros");
    REQUIRE(expand("50 \u20ac") == "fifty euros");
}

TEST_CASE("alphanumeric tokens are left for the existing rules", "[unit][kokoro][g2p]") {
    // A digit inside a word is not a number. "x64" regressed to "x6 four" when
    // the token-start guard tested isalpha instead of isalnum: the '6' was
    // skipped for its alpha predecessor, then the '4' saw a DIGIT before it and
    // called itself a fresh token.
    REQUIRE(expand("mp3") == "mp3");
    REQUIRE(expand("x64") == "x64");
    REQUIRE(expand("h264") == "h264");
    REQUIRE(expand("no digits here") == "no digits here");
}

// ── phoneme alphabet ────────────────────────────────────────────────────────

using core_phoneme::convert;
using core_phoneme::Dialect;
using core_phoneme::to_misaki;

TEST_CASE("textbook IPA is rewritten into Kokoro's alphabet", "[unit][kokoro][g2p]") {
    // Kokoro's vocab contains BOTH spellings, so the wrong one is silently
    // accepted — nothing is dropped, the model just receives tokens it was
    // never trained on. That is why this needed a test rather than an assert.
    REQUIRE(to_misaki("tʃ") == "ʧ");
    REQUIRE(to_misaki("dʒ") == "ʤ");
    REQUIRE(to_misaki("eɪ") == "A");
    REQUIRE(to_misaki("aɪ") == "I");
    REQUIRE(to_misaki("oʊ") == "O");
    REQUIRE(to_misaki("əʊ") == "O"); // the en-gb spelling of the same vowel
    REQUIRE(to_misaki("aʊ") == "W");
    REQUIRE(to_misaki("ɔɪ") == "Y");
    REQUIRE(to_misaki("ɾ") == "T");
}

TEST_CASE("length marks are dropped — misaki's US output has none", "[unit][kokoro][g2p]") {
    // `ː` is the RP length mark; feeding it to a US voice is a large part of
    // the reported "she becomes British".
    REQUIRE(to_misaki("tˈuː") == "tˈu");
    REQUIRE(to_misaki("spˈiːtʃ") == "spˈiʧ"); // byte-identical to misaki
}

TEST_CASE("the rhotic schwa expands without doubling the r", "[unit][kokoro][g2p]") {
    // ɚ → əɹ, but our G2P already emits an ɹ after it in "parameters"
    // (pɚɹ…), so the naive rewrite produced əɹɹ, which misaki never emits.
    REQUIRE(to_misaki("ɚ") == "əɹ");
    REQUIRE(to_misaki("pɚɹˈæməɾɚz") == "pəɹˈæməTəɹz"); // byte-identical to misaki
}

TEST_CASE("a replacement is never re-examined", "[unit][kokoro][g2p]") {
    // tʃ → ʧ must not then be re-scanned and hit a bare-t rule; the pass is
    // left-to-right longest-match-first with no cascade.
    REQUIRE(to_misaki("ʧ") == "ʧ");
    REQUIRE(to_misaki("T") == "T");
    REQUIRE(to_misaki("") == "");
    // Plain ASCII with no mapped symbol is untouched. (I first wrote "hɛllo"
    // here, which was simply wrong — there is no rule that fabricates a vowel.)
    REQUIRE(to_misaki("hello") == "hello");
    // The one ASCII rule that DOES fire: espeak spells the approximant `r`.
    REQUIRE(to_misaki("ɹɛd") == "ɹɛd");
    REQUIRE(to_misaki("rɛd") == "ɹɛd");
}

TEST_CASE("EspeakIpa is the identity — piper must not move", "[unit][kokoro][g2p]") {
    // The whole point of the dialect enum: one G2P feeds several models. piper
    // is trained on the espeak spelling, so selecting its dialect must return
    // the G2P's output byte-for-byte.
    for (const char* s : {"spˈiːtʃ", "wˈɜːld", "pɚɹˈæməɾɚz", "hɛlˈoʊ", ""})
        REQUIRE(convert(s, Dialect::EspeakIpa) == std::string(s));
    // …and selecting Misaki is the same as calling the converter directly.
    REQUIRE(convert("spˈiːtʃ", Dialect::Misaki) == to_misaki("spˈiːtʃ"));
}

TEST_CASE("NURSE keeps its r for Kokoro", "[unit][kokoro][g2p]") {
    // CMUdict's ER reaches us as the non-rhotic RP `ɜː`; dropping the length
    // mark alone left "world" as wˈɜld, an r short of misaki's wˈɜɹld. Measured
    // as the single largest remaining divergence over a 1508-word corpus.
    REQUIRE(to_misaki("wˈɜːld") == "wˈɜɹld");
    REQUIRE(to_misaki("ɜː") == "ɜɹ");
}


// ── corpus-derived coverage ─────────────────────────────────────────────────
//
// "Do we cover every phoneme?" — measured, not asserted by hand. Over a
// 1508-word corpus (every 40th CMUdict headword, run through both our G2P and
// misaki 0.9.4):
//
//   symbols we emit that are NOT in Kokoro's vocab       : none
//   symbols we emit that misaki NEVER emits              : none
//   symbols misaki emits that we never do                : ᵊ, ᵻ (its two
//                                                          REDUCED vowels)
//   exact whole-word phoneme match                       : 58.3% (879/1508)
//
// The first two lines are the ones that matter for correctness: we never hand
// Kokoro a token outside its vocabulary (which would be silently dropped) or
// outside its training distribution (which is what made it drift). The 41.7%
// that still differ are dictionary-level disagreements — CMUdict's stress and
// unstressed-vowel choices vs misaki's lexicon — not spelling-system errors;
// modelling ᵊ/ᵻ alone would take the match to ~63%.
//
// Each pair below is REAL output: the left side is what our G2P produced for
// that word, the right side is what the conversion makes of it — and every one
// was verified equal to misaki's output for the same word. Between them they
// exercise all 43 symbols our pipeline can emit.

namespace {
struct GoldenPair {
    const char* espeak_ipa;
    const char* misaki;
};
inline const GoldenPair kGolden[] = {
    {"ɔːltˈɜːnəɾɪv", "ɔltˈɜɹnəTɪv"},             // alternative
    {"ˈæmbjələtˌɔːɹi", "ˈæmbjələtˌɔɹi"},         // ambulatory
    {"bˈækɡɹˌaʊndz", "bˈækɡɹˌWndz"},             // backgrounds
    {"sˌɜːɾəfəkˈeɪʃənz", "sˌɜɹTəfəkˈAʃənz"},     // certifications
    {"kənstˈɪtʃʊənsi", "kənstˈɪʧʊənsi"},         // constituency
    {"kənvˈɜːʒənz", "kənvˈɜɹʒənz"},              // conversions
    {"fˈɑːɹðɪŋɡˌeɪl", "fˈɑɹðɪŋɡˌAl"},            // farthingale
    {"hˈɛmɚɹˌɔɪdz", "hˈɛməɹˌYdz"},               // hemorrhoids
    {"ˌɪmpɚfˈɛkʃənz", "ˌɪmpəɹfˈɛkʃənz"},         // imperfections
    {"dʒˌʌkstəpəzˈɪʃən", "ʤˌʌkstəpəzˈɪʃən"},     // juxtaposition
    {"nˈɛtwˌɜːks", "nˈɛtwˌɜɹks"},                // networks
    {"pˈɛtɹoʊdˌɑːlɚz", "pˈɛtɹOdˌɑləɹz"},         // petrodollars
    {"sˌækɹəmˈɛntoʊz", "sˌækɹəmˈɛntOz"},         // sacramento's
    {"stɹˌæŋɡjəlˈeɪʃənz", "stɹˌæŋɡjəlˈAʃənz"},   // strangulations
    {"sˌuːpɚhˈɛɾɚɹədˌaɪn", "sˌupəɹhˈɛTəɹədˌIn"}, // superheterodyne
    {"sˈɪmpəθˌaɪzɪŋ", "sˈɪmpəθˌIzɪŋ"},           // sympathizing
};
} // namespace

TEST_CASE("real G2P output converts byte-identically to misaki", "[unit][kokoro][g2p]") {
    for (const auto& g : kGolden) {
        INFO("input: " << g.espeak_ipa);
        REQUIRE(to_misaki(g.espeak_ipa) == std::string(g.misaki));
    }
}

TEST_CASE("no espeak-dialect residue survives the conversion", "[unit][kokoro][g2p]") {
    // The invariant the dialect exists to guarantee: none of the espeak-only
    // spellings may reach Kokoro. They are all IN its vocab, so a leak is
    // silent — no drop, no error, just a token the model never trained on.
    static const char* kEspeakOnly[] = {"ː", "tʃ", "dʒ", "eɪ", "aɪ", "oʊ", "aʊ", "ɔɪ", "ɚ", "ɝ", "ɾ"};
    for (const auto& g : kGolden) {
        const std::string conv = to_misaki(g.espeak_ipa);
        for (const char* bad : kEspeakOnly) {
            INFO("word IPA: " << g.espeak_ipa << "  residue: " << bad);
            REQUIRE(conv.find(bad) == std::string::npos);
        }
    }
}

// ── morphological fallback ──────────────────────────────────────────────────
//
// misaki's lexicon stores STEMS: only 46% of inflected forms are listed
// verbatim (CMUdict lists 100%). Recovering the rest by rule took whole-word
// agreement from 88.5% to 98.1%. The suffix phonemes below are misaki's own
// `_s`, read from its source.

#include "core/g2p_inflect.h"

namespace {
// A stand-in lexicon holding only stems, which is the situation being tested.
std::string misaki_stub(const std::string& w) {
    if (w == "believe")
        return "bəlˈiv";
    if (w == "airbase")
        return "ˈɛɹbˌAs";
    if (w == "abdicate")
        return "ˈæbdəkˌAt";
    if (w == "walk")
        return "wˈɔk";
    if (w == "cat")
        return "kˈæt";
    if (w == "run")
        return "ɹˈʌn";
    if (w == "carry")
        return "kˈæɹi";
    return "";
}
core_g2p_inflect::Params misaki_params() {
    core_g2p_inflect::Params p;
    p.reduced_vowel = "ᵻ";
    p.flap = "T";
    return p;
}
} // namespace

TEST_CASE("regular -s follows voicing and sibilance", "[unit][kokoro][g2p]") {
    const auto p = misaki_params();
    REQUIRE(core_g2p_inflect::inflect("believes", misaki_stub, p) == "bəlˈivz");   // voiced -> z
    REQUIRE(core_g2p_inflect::inflect("cats", misaki_stub, p) == "kˈæts");         // voiceless -> s
    REQUIRE(core_g2p_inflect::inflect("airbases", misaki_stub, p) == "ˈɛɹbˌAsᵻz"); // sibilant -> ᵻz
}

TEST_CASE("regular -ed, and the flap it creates", "[unit][kokoro][g2p]") {
    const auto p = misaki_params();
    REQUIRE(core_g2p_inflect::inflect("walked", misaki_stub, p) == "wˈɔkt"); // voiceless -> t
    // The one a naive concatenation gets wrong: "abdicate"+"ᵻd" puts /t/
    // between vowels, where American English flaps it. Flapping must run
    // AFTER attachment or this comes out `…ˌAtᵻd`.
    REQUIRE(core_g2p_inflect::inflect("abdicated", misaki_stub, p) == "ˈæbdəkˌATᵻd");
}

TEST_CASE("orthographic stem recovery", "[unit][kokoro][g2p]") {
    const auto p = misaki_params();
    REQUIRE(core_g2p_inflect::inflect("running", misaki_stub, p) == "ɹˈʌnɪŋ"); // undouble
    REQUIRE(core_g2p_inflect::inflect("carries", misaki_stub, p) == "kˈæɹiz"); // i -> y
    REQUIRE(core_g2p_inflect::inflect("nonsenses", misaki_stub, p).empty());   // unknown stem
}

TEST_CASE("the dialect decides the suffix symbols", "[unit][kokoro][g2p]") {
    // Same rules, different alphabet — this is why the helper is shared and
    // parameterised rather than duplicated per backend.
    core_g2p_inflect::Params espeak; // defaults: ɪ and ɾ
    REQUIRE(core_g2p_inflect::inflect("airbases", misaki_stub, espeak) == "ˈɛɹbˌAsɪz");
    REQUIRE(core_g2p_inflect::inflect("abdicated", misaki_stub, espeak) == "ˈæbdəkˌAɾɪd");
}

// ── contextual function words + capitalisation stress ───────────────────────

#include "core/g2p_ctxwords.h"

using core_g2p_ctxwords::NextVowel;

TEST_CASE("the/to reduce according to the FOLLOWING word", "[unit][kokoro][g2p]") {
    // misaki's future_vowel rule. Reading the citation form in every position
    // ("ðiː bɑks") is the "old English" diction reported in #316.
    REQUIRE(core_g2p_ctxwords::lookup("the", NextVowel::Yes, "ði") == "ði");    // the apple
    REQUIRE(core_g2p_ctxwords::lookup("the", NextVowel::No, "ði") == "ðə");     // the box
    REQUIRE(core_g2p_ctxwords::lookup("to", NextVowel::Yes, "tu") == "tʊ");     // to open
    REQUIRE(core_g2p_ctxwords::lookup("to", NextVowel::No, "tu") == "tə");      // to walk
    REQUIRE(core_g2p_ctxwords::lookup("to", NextVowel::Unknown, "tu") == "tu"); // isolated
    REQUIRE(core_g2p_ctxwords::lookup("a", NextVowel::No, "A") == "ɐ");         // article, never `A`
    REQUIRE(core_g2p_ctxwords::lookup("box", NextVowel::No, "bˈɑks").empty());  // not special
}

TEST_CASE("a vowel is detected past any stress mark", "[unit][kokoro][g2p]") {
    REQUIRE(core_g2p_ctxwords::starts_with_vowel("ˈæpᵊl"));
    REQUIRE(core_g2p_ctxwords::starts_with_vowel("ɐ"));
    REQUIRE_FALSE(core_g2p_ctxwords::starts_with_vowel("bˈɑks"));
    REQUIRE_FALSE(core_g2p_ctxwords::starts_with_vowel(""));
}

TEST_CASE("capitalisation drives the stress, not part of speech", "[unit][kokoro][g2p]") {
    // Verified against misaki in an identical frame:
    //   and -> ænd     And -> ˌænd     AND -> ˈænd
    // I had assumed this needed a POS tagger; it does not.
    REQUIRE(core_g2p_ctxwords::apply_caps_stress("and", "ænd") == "ænd");
    REQUIRE(core_g2p_ctxwords::apply_caps_stress("And", "ænd") == "ˌænd");
    REQUIRE(core_g2p_ctxwords::apply_caps_stress("AND", "ænd") == "ˈænd");
    // The mark lands before the first VOWEL, matching misaki's restress:
    // "she" is ʃˌi, never ˌʃi.
    REQUIRE(core_g2p_ctxwords::apply_caps_stress("She", "ʃi") == "ʃˌi");
    // An existing mark wins — misaki only inserts when there is none.
    REQUIRE(core_g2p_ctxwords::apply_caps_stress("Box", "bˈɑks") == "bˈɑks");
    // No vowel, nothing to stress.
    REQUIRE(core_g2p_ctxwords::apply_caps_stress("Hmm", "hm") == "hm");
}

// ── the rules reach the OUTPUT, not just their own header ───────────────────
//
// Everything above tests `core_g2p_ctxwords` in isolation, and every one of
// those assertions passed while the product shipped `ði bɑks` and read the
// article "a" as the LETTER, `ˈA`: nothing anywhere set `context_words`, so
// `text_to_ipa` never called any of it. These tests run the pipeline.

#include "core/g2p_en.h"

namespace {

// A stand-in for misaki's lexicon: the same entries, verbatim, for the handful
// of words below. Keeps the test hermetic — no download, no 178K-entry file.
g2p_en::context mini_lexicon() {
    g2p_en::context ctx;
    ctx.espeak_ipa.entries = {
        {"a", "A"}, // misaki's citation form IS the letter, "eɪ"
        {"the", "ði"},      {"to", "tu"},
        {"i", "ˈI"},        {"box", "bˈɑks"},
        {"apple", "ˈæpᵊl"}, {"dramatic", "dɹəmˈæɾɪk"},
        {"high", "hˈI"},    {"contrast", "kˈɑntɹˌæst"},
        {"dark", "dˈɑɹk"},  {"mr.", "mˈɪstəɹ"},
    };
    ctx.espeak_ipa.loaded = true;
    return ctx;
}

std::string misaki(const g2p_en::context& ctx, const std::string& text) {
    return g2p_en::text_to_ipa(ctx, text, g2p_en::misaki_style());
}

} // namespace

TEST_CASE("configure_for_misaki turns on everything Kokoro needs", "[unit][kokoro][g2p]") {
    // The wiring contract. The first round of #316 wrote the contextual-word
    // rules and left them switched off in the product for a whole release.
    g2p_en::context ctx;
    REQUIRE_FALSE(ctx.context_words);
    g2p_en::configure_for_misaki(ctx);
    REQUIRE(ctx.context_words);
    REQUIRE(ctx.emit_punctuation);
    REQUIRE(ctx.join_hyphenated);
    REQUIRE(ctx.consumer().context_words);
}

TEST_CASE("the contextual forms come out of text_to_ipa", "[unit][kokoro][g2p]") {
    const auto ctx = mini_lexicon();
    // The reported "old English" diction: the citation form in every position.
    REQUIRE(misaki(ctx, "the box") == "ðə bˈɑks");
    REQUIRE(misaki(ctx, "the apple") == "ði ˈæpᵊl");
    // "and a dark background" — the article was read as the LETTER A, with
    // PRIMARY stress: the "unnecessary emphasis on a" in the #316 follow-up.
    REQUIRE(misaki(ctx, "a dark box") == "ɐ dˈɑɹk bˈɑks");
    // ...and the pronoun carries SECONDARY stress, not primary.
    REQUIRE(misaki(ctx, "I box") == "ˌI bˈɑks");
    // Without the misaki conventions the same dictionary keeps its citation
    // forms — that is what piper wants and must not change.
    REQUIRE(g2p_en::text_to_ipa(ctx, "the box") == "ði bˈɑks");
}

TEST_CASE("punctuation survives into the phoneme string", "[unit][kokoro][g2p]") {
    const auto ctx = mini_lexicon();
    // Kokoro's 178-symbol vocabulary contains `,` and `.`; they are how the
    // model knows to pause. Dropping them delivered a whole paragraph in one
    // breath.
    REQUIRE(misaki(ctx, "the box, the apple.") == "ðə bˈɑks, ði ˈæpᵊl.");
    // A mark sits flush against the word it follows, exactly as misaki emits it.
    REQUIRE(misaki(ctx, "box; box: box!") == "bˈɑks; bˈɑks: bˈɑks!");
    // The piper consumer still gets none of it — and gets ONE space where the
    // old loop emitted two.
    REQUIRE(g2p_en::text_to_ipa(ctx, "the box, the apple.") == "ði bˈɑks ði ˈæpᵊl");
}

TEST_CASE("a quoted word still reaches the lexicon", "[unit][kokoro][g2p]") {
    const auto ctx = mini_lexicon();
    // `"dramatic"` used to arrive at every lookup tier WEARING ITS QUOTES,
    // missed all of them, and came back out of the letter-to-sound rules as
    // dɹˈæmætɪk — first-syllable stress. That is the "strange pronunciation of
    // dramatic" in the #316 follow-up report.
    REQUIRE(misaki(ctx, "\"dramatic\"") == "“dɹəmˈæɾɪk”");
    REQUIRE(misaki(ctx, "(box)") == "(bˈɑks)");
    // Straight quotes alternate into the directional pair misaki produces and
    // Kokoro was trained on.
    REQUIRE(misaki(ctx, "\"box\" \"apple\"") == "“bˈɑks” “ˈæpᵊl”");
}

TEST_CASE("a hyphenated compound is one word with one primary stress", "[unit][kokoro][g2p]") {
    const auto ctx = mini_lexicon();
    // misaki: high-contrast -> hˌIkˈɑntɹˌæst. Splitting it gave two primary
    // stresses and a pause through the middle of the word.
    REQUIRE(misaki(ctx, "high-contrast") == "hˌIkˈɑntɹˌæst");
    // piper keeps the split (its dictionaries have no compound entries).
    REQUIRE(g2p_en::text_to_ipa(ctx, "high-contrast") == "hˈI kˈɑntɹˌæst");
}

TEST_CASE("an abbreviation's period is not a full stop", "[unit][kokoro][g2p]") {
    const auto ctx = mini_lexicon();
    // misaki's lexicon lists `Mr.` WITH the dot; splitting it off put a
    // sentence-length pause after every "Mister".
    REQUIRE(misaki(ctx, "Mr. Box") == "mˈɪstəɹ bˈɑks");
    // A real sentence end is still a sentence end.
    REQUIRE(misaki(ctx, "the box.") == "ðə bˈɑks.");
}

TEST_CASE("compound de-stressing follows misaki's rule", "[unit][kokoro][g2p]") {
    using core_g2p_ctxwords::destress;
    using core_g2p_ctxwords::join_compound;
    // apply_stress(ps, -0.5): primary becomes secondary, secondaries go.
    REQUIRE(destress("hˈI") == "hˌI");
    REQUIRE(destress("kˈɑntɹˌæst") == "kˌɑntɹæst");
    REQUIRE(destress("ænd") == "ænd"); // no primary — untouched
    // The lighter half yields: "high" (weight 4) loses to "contrast" (10).
    REQUIRE(join_compound({"hˈI", "kˈɑntɹˌæst"}) == "hˌIkˈɑntɹˌæst");
    // Only one primary between them: nothing to resolve, join as-is.
    REQUIRE(join_compound({"ænd", "bˈɑks"}) == "ændbˈɑks");
}

TEST_CASE("ALLCAPS promotes an existing secondary stress", "[unit][kokoro][g2p]") {
    // misaki's apply_stress, the `stress >= 1 and PRIMARY not in ps` branch.
    REQUIRE(core_g2p_ctxwords::apply_caps_stress("HIGH", "hˌI") == "hˈI");
    REQUIRE(core_g2p_ctxwords::apply_caps_stress("High", "hˌI") == "hˌI");
}

// ── acronyms: misaki reads them out, letter by letter ───────────────────────

namespace {

// misaki's own letter readings (us_gold.json, uppercase keys).
g2p_en::context lexicon_with_letters() {
    g2p_en::context ctx = mini_lexicon();
    ctx.letters.entries = {
        {"a", "ˈA"}, {"b", "bˈi"}, {"c", "sˈi"}, {"d", "dˈi"}, {"e", "ˈi"},  {"f", "ˈɛf"}, {"g", "ʤˈi"},  {"h", "ˈAʧ"},
        {"i", "ˈI"}, {"n", "ˈɛn"}, {"p", "pˈi"}, {"s", "ˈɛs"}, {"u", "jˈu"}, {"v", "vˈi"}, {"x", "ˈɛks"},
    };
    ctx.letters.loaded = true;
    return ctx;
}

} // namespace

TEST_CASE("a dotted acronym is spelled, not pronounced", "[unit][kokoro][g2p]") {
    const auto ctx = lexicon_with_letters();
    // Verified against misaki 0.9.4 in an identical frame. The old output was
    // `jˈu.ˈɛs.ɐ.` — full stops INSIDE the word, so the model paused between
    // the letters, and the trailing "a" was read as the article.
    REQUIRE(misaki(ctx, "U.S.A.") == "jˌuˌɛsˈA");
    REQUIRE(misaki(ctx, "e.g.") == "ˌiʤˈi");
    // Every letter is demoted and the LAST one promoted back — one word, one
    // stress peak (misaki's get_NNP).
    REQUIRE(misaki(ctx, "U.S.") == "jˌuˈɛs");
}

TEST_CASE("an unknown ALLCAPS word is spelled; a known one is not", "[unit][kokoro][g2p]") {
    const auto ctx = lexicon_with_letters();
    REQUIRE(misaki(ctx, "PDF") == "pˌidˌiˈɛf");
    // misaki lowercases an ALLCAPS word and looks THAT up first, so a word in
    // the dictionary is still a word however it is capitalised.
    REQUIRE(misaki(ctx, "BOX") == "bˈɑks");
    // A single letter is not an acronym — a lone "A" stays the article.
    REQUIRE(misaki(ctx, "A dark box") == "ɐ dˈɑɹk bˈɑks");
}

TEST_CASE("spelling is inert without a letters table", "[unit][kokoro][g2p]") {
    // The espeak/piper dicts have no letter entries, which is what keeps this
    // whole path off for that consumer — no separate flag needed.
    const auto ctx = mini_lexicon();
    REQUIRE(ctx.letters.entries.empty());
    REQUIRE(g2p_en::spell_out(ctx, "PDF").empty());
    REQUIRE_FALSE(g2p_en::wants_spelling(ctx, "PDF"));
}

TEST_CASE("is_dotted_acronym matches misaki's shape test", "[unit][kokoro][g2p]") {
    using core_g2p_ctxwords::is_dotted_acronym;
    REQUIRE(is_dotted_acronym("U.S."));
    REQUIRE(is_dotted_acronym("e.g."));
    REQUIRE(is_dotted_acronym("Ph.D."));     // longest run is 2, still under 3
    REQUIRE_FALSE(is_dotted_acronym("Mr.")); // no dot INSIDE the word
    REQUIRE_FALSE(is_dotted_acronym("etc.."));
    REQUIRE_FALSE(is_dotted_acronym("box"));
    REQUIRE_FALSE(is_dotted_acronym("3.14"));
}

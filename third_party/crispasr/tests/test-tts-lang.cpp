// test-tts-lang.cpp — language tags + reference-transcript language ID (#329).
//
// Hermetic: no model, no network, no GPU. Everything here sits DOWNSTREAM of
// the logits, where the per-stage diff harness is blind: `is_cross_lingual()`
// decides whether CosyVoice3 drops the reference transcript, and a wrong answer
// produces a graph that still computes cos 1.000000 while the user hears an
// accented clone. Only a table test catches that.
//
// The regression that motivated the file is at the bottom: before #329 the
// reference-language detector only knew Hangul/Kana/Han/Cyrillic, so EVERY
// Latin-script pair (en↔de, en↔fr, es↔it — the subtitle-dubbing cases) resolved
// to "unknown" and cross-lingual synthesis never engaged.

#include "core/tts_lang.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace core_tts_lang;

TEST_CASE("tts-lang: tags normalize to a comparable 2-letter base", "[unit][tts-lang]") {
    REQUIRE(norm("en") == "en");
    REQUIRE(norm("en-US") == "en");
    REQUIRE(norm("en_GB") == "en");
    REQUIRE(norm("EN") == "en");
    REQUIRE(norm("cmn") == "zh");
    REQUIRE(norm("zho") == "zh");
    REQUIRE(norm("jpn") == "ja");
    REQUIRE(norm("kor") == "ko");
    REQUIRE(norm("deu") == "de");
    REQUIRE(norm("ger") == "de");

    // The session ABI spells languages out for LLM prompts, so a caller can
    // hand us an English name rather than a code (core_lang::iso_to_english).
    REQUIRE(norm("German") == "de");
    REQUIRE(norm("Chinese") == "zh");
    REQUIRE(norm("Portuguese") == "pt");

    // "auto" and empty are both "no opinion", never a language.
    REQUIRE(norm("auto").empty());
    REQUIRE(norm("").empty());
}

TEST_CASE("tts-lang: an unknown tag only ever matches itself", "[unit][tts-lang]") {
    // Truncating an unrecognised 3-letter tag to 2 characters must not collide
    // with a real language: "swe" -> "sw" is not Swahili's behaviour we rely
    // on, but it must at least stay self-consistent and not equal "sv"/"en".
    REQUIRE(norm("swe") == norm("swe"));
    REQUIRE(norm("swe") != norm("en"));
    REQUIRE(same_language("swe", "swe"));
    REQUIRE_FALSE(same_language("swe", "en"));
}

TEST_CASE("tts-lang: scripts that only one language uses decide alone", "[unit][tts-lang]") {
    REQUIRE(script_language("안녕하세요 반갑습니다") == "ko");
    REQUIRE(script_language("こんにちは、元気ですか") == "ja");
    REQUIRE(script_language("今天天气很好") == "zh");
    REQUIRE(script_language("Привет, как дела") == "ru");
    REQUIRE(script_language("مرحبا كيف حالك") == "ar");

    // Korean and Japanese both embed Han characters, so the script that only
    // ONE language writes wins over the shared one.
    REQUIRE(script_language("日本語のテスト") == "ja");
    REQUIRE(script_language("한국어 테스트 漢字") == "ko");

    // Latin text has no script answer at all — that is what latin_language is
    // for, and answering "en" here would be a guess dressed as a fact.
    REQUIRE(script_language("The quick brown fox jumps over the lazy dog").empty());
}

TEST_CASE("tts-lang: Latin-script languages separate on function words", "[unit][tts-lang]") {
    REQUIRE(latin_language("The weather is nice today and the sun is shining over the city") == "en");
    REQUIRE(latin_language("Das Wetter ist heute schön und die Sonne scheint über der Stadt") == "de");
    REQUIRE(latin_language("Le temps est beau aujourd'hui et le soleil brille sur la ville") == "fr");
    REQUIRE(latin_language("El tiempo es agradable hoy y el sol brilla sobre la ciudad") == "es");
    REQUIRE(latin_language("Il tempo è bello oggi e il sole splende sulla città") == "it");
    REQUIRE(latin_language("O tempo está bom hoje e o sol brilha sobre a cidade") == "pt");
    REQUIRE(latin_language("Het weer is vandaag mooi en de zon schijnt over de stad") == "nl");
    REQUIRE(latin_language("Pogoda jest dzisiaj ładna i słońce świeci nad miastem") == "pl");
}

TEST_CASE("tts-lang: Romance neighbours are not confused for each other", "[unit][tts-lang]") {
    // es/it/pt share most of their short function words ("la", "e", "que",
    // "un", "con"), which is why the scorer weights each hit by how many
    // languages claim it. Without that these three collapse into one bucket.
    REQUIRE(latin_language("No creo que esto sea una buena idea para todos nosotros") == "es");
    REQUIRE(latin_language("Non credo che questa sia una buona idea per tutti noi") == "it");
    REQUIRE(latin_language("Não acho que isso seja uma boa ideia para todos nós") == "pt");
}

TEST_CASE("tts-lang: thin evidence answers 'unknown', not a guess", "[unit][tts-lang]") {
    // A wrong answer here silently changes synthesis behaviour, so the detector
    // must decline rather than pick a winner from noise.
    REQUIRE(latin_language("").empty());
    REQUIRE(latin_language("Hello").empty());
    REQUIRE(latin_language("one two").empty());
    REQUIRE(latin_language("Xyzzy plugh frobnitz quux blorp").empty());
    REQUIRE(latin_language("1234 5678 9012 3456").empty());
}

TEST_CASE("tts-lang: cross-lingual needs BOTH sides known and different", "[unit][tts-lang]") {
    REQUIRE(is_cross_lingual("de", "en"));
    REQUIRE(is_cross_lingual("en", "zh"));
    REQUIRE(is_cross_lingual("de-DE", "eng"));

    REQUIRE_FALSE(is_cross_lingual("en", "en"));
    REQUIRE_FALSE(is_cross_lingual("en-US", "en-GB"));

    // Unknown on either side must never engage it: the safe default is the
    // plain zero-shot path, and "I could not tell" is not "they differ".
    REQUIRE_FALSE(is_cross_lingual("", "en"));
    REQUIRE_FALSE(is_cross_lingual("de", ""));
    REQUIRE_FALSE(is_cross_lingual("auto", "en"));
    REQUIRE_FALSE(is_cross_lingual("", ""));
}

TEST_CASE("tts-lang: reference language resolves explicit > bank > detected", "[unit][tts-lang]") {
    // 1. The caller's own statement about their own recording outranks
    //    everything we could infer from it.
    REQUIRE(resolve_reference_language("de", "en", "The weather is nice today and the sun is shining") == "de");
    REQUIRE(resolve_reference_language("fr", "", "Das Wetter ist heute schön und die Sonne scheint") == "fr");

    // 2. A baked bank voice names its own language.
    REQUIRE(resolve_reference_language("", "de", "The weather is nice today and the sun is shining") == "de");
    REQUIRE(resolve_reference_language("auto", "ja", "") == "ja");

    // 3. Otherwise fall back to the transcript.
    REQUIRE(resolve_reference_language("", "", "Das Wetter ist heute schön und die Sonne scheint") == "de");
    REQUIRE(resolve_reference_language("", "", "今天天气很好") == "zh");

    // Nothing to go on stays unknown.
    REQUIRE(resolve_reference_language("", "", "").empty());
    REQUIRE(resolve_reference_language("auto", "", "Hello").empty());
}

TEST_CASE("tts-lang: #329 regression — Latin-script pairs engage cross-lingual", "[unit][tts-lang]") {
    // The #304 detector knew only Hangul/Kana/Han/Cyrillic, so an English
    // reference transcript resolved to "" and `is_cross_lingual("de", "")` was
    // false: asking CosyVoice3 for German output from an English reference kept
    // the English transcript in the LM prompt and produced the accented clone
    // reported in #329. Each pair below is one that used to be missed.
    const std::string en = "The weather is nice today and the sun is shining over the city";
    const std::string de = "Das Wetter ist heute schön und die Sonne scheint über der Stadt";
    const std::string fr = "Le temps est beau aujourd'hui et le soleil brille sur la ville";

    REQUIRE(is_cross_lingual("de", detect(en)));
    REQUIRE(is_cross_lingual("en", detect(de)));
    REQUIRE(is_cross_lingual("fr", detect(en)));
    REQUIRE(is_cross_lingual("es", detect(fr)));

    // …and the same-language case must still stay on the zero-shot default.
    REQUIRE_FALSE(is_cross_lingual("en", detect(en)));
    REQUIRE_FALSE(is_cross_lingual("de", detect(de)));

    // An explicit tag rescues the case detection cannot call: a two-word
    // reference transcript is below the evidence floor, but the caller can say
    // what it is (CLI --source-lang / server "source_lang").
    REQUIRE(detect("Guten Tag").empty());
    REQUIRE(is_cross_lingual("en", resolve_reference_language("de", "", "Guten Tag")));
}

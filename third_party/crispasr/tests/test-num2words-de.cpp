// test-num2words-de.cpp — German number expansion for the built-in G2P (#316).
//
// Hermetic: no model, no network, no GPU. This is the harness-blind zone —
// a wrong entry in these tables produces perfectly well-formed phonemes for the
// WRONG WORD, so every numeric check we have passes while the audio is wrong.
// Only a table test catches it.
//
// The regression at the bottom is the reason the file exists: before this,
// `g2p_de` had no number path at all, so "Ich habe 82 Euro" phonemized to
// "ɪç hɑːbə  ɔʏ̯roː" — the 82 silently gone, no error, no warning.

#include "core/g2p_de.h"
#include "core/num2words_de.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

using core_num2words_de::cardinal;
using core_num2words_de::expand;
using core_num2words_de::ordinal;
using core_num2words_de::year;

TEST_CASE("num2words-de: units come FIRST and the compound is one word", "[unit][num2words-de]") {
    // The rule that makes German not-English. Reversing it is audible and wrong,
    // not merely accented.
    REQUIRE(cardinal(21) == "einundzwanzig");
    REQUIRE(cardinal(82) == "zweiundachtzig");
    REQUIRE(cardinal(32) == "zweiunddreißig");
    REQUIRE(cardinal(99) == "neunundneunzig");

    // "ein" inside a compound, "eins" standing alone — never "einsundzwanzig".
    REQUIRE(cardinal(1) == "eins");
    REQUIRE(cardinal(21).substr(0, 3) == "ein");
    REQUIRE(cardinal(21).find("eins") == std::string::npos);
}

TEST_CASE("num2words-de: the stems that are not plain concatenation", "[unit][num2words-de]") {
    // Each of these is a word a naive builder gets wrong in the obvious way.
    REQUIRE(cardinal(16) == "sechzehn"); // not sechszehn
    REQUIRE(cardinal(17) == "siebzehn"); // not siebenzehn
    REQUIRE(cardinal(60) == "sechzig");  // not sechszig
    REQUIRE(cardinal(70) == "siebzig");  // not siebenzig
    REQUIRE(cardinal(30) == "dreißig");  // ß, not dreissig or dreizig

    // …and the neighbours that ARE regular, so the special-casing is not too eager.
    REQUIRE(cardinal(6) == "sechs");
    REQUIRE(cardinal(7) == "sieben");
    REQUIRE(cardinal(13) == "dreizehn");
    REQUIRE(cardinal(40) == "vierzig");
}

TEST_CASE("num2words-de: hundreds and thousands glue, scale words do not", "[unit][num2words-de]") {
    REQUIRE(cardinal(100) == "einhundert");
    REQUIRE(cardinal(101) == "einhunderteins");
    REQUIRE(cardinal(111) == "einhundertelf");
    REQUIRE(cardinal(200) == "zweihundert");
    REQUIRE(cardinal(999) == "neunhundertneunundneunzig");

    REQUIRE(cardinal(1000) == "eintausend");
    REQUIRE(cardinal(1005) == "eintausendfünf");
    REQUIRE(cardinal(12345) == "zwölftausenddreihundertfünfundvierzig");

    // Million/Milliarde are separate words and inflect for number.
    REQUIRE(cardinal(1000000) == "eine Million");
    REQUIRE(cardinal(2000000) == "zwei Millionen");
    REQUIRE(cardinal(1000000000) == "eine Milliarde");

    REQUIRE(cardinal(0) == "null");
}

TEST_CASE("num2words-de: years switch to the cardinal reading at 2000", "[unit][num2words-de]") {
    // German says "neunzehnhundert…" for 1900s but "zweitausend…" for 2000s —
    // there is no German equivalent of English's "twenty twenty six".
    REQUIRE(year(1984) == "neunzehnhundertvierundachtzig");
    REQUIRE(year(1900) == "neunzehnhundert");
    REQUIRE(year(1100) == "elfhundert");
    REQUIRE(year(2000) == "zweitausend");
    REQUIRE(year(2026) == "zweitausendsechsundzwanzig");
}

TEST_CASE("num2words-de: ordinal stems", "[unit][num2words-de]") {
    REQUIRE(ordinal(1) == "erste");
    REQUIRE(ordinal(3) == "dritte");
    REQUIRE(ordinal(7) == "siebte"); // not siebente
    REQUIRE(ordinal(8) == "achte");  // not achtte
    REQUIRE(ordinal(2) == "zweite");
    REQUIRE(ordinal(20) == "zwanzigste"); // -ste from 20 up
}

TEST_CASE("num2words-de: German separators are INVERTED vs English", "[unit][num2words-de]") {
    // The trap most likely to produce plausible-but-wrong audio: in German text
    // the comma is the decimal mark and the period groups thousands.
    REQUIRE(expand("3,14") == "drei Komma eins vier");
    REQUIRE(expand("1.000 Stück") == "eintausend Stück");

    // A period NOT followed by exactly three digits is not a separator, so a
    // sentence-final full stop survives as punctuation.
    REQUIRE(expand("Es kostet 5.") == "Es kostet fünf.");
}

TEST_CASE("num2words-de: signs, percent and non-numeric tokens", "[unit][num2words-de]") {
    REQUIRE(expand("-5 Grad") == "minus fünf Grad");
    REQUIRE(expand("50%") == "fünfzig Prozent");
    REQUIRE(expand("Am 1. Mai") == "Am erste Mai");

    // Alphanumerics are left for the word rules — "mp3" must not become "mp drei".
    REQUIRE(expand("mp3 und x64") == "mp3 und x64");

    // Nothing numeric: byte-identical passthrough.
    REQUIRE(expand("Guten Morgen") == "Guten Morgen");
    REQUIRE(expand("") == "");
}

TEST_CASE("num2words-de: the currency unit is spoken, and is invariable", "[unit][num2words-de]") {
    // The unit vanished exactly like the digits did — it is in no dictionary
    // and no letter-to-sound rule. Both written forms occur.
    REQUIRE(expand("\u20ac50") == "f\u00fcnfzig Euro");
    REQUIRE(expand("50\u20ac") == "f\u00fcnfzig Euro");
    REQUIRE(expand("50 \u20ac") == "f\u00fcnfzig Euro");
    REQUIRE(expand("$50") == "f\u00fcnfzig Dollar");
    REQUIRE(expand("\u00a350") == "f\u00fcnfzig Pfund");

    // German units do NOT pluralise: 50 Euro, never 50 Euros.
    REQUIRE(expand("2\u20ac") == "zwei Euro");

    // …but the NUMBER takes its attributive form before the noun: ein Euro,
    // not eins Euro. Only a trailing standalone "eins" changes.
    REQUIRE(expand("1\u20ac") == "ein Euro");
    REQUIRE(expand("101\u20ac") == "einhundertein Euro");
    REQUIRE(expand("21\u20ac") == "einundzwanzig Euro"); // untouched
}

TEST_CASE("num2words-de: #316 regression — German G2P must not DROP numbers", "[unit][num2words-de]") {
    // The bug: g2p_de had no number path, digits are in no dictionary and no
    // letter-to-sound rule, so a numeric token phonemized to "" and vanished.
    // Reproduced on main as:
    //     "Ich habe 82 Euro" -> "ɪç hɑːbə  ɔʏ̯roː"      (note the double space)
    //     "82"               -> ""
    // These assertions fail against an unwired g2p_de — that is the point.
    g2p_de::context ctx;

    const std::string bare = g2p_de::text_to_ipa(ctx, "82");
    INFO("phonemes for '82': '" << bare << "'");
    REQUIRE_FALSE(bare.empty());

    const std::string sentence = g2p_de::text_to_ipa(ctx, "Ich habe 82 Euro");
    INFO("phonemes for 'Ich habe 82 Euro': '" << sentence << "'");
    // Longer than the same sentence without the number, i.e. the number
    // contributed phonemes rather than disappearing.
    const std::string without = g2p_de::text_to_ipa(ctx, "Ich habe Euro");
    REQUIRE(sentence.size() > without.size());

    // And every digit-bearing sentence must produce SOME phonemes for the digits.
    for (const char* t : {"3 Euro", "Im Jahr 1984", "50%", "Am 1. Mai", "3,14"}) {
        const std::string ipa = g2p_de::text_to_ipa(ctx, t);
        INFO("input '" << t << "' -> '" << ipa << "'");
        REQUIRE_FALSE(ipa.empty());
    }
}

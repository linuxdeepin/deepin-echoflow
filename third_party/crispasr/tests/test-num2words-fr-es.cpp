// test-num2words-fr-es.cpp — French + Spanish number expansion (#316).
//
// Hermetic: no model, no network, no GPU. Same reason as the German file — a
// wrong entry here produces perfectly well-formed phonemes for the WRONG WORD,
// so every numeric check we have passes while the audio is wrong.
//
// The two regressions at the bottom are why the file exists: before this,
// `g2p_fr` and `g2p_es` had no number path, so
//     "J'ai 82 euros"  -> "ʒ ɛ  øʁɔ"
//     "Tengo 82 euros" -> "tenɡo  euɾos"
// with the 82 silently gone in both.

#include "core/g2p_es.h"
#include "core/g2p_fr.h"
#include "core/num2words_es.h"
#include "core/num2words_fr.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

// ── French ──────────────────────────────────────────────────────────────────

TEST_CASE("num2words-fr: counting goes vigesimal above 60", "[unit][num2words-fr]") {
    // The single reason French is not a table lookup.
    REQUIRE(core_num2words_fr::cardinal(70) == "soixante-dix");
    REQUIRE(core_num2words_fr::cardinal(72) == "soixante-douze");
    REQUIRE(core_num2words_fr::cardinal(79) == "soixante-dix-neuf");
    REQUIRE(core_num2words_fr::cardinal(80) == "quatre-vingts");
    REQUIRE(core_num2words_fr::cardinal(82) == "quatre-vingt-deux");
    REQUIRE(core_num2words_fr::cardinal(90) == "quatre-vingt-dix");
    REQUIRE(core_num2words_fr::cardinal(95) == "quatre-vingt-quinze");
    REQUIRE(core_num2words_fr::cardinal(99) == "quatre-vingt-dix-neuf");

    // 17-19 are compounds too, but 16 is lexical — the boundary a loop gets wrong.
    REQUIRE(core_num2words_fr::cardinal(16) == "seize");
    REQUIRE(core_num2words_fr::cardinal(17) == "dix-sept");
}

TEST_CASE("num2words-fr: 'et' joins 21..61 and 71, never 81 or 91", "[unit][num2words-fr]") {
    REQUIRE(core_num2words_fr::cardinal(21) == "vingt et un");
    REQUIRE(core_num2words_fr::cardinal(31) == "trente et un");
    REQUIRE(core_num2words_fr::cardinal(61) == "soixante et un");
    REQUIRE(core_num2words_fr::cardinal(71) == "soixante et onze"); // the last one

    // …and the vigesimal decades take a hyphen instead.
    REQUIRE(core_num2words_fr::cardinal(81) == "quatre-vingt-un");
    REQUIRE(core_num2words_fr::cardinal(91) == "quatre-vingt-onze");
    REQUIRE(core_num2words_fr::cardinal(22) == "vingt-deux");
}

TEST_CASE("num2words-fr: the plural s drops as soon as anything follows", "[unit][num2words-fr]") {
    // `vingt` and `cent` pluralise only when they END the number — including
    // when a SCALE WORD follows, which is the case a naive port misses.
    REQUIRE(core_num2words_fr::cardinal(80) == "quatre-vingts");
    REQUIRE(core_num2words_fr::cardinal(80000) == "quatre-vingt mille"); // no s
    REQUIRE(core_num2words_fr::cardinal(200) == "deux cents");
    REQUIRE(core_num2words_fr::cardinal(201) == "deux cent un");       // no s
    REQUIRE(core_num2words_fr::cardinal(200000) == "deux cent mille"); // no s

    // `mille` is invariable — "deux milles" is always wrong.
    REQUIRE(core_num2words_fr::cardinal(1000) == "mille"); // never "un mille"
    REQUIRE(core_num2words_fr::cardinal(2000) == "deux mille");
    REQUIRE(core_num2words_fr::cardinal(1000000) == "un million");
    REQUIRE(core_num2words_fr::cardinal(2000000) == "deux millions");
    REQUIRE(core_num2words_fr::cardinal(0) == "zéro");
}

TEST_CASE("num2words-fr: expansion, separators and ordinals", "[unit][num2words-fr]") {
    REQUIRE(core_num2words_fr::expand("J'ai 82 euros") == "J'ai quatre-vingt-deux euros");
    REQUIRE(core_num2words_fr::expand("3,14") == "trois virgule un quatre");    // comma decimal
    REQUIRE(core_num2words_fr::expand("1 000 personnes") == "mille personnes"); // space thousands
    REQUIRE(core_num2words_fr::expand("50%") == "cinquante pour cent");
    REQUIRE(core_num2words_fr::expand("-5 degres") == "moins cinq degres");
    REQUIRE(core_num2words_fr::expand("le 1er mai") == "le premier mai");
    REQUIRE(core_num2words_fr::expand("le 2e jour") == "le deuxième jour");

    REQUIRE(core_num2words_fr::ordinal(5) == "cinquième");         // cinq gains a u
    REQUIRE(core_num2words_fr::ordinal(9) == "neuvième");          // neuf: f -> v
    REQUIRE(core_num2words_fr::ordinal(4) == "quatrième");         // silent e drops
    REQUIRE(core_num2words_fr::ordinal(80) == "quatre-vingtième"); // and so does the s

    // Alphanumerics stay for the word rules; nothing numeric passes through.
    REQUIRE(core_num2words_fr::expand("mp3 et x64") == "mp3 et x64");
    REQUIRE(core_num2words_fr::expand("Bonjour") == "Bonjour");
}

// ── Spanish ─────────────────────────────────────────────────────────────────

TEST_CASE("num2words-es: 16-29 fuse into one accented word", "[unit][num2words-es]") {
    REQUIRE(core_num2words_es::cardinal(16) == "dieciséis"); // not "diez y seis"
    REQUIRE(core_num2words_es::cardinal(17) == "diecisiete");
    REQUIRE(core_num2words_es::cardinal(21) == "veintiuno");
    REQUIRE(core_num2words_es::cardinal(22) == "veintidós");
    REQUIRE(core_num2words_es::cardinal(23) == "veintitrés");
    REQUIRE(core_num2words_es::cardinal(26) == "veintiséis");

    // "y" appears only from 31 up — never inside the fused range.
    REQUIRE(core_num2words_es::cardinal(31) == "treinta y uno");
    REQUIRE(core_num2words_es::cardinal(82) == "ochenta y dos");
    REQUIRE(core_num2words_es::cardinal(99) == "noventa y nueve");
    REQUIRE(core_num2words_es::cardinal(20) == "veinte");
    REQUIRE(core_num2words_es::cardinal(30) == "treinta");
}

TEST_CASE("num2words-es: four hundreds are irregular, and 100 has two forms", "[unit][num2words-es]") {
    REQUIRE(core_num2words_es::cardinal(500) == "quinientos");  // not cincocientos
    REQUIRE(core_num2words_es::cardinal(700) == "setecientos"); // not sietecientos
    REQUIRE(core_num2words_es::cardinal(900) == "novecientos"); // not nuevecientos
    REQUIRE(core_num2words_es::cardinal(200) == "doscientos");  // …and the regular ones
    REQUIRE(core_num2words_es::cardinal(600) == "seiscientos");

    REQUIRE(core_num2words_es::cardinal(100) == "cien");       // standing alone
    REQUIRE(core_num2words_es::cardinal(101) == "ciento uno"); // with anything after
    REQUIRE(core_num2words_es::cardinal(110) == "ciento diez");
}

TEST_CASE("num2words-es: uno apocopates before a scale word", "[unit][num2words-es]") {
    // Standing alone the full form is right…
    REQUIRE(core_num2words_es::cardinal(21) == "veintiuno");
    REQUIRE(core_num2words_es::cardinal(41) == "cuarenta y uno");
    // …but before mil/millones it shortens, and veintiuno takes its accent.
    REQUIRE(core_num2words_es::cardinal(21000) == "veintiún mil");
    REQUIRE(core_num2words_es::cardinal(31000) == "treinta y un mil");
    REQUIRE(core_num2words_es::cardinal(101000) == "ciento un mil");

    // mil takes no article; a million does.
    REQUIRE(core_num2words_es::cardinal(1000) == "mil"); // never "un mil"
    REQUIRE(core_num2words_es::cardinal(2000) == "dos mil");
    REQUIRE(core_num2words_es::cardinal(1000000) == "un millón");
    REQUIRE(core_num2words_es::cardinal(2000000) == "dos millones");
    REQUIRE(core_num2words_es::cardinal(0) == "cero");
}

TEST_CASE("num2words-es: expansion and separators", "[unit][num2words-es]") {
    REQUIRE(core_num2words_es::expand("Tengo 82 euros") == "Tengo ochenta y dos euros");
    REQUIRE(core_num2words_es::expand("3,14") == "tres coma uno cuatro"); // comma decimal
    REQUIRE(core_num2words_es::expand("1.000 personas") == "mil personas");
    REQUIRE(core_num2words_es::expand("50%") == "cincuenta por ciento");
    REQUIRE(core_num2words_es::expand("-5 grados") == "menos cinco grados");
    // A period not before exactly three digits stays punctuation.
    REQUIRE(core_num2words_es::expand("Cuesta 5.") == "Cuesta cinco.");
    REQUIRE(core_num2words_es::expand("mp3 y x64") == "mp3 y x64");
    REQUIRE(core_num2words_es::expand("Hola") == "Hola");
}

TEST_CASE("num2words-fr: the currency unit is spoken", "[unit][num2words-fr]") {
    REQUIRE(core_num2words_fr::expand("\u20ac50") == "cinquante euros");
    REQUIRE(core_num2words_fr::expand("50\u20ac") == "cinquante euros");
    REQUIRE(core_num2words_fr::expand("50 \u20ac") == "cinquante euros");
    REQUIRE(core_num2words_fr::expand("1\u20ac") == "un euro");
    REQUIRE(core_num2words_fr::expand("$50") == "cinquante dollars");
    REQUIRE(core_num2words_fr::expand("\u00a350") == "cinquante livres");
}

TEST_CASE("num2words-es: the currency unit is spoken, and uno apocopates", "[unit][num2words-es]") {
    REQUIRE(core_num2words_es::expand("\u20ac50") == "cincuenta euros");
    REQUIRE(core_num2words_es::expand("50\u20ac") == "cincuenta euros");
    REQUIRE(core_num2words_es::expand("50 \u20ac") == "cincuenta euros");
    // The dollar singular carries an accent the plural loses.
    REQUIRE(core_num2words_es::expand("$50") == "cincuenta d\u00f3lares");
    REQUIRE(core_num2words_es::expand("$1") == "un d\u00f3lar");
    // A unit noun triggers the same apocopation `mil` does.
    REQUIRE(core_num2words_es::expand("1\u20ac") == "un euro");
    REQUIRE(core_num2words_es::expand("21\u20ac") == "veinti\u00fan euros");
}

// ── The regressions ─────────────────────────────────────────────────────────

TEST_CASE("num2words-fr: #316 regression — French G2P must not DROP numbers", "[unit][num2words-fr]") {
    // Fails against an unwired g2p_fr — that is the point.
    g2p_fr::context ctx;
    const std::string bare = g2p_fr::text_to_ipa(ctx, "82");
    INFO("phonemes for '82': '" << bare << "'");
    REQUIRE_FALSE(bare.empty());

    const std::string with = g2p_fr::text_to_ipa(ctx, "J'ai 82 euros");
    const std::string without = g2p_fr::text_to_ipa(ctx, "J'ai euros");
    INFO("with='" << with << "' without='" << without << "'");
    REQUIRE(with.size() > without.size());

    for (const char* t : {"3 euros", "50%", "3,14"}) {
        const std::string ipa = g2p_fr::text_to_ipa(ctx, t);
        INFO("input '" << t << "' -> '" << ipa << "'");
        REQUIRE_FALSE(ipa.empty());
    }
}

TEST_CASE("num2words-es: #316 regression — Spanish G2P must not DROP numbers", "[unit][num2words-es]") {
    g2p_es::context ctx;
    const std::string bare = g2p_es::text_to_ipa(ctx, "82");
    INFO("phonemes for '82': '" << bare << "'");
    REQUIRE_FALSE(bare.empty());

    const std::string with = g2p_es::text_to_ipa(ctx, "Tengo 82 euros");
    const std::string without = g2p_es::text_to_ipa(ctx, "Tengo euros");
    INFO("with='" << with << "' without='" << without << "'");
    REQUIRE(with.size() > without.size());

    for (const char* t : {"3 euros", "50%", "3,14"}) {
        const std::string ipa = g2p_es::text_to_ipa(ctx, t);
        INFO("input '" << t << "' -> '" << ipa << "'");
        REQUIRE_FALSE(ipa.empty());
    }
}

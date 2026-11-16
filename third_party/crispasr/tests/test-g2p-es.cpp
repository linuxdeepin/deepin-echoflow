// tests/test-g2p-es.cpp — unit tests for Spanish G2P.

#include <catch2/catch_test_macros.hpp>
#include "core/g2p_es.h"
#include <string>

TEST_CASE("Spanish LTS: ch/ll/rr/ñ", "[g2p_es][lts]") {
    SECTION("ch → tʃ") {
        std::string ipa = g2p_es::lts_word_to_ipa("chico");
        CHECK(ipa.find("t\xCA\x83") != std::string::npos);
    }
    SECTION("ll → ʝ (yeísmo)") {
        std::string ipa = g2p_es::lts_word_to_ipa("calle");
        CHECK(ipa.find("\xCA\x9D") != std::string::npos);
    }
    SECTION("rr → trill") {
        std::string ipa = g2p_es::lts_word_to_ipa("perro");
        CHECK(ipa.find("r") != std::string::npos);
    }
    SECTION("ñ → ɲ") {
        std::string ipa = g2p_es::lts_word_to_ipa("espa\xC3\xB1"
                                                  "a");
        CHECK(ipa.find("\xC9\xB2") != std::string::npos);
    }
}

TEST_CASE("Spanish LTS: b/d/g lenition", "[g2p_es][lts]") {
    SECTION("initial b → stop") {
        std::string ipa = g2p_es::lts_word_to_ipa("bueno");
        CHECK(ipa[0] == 'b');
    }
    SECTION("intervocalic b → β") {
        std::string ipa = g2p_es::lts_word_to_ipa("haba");
        CHECK(ipa.find("\xCE\xB2") != std::string::npos);
    }
    SECTION("initial d → stop") {
        std::string ipa = g2p_es::lts_word_to_ipa("dia");
        CHECK(ipa[0] == 'd');
    }
    SECTION("intervocalic d → ð") {
        std::string ipa = g2p_es::lts_word_to_ipa("nada");
        CHECK(ipa.find("\xC3\xB0") != std::string::npos);
    }
}

TEST_CASE("Spanish LTS: c/z seseo, g/j jota", "[g2p_es][lts]") {
    SECTION("ce → s (seseo)") {
        std::string ipa = g2p_es::lts_word_to_ipa("cena");
        CHECK(ipa[0] == 's');
    }
    SECTION("z → s") {
        std::string ipa = g2p_es::lts_word_to_ipa("zapato");
        CHECK(ipa[0] == 's');
    }
    SECTION("j → x (jota)") {
        std::string ipa = g2p_es::lts_word_to_ipa("jugar");
        CHECK(ipa[0] == 'x');
    }
    SECTION("ge → x") {
        std::string ipa = g2p_es::lts_word_to_ipa("gente");
        CHECK(ipa[0] == 'x');
    }
    SECTION("ga → ɡ") {
        std::string ipa = g2p_es::lts_word_to_ipa("gato");
        CHECK(ipa.find("\xC9\xA1") != std::string::npos);
    }
}

TEST_CASE("Spanish LTS: silent h, qu", "[g2p_es][lts]") {
    SECTION("h is silent") {
        std::string ipa = g2p_es::lts_word_to_ipa("hola");
        CHECK(ipa.find("h") == std::string::npos);
    }
    SECTION("qu → k") {
        std::string ipa = g2p_es::lts_word_to_ipa("que");
        CHECK(ipa[0] == 'k');
    }
}

TEST_CASE("Spanish LTS: vowels and r", "[g2p_es][lts]") {
    SECTION("initial r → trill") {
        std::string ipa = g2p_es::lts_word_to_ipa("rojo");
        CHECK(ipa[0] == 'r');
    }
    SECTION("intervocalic r → tap ɾ") {
        std::string ipa = g2p_es::lts_word_to_ipa("pero");
        CHECK(ipa.find("\xC9\xBE") != std::string::npos);
    }
    SECTION("word-final y → i") {
        std::string ipa = g2p_es::lts_word_to_ipa("hoy");
        bool has_i = ipa.back() == 'i' || ipa.find("i") != std::string::npos;
        CHECK(has_i);
    }
}

TEST_CASE("Spanish text_to_ipa", "[g2p_es][sentence]") {
    g2p_es::context ctx;
    SECTION("hola mundo") {
        std::string ipa = g2p_es::text_to_ipa(ctx, "hola mundo");
        CHECK(!ipa.empty());
        CHECK(ipa.find(' ') != std::string::npos);
        // h silent, so first sound should be 'o'
        CHECK(ipa[0] == 'o');
    }
}

// ── #316 round 2: punctuation is the consumer's choice ──────────────────────
//
// Kokoro's 178-symbol vocabulary contains `,.;:!?` and they are how it pauses;
// dropping them delivered a paragraph in one breath. English proved it against
// misaki; this is the same defect in the same shape, one language over. Off by
// default so piper — whose espeak inventory has never been fed punctuation — is
// unchanged.

TEST_CASE("es: punctuation is dropped by default, kept on request", "[g2p_es][unit][punct]") {
    g2p_es::context ctx;
    const std::string plain = g2p_es::text_to_ipa(ctx, "a b, c.");
    CHECK(plain.find(',') == std::string::npos);
    CHECK(plain.find('.') == std::string::npos);
    // A dropped mark still separates its neighbours — with ONE space, not the
    // two the old loop emitted around every comma.
    CHECK(plain.find("  ") == std::string::npos);

    ctx.emit_punctuation = true;
    const std::string kept = g2p_es::text_to_ipa(ctx, "a b, c.");
    CHECK(kept.find(',') != std::string::npos);
    CHECK(kept.find('.') != std::string::npos);
    // A mark sits flush against the word before it and is followed by a space.
    CHECK(kept.find(" ,") == std::string::npos);
    CHECK(kept.find(", ") != std::string::npos);
}

TEST_CASE("es: the hyphen is a separator, never a mark", "[g2p_es][unit][punct]") {
    g2p_es::context ctx;
    ctx.emit_punctuation = true;
    // No TTS vocabulary here has an ASCII hyphen, and misaki drops it.
    CHECK(g2p_es::text_to_ipa(ctx, "a-b").find('-') == std::string::npos);
}

TEST_CASE("es: a quoted word is looked up without its quotes", "[g2p_es][unit][punct]") {
    // The tokenizer split on ,.!?;:- only, so a quoted word reached every
    // lookup tier WEARING its quotes and fell through to the letter-to-sound
    // rules — the same defect that made English read "dramatic" as DRAM-atic.
    g2p_es::context ctx;
    const std::string quoted = g2p_es::text_to_ipa(ctx, "\"ab\"");
    const std::string bare = g2p_es::text_to_ipa(ctx, "ab");
    CHECK(quoted == bare);
}

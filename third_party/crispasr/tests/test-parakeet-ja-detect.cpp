// test-parakeet-ja-detect.cpp — issue #257 regression guard.
//
// The parakeet JA-model detection must key on vocab CONTENT, not size, so a
// small-vocab ENGLISH model (parakeet-tdt-1.1b, vocab 1024) is not misclassified
// as Japanese and forced onto the JA short-chunk decode path.

#include <catch2/catch_test_macros.hpp>

#include "parakeet_ja_detect.h"

#include <string>
#include <vector>

using crispasr_parakeet::token_has_japanese;
using crispasr_parakeet::vocab_looks_japanese;

TEST_CASE("token_has_japanese: kana and kanji are detected, Latin is not", "[unit][parakeet][issue-257]") {
    REQUIRE(token_has_japanese("\xE3\x81\x82"));     // U+3042 あ (hiragana)
    REQUIRE(token_has_japanese("\xE3\x82\xAB"));     // U+30AB カ (katakana)
    REQUIRE(token_has_japanese("\xE6\x97\xA5"));     // U+65E5 日 (kanji)
    REQUIRE(token_has_japanese("\xE6\x97\xA5\x62")); // 日b — mixed still JA
    REQUIRE_FALSE(token_has_japanese("hello"));
    REQUIRE_FALSE(token_has_japanese("don't"));
    REQUIRE_FALSE(token_has_japanese(""));
}

TEST_CASE("vocab_looks_japanese: JA-only vocab is Japanese", "[unit][parakeet][issue-257]") {
    // ~97% kana/kanji like parakeet-tdt-0.6b-ja (vocab_size ~3073).
    std::vector<std::string> ja = {"<blank>", "\xE6\x97\xA5", "\xE6\x99\x82",       "\xE3\x83\x8A",
                                   "\xE3\x81\x84\xE3\x81\xA6", "\xE6\x97\xA5\xE6\x9C\xAC", "\xE3\x81\x93"};
    REQUIRE(vocab_looks_japanese(ja));
}

TEST_CASE("vocab_looks_japanese: small English vocab is NOT Japanese (parakeet-tdt-1.1b)",
          "[unit][parakeet][issue-257]") {
    // vocab_size ~1024, all Latin — the exact case that was misclassified.
    std::vector<std::string> en = {"<blank>", " okay",  " i",   " don't", " understand",
                                   " why",    " nick",  " here", "ing",    "'s"};
    REQUIRE_FALSE(vocab_looks_japanese(en));
}

TEST_CASE("vocab_looks_japanese: multilingual vocab with a few JA language tags is NOT Japanese",
          "[unit][parakeet][issue-257]") {
    // parakeet-tdt-0.6b-v3 (vocab 8192): language tags like <|ja|> are skipped as
    // specials, and the body is Latin/other scripts.
    std::vector<std::string> ml = {"<|ja|>", "<|en|>",   "<|zh|>", " hello", " world",
                                   " bonjour", " country", " ask",   "ing",    "'s"};
    REQUIRE_FALSE(vocab_looks_japanese(ml));
}

TEST_CASE("vocab_looks_japanese: empty / all-special vocab is not Japanese", "[unit][parakeet][issue-257]") {
    REQUIRE_FALSE(vocab_looks_japanese({}));
    REQUIRE_FALSE(vocab_looks_japanese({"<blank>", "<pad>", "<|ja|>"}));
}

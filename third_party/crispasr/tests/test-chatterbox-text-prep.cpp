// test-chatterbox-text-prep.cpp - regression tests for Chatterbox text prep.

#include "chatterbox_text_prep.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("chatterbox text prep normalizes punctuation and whitespace", "[unit][chatterbox][text-prep]") {
    REQUIRE(chatterbox_text_prep::normalize("  hello   world  ", false) == "Hello world.");
    REQUIRE(chatterbox_text_prep::normalize("hello...   world", false) == "Hello, world.");
    REQUIRE(chatterbox_text_prep::normalize("hello\u00a0world", false) == "Hello world.");
}

TEST_CASE("chatterbox multilingual prep lowercases ascii after punctuation normalization",
          "[unit][chatterbox][text-prep]") {
    REQUIRE(chatterbox_text_prep::normalize("  Justice   Justice  ", true) == "justice justice.");
    REQUIRE(chatterbox_text_prep::normalize("Fur, justice", true) == "fur, justice.");
}

// Issue #170: the multilingual path must apply NFKD like upstream
// MTLTokenizer.preprocess_text, or precomposed graphemes tokenize to the
// wrong id and the model emits spurious onset phonemes.
TEST_CASE("chatterbox multilingual prep applies NFKD normalization (#170)", "[unit][chatterbox][text-prep]") {
    // U+0623 ARABIC LETTER ALEF WITH HAMZA ABOVE decomposes to
    // U+0627 ALEF + U+0654 COMBINING HAMZA ABOVE (the bug's first word).
    REQUIRE(chatterbox_text_prep::normalize("\xd8\xa3", true) == "\xd8\xa7\xd9\x94.");
    // NFKD is a no-op for text without precomposed forms (the working case).
    REQUIRE(chatterbox_text_prep::normalize("\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7", true) ==
            "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7.");
    // Accented Latin: U+00C9 "É" decomposes to "E" + U+0301 combining acute,
    // then ASCII-lowercases to "e" + combining acute.
    REQUIRE(chatterbox_text_prep::normalize("\xc3\x89ric", true) == "\x65\xcc\x81ric.");
    // NFKD must NOT fire on the non-multilingual (English) path.
    REQUIRE(chatterbox_text_prep::normalize("\xc3\x89ric", false) == "\xc3\x89ric.");
}

// test-uroman.cpp — unit tests for core/uroman.h romanization.
// Validates Arabic, Cyrillic, and mixed-script romanization for CTC alignment.

#include <catch2/catch_test_macros.hpp>
#include "core/uroman.h"

using namespace core_uroman;

TEST_CASE("uroman: ASCII passes through unchanged", "[unit][uroman]") {
    REQUIRE(romanize("Hello world") == "Hello world");
    REQUIRE(romanize("test 123 !@#") == "test 123 !@#");
}

TEST_CASE("uroman: Arabic consonants romanize", "[unit][uroman]") {
    // بسم = b-s-m
    REQUIRE(romanize("\xd8\xa8\xd8\xb3\xd9\x85") == "bsm");
    // كتاب = k-t-a-b
    REQUIRE(romanize("\xd9\x83\xd8\xaa\xd8\xa7\xd8\xa8") == "ktab");
}

TEST_CASE("uroman: Arabic sentence romanizes", "[unit][uroman]") {
    // سأرى = s-a-r-a (alef hamza above + ra + alef maksura)
    std::string arabic = "\xd8\xb3\xd8\xa3\xd8\xb1\xd9\x89";
    std::string rom = romanize(arabic);
    REQUIRE(!rom.empty());
    REQUIRE(rom.find_first_of("sr") != std::string::npos);
}

TEST_CASE("uroman: Cyrillic romanizes", "[unit][uroman]") {
    // Привет = Privet
    std::string cyrillic = "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82";
    REQUIRE(romanize(cyrillic) == "privet");
}

TEST_CASE("uroman: mixed script preserves spaces", "[unit][uroman]") {
    // "مرحبا world" → "mrhba world"
    std::string mixed = "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 world";
    std::string rom = romanize(mixed);
    REQUIRE(rom.find(" world") != std::string::npos);
    REQUIRE(rom[0] == 'm'); // mim
}

TEST_CASE("uroman: needs_romanization detects Arabic", "[unit][uroman]") {
    REQUIRE(needs_romanization("\xd8\xa8\xd8\xb3\xd9\x85"));
    REQUIRE_FALSE(needs_romanization("Hello world"));
    REQUIRE_FALSE(needs_romanization("123 test"));
}

TEST_CASE("uroman: needs_romanization detects Cyrillic", "[unit][uroman]") {
    REQUIRE(needs_romanization("\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82"));
}

TEST_CASE("uroman: empty string", "[unit][uroman]") {
    REQUIRE(romanize("") == "");
    REQUIRE_FALSE(needs_romanization(""));
}

TEST_CASE("uroman: per-word romanization is label-safe (#419)", "[unit][uroman]") {
    // The aligner romanizes PER WORD to build CTC labels while returning the
    // original-script words to callers (crispasr_align_words::restore_text).
    // That 1:1 mapping requires romanize(word) to stay a single, non-empty,
    // whitespace-free token for the scripts we align — a romanization that
    // split or emptied a word would desync labels from originals and the
    // restore would bail to the aligner's own labels (the #419 translit).
    const char* words[] = {
        "\xd0\xb2\xd0\xb8\xd0\xba\xd0\xb8\xd0\xbd\xd0\xb3\xd0\xb8",         // викинги
        "\xd0\xbe\xd1\x82\xd0\xb2\xd0\xb0\xd0\xb6\xd0\xbd\xd1\x8b\xd0\xb5", // отважные
        "\xd0\xb2\xd0\xbe\xd0\xb8\xd0\xbd\xd1\x8b,",                        // воины, (with punct)
        "\xd0\x95\xd0\xb2\xd1\x80\xd0\xbe\xd0\xbf\xd1\x8b.",                // Европы.
    };
    for (const char* w : words) {
        REQUIRE(needs_romanization(w));
        const std::string r = romanize(w);
        REQUIRE_FALSE(r.empty());
        REQUIRE(r.find(' ') == std::string::npos);
        REQUIRE(r.find('\t') == std::string::npos);
    }
}

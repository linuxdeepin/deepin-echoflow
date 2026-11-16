// test-asr-parity.cpp — unit tests for core_parity::segments_equal, the
// canonical cross-surface parity predicate (improvements Phase 0).

#include <catch2/catch_test_macros.hpp>

#include "core/asr_parity.h"

using core_parity::norm_text;
using core_parity::segments_equal;
using core_parity::SegView;

TEST_CASE("norm_text trims and collapses whitespace", "[unit][parity]") {
    REQUIRE(norm_text("  hello   world  ") == "hello world");
    REQUIRE(norm_text("a\tb\nc") == "a b c");
    REQUIRE(norm_text("nochange") == "nochange");
    REQUIRE(norm_text("") == "");
    REQUIRE(norm_text("   ") == "");
}

TEST_CASE("segments_equal: identical lists match", "[unit][parity]") {
    std::vector<SegView> a = {{"hello world", 0, 100}, {"foo", 100, 200}};
    std::vector<SegView> b = a;
    REQUIRE(segments_equal(a, b));
}

TEST_CASE("segments_equal: whitespace-only text diff still matches", "[unit][parity]") {
    std::vector<SegView> a = {{"hello world", 0, 100}};
    std::vector<SegView> b = {{"  hello   world\n", 0, 100}};
    REQUIRE(segments_equal(a, b));
}

TEST_CASE("segments_equal: case / punctuation diff does NOT match", "[unit][parity]") {
    REQUIRE_FALSE(segments_equal({{"Hello", 0, 1}}, {{"hello", 0, 1}}));
    REQUIRE_FALSE(segments_equal({{"hi.", 0, 1}}, {{"hi", 0, 1}}));
}

TEST_CASE("segments_equal: length mismatch fails", "[unit][parity]") {
    std::vector<SegView> a = {{"a", 0, 1}, {"b", 1, 2}};
    std::vector<SegView> b = {{"a", 0, 1}};
    REQUIRE_FALSE(segments_equal(a, b));
}

TEST_CASE("segments_equal: offset tolerance", "[unit][parity]") {
    std::vector<SegView> a = {{"x", 100, 200}};
    std::vector<SegView> b = {{"x", 103, 197}};
    REQUIRE_FALSE(segments_equal(a, b, 0)); // exact
    REQUIRE_FALSE(segments_equal(a, b, 2)); // within 2 cs? 3 > 2 → no
    REQUIRE(segments_equal(a, b, 3));       // within 3 cs → yes
}

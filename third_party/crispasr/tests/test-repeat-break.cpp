// test-repeat-break.cpp — unit tests for core_repeat::tail_is_repetition, the
// decode-time repetition-loop detector (fix/moonshine-repeat-break).

#include <catch2/catch_test_macros.hpp>

#include "core/repeat_break.h"

#include <vector>

using core_repeat::tail_is_repetition;

namespace {
std::vector<int> seq(std::initializer_list<int> l) {
    return std::vector<int>(l);
}
// n copies of a block appended to a prefix.
std::vector<int> loop(std::vector<int> prefix, std::vector<int> block, int n) {
    for (int i = 0; i < n; ++i)
        prefix.insert(prefix.end(), block.begin(), block.end());
    return prefix;
}
} // namespace

TEST_CASE("clean sequence is not a loop", "[unit][repeat-break]") {
    REQUIRE_FALSE(tail_is_repetition(seq({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})));
    REQUIRE_FALSE(tail_is_repetition(std::vector<int>{}));
}

TEST_CASE("period-1 token repeated >= 4x is a loop", "[unit][repeat-break]") {
    REQUIRE(tail_is_repetition(seq({5, 5, 5, 5})));
    REQUIRE(tail_is_repetition(seq({1, 2, 5, 5, 5, 5}))); // loop at the tail
    REQUIRE_FALSE(tail_is_repetition(seq({5, 5, 5})));    // only 3x
}

TEST_CASE("period-2 block repeated 4x (the moonshine 'I'm sorry' case)", "[unit][repeat-break]") {
    // block {7,8} = "I'm sorry" repeated 4x at the tail.
    REQUIRE(tail_is_repetition(loop({1, 2, 3}, {7, 8}, 4)));
    REQUIRE_FALSE(tail_is_repetition(loop({1, 2, 3}, {7, 8}, 3))); // 3x below threshold
}

TEST_CASE("period-3 block repeated 4x", "[unit][repeat-break]") {
    REQUIRE(tail_is_repetition(loop({0}, {4, 5, 6}, 4)));
}

TEST_CASE("natural short repeat (2x) is not flagged", "[unit][repeat-break]") {
    REQUIRE_FALSE(tail_is_repetition(seq({9, 9})));          // "very very"
    REQUIRE_FALSE(tail_is_repetition(loop({1}, {3, 4}, 2))); // block 2x
}

TEST_CASE("min_reps / max_period parameters respected", "[unit][repeat-break]") {
    // 3x flagged when min_reps lowered to 3.
    REQUIRE(tail_is_repetition(seq({5, 5, 5}), /*min_reps=*/3));
    // period-10 block not detected with default max_period=8.
    auto p10 = loop({}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 4);
    REQUIRE_FALSE(tail_is_repetition(p10));
    REQUIRE(tail_is_repetition(p10, 4, /*max_period=*/10));
}

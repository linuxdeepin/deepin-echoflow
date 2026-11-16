// test-segment-group.cpp — unit tests for core_segment::group_by_window, the
// output-segment grouping used to split parakeet's single coherent decode into
// ~N-second segments (issue #257).

#include <catch2/catch_test_macros.hpp>

#include "core/asr_segment_group.h"

#include <cstdint>
#include <vector>

using core_segment::group_by_window;

TEST_CASE("group_by_window: empty input yields no segments", "[unit][segment-group][issue-257]") {
    REQUIRE(group_by_window({}, 700).empty());
}

TEST_CASE("group_by_window: single segment when everything fits the window", "[unit][segment-group][issue-257]") {
    // words at 0.0, 1.0, 2.0, 3.0 s; 7 s window → all in one segment.
    std::vector<int64_t> t0 = {0, 100, 200, 300};
    auto s = group_by_window(t0, 700);
    REQUIRE(s == std::vector<int>{0});
}

TEST_CASE("group_by_window: splits at the window boundary", "[unit][segment-group][issue-257]") {
    // words every 1 s over 20 s, 7 s window. Segment 0 starts at word 0 (t=0);
    // word 7 is at t=700 cs = exactly 7 s past → new segment; word 14 at 1400 →
    // third segment.
    std::vector<int64_t> t0;
    for (int i = 0; i < 20; ++i)
        t0.push_back((int64_t)i * 100);
    auto s = group_by_window(t0, 700);
    REQUIRE(s == std::vector<int>{0, 7, 14});
}

TEST_CASE("group_by_window: boundary is measured from each segment start, not global",
          "[unit][segment-group][issue-257]") {
    // Sparse tail: words at 0,1,2,3 s then a gap to 10,11 s. 7 s window.
    // seg0 = {0,1,2,3} (10 s - 0 = 10 >= 7 → new seg at the 10 s word).
    std::vector<int64_t> t0 = {0, 100, 200, 300, 1000, 1100};
    auto s = group_by_window(t0, 700);
    REQUIRE(s == std::vector<int>{0, 4});
}

TEST_CASE("group_by_window: a word longer than the window still starts one segment",
          "[unit][segment-group][issue-257]") {
    // Consecutive far-apart words each exceed the window → one segment each,
    // never a stall, never a dropped word.
    std::vector<int64_t> t0 = {0, 1000, 2000};
    auto s = group_by_window(t0, 700);
    REQUIRE(s == std::vector<int>{0, 1, 2});
}

TEST_CASE("group_by_window: non-positive window collapses to a single segment", "[unit][segment-group][issue-257]") {
    std::vector<int64_t> t0 = {0, 100, 200};
    REQUIRE(group_by_window(t0, 0) == std::vector<int>{0});
    REQUIRE(group_by_window(t0, -5) == std::vector<int>{0});
}

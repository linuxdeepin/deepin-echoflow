// #375 follow-up — canary long-form now follows canary-1b-v2's own
// `.transcribe()` dynamic-chunking blueprint instead of the parakeet-style
// 8 s / 2 s machinery that produced the "ask not Ask not" seam duplications.
//
// These tests pin the two ported primitives against vectors generated from
// the ACTUAL Python functions (nemo 2.7.3):
//   - streaming_utils.longest_common_subsequence_merge
//   - PromptedAudioToTextLhotseDataset._find_optimal_chunk_size
// Cross-implementation on purpose: a test that re-derives the expectation
// from the port itself would move with the port and stay green over a
// divergence.
#include "core/canary_chunk_merge.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

namespace {

struct LcsCase {
    std::vector<int> X, Y;
    int i, j, len;
};

} // namespace

TEST_CASE("nemo_lcs_merge matches NeMo's longest_common_subsequence_merge", "[unit][canary][chunk-merge]") {
    // Generated 2026-08-19 by running nemo 2.7.3's
    // streaming_utils.longest_common_subsequence_merge on each (X, Y) pair
    // (crafted cases + random.seed(42) cases at the realistic <=24 x <=7
    // sizes the 1 s-overlap merge produces).
    const LcsCase cases[] = {
        {{1, 2, 3, 4, 5, 6, 7, 8}, {6, 7, 8, 9, 10}, 5, 0, 3},
        {{1, 2, 3, 4, 5}, {9, 9, 9}, 5, 0, 0},
        {{1, 2, 3, 4, 5, 6, 7, 8}, {5, 99, 7, 8, 9}, 6, 2, 2},
        {{1, 2, 3, 4, 5, 6, 7, 8}, {7, 8}, 6, 0, 2},
        {{1, 2, 3, 4}, {1, 2, 3, 4}, 0, 0, 4},
        {{3, 3, 3, 3, 3}, {3, 3}, 3, 0, 2},
        {{1, 2, 3, 9, 9, 6, 7, 8}, {6, 7, 8, 20, 21}, 5, 0, 3},
        {{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33},
         {31, 32, 33, 40, 41, 42, 43},
         21,
         0,
         3},
        {{1, 12, 5, 4, 4, 3, 12, 2, 11, 12, 15, 9, 2, 10, 7, 1, 1, 2, 4, 4, 9}, {10}, 14, 0, 0},
        {{4}, {12, 11, 12, 9, 7}, 1, 0, 0},
        {{10, 5, 13, 14, 1, 13, 13, 3}, {12, 7, 6, 5}, 2, 0, 0},
        {{13, 6, 2, 2, 7}, {2, 6}, 4, 0, 0},
        {{5, 13, 1, 12, 8, 9, 2, 15, 7, 2, 9, 5}, {14, 11, 10, 15, 14}, 8, 0, 0},
        {{4, 12, 2, 1, 11, 4, 13, 5, 2, 14, 4, 14}, {2, 7, 5, 8, 11}, 9, 0, 0},
        {{6, 6, 4, 11, 5, 12, 15, 11, 11, 2, 10, 11}, {3, 9}, 12, 0, 0},
        {{3, 8, 7, 5, 15, 11, 12, 9, 4, 11, 6, 14, 13, 13, 1, 4, 14, 1, 13, 6, 7, 5, 2, 4}, {15, 10}, 5, 0, 0},
    };
    for (const auto& c : cases) {
        int i = -1, j = -1, len = -1;
        core_canary_chunk::nemo_lcs_merge(c.X, c.Y, i, j, len);
        INFO("X[0]=" << (c.X.empty() ? -1 : c.X[0]) << " |X|=" << c.X.size() << " |Y|=" << c.Y.size());
        CHECK(i == c.i);
        CHECK(j == c.j);
        CHECK(len == c.len);
    }
}

TEST_CASE("optimal_chunk_samples matches NeMo's _find_optimal_chunk_size", "[unit][canary][chunk-merge]") {
    // Generated 2026-08-19 from the exact _find_optimal_chunk_size (defaults:
    // 30..40 s, 16 kHz, 1 s overlap). total_len < 40 s returns total_len,
    // i.e. a single pass.
    const int cases[][2] = {
        {160000, 160000},   // 10 s
        {638400, 638400},   // 39.9 s
        {640000, 480000},   // 40 s -> 30 s chunks
        {656000, 480000},   // 41 s
        {960000, 496000},   // 60 s
        {1200000, 624000},  // 75 s
        {2112000, 544000},  // 132 s (jfk_x12)
        {4800000, 624000},  // 300 s
        {9504000, 576000},  // 594 s (fleurs_600s)
        {57600000, 592000}, // 1 h
    };
    for (const auto& c : cases) {
        CHECK(core_canary_chunk::optimal_chunk_samples(c[0], 16000) == c[1]);
    }
}

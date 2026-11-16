// Unit tests for core/powerset.h — the pyannote-seg-3.0 class layout.
//
// This file exists because the layout was wrong in production and nothing
// caught it. `SPK_MASK` in crispasr_diarize.cpp had classes 3 and 4 swapped,
// so every frame in which the THIRD local speaker was talking alone got
// scored as the first two speakers overlapping. On the VoxConverse dev shard
// that cost ~15 DER points (48.21% -> 33.37%).
//
// It survived because every existing test used at most two speakers, where
// the wrong table and the right one agree. So the tests below deliberately
// exercise the three-speaker classes and the structural invariants (the
// singleton/pair split, bijectivity, permutation closure) rather than just
// spot-checking a couple of indices.

#include "../src/core/powerset.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

using namespace core_powerset;

TEST_CASE("powerset: classes are ordered singletons-then-pairs", "[unit][powerset][diarize]") {
    // This is the property that was violated: class 3 must be a LONE speaker
    // (spk2), and class 4 must be the first PAIR. Getting these two backwards
    // is exactly the shipped bug.
    REQUIRE(members(0) == 0x0); // silence
    REQUIRE(members(1) == 0x1); // {spk0}
    REQUIRE(members(2) == 0x2); // {spk1}
    REQUIRE(members(3) == 0x4); // {spk2}      <-- NOT {spk0,spk1}
    REQUIRE(members(4) == 0x3); // {spk0,spk1} <-- NOT {spk2}
    REQUIRE(members(5) == 0x5); // {spk0,spk2}
    REQUIRE(members(6) == 0x6); // {spk1,spk2}

    auto popcount = [](uint8_t m) { return (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1); };
    for (int c = 1; c <= 3; c++)
        REQUIRE(popcount(members(c)) == 1);
    for (int c = 4; c <= 6; c++)
        REQUIRE(popcount(members(c)) == 2);
}

TEST_CASE("powerset: class_of inverts members", "[unit][powerset][diarize]") {
    for (int c = 0; c < kClasses; c++)
        REQUIRE(class_of(members(c)) == c);
    // Every representable subset (all but "all three at once") is hit exactly
    // once — so no two classes collide and none is unreachable.
    std::vector<int> seen;
    for (uint8_t m = 0; m < 8; m++)
        if (m != 0x7)
            seen.push_back(class_of(m));
    std::sort(seen.begin(), seen.end());
    REQUIRE(seen == std::vector<int>{0, 1, 2, 3, 4, 5, 6});
    REQUIRE(class_of(0x7) == -1); // 3 simultaneous speakers is not in the head
}

TEST_CASE("powerset: covers() credits BOTH speakers of an overlap class", "[unit][powerset][diarize]") {
    // The #107 fix: an overlap class is activity for each speaker it names.
    REQUIRE(covers(4, 0));
    REQUIRE(covers(4, 1));
    REQUIRE_FALSE(covers(4, 2));

    // spk2's classes are 3 (alone), 5 (with spk0), 6 (with spk1) — and
    // critically NOT class 4.
    REQUIRE(covers(3, 2));
    REQUIRE(covers(5, 2));
    REQUIRE(covers(6, 2));
    REQUIRE_FALSE(covers(4, 2));

    // Silence credits nobody.
    for (int s = 0; s < kSpeakers; s++)
        REQUIRE_FALSE(covers(0, s));

    // Each speaker is covered by exactly 3 classes: alone, and paired with
    // each of the other two.
    for (int s = 0; s < kSpeakers; s++) {
        int n = 0;
        for (int c = 0; c < kClasses; c++)
            n += covers(c, s) ? 1 : 0;
        REQUIRE(n == 3);
    }
}

TEST_CASE("powerset: permute_class is a bijection that fixes silence", "[unit][powerset][diarize]") {
    for (int pi = 0; pi < 6; pi++) {
        const int* p = kPerms[pi];
        std::vector<int> image;
        for (int c = 0; c < kClasses; c++) {
            const int nc = permute_class(c, p);
            REQUIRE(nc >= 0);
            REQUIRE(nc < kClasses);
            image.push_back(nc);
        }
        std::sort(image.begin(), image.end());
        REQUIRE(image == std::vector<int>{0, 1, 2, 3, 4, 5, 6}); // a permutation
        REQUIRE(permute_class(0, p) == 0);                       // silence is silence
        // Relabelling never changes HOW MANY speakers are active, only which.
        for (int c = 0; c < kClasses; c++) {
            auto pc = [](uint8_t m) { return (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1); };
            REQUIRE(pc(members(permute_class(c, p))) == pc(members(c)));
        }
    }
}

TEST_CASE("powerset: identity permutation is first and is a no-op", "[unit][powerset][diarize]") {
    // The chunk stitcher relies on this: on a tie it keeps the existing
    // numbering instead of shuffling labels for no reason.
    REQUIRE(kPerms[0][0] == 0);
    REQUIRE(kPerms[0][1] == 1);
    REQUIRE(kPerms[0][2] == 2);
    for (int c = 0; c < kClasses; c++)
        REQUIRE(permute_class(c, kPerms[0]) == c);
}

TEST_CASE("powerset: swapping two speakers swaps exactly their classes", "[unit][powerset][diarize]") {
    const int swap01[3] = {1, 0, 2};
    REQUIRE(permute_class(1, swap01) == 2); // {spk0} -> {spk1}
    REQUIRE(permute_class(2, swap01) == 1); // {spk1} -> {spk0}
    REQUIRE(permute_class(3, swap01) == 3); // {spk2} untouched
    REQUIRE(permute_class(4, swap01) == 4); // {spk0,spk1} maps to itself
    REQUIRE(permute_class(5, swap01) == 6); // {spk0,spk2} -> {spk1,spk2}
    REQUIRE(permute_class(6, swap01) == 5);
}

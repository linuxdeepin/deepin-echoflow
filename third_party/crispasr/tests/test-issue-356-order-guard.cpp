// test-issue-356-order-guard.cpp — issue #356 time-order guard.
//
// #356 reached a user as a subtitle-rendering complaint because nothing
// between the producer (the #89 gap-fill second pass) and the SRT/VTT writers
// ever checked that the transcript came out in time order: merge_segments
// concatenated slice results with no check, and the core_seg_hygiene pass is
// opt-in and documented as "never reorders". crispasr_first_backward_segment
// is that missing check.
//
// The subtlety it has to get right is WHICH timestamp orders the output:
// crispasr_make_disp_segments builds every cue from the WORD timestamps of a
// segment that carries words and only falls back to seg.t0/seg.t1 for a
// text-only segment. A guard written against segment spans would have passed
// the exact list that shipped the bug.
//
// Pure CPU, no model load.

#include "crispasr_output.h"

#include "core/asr_time_order.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

namespace {

crispasr_word make_word(const char* text, int64_t t0_cs, int64_t t1_cs) {
    crispasr_word w;
    w.text = text;
    w.t0 = t0_cs;
    w.t1 = t1_cs;
    return w;
}

crispasr_segment with_words(std::vector<crispasr_word> words) {
    crispasr_segment seg;
    seg.words = std::move(words);
    seg.t0 = seg.words.front().t0;
    seg.t1 = seg.words.back().t1;
    for (const auto& w : seg.words)
        seg.text += w.text;
    return seg;
}

crispasr_segment text_only(const char* text, int64_t t0_cs, int64_t t1_cs) {
    crispasr_segment seg;
    seg.text = text;
    seg.t0 = t0_cs;
    seg.t1 = t1_cs;
    return seg;
}

} // namespace

TEST_CASE("issue #356: an in-order transcript is not flagged", "[unit][order-guard][issue-356]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(with_words({make_word("a", 100, 150), make_word("b", 150, 200)}));
    segs.push_back(with_words({make_word("c", 300, 400)}));
    segs.push_back(text_only("no words here", 400, 500));
    REQUIRE(crispasr_first_backward_segment(segs) == -1);
}

TEST_CASE("issue #356: cues that merely overlap at the seam are not flagged", "[unit][order-guard][issue-356]") {
    // A refill window re-hears the audio either side of a hole, so a recovered
    // word can END up to kEdgePadCs (0.2 s) after the next word STARTS. That is
    // word-timestamp jitter, not a reordering, and it must stay quiet or the
    // warning is noise on every gap-filled file.
    std::vector<crispasr_segment> segs;
    segs.push_back(with_words({make_word("beta", 500, 620)}));
    segs.push_back(with_words({make_word("omega", 600, 900)}));
    REQUIRE(crispasr_first_backward_segment(segs) == -1);
}

TEST_CASE("issue #356: the reported backward jump is flagged", "[unit][order-guard][issue-356]") {
    // The reporter's excerpt: a covering segment 32.66..43.30 followed by the
    // recovery that was lifted out of its middle.
    std::vector<crispasr_segment> segs;
    segs.push_back(with_words({make_word("まずね", 3266, 3310), make_word("暮らすのは好き", 3946, 4330)}));
    segs.push_back(with_words({make_word("日本にもそんないない", 3318, 3454)}));

    int64_t prev = 0, cur = 0;
    REQUIRE(crispasr_first_backward_segment(segs, &prev, &cur) == 1);
    REQUIRE(prev == 3946);
    REQUIRE(cur == 3318);
}

TEST_CASE("issue #356: the guard reads words, not segment spans", "[unit][order-guard][issue-356]") {
    // Exactly the list the clamp fallback used to produce: the segment SPANS
    // are monotone (100..500 then 500..620) while the word stream still reads
    // 100, 600, 500. A span-based guard passes this; the writers do not.
    std::vector<crispasr_segment> segs;
    auto covering = with_words({make_word("alpha", 100, 150), make_word("omega", 600, 900)});
    covering.t1 = 500; // clamped
    segs.push_back(std::move(covering));
    segs.push_back(with_words({make_word("beta", 500, 620)}));

    REQUIRE(segs[1].t0 >= segs[0].t1); // spans look fine
    REQUIRE(crispasr_first_backward_segment(segs) == 1);
}

TEST_CASE("issue #356: out-of-order text-only segments are flagged too", "[unit][order-guard][issue-356]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(text_only("second", 900, 1000));
    segs.push_back(text_only("first", 100, 200));
    REQUIRE(crispasr_first_backward_segment(segs) == 1);
}

TEST_CASE("issue #356: empty segments are skipped, not treated as position 0", "[unit][order-guard][issue-356]") {
    // The writers drop a text-less segment, so it must not reset the cursor and
    // fake a backward jump on the segment after it.
    std::vector<crispasr_segment> segs;
    segs.push_back(with_words({make_word("a", 500, 600)}));
    segs.push_back(text_only("", 0, 0));
    segs.push_back(with_words({make_word("b", 700, 800)}));
    REQUIRE(crispasr_first_backward_segment(segs) == -1);
}

// The session C ABI carries its own segment struct (crispasr_session_seg, a
// private type in src/crispasr_c_api.cpp) and reimplements every backend's
// transcribe inline, so it cannot call the CLI's copy of anything — which is
// exactly why apply_session_hygiene exists. The predicate is shared through
// src/core/asr_time_order.h instead of being duplicated; this pins that it
// still compiles and behaves on a session-shaped segment, so the two surfaces
// cannot drift the way #308's punctuation fix did.
namespace {

struct SessionLikeWord {
    std::string text;
    int64_t t0 = 0;
    int64_t t1 = 0;
};

struct SessionLikeSeg {
    std::string text;
    int64_t t0 = 0;
    int64_t t1 = 0;
    std::vector<SessionLikeWord> words;
};

} // namespace

TEST_CASE("issue #356: the guard is shared with the session ABI's segment type", "[unit][order-guard][issue-356]") {
    std::vector<SessionLikeSeg> segs;
    segs.push_back({"in order", 100, 300, {{"a", 100, 150}, {"b", 200, 300}}});
    segs.push_back({"still fine", 400, 500, {{"c", 400, 500}}});
    REQUIRE(core_time_order::first_backward(segs) == -1);

    segs.push_back({"backwards", 200, 250, {{"d", 200, 250}}});
    int64_t prev = 0, cur = 0;
    REQUIRE(core_time_order::first_backward(segs, &prev, &cur) == 2);
    REQUIRE(prev == 400);
    REQUIRE(cur == 200);

    // And the CLI type resolves to the same template, not a second copy.
    std::vector<crispasr_segment> cli;
    cli.push_back(with_words({make_word("a", 400, 500)}));
    cli.push_back(with_words({make_word("b", 200, 250)}));
    REQUIRE(core_time_order::first_backward(cli) == crispasr_first_backward_segment(cli));
    REQUIRE(crispasr_first_backward_segment(cli) == 1);
}

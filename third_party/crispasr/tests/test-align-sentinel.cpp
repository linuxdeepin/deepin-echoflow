// test-align-sentinel.cpp — forced-alignment collapse detector (PLAN.md §W3).
//
// The fixtures are not invented shapes: they are what `ctc_forced_align()`
// (src/align.cpp:285) actually emits on its two silent-zero paths. A word whose
// characters are absent from the CTC vocab, or one the Viterbi path never
// visits, comes back as t0 == t1 == 0 inside an otherwise successful return.
//
// A detector is only worth anything if it discriminates, so every "fires" case
// below has a near-identical healthy twin that must NOT fire. If you loosen a
// threshold, the twin is what stops you loosening it into uselessness.

#include <catch2/catch_test_macros.hpp>

#include "core/align_sentinel.h"

#include <string>
#include <vector>

using core_align_sentinel::assess;
using core_align_sentinel::Thresholds;
using core_align_sentinel::Word;

namespace {

// A healthy alignment: 10 words spread evenly over 5 s of audio.
std::vector<Word> healthy() {
    const char* text[] = {"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "today"};
    std::vector<Word> w;
    for (int i = 0; i < 10; i++)
        w.push_back({text[i], 0.1f + 0.45f * (float)i, 0.1f + 0.45f * (float)i + 0.35f});
    return w;
}

} // namespace

TEST_CASE("a healthy alignment is not flagged", "[align-sentinel]") {
    const auto a = assess(healthy(), 5.0f);
    INFO(core_align_sentinel::describe(a));
    REQUIRE_FALSE(a.collapsed);
    REQUIRE(a.reasons.empty());
    REQUIRE(a.zero_position_words == 0);
    REQUIRE(a.degenerate_words == 0);
    REQUIRE(a.coverage > 0.8f);
}

TEST_CASE("out-of-vocab words at (0,0) are caught", "[align-sentinel]") {
    // align.h's documented case: "Words whose characters are entirely absent
    // from the CTC vocab get t0 == t1 == 0." A Chinese transcript against a
    // Latin-vocab CTC model returns EVERY word this way, rc=success.
    std::vector<Word> w;
    for (const char* s : {"今日", "は", "いい", "天気", "です", "ね"})
        w.push_back({s, 0.0f, 0.0f});

    const auto a = assess(w, 5.0f);
    INFO(core_align_sentinel::describe(a));
    REQUIRE(a.collapsed);
    REQUIRE(a.zero_position_words == 6);
    REQUIRE_FALSE(a.reasons.empty());
}

TEST_CASE("a partial out-of-vocab run is caught, a single leading word is not", "[align-sentinel]") {
    // The realistic mixed case: a few words fall out of vocab, the rest align.
    // 3 of 10 at (0,0) is 30% — well over the 10% bar.
    {
        std::vector<Word> w = healthy();
        for (int i : {2, 5, 7})
            w[(size_t)i] = {w[(size_t)i].text, 0.0f, 0.0f};
        const auto a = assess(w, 5.0f);
        INFO(core_align_sentinel::describe(a));
        REQUIRE(a.collapsed);
        REQUIRE(a.zero_position_words == 3);
    }
    // The discriminating twin: ONE word at (0,0) out of ten is 10%, not over
    // the bar. A leading word legitimately starting at 0.0 must not condemn an
    // otherwise good alignment.
    {
        std::vector<Word> w = healthy();
        w[0] = {w[0].text, 0.0f, 0.0f};
        const auto a = assess(w, 5.0f);
        INFO(core_align_sentinel::describe(a));
        REQUIRE_FALSE(a.collapsed);
    }
}

TEST_CASE("the ~100ms pile-up is caught by span and chars/sec", "[align-sentinel]") {
    // Every word crammed into a 100 ms window of a 30 s clip — the canonical
    // collapse. Two independent signals should fire, not one.
    std::vector<Word> w;
    const char* text[] = {"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "today"};
    for (int i = 0; i < 10; i++)
        w.push_back({text[i], 1.00f + 0.001f * (float)i, 1.01f + 0.001f * (float)i});

    const auto a = assess(w, 30.0f);
    INFO(core_align_sentinel::describe(a));
    REQUIRE(a.collapsed);
    REQUIRE(a.reasons.size() >= 2);
    REQUIRE(a.chars_per_sec > 50.0f);
}

TEST_CASE("zero-length spans are caught", "[align-sentinel]") {
    // Positioned but durationless: t1 == t0. Distinct from the (0,0) case and
    // it needs its own signal — these words carry a plausible start time.
    std::vector<Word> w = healthy();
    for (size_t i = 0; i < w.size(); i++)
        if (i % 2 == 0)
            w[i].t1 = w[i].t0; // 5 of 10 = 50% > 40%
    const auto a = assess(w, 5.0f);
    INFO(core_align_sentinel::describe(a));
    REQUIRE(a.collapsed);
    REQUIRE(a.degenerate_words == 5);
}

TEST_CASE("short clips and sparse speech are NOT false-positived", "[align-sentinel]") {
    // The three ways a naive port of these thresholds corrupts good output.

    // (a) A genuinely short clip: 3 words in 0.4 s of 0.5 s audio. The span is
    // under min_span_sec but the AUDIO is too, so the short-span signal must
    // stand down.
    {
        std::vector<Word> w = {{"yes", 0.05f, 0.15f}, {"I", 0.16f, 0.22f}, {"do", 0.23f, 0.40f}};
        const auto a = assess(w, 0.5f);
        INFO(core_align_sentinel::describe(a));
        REQUIRE_FALSE(a.collapsed);
    }
    // (b) Very little text: 2 short words. Below min_chars_to_assess, so the
    // rate-based signals must not fire even though coverage is tiny.
    {
        std::vector<Word> w = {{"ok", 0.10f, 0.20f}, {"no", 0.21f, 0.30f}};
        const auto a = assess(w, 60.0f);
        INFO(core_align_sentinel::describe(a));
        REQUIRE_FALSE(a.collapsed);
    }
    // (c) Unknown audio duration: coverage and short-span are unmeasurable and
    // must be skipped rather than computed against 0.
    {
        const auto a = assess(healthy(), -1.0f);
        INFO(core_align_sentinel::describe(a));
        REQUIRE_FALSE(a.collapsed);
    }
}

TEST_CASE("chars/sec counts code points, not UTF-8 bytes", "[align-sentinel]") {
    // A byte count reads 3x high on CJK and would condemn every correct
    // Japanese alignment. 12 kana over 3 s is 4 chars/s — fine; the same text
    // measured in bytes is 36 bytes -> 12/s, still fine, but scale it up and
    // the two verdicts diverge. Pin the code-point count directly.
    std::vector<Word> w;
    for (const char* s : {"きょうは", "いいてんき", "ですね"}) // 4 + 5 + 3 = 12 code points
        w.push_back({s, 0.0f, 0.0f});
    // Give them real positions spread over 3 s.
    w[0] = {w[0].text, 0.0f, 1.0f};
    w[1] = {w[1].text, 1.0f, 2.0f};
    w[2] = {w[2].text, 2.0f, 3.0f};

    const auto a = assess(w, 3.0f);
    INFO(core_align_sentinel::describe(a));
    REQUIRE(a.char_count == 12);
    REQUIRE_FALSE(a.collapsed);
}

TEST_CASE("thresholds are configurable and the gate can be driven red", "[align-sentinel]") {
    // Proof the detector is actually consulting its thresholds rather than
    // hard-coding a verdict: the healthy fixture flips to collapsed under an
    // absurd bar, and back to clean under a permissive one.
    Thresholds strict;
    strict.max_chars_per_sec = 1.0f; // no speech is this slow
    const auto red = assess(healthy(), 5.0f, strict);
    REQUIRE(red.collapsed);

    Thresholds loose;
    loose.max_zero_position_ratio = 1.0f;
    loose.max_degenerate_ratio = 1.0f;
    loose.min_coverage = 0.0f;
    loose.min_span_sec = 0.0f;
    std::vector<Word> allzero(6, Word{"word", 0.0f, 0.0f});
    const auto green = assess(allzero, 5.0f, loose);
    REQUIRE_FALSE(green.collapsed);
}

TEST_CASE("redistribute spreads words proportionally and pins the end", "[align-sentinel]") {
    std::vector<Word> collapsed(4, Word{"", 0.0f, 0.0f});
    collapsed[0].text = "a";    // 1
    collapsed[1].text = "bb";   // 2
    collapsed[2].text = "cccc"; // 4
    collapsed[3].text = "ddd";  // 3  -> total 10

    const auto out = core_align_sentinel::redistribute(collapsed, 2.0f, 12.0f);
    REQUIRE(out.size() == 4);
    REQUIRE(out[0].t0 == 2.0f);
    REQUIRE(out[3].t1 == 12.0f);
    // Monotonic, contiguous, and proportional to character count.
    for (size_t i = 1; i < out.size(); i++)
        REQUIRE(out[i].t0 >= out[i - 1].t1 - 1e-4f);
    const float d0 = out[0].t1 - out[0].t0;
    const float d2 = out[2].t1 - out[2].t0;
    REQUIRE(d2 > 3.5f * d0); // "cccc" is 4x "a"
    REQUIRE(d2 < 4.5f * d0);

    // An empty word list and an inverted range are no-ops, not crashes.
    REQUIRE(core_align_sentinel::redistribute({}, 0.0f, 1.0f).empty());
    const auto noop = core_align_sentinel::redistribute(collapsed, 5.0f, 5.0f);
    REQUIRE(noop[0].t0 == 0.0f); // unchanged
}

TEST_CASE("an empty word list is safe", "[align-sentinel]") {
    const auto a = assess({}, 10.0f);
    REQUIRE_FALSE(a.collapsed);
    REQUIRE(a.word_count == 0);
    REQUIRE(core_align_sentinel::describe(a).empty());
}

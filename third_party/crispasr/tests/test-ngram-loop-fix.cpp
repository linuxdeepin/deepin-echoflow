// test-ngram-loop-fix.cpp — Catch2 unit tests for core_ngram::fix_loops, the
// shared greedy-n-gram-loop collapse used by the higgs-audio-v3-stt and
// moss-transcribe backends (issue #218).
//
// Pure text transform — no model load. The fixtures are the exact degenerate
// outputs observed on the issue #218 sample (t32-145s.wav) plus invariants
// (clean text untouched, empty input safe).

#include <catch2/catch_test_macros.hpp>

#include "core/ngram_loop_fix.h"

#include <string>
#include <vector>

using core_ngram::fix_loops;

TEST_CASE("clean transcripts pass through byte-identical", "[ngram-loop]") {
    // The transform must be a no-op on non-degenerate text so it never harms a
    // good slice.
    const std::string clean = "Fast! This is some sort of a test.";
    REQUIRE(fix_loops(clean) == clean);

    const std::string clean2 = "Come on! Don't move much of the fence.";
    REQUIRE(fix_loops(clean2) == clean2);

    // A legitimate short repeat (2 reps of a bigram, 3 of a unigram) is under
    // the collapse threshold and survives.
    const std::string ok_rep = "no no no more games";
    REQUIRE(fix_loops(ok_rep) == ok_rep);
}

TEST_CASE("degenerate unigram loop collapses to a bounded run", "[ngram-loop]") {
    // issue #218 slice 2: "Hey," repeated ~490× until the token cap.
    std::string loop = "just get him moving.";
    for (int i = 0; i < 40; i++)
        loop += " hey,";
    const std::string out = fix_loops(loop);
    // The good prefix survives; the "hey," run is trimmed to at most a few reps.
    REQUIRE(out.find("just get him moving.") == 0);
    // Count surviving "hey," tokens — must be a small bounded number, not 40.
    size_t count = 0;
    for (size_t p = out.find("hey,"); p != std::string::npos; p = out.find("hey,", p + 1))
        count++;
    REQUIRE(count <= 3);
    REQUIRE(out.size() < loop.size());
}

TEST_CASE("long single-word chunk loop collapses after larger n-grams", "[ngram-loop]") {
    std::string loop = "just get him moving";
    for (int i = 0; i < 80; i++)
        loop += " back";
    loop += " It is a joke.";
    const std::string out = fix_loops(loop);
    REQUIRE(out.find("just get him moving") == 0);
    REQUIRE(out.find("It is a joke.") != std::string::npos);
    REQUIRE(out.find("back back back back") == std::string::npos);
    REQUIRE(out.size() < loop.size());
}

TEST_CASE("degenerate multi-word cycle collapses", "[ngram-loop]") {
    // issue #218 slice 5: "run hey hey hey hey hey run" cycles.
    const std::string loop =
        "run hey hey hey hey hey run hey hey hey hey hey run hey hey hey hey hey run Alex you okay?";
    const std::string out = fix_loops(loop);
    REQUIRE(out.size() < loop.size());
    REQUIRE(out.find("Alex you okay?") != std::string::npos);
    // No five-in-a-row "hey" survives the unigram collapse.
    REQUIRE(out.find("hey hey hey hey hey") == std::string::npos);
}

TEST_CASE("degenerate phrase repetition collapses", "[ngram-loop]") {
    // issue #218 follow-ups: multiple AR ASR backends can loop on short
    // phrases such as "come on" after a valid prefix.
    const std::string loop = "move much of the fence. come on come on come on come on come on come on come on come on";
    const std::string out = fix_loops(loop);
    REQUIRE(out.find("move much of the fence.") == 0);
    REQUIRE(out.find("come on come on come on come on") == std::string::npos);
    REQUIRE(out.size() < loop.size());
}

TEST_CASE("edge cases are safe", "[ngram-loop]") {
    REQUIRE(fix_loops("") == "");
    REQUIRE(fix_loops("word") == "word");
    // Whitespace is normalised to single spaces on rejoin.
    REQUIRE(fix_loops("  a   b  ") == "a b");
}

// issue #218 follow-up (AppleSheeple, 2026-07-09): fix_loops() cleans the
// flat `seg.text`, but backends build `seg.words`/tokens independently from
// the raw (un-collapsed) token stream, so word-level output (SRT/VTT, JSON
// `words`) still shows the duplicates even after the visible text looks
// clean. fix_loops_keep_indices() lets callers filter that parallel array
// in lockstep with the text collapse.
TEST_CASE("fix_loops_keep_indices matches fix_loops's own collapse decisions", "[ngram-loop][word-level]") {
    using core_ngram::fix_loops_keep_indices;
    using core_ngram::split_words;

    const std::string loop = "move much of the fence. come on come on come on come on come on come on come on come on";
    const std::vector<std::string> words = split_words(loop);
    const std::vector<int> keep = fix_loops_keep_indices(words);

    // The kept-index subsequence, rejoined, must equal fix_loops()'s own
    // output on the same text — no separate/inconsistent decision.
    std::string rebuilt;
    for (size_t k = 0; k < keep.size(); k++) {
        if (k)
            rebuilt += ' ';
        rebuilt += words[keep[k]];
    }
    REQUIRE(rebuilt == fix_loops(loop));

    // Indices are strictly ascending (a valid subsequence, not a reordering).
    for (size_t k = 1; k < keep.size(); k++)
        REQUIRE(keep[k] > keep[k - 1]);

    // Drastically fewer words survive than the degenerate input had.
    REQUIRE(keep.size() < words.size());
}

TEST_CASE("fix_loops_keep_indices filters a parallel per-word array", "[ngram-loop][word-level]") {
    // Simulates seg.words: one entry per word, built from the SAME raw
    // (un-collapsed) token stream as the degenerate text. This is the
    // exact fixture used for the phrase-repetition text-level test above,
    // to prove filtering seg.words by the returned indices reduces it to
    // the same word count as the collapsed text.
    using core_ngram::fix_loops_keep_indices;
    using core_ngram::split_words;

    const std::string loop =
        "run hey hey hey hey hey run hey hey hey hey hey run hey hey hey hey hey run Alex you okay?";
    const std::vector<std::string> words = split_words(loop);

    struct FakeWord {
        std::string text;
        int t0;
    };
    std::vector<FakeWord> seg_words;
    for (size_t i = 0; i < words.size(); i++)
        seg_words.push_back({words[i], (int)i});

    const std::vector<int> keep = fix_loops_keep_indices(words);
    std::vector<FakeWord> filtered;
    for (int idx : keep)
        filtered.push_back(seg_words[idx]);

    REQUIRE(filtered.size() == keep.size());
    REQUIRE(filtered.size() < seg_words.size());
    // "Alex you okay?" (the last 3 words) must survive untouched, in order,
    // with their original timestamps intact (t0 == original index).
    REQUIRE(filtered.size() >= 3);
    REQUIRE(filtered[filtered.size() - 3].text == "Alex");
    REQUIRE(filtered[filtered.size() - 2].text == "you");
    REQUIRE(filtered[filtered.size() - 1].text == "okay?");
    REQUIRE(filtered[filtered.size() - 1].t0 == (int)words.size() - 1);
}

// test-segment-hygiene.cpp — §W2 length cap, §W5 repeat merge, §W6 filters.
//
// This is harness-blind territory: everything here happens downstream of the
// logits, so crispasr-diff reports cos 1.000000 whether it works or not. These
// tests are the only place the behaviour is checked.
//
// Each transform is OFF by default, so the first thing every section asserts is
// that a default-constructed config changes nothing.

#include <catch2/catch_test_macros.hpp>

#include "core/segment_hygiene.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "portable_env.h"

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

using namespace core_seg_hygiene;

namespace {

std::string rep(const std::string& u, int n) {
    std::string s;
    for (int i = 0; i < n; i++)
        s += u;
    return s;
}

Seg seg(const std::string& text, int64_t t0, int64_t t1) {
    Seg s;
    s.text = text;
    s.t0 = t0;
    s.t1 = t1;
    return s;
}

} // namespace

// ---------------------------------------------------------------------------
// §W2 length cap
// ---------------------------------------------------------------------------

TEST_CASE("length cap is off by default", "[seg-hygiene]") {
    const std::string huge = rep("あ", 500);
    REQUIRE(cap_length(huge, LengthCapConfig{}) == huge);
}

TEST_CASE("an over-long line is cut at a sentence boundary", "[seg-hygiene]") {
    LengthCapConfig cfg;
    cfg.max_codepoints = 40;
    // The boundary has to sit at or past the keep floor (0.75 * 40 = 30 code
    // points) for the back-up to be accepted — below that the hard cut wins,
    // which is the next test. 35 real chars ending in 。 then 60 of garbage.
    const std::string good = "今日はいい天気ですね散歩に行きましょうか楽しみにしていますよ。"; // 30 cp
    REQUIRE(core_seg_hygiene::detail::count_codepoints(good) >= 30);
    const std::string line = good + rep("あ", 60);
    const std::string out = cap_length(line, cfg);

    REQUIRE(out == good); // backed up to the 。
    REQUIRE(out.size() < line.size());
}

TEST_CASE("the cut never falls below the keep floor", "[seg-hygiene]") {
    // A "。" at code point 3 is well under 75% of a 40-cap, so backing up to it
    // would throw away a legitimate line. The hard cut must win instead.
    LengthCapConfig cfg;
    cfg.max_codepoints = 40;
    const std::string line = "はい。" + rep("う", 100);
    const std::string out = cap_length(line, cfg);

    REQUIRE(out != "はい。");
    REQUIRE(core_seg_hygiene::detail::count_codepoints(out) == 40);
}

TEST_CASE("a line within the cap is untouched", "[seg-hygiene]") {
    LengthCapConfig cfg;
    cfg.max_codepoints = 40;
    const std::string ok = "今日はいい天気ですね。";
    REQUIRE(cap_length(ok, cfg) == ok);
    // Exactly at the cap is within it.
    const std::string exact = rep("あ", 40);
    REQUIRE(cap_length(exact, cfg) == exact);
}

TEST_CASE("the cap counts code points, not bytes", "[seg-hygiene]") {
    // 20 kana = 60 UTF-8 bytes. A byte-based cap of 40 would cut this correct
    // line in half — and would also cut mid-character, producing mojibake.
    LengthCapConfig cfg;
    cfg.max_codepoints = 40;
    const std::string line = rep("あ", 20);
    REQUIRE(cap_length(line, cfg) == line);
}

// ---------------------------------------------------------------------------
// §W6 filters
// ---------------------------------------------------------------------------

TEST_CASE("filters are off by default", "[seg-hygiene]") {
    Seg s = seg("[Music]", 0, 100);
    s.avg_logprob = -99.0f;
    s.has_logprob = true;
    REQUIRE(should_drop(s, 1.0, FilterConfig{}) == DropReason::None);
}

TEST_CASE("low-logprob segments drop, and short ones get a tighter bar", "[seg-hygiene]") {
    FilterConfig cfg;
    cfg.enabled = true;
    cfg.use_logprob = true;
    cfg.logprob_threshold = -1.0f;
    cfg.short_segment_margin = 0.5f; // short segments must beat -1.5
    cfg.short_segment_sec = 1.6;

    Seg s = seg("hello", 0, 300);
    s.has_logprob = true;

    s.avg_logprob = -0.5f;
    REQUIRE(should_drop(s, 3.0, cfg) == DropReason::None);
    s.avg_logprob = -1.2f;
    REQUIRE(should_drop(s, 3.0, cfg) == DropReason::LowLogprob);

    // The SAME -1.2 on a short segment survives, because the threshold moved
    // DOWN to -1.5: a short segment's mean logprob is noisier, so it gets more
    // room, not less.
    REQUIRE(should_drop(s, 1.0, cfg) == DropReason::None);
    s.avg_logprob = -1.9f;
    REQUIRE(should_drop(s, 1.0, cfg) == DropReason::LowLogprob);

    // Unknown duration => no margin applied.
    s.avg_logprob = -1.2f;
    REQUIRE(should_drop(s, -1.0, cfg) == DropReason::LowLogprob);
}

TEST_CASE("a segment with no logprob is never dropped for logprob", "[seg-hygiene]") {
    FilterConfig cfg;
    cfg.enabled = true;
    cfg.use_logprob = true;
    cfg.logprob_threshold = -1.0f;
    Seg s = seg("hello", 0, 300); // has_logprob stays false
    REQUIRE(should_drop(s, 3.0, cfg) == DropReason::None);
}

TEST_CASE("bracketed non-verbal markers are recognised", "[seg-hygiene]") {
    REQUIRE(looks_nonverbal("[Music]"));
    REQUIRE(looks_nonverbal("(applause)"));
    REQUIRE(looks_nonverbal("[LAUGHTER]"));
    REQUIRE(looks_nonverbal("(moaning)"));
    REQUIRE(looks_nonverbal("（喘ぎ声）"));
    REQUIRE(looks_nonverbal("♪"));
    REQUIRE(looks_nonverbal("♪♪♪"));
    REQUIRE(looks_nonverbal("  ♪ ♪  "));
}

TEST_CASE("running speech containing a marker word is NOT dropped", "[seg-hygiene]") {
    // The load-bearing difference from WhisperJAV, which substring-matches its
    // keyword list against the whole line and so deletes any sentence
    // containing "music", "laugh", "breath"...
    REQUIRE_FALSE(looks_nonverbal("The music started and everyone danced."));
    REQUIRE_FALSE(looks_nonverbal("I could hear applause from the hall."));
    REQUIRE_FALSE(looks_nonverbal("Take a deep breath and relax."));
    REQUIRE_FALSE(looks_nonverbal("She began to laugh."));
    REQUIRE_FALSE(looks_nonverbal("音楽が好きです。"));
    // A note mid-sentence is lyric punctuation, not a music marker.
    REQUIRE_FALSE(looks_nonverbal("♪ and then she sang ♪ loudly"));
    REQUIRE_FALSE(looks_nonverbal(""));
    REQUIRE_FALSE(looks_nonverbal("   "));
}

TEST_CASE("a bracketed line that is real speech survives", "[seg-hygiene]") {
    // Brackets alone are not enough — the contents must match a marker.
    REQUIRE_FALSE(looks_nonverbal("(I think so)"));
    REQUIRE_FALSE(looks_nonverbal("[John] Hello there"));
}

// ---------------------------------------------------------------------------
// §W5 cross-segment merge
// ---------------------------------------------------------------------------

TEST_CASE("merge is off by default", "[seg-hygiene]") {
    std::vector<Seg> in = {seg("yes", 0, 100), seg("yes", 100, 200), seg("yes", 200, 300), seg("yes", 300, 400)};
    REQUIRE(merge_repeats(in, MergeConfig{}).size() == 4);
}

TEST_CASE("a run of identical adjacent segments collapses to one", "[seg-hygiene]") {
    MergeConfig cfg;
    cfg.enabled = true;
    std::vector<Seg> in;
    for (int i = 0; i < 8; i++)
        in.push_back(seg("はい、はい。", i * 100, i * 100 + 100));

    const auto out = merge_repeats(in, cfg);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0].text == "はい、はい。");
    REQUIRE(out[0].t0 == 0);
    REQUIRE(out[0].t1 == 800); // the run's full span is preserved
}

TEST_CASE("a short repeat is NOT merged", "[seg-hygiene]") {
    // Two identical lines in a row is ordinary speech, not a loop. min_run is
    // what separates them.
    MergeConfig cfg;
    cfg.enabled = true;
    std::vector<Seg> in = {seg("yes.", 0, 100), seg("yes.", 100, 200), seg("no.", 200, 300)};
    const auto out = merge_repeats(in, cfg);
    REQUIRE(out.size() == 3);
}

TEST_CASE("a wide time gap blocks the merge", "[seg-hygiene]") {
    // The same line three times, but a minute apart each: three real
    // utterances, not a decode loop.
    MergeConfig cfg;
    cfg.enabled = true;
    std::vector<Seg> in = {seg("thank you.", 0, 100), seg("thank you.", 6000, 6100), seg("thank you.", 12000, 12100)};
    const auto out = merge_repeats(in, cfg);
    REQUIRE(out.size() == 3);
}

TEST_CASE("distinct segments are left alone", "[seg-hygiene]") {
    MergeConfig cfg;
    cfg.enabled = true;
    std::vector<Seg> in = {seg("the first thing", 0, 100), seg("a second thing", 100, 200),
                           seg("something else entirely", 200, 300), seg("and now for more", 300, 400)};
    const auto out = merge_repeats(in, cfg);
    REQUIRE(out.size() == 4);
    for (size_t i = 0; i < out.size(); i++)
        REQUIRE(out[i].text == in[i].text);
}

TEST_CASE("a drifting chain cannot merge unboundedly", "[seg-hygiene]") {
    // Each line is similar to its neighbour but the first and last share
    // nothing. Comparing against the run's FIRST text (not the predecessor)
    // stops the run before it swallows unrelated content.
    MergeConfig cfg;
    cfg.enabled = true;
    cfg.similarity = 0.8;
    std::vector<Seg> in = {seg("aaaaaaaaaa", 0, 100), seg("aaaaaaaaab", 100, 200), seg("aaaaaaaabb", 200, 300),
                           seg("aaaaaaabbb", 300, 400), seg("bbbbbbbbbb", 400, 500)};
    const auto out = merge_repeats(in, cfg);
    REQUIRE(out.size() >= 2);
    // The last, wholly-different line must survive intact.
    REQUIRE(out.back().text == "bbbbbbbbbb");
}

TEST_CASE("similarity is measured in code points", "[seg-hygiene]") {
    // One kana differing out of five is 1 difference, not 3 (its byte length).
    // A byte-level ratio would score this pair lower and miss real duplicates.
    const double s = similarity("こんにちは", "こんにちわ");
    REQUIRE(s > 0.75);
    REQUIRE(s < 1.0);
    REQUIRE(similarity("abc", "abc") == 1.0);
    REQUIRE(similarity("", "abc") == 0.0);
    REQUIRE(similarity("  hello  ", "hello") == 1.0); // trimmed before compare
}

TEST_CASE("merge handles empty and single-segment input", "[seg-hygiene]") {
    MergeConfig cfg;
    cfg.enabled = true;
    REQUIRE(merge_repeats({}, cfg).empty());
    std::vector<Seg> one = {seg("hello", 0, 100)};
    REQUIRE(merge_repeats(one, cfg).size() == 1);
}

// ---------------------------------------------------------------------------
// Env configuration + the combined apply
// ---------------------------------------------------------------------------

namespace {

// RAII: set env vars for one test and restore the previous state after, so a
// leaked variable cannot silently change the verdict of a later test.
struct ScopedEnv {
    std::vector<std::pair<std::string, bool>> saved; // name, was-set
    std::vector<std::string> old_values;
    void set(const char* k, const char* v) {
        const char* prev = getenv(k);
        saved.push_back({k, prev != nullptr});
        old_values.push_back(prev ? prev : "");
        setenv(k, v, 1);
    }
    ~ScopedEnv() {
        for (size_t i = 0; i < saved.size(); i++) {
            if (saved[i].second)
                setenv(saved[i].first.c_str(), old_values[i].c_str(), 1);
            else
                unsetenv(saved[i].first.c_str());
        }
    }
};

} // namespace

TEST_CASE("with no env set, nothing is enabled and apply_all is identity", "[seg-hygiene]") {
    // The default posture. Every stage can delete user-visible text, so none of
    // them may switch on by surprise.
    const auto cfg = config_from_env();
    REQUIRE_FALSE(any_enabled(cfg));

    std::vector<Seg> in = {seg("[Music]", 0, 100), seg(rep("あ", 300), 100, 200), seg("yes", 200, 300),
                           seg("yes", 300, 400), seg("yes", 400, 500)};
    int dropped = -1;
    const auto out = apply_all(in, cfg, &dropped);
    REQUIRE(out.size() == in.size());
    REQUIRE(dropped == 0);
    for (size_t i = 0; i < out.size(); i++)
        REQUIRE(out[i].text == in[i].text);
}

TEST_CASE("each env knob enables exactly its own stage", "[seg-hygiene]") {
    {
        ScopedEnv e;
        e.set("CRISPASR_SEG_MAX_CHARS", "40");
        const auto c = config_from_env();
        REQUIRE(c.cap.max_codepoints == 40);
        REQUIRE_FALSE(c.filter.enabled);
        REQUIRE_FALSE(c.merge.enabled);
    }
    {
        ScopedEnv e;
        e.set("CRISPASR_SEG_DROP_NONVERBAL", "1");
        const auto c = config_from_env();
        REQUIRE(c.filter.enabled);
        REQUIRE(c.filter.drop_nonverbal);
        REQUIRE_FALSE(c.filter.use_logprob); // no threshold given
        REQUIRE(c.cap.max_codepoints == 0);
    }
    {
        ScopedEnv e;
        e.set("CRISPASR_SEG_LOGPROB_THOLD", "-0.8");
        e.set("CRISPASR_SEG_LOGPROB_MARGIN", "0.4");
        const auto c = config_from_env();
        REQUIRE(c.filter.enabled);
        REQUIRE(c.filter.use_logprob);
        REQUIRE(c.filter.logprob_threshold == -0.8f);
        REQUIRE(c.filter.short_segment_margin == 0.4f);
        REQUIRE_FALSE(c.filter.drop_nonverbal);
    }
    {
        ScopedEnv e;
        e.set("CRISPASR_SEG_MERGE_REPEATS", "1");
        e.set("CRISPASR_SEG_MERGE_MIN_RUN", "5");
        e.set("CRISPASR_SEG_MERGE_GAP_CS", "500");
        const auto c = config_from_env();
        REQUIRE(c.merge.enabled);
        REQUIRE(c.merge.min_run == 5);
        REQUIRE(c.merge.max_gap_cs == 500);
    }
}

TEST_CASE("a zero or unparseable env value does not enable a stage", "[seg-hygiene]") {
    ScopedEnv e;
    e.set("CRISPASR_SEG_MAX_CHARS", "0");
    e.set("CRISPASR_SEG_DROP_NONVERBAL", "0");
    e.set("CRISPASR_SEG_MERGE_REPEATS", "0");
    const auto c = config_from_env();
    REQUIRE_FALSE(any_enabled(c));
}

TEST_CASE("apply_all runs cap before merge, which is what makes merging work", "[seg-hygiene]") {
    // Two runaway lines that share a clean 30-char head and then diverge into
    // different garbage. Capping FIRST reduces both to the same head so they
    // merge; merging first would compare two different 300-char lines that
    // score far below the similarity bar and never collapse.
    Config cfg;
    cfg.cap.max_codepoints = 30;
    cfg.merge.enabled = true;
    cfg.merge.min_run = 3;
    cfg.merge.max_gap_cs = 200;

    const std::string head = rep("は", 30);
    std::vector<Seg> in = {seg(head + rep("あ", 200), 0, 100), seg(head + rep("い", 200), 100, 200),
                           seg(head + rep("う", 200), 200, 300)};

    const auto out = apply_all(in, cfg);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0].text == head);
    REQUIRE(out[0].t1 == 300);
}

TEST_CASE("apply_all reports how many segments it dropped", "[seg-hygiene]") {
    // Silent loss is the failure mode here: a user whose subtitles lost three
    // lines needs to be told, not left to notice.
    Config cfg;
    cfg.filter.enabled = true;
    cfg.filter.drop_nonverbal = true;

    std::vector<Seg> in = {seg("hello there", 0, 100), seg("[Music]", 100, 200), seg("(applause)", 200, 300),
                           seg("goodbye", 300, 400), seg("♪", 400, 500)};
    int dropped = 0;
    const auto out = apply_all(in, cfg, &dropped);
    REQUIRE(dropped == 3);
    REQUIRE(out.size() == 2);
    REQUIRE(out[0].text == "hello there");
    REQUIRE(out[1].text == "goodbye");
}

TEST_CASE("apply_all derives segment duration from the timestamps", "[seg-hygiene]") {
    // The short-segment margin only means anything if the duration reaching
    // should_drop() is real. t1-t0 is in centiseconds; 120 cs = 1.2 s, which is
    // inside the 1.6 s short window, so the tightened bar applies.
    Config cfg;
    cfg.filter.enabled = true;
    cfg.filter.use_logprob = true;
    cfg.filter.logprob_threshold = -1.0f;
    cfg.filter.short_segment_margin = 0.5f;

    Seg shortSeg = seg("um", 0, 120);
    shortSeg.avg_logprob = -1.2f;
    shortSeg.has_logprob = true;

    Seg longSeg = seg("a full sentence here", 200, 700); // 5.0 s
    longSeg.avg_logprob = -1.2f;
    longSeg.has_logprob = true;

    int dropped = 0;
    const auto out = apply_all({shortSeg, longSeg}, cfg, &dropped);
    // The SHORT one survives (-1.2 beats the loosened -1.5); the LONG one is
    // dropped (-1.2 fails the flat -1.0). If the duration were not being
    // computed, both would go the same way.
    REQUIRE(dropped == 1);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0].text == "um");
}

// ---------------------------------------------------------------------------
// Wiring — does the PRODUCT call any of this?
// ---------------------------------------------------------------------------

TEST_CASE("merge_segments actually calls the hygiene pass", "[seg-hygiene][wiring]") {
    // §W1b's lesson, applied before it can bite: firered-asr's loop collapse
    // was correct, unit-tested and NEVER CALLED for its entire life. A pure
    // predicate test cannot catch that — the predicate is fine, the join is
    // missing. So this asserts the CALL, in the one function all four
    // `merge_segments(...)` sites in crispasr_run.cpp pass through.
    //
    // If this goes red, the hygiene has gone inert: restore the call rather
    // than deleting the test.
    const std::string path = std::string(CRISPASR_SOURCE_DIR) + "/examples/cli/crispasr_run.cpp";
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();

    // The include alone proves nothing — that is exactly the state §W1b found.
    REQUIRE(src.find("core_seg_hygiene::config_from_env(") != std::string::npos);
    REQUIRE(src.find("core_seg_hygiene::apply_all(") != std::string::npos);

    // And it must sit inside merge_segments(), not merely somewhere in the file.
    const size_t fn = src.find("std::vector<crispasr_segment> merge_segments(");
    REQUIRE(fn != std::string::npos);
    const size_t call = src.find("core_seg_hygiene::apply_all(", fn);
    REQUIRE(call != std::string::npos);
    // Within a reasonable body length of the function head.
    REQUIRE(call - fn < 4000);
}

TEST_CASE("the session ABI has its own hygiene arm", "[seg-hygiene][wiring]") {
    // crispasr_c_api.cpp REIMPLEMENTS every backend's transcribe inline and
    // never calls the CLI adapter, so the merge_segments() wiring above reaches
    // nothing here — bindings and the server would silently miss all of §W2,
    // §W5 and §W6. This asserts the session arm exists and is actually invoked,
    // not merely defined.
    const std::string path = std::string(CRISPASR_SOURCE_DIR) + "/src/crispasr_c_api.cpp";
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();

    REQUIRE(src.find("core_seg_hygiene::apply_all(") != std::string::npos);
    // Defined AND called — a helper nobody invokes is the §W1b failure exactly.
    REQUIRE(src.find("static void apply_session_hygiene(") != std::string::npos);
    size_t calls = 0;
    for (size_t p = src.find("apply_session_hygiene("); p != std::string::npos;
         p = src.find("apply_session_hygiene(", p + 1))
        calls++;
    REQUIRE(calls >= 4); // 1 definition + >=3 call sites

    // The VAD path must DEFER the merge: it transcribes a stitched buffer where
    // every gap is a uniform 0.1 s join, so merging before the timestamps are
    // remapped would collapse utterances that are minutes apart.
    REQUIRE(src.find("hygiene_defer_merge") != std::string::npos);
}

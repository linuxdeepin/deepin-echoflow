// test-issue-356-gap-fill-overlap.cpp — issue #356 regression guard.
//
// The issue #89 gap-fill second pass recovers speech the parakeet-ja encoder
// blanks inside a slice, but it appended the recovered segments and sorted by
// t0 only. The first pass emits ONE segment whose sparse words straddle the
// hole, so its t0..t1 encloses every recovery — sorting cannot fix that, and
// the SRT/VTT output carried cues jumping backwards by up to the hole length
// (10.5 s on the downstream reporter's file; the 32.66→43.30 s excerpt below
// mirrors it). crispasr_gap_fill_resolve_overlaps now splits the covering
// segment at each recovery so the list comes out monotone with no text lost.
//
// Pure CPU, no model load: the backend is a scripted fake that "recovers"
// preset words for whatever window it is asked to transcribe.

#include "crispasr_gap_fill.h"
#include "whisper_params.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;

crispasr_word make_word(const char* text, int64_t t0_cs, int64_t t1_cs) {
    crispasr_word w;
    w.text = text;
    w.t0 = t0_cs;
    w.t1 = t1_cs;
    return w;
}

crispasr_segment make_segment(std::vector<crispasr_word> words) {
    crispasr_segment seg;
    seg.words = std::move(words);
    seg.text = crispasr_rebuild_text_from_words(seg.words);
    seg.t0 = seg.words.front().t0;
    seg.t1 = seg.words.back().t1;
    return seg;
}

// Backend whose "audio" contains a fixed script of absolute-timed words: a
// transcribe() call returns every scripted word that falls inside the
// requested window. This is exactly the contract gap-fill relies on when it
// re-transcribes a hole in isolation.
class ScriptedBackend : public CrispasrBackend {
public:
    explicit ScriptedBackend(std::vector<crispasr_word> script) : script_(std::move(script)) {}

    const char* name() const override { return "scripted"; }
    uint32_t capabilities() const override { return 0; }
    bool init(const whisper_params&) override { return true; }
    void shutdown() override {}

    std::vector<crispasr_segment> transcribe(const float*, int n_samples, int64_t t_offset_cs,
                                             const whisper_params&) override {
        const int64_t win0 = t_offset_cs;
        const int64_t win1 = t_offset_cs + (int64_t)n_samples * 100 / kSampleRate;
        crispasr_segment seg;
        for (const auto& w : script_) {
            if (w.t0 >= win0 && w.t1 <= win1) {
                seg.words.push_back(w);
                crispasr_token tk;
                tk.text = w.text;
                tk.t0 = w.t0;
                tk.t1 = w.t1;
                seg.tokens.push_back(tk);
            }
        }
        if (seg.words.empty())
            return {};
        seg.t0 = seg.words.front().t0;
        seg.t1 = seg.words.back().t1;
        seg.text = crispasr_rebuild_text_from_words(seg.words);
        return {seg};
    }

private:
    std::vector<crispasr_word> script_;
};

void require_monotone(const std::vector<crispasr_segment>& segs) {
    for (size_t i = 1; i < segs.size(); i++) {
        INFO("segment " << i << " [" << segs[i].t0 << ".." << segs[i].t1 << "] vs previous [" << segs[i - 1].t0 << ".."
                        << segs[i - 1].t1 << "]");
        REQUIRE(segs[i].t0 >= segs[i - 1].t1);
    }
}

// The property #356 is actually about. --output-srt / --output-vtt go through
// crispasr_make_disp_segments, which — for any segment that carries words —
// builds every cue from the WORD timestamps in segment order and never reads
// seg.t0/seg.t1. So a list whose segment spans are monotone can still emit
// backward-jumping cues; what has to be non-decreasing is the word stream read
// in segment order. require_monotone above is the weaker, structural check.
void require_words_monotone_in_reading_order(const std::vector<crispasr_segment>& segs) {
    int64_t prev_t0 = -1;
    std::string prev_text;
    for (size_t i = 0; i < segs.size(); i++) {
        for (const auto& w : segs[i].words) {
            INFO("segment " << i << ": word '" << w.text << "' at [" << w.t0 << ".." << w.t1 << "] follows '"
                            << prev_text << "' starting at " << prev_t0);
            REQUIRE(w.t0 >= prev_t0);
            prev_t0 = w.t0;
            prev_text = w.text;
        }
    }
}

// A segment's span must still contain the words whose text it carries.
void require_words_inside_their_cue(const std::vector<crispasr_segment>& segs) {
    for (size_t i = 0; i < segs.size(); i++)
        for (const auto& w : segs[i].words) {
            INFO("segment " << i << " [" << segs[i].t0 << ".." << segs[i].t1 << "] text='" << segs[i].text
                            << "' carries word '" << w.text << "' at [" << w.t0 << ".." << w.t1 << "]");
            REQUIRE(w.t0 <= segs[i].t1);
            REQUIRE(w.t1 >= segs[i].t0);
        }
}

std::string joined_text(const std::vector<crispasr_segment>& segs) {
    std::string all;
    for (const auto& s : segs)
        all += s.text;
    return all;
}

} // namespace

TEST_CASE("issue #356: recovery inside a straddling segment splits it, output monotone",
          "[unit][gap-fill][issue-356]") {
    // Mirrors the reporter's 32.66→43.30 s slice: the first pass emitted one
    // segment whose words cover the edges but skip 33.1→39.4 s; gap-fill
    // recovers the middle from the scripted backend.
    const crispasr_audio_slice sl = {3250 * kSampleRate / 100, 4330 * kSampleRate / 100, 3250, 4330};
    std::vector<float> samples((size_t)sl.end, 0.0f);

    std::vector<crispasr_segment> segs;
    segs.push_back(make_segment({
        make_word("まずね", 3266, 3310),
        make_word("暮らすのは好きなんです。", 3946, 4330),
    }));

    ScriptedBackend be({
        make_word("日本にもそんないない", 3318, 3454),
        make_word("です。", 3460, 3518),
        make_word("えっと。", 3686, 3782),
    });
    whisper_params params;

    crispasr_gap_fill_slice(be, params, samples.data(), sl.end, kSampleRate, sl, segs);

    // All three parts present, in reading order, with no overlap: the
    // covering segment must have been split around the recovery instead of
    // being left spanning it.
    require_monotone(segs);
    require_words_monotone_in_reading_order(segs);
    REQUIRE(segs.size() == 3);
    REQUIRE(segs[0].text == "まずね");
    REQUIRE(segs[1].text == "日本にもそんないないです。えっと。");
    REQUIRE(segs[2].text == "暮らすのは好きなんです。");
    REQUIRE(segs[0].t1 <= segs[1].t0);
    REQUIRE(segs[1].t1 <= segs[2].t0);

    // The fill's tokens ride along with the kept words: token-level outputs
    // read seg.tokens, and a recovery with an empty list vanishes from them.
    REQUIRE(segs[1].tokens.size() == 3);
    REQUIRE(segs[1].tokens.front().text == "日本にもそんないない");
    REQUIRE(segs[1].tokens.back().text == "えっと。");
}

TEST_CASE("issue #356: segment enclosing two recoveries is split at each", "[unit][gap-fill][issue-356]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_segment({
        make_word("head", 100, 150),
        make_word("mid", 700, 780),
        make_word("tail", 1400, 1500),
    }));
    segs.push_back(make_segment({make_word("first-recovery", 300, 500)}));
    segs.push_back(make_segment({make_word("second-recovery", 900, 1200)}));

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    require_words_monotone_in_reading_order(segs);
    REQUIRE(segs.size() == 5);
    REQUIRE(segs[0].text == "head");
    REQUIRE(segs[1].text == "first-recovery");
    REQUIRE(segs[2].text == "mid");
    REQUIRE(segs[3].text == "second-recovery");
    REQUIRE(segs[4].text == "tail");
}

TEST_CASE("issue #356: recovery before the first word moves the covering segment after it",
          "[unit][gap-fill][issue-356]") {
    // The covering segment's span starts at/before the recovery but all its
    // words sit past the recovery's end — nothing to keep on the left, so the
    // segment itself becomes the tail.
    std::vector<crispasr_segment> segs;
    auto covering = make_segment({make_word("late-words", 800, 1000)});
    covering.t0 = 200; // span inflated to the left (e.g. by an earlier aligner pass)
    segs.push_back(covering);
    segs.push_back(make_segment({make_word("recovery", 250, 600)}));

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].text == "recovery");
    REQUIRE(segs[1].text == "late-words");
    REQUIRE(segs[1].t0 == 800);
}

TEST_CASE("issue #356: word-less covering segment is clamped at the recovery", "[unit][gap-fill][issue-356]") {
    std::vector<crispasr_segment> segs;
    crispasr_segment plain;
    plain.text = "segment-level only";
    plain.t0 = 100;
    plain.t1 = 900;
    segs.push_back(plain);
    segs.push_back(make_segment({make_word("recovery", 400, 700)}));

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].text == "segment-level only");
    REQUIRE(segs[0].t1 == 400);
}

TEST_CASE("issue #356: already-monotone segments come through unchanged", "[unit][gap-fill][issue-356]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_segment({make_word("one", 100, 200)}));
    segs.push_back(make_segment({make_word("two", 200, 300)}));
    segs.push_back(make_segment({make_word("three", 500, 600)}));
    const auto before = joined_text(segs);

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    REQUIRE(segs.size() == 3);
    REQUIRE(joined_text(segs) == before);
    REQUIRE(segs[0].t0 == 100);
    REQUIRE(segs[2].t1 == 600);
}

TEST_CASE("issue #356: no text is lost when splitting", "[unit][gap-fill][issue-356]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_segment({
        make_word("alpha", 0, 100),
        make_word("omega", 2000, 2100),
    }));
    segs.push_back(make_segment({make_word("beta", 500, 900)}));
    const auto before = std::string("alpha") + "beta" + "omega"; // reading order

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    REQUIRE(joined_text(segs) == before);
}

TEST_CASE("issue #356: a recovery ending past the next word's start is merged, not clamped",
          "[unit][gap-fill][issue-356]") {
    // The clamp fallback used to fire here: the covering segment's LAST word
    // starts at 600, the recovery ends at 620, so `words.back().t0 >= cur.t1`
    // is false and there is nothing to split off. Capping the covering
    // segment's t1 at 500 left "omega" (600..900) inside a cue ending at 500
    // AND left the word stream reading 100, 600, 500 — i.e. #356 itself,
    // hidden from a check on segment spans.
    std::vector<crispasr_segment> segs;
    segs.push_back(make_segment({
        make_word("alpha", 100, 150),
        make_word("omega", 600, 900),
    }));
    segs.push_back(make_segment({make_word("beta", 500, 620)}));

    crispasr_gap_fill_resolve_overlaps(segs);

    require_words_monotone_in_reading_order(segs);
    require_monotone(segs);
    require_words_inside_their_cue(segs);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].text == "alpha beta omega");
    REQUIRE(segs[0].t0 == 100);
    REQUIRE(segs[0].t1 == 900);
}

TEST_CASE("issue #356: the merge case is reachable end-to-end through gap_fill_slice", "[unit][gap-fill][issue-356]") {
    // Nothing synthetic about the shape: the refill window for a hole runs to
    // g.second + kEdgePadCs (0.2 s), so a recovered word may legitimately end
    // after the next first-pass word starts. Slice 0..1000 cs, hole 150..600.
    const crispasr_audio_slice sl = {0, 1000 * kSampleRate / 100, 0, 1000};
    std::vector<float> samples((size_t)sl.end, 0.0f);

    std::vector<crispasr_segment> segs;
    segs.push_back(make_segment({
        make_word("alpha", 100, 150),
        make_word("omega", 600, 900),
    }));

    // mid = 560 — inside the hole and not covered by the merged first-pass
    // intervals ([100,150], [600,900]), so gap-fill keeps it.
    ScriptedBackend be({make_word("beta", 500, 620)});
    whisper_params params;

    crispasr_gap_fill_slice(be, params, samples.data(), sl.end, kSampleRate, sl, segs);

    require_words_monotone_in_reading_order(segs);
    require_monotone(segs);
    require_words_inside_their_cue(segs);
    REQUIRE(joined_text(segs) == "alpha beta omega");
}

TEST_CASE("issue #356: splitting keeps the speaker turn on the tail only", "[unit][gap-fill][issue-356]") {
    // tinydiarize's speaker_turn_next means "a turn follows THIS segment". The
    // tail is what follows the head, so copying the segment's metadata onto
    // both halves would announce a turn in the middle of one utterance.
    std::vector<crispasr_segment> segs;
    auto covering = make_segment({
        make_word("head", 100, 150),
        make_word("tail", 1400, 1500),
    });
    covering.speaker = "speaker 1";
    covering.speaker_turn_next = true;
    covering.chunk_id = 7;
    segs.push_back(std::move(covering));
    segs.push_back(make_segment({make_word("recovery", 300, 500)}));

    crispasr_gap_fill_resolve_overlaps(segs);

    require_words_monotone_in_reading_order(segs);
    REQUIRE(segs.size() == 3);
    REQUIRE(segs[0].text == "head");
    REQUIRE(segs[2].text == "tail");
    REQUIRE_FALSE(segs[0].speaker_turn_next);
    REQUIRE(segs[2].speaker_turn_next);
    // Everything else the split copies is still carried by both halves.
    REQUIRE(segs[0].speaker == "speaker 1");
    REQUIRE(segs[2].speaker == "speaker 1");
    REQUIRE(segs[0].chunk_id == 7);
    REQUIRE(segs[2].chunk_id == 7);
}

TEST_CASE("issue #356: untimed tokens survive a split that empties the head", "[unit][gap-fill][issue-356]") {
    // A token with t0 == -1 joins no word, so the t0-based partition would put
    // it in the head — and the head-is-empty branch discards the head outright.
    std::vector<crispasr_segment> segs;
    auto covering = make_segment({make_word("late-words", 800, 1000)});
    covering.t0 = 200; // span inflated to the left
    crispasr_token untimed;
    untimed.text = "no-timestamp";
    covering.tokens.push_back(untimed);
    crispasr_token timed;
    timed.text = "late";
    timed.t0 = 800;
    timed.t1 = 1000;
    covering.tokens.push_back(timed);
    segs.push_back(std::move(covering));
    segs.push_back(make_segment({make_word("recovery", 250, 600)}));

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[1].text == "late-words");
    REQUIRE(segs[1].tokens.size() == 2);
    REQUIRE(segs[1].tokens.front().text == "no-timestamp");
    REQUIRE(segs[1].tokens.back().text == "late");
}

// tests/test_diarize_align_order.cpp — unit tests for issue #267.
//
// Verifies that crispasr_apply_diarize correctly:
//   - splits segments at speaker-turn boundaries when word timestamps
//     are present (simulating post-alignment state);
//   - falls back to segment-level dominant-speaker assignment when
//     words are absent (no aligner, or alignment failed);
//   - handles edge cases: ties, short words, multiple turns, empty input.
//
// Pure CPU, no model load, no subprocess — all inputs are synthetic.
// Uses a pre-built CrispasrSherpaCache to exercise the global-sherpa
// diarize path without spawning the sherpa subprocess.
//
// Run: ctest -R test-diarize-align-order --output-on-failure

#include <catch2/catch_test_macros.hpp>

#include "crispasr_backend.h"
#include "crispasr_diarize_cli.h"
#include "whisper_params.h"

#include <string>
#include <vector>

// ── Helpers ─────────────────────────────────────────────────────────

static crispasr_segment make_seg(int64_t t0, int64_t t1, const std::string& text,
                                 const std::vector<std::pair<int64_t, int64_t>>& word_times = {}) {
    crispasr_segment s;
    s.t0 = t0;
    s.t1 = t1;
    s.text = text;
    std::vector<std::string> words_text;
    {
        std::string w;
        for (char c : text) {
            if (c == ' ') {
                if (!w.empty())
                    words_text.push_back(w);
                w.clear();
            } else {
                w += c;
            }
        }
        if (!w.empty())
            words_text.push_back(w);
    }
    for (size_t i = 0; i < words_text.size(); i++) {
        crispasr_word w;
        w.text = words_text[i];
        if (i < word_times.size()) {
            w.t0 = word_times[i].first;
            w.t1 = word_times[i].second;
        }
        s.words.push_back(w);
    }
    return s;
}

static CrispasrSherpaCache make_sherpa_cache(const std::vector<std::tuple<double, double, int>>& regions) {
    CrispasrSherpaCache cache;
    for (auto& [t0, t1, spk] : regions)
        cache.segments.push_back({t0, t1, spk});
    return cache;
}

static whisper_params make_sherpa_params() {
    whisper_params p{};
    p.diarize = true;
    p.diarize_method = "sherpa";
    p.no_prints = true;
    return p;
}

// ── #267 Test 1: words present → split at speaker turn ──────────────

TEST_CASE("issue267: diarize with words splits at speaker turn", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 5.0, 0}, {5.0, 10.0, 1}});

    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg(0, 1000, "Hello everyone I will now present the project",
                            {
                                {0, 100},
                                {100, 200},
                                {200, 350},
                                {350, 500},
                                {500, 600},
                                {600, 750},
                                {750, 850},
                                {850, 1000},
                            }));

    auto p = make_sherpa_params();
    std::vector<float> dummy(160000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);

    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].speaker.find("speaker 0") != std::string::npos);
    REQUIRE(segs[1].speaker.find("speaker 1") != std::string::npos);
    REQUIRE(segs[0].text.find("Hello") != std::string::npos);
    REQUIRE(segs[0].text.find("will") != std::string::npos);
    REQUIRE(segs[1].text.find("now") != std::string::npos);
    REQUIRE(segs[1].text.find("project") != std::string::npos);
    REQUIRE(!segs[0].words.empty());
    REQUIRE(!segs[1].words.empty());
}

// ── #267 Test 2: no words → dominant speaker, no crash ──────────────

TEST_CASE("issue267: diarize without words assigns dominant speaker", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 7.0, 0}, {7.0, 10.0, 1}});

    std::vector<crispasr_segment> segs;
    crispasr_segment seg;
    seg.t0 = 0;
    seg.t1 = 1000;
    seg.text = "Hello everyone I will now present the project";
    segs.push_back(seg);

    auto p = make_sherpa_params();
    std::vector<float> dummy(160000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].speaker.find("speaker 0") != std::string::npos);
}

// ── #267 Test 3: multiple speaker turns inside one segment ──────────

TEST_CASE("issue267: three speaker turns in one segment", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({
        {0.0, 3.0, 0},
        {3.0, 6.0, 1},
        {6.0, 9.0, 0},
    });

    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg(0, 900, "a b c d e f g h i",
                            {
                                {0, 100},
                                {100, 200},
                                {200, 300},
                                {300, 400},
                                {400, 500},
                                {500, 600},
                                {600, 700},
                                {700, 800},
                                {800, 900},
                            }));

    auto p = make_sherpa_params();
    std::vector<float> dummy(160000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    REQUIRE(segs.size() == 3);
    REQUIRE(segs[0].speaker.find("speaker 0") != std::string::npos);
    REQUIRE(segs[1].speaker.find("speaker 1") != std::string::npos);
    REQUIRE(segs[2].speaker.find("speaker 0") != std::string::npos);
}

// ── #267 Test 4: word at speaker boundary assigned by overlap ───────

TEST_CASE("issue267: word straddling speaker boundary", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 5.0, 0}, {5.0, 10.0, 1}});

    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg(0, 1000, "before straddling after",
                            {
                                {0, 300},
                                {450, 550},
                                {700, 1000},
                            }));

    auto p = make_sherpa_params();
    std::vector<float> dummy(160000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].speaker.find("speaker 0") != std::string::npos);
    REQUIRE(segs[0].text.find("straddling") != std::string::npos);
    REQUIRE(segs[1].speaker.find("speaker 1") != std::string::npos);
    REQUIRE(segs[1].text.find("after") != std::string::npos);
}

// ── #267 Test 5: all words same speaker → no split ──────────────────

TEST_CASE("issue267: single speaker does not split", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 10.0, 0}});

    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg(0, 500, "hello world foo bar",
                            {
                                {0, 100},
                                {100, 200},
                                {200, 300},
                                {300, 500},
                            }));

    auto p = make_sherpa_params();
    std::vector<float> dummy(160000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].speaker.find("speaker 0") != std::string::npos);
}

// ── #267 Test 6: native words also split correctly ──────────────────

TEST_CASE("issue267: native word timestamps also split correctly", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 2.0, 0}, {2.0, 4.0, 1}});

    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg(0, 400, "one two three four",
                            {
                                {0, 100},
                                {100, 200},
                                {200, 300},
                                {300, 400},
                            }));

    auto p = make_sherpa_params();
    std::vector<float> dummy(64000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].speaker.find("speaker 0") != std::string::npos);
    REQUIRE(segs[1].speaker.find("speaker 1") != std::string::npos);
}

// ── #267 Test 7: empty segments handled ─────────────────────────────

TEST_CASE("issue267: empty segment list is a no-op", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 10.0, 0}});
    std::vector<crispasr_segment> segs;

    auto p = make_sherpa_params();
    std::vector<float> dummy(16000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    REQUIRE(segs.empty());
}

// ── #267 Test 8: split preserves word timestamps ────────────────────

TEST_CASE("issue267: split sub-segments carry correct word timestamps", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 5.0, 0}, {5.0, 10.0, 1}});

    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg(0, 1000, "a b c d",
                            {
                                {0, 200},
                                {200, 500},
                                {500, 700},
                                {700, 1000},
                            }));

    auto p = make_sherpa_params();
    std::vector<float> dummy(160000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].words.size() == 2);
    REQUIRE(segs[0].words[0].text == "a");
    REQUIRE(segs[0].words[0].t0 == 0);
    REQUIRE(segs[0].words[1].text == "b");
    REQUIRE(segs[0].words[1].t1 == 500);
    REQUIRE(segs[1].words.size() == 2);
    REQUIRE(segs[1].words[0].text == "c");
    REQUIRE(segs[1].words[0].t0 == 500);
    REQUIRE(segs[1].words[1].text == "d");
    REQUIRE(segs[1].words[1].t1 == 1000);
}

// ── #267 Test 9: segment timestamps updated on split ────────────────

TEST_CASE("issue267: split sub-segment timestamps match word boundaries", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 3.0, 0}, {3.0, 6.0, 1}});

    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg(0, 600, "hello world",
                            {
                                {50, 250},
                                {350, 550},
                            }));

    auto p = make_sherpa_params();
    std::vector<float> dummy(96000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].t0 == 50);
    REQUIRE(segs[0].t1 == 250);
    REQUIRE(segs[1].t0 == 350);
    REQUIRE(segs[1].t1 == 550);
}

// ── #267 Test 10: deterministic dominant speaker on equal overlap ────

TEST_CASE("issue267: tied overlap picks first speaker deterministically", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 5.0, 0}, {5.0, 10.0, 1}});

    crispasr_segment seg;
    seg.t0 = 0;
    seg.t1 = 1000;
    seg.text = "equal overlap segment";
    std::vector<crispasr_segment> segs;
    segs.push_back(seg);

    auto p = make_sherpa_params();
    std::vector<float> dummy(160000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    REQUIRE(segs.size() == 1);
    // Equal overlap → first speaker wins (strict > in assign_speakers_from_sherpa)
    REQUIRE(segs[0].speaker.find("speaker 0") != std::string::npos);
}

// ── #267 Test 11: short words at diarize boundary ───────────────────

TEST_CASE("issue267: very short words near boundary", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 5.0, 0}, {5.0, 10.0, 1}});

    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg(0, 1000, "x y z",
                            {
                                {490, 500},
                                {500, 510},
                                {800, 1000},
                            }));

    auto p = make_sherpa_params();
    std::vector<float> dummy(160000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    REQUIRE(!segs.empty());
    for (const auto& s : segs)
        REQUIRE(s.speaker.find("speaker") != std::string::npos);
}

// ── #267 Test 12: JSON output consistency after split ────────────────

TEST_CASE("issue267: split segments produce consistent timing", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 5.0, 0}, {5.0, 10.0, 1}});

    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg(0, 1000, "hello world goodbye world",
                            {
                                {0, 200},
                                {200, 500},
                                {500, 700},
                                {700, 1000},
                            }));

    auto p = make_sherpa_params();
    std::vector<float> dummy(160000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);

    for (size_t i = 0; i < segs.size(); i++) {
        REQUIRE(segs[i].t1 >= segs[i].t0);
        REQUIRE(!segs[i].text.empty());
        REQUIRE(!segs[i].speaker.empty());
        for (const auto& w : segs[i].words) {
            REQUIRE(w.t0 >= segs[i].t0);
            REQUIRE(w.t1 <= segs[i].t1);
            REQUIRE(w.t1 >= w.t0);
        }
    }
    for (size_t i = 1; i < segs.size(); i++)
        REQUIRE(segs[i].t0 >= segs[i - 1].t0);
}

// ── #267 Test 13: mixed — some segs have words, some don't ──────────

TEST_CASE("issue267: mixed segments with and without words", "[diarize][unit][issue267]") {
    auto cache = make_sherpa_cache({{0.0, 5.0, 0}, {5.0, 15.0, 1}});

    std::vector<crispasr_segment> segs;
    // Segment 1: has words spanning speaker boundary
    segs.push_back(make_seg(0, 800, "a b c d",
                            {
                                {0, 200},
                                {200, 500},
                                {500, 650},
                                {650, 800},
                            }));
    // Segment 2: no words at all
    crispasr_segment seg2;
    seg2.t0 = 800;
    seg2.t1 = 1500;
    seg2.text = "no words here";
    segs.push_back(seg2);

    auto p = make_sherpa_params();
    std::vector<float> dummy(240000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);

    // Seg1 should be split (has words across boundary).
    // Seg2 should stay as one segment with dominant speaker.
    REQUIRE(segs.size() >= 2);
    // Last segment (from seg2) should be speaker 1 (800-1500 cs = 8-15s)
    REQUIRE(segs.back().speaker.find("speaker 1") != std::string::npos);
    REQUIRE(segs.back().text.find("no words here") != std::string::npos);
}

// ── #267 Test 14: alignment failure fallback ────────────────────────

TEST_CASE("issue267: alignment failure simulated — empty words treated as no-word", "[diarize][unit][issue267]") {
    // Simulate what happens when the external aligner was requested but
    // produced no valid words. The segment reaches diarize with
    // words.empty()==true. This is the fallback path.
    auto cache = make_sherpa_cache({{0.0, 3.0, 0}, {3.0, 6.0, 1}});

    crispasr_segment seg;
    seg.t0 = 0;
    seg.t1 = 600;
    seg.text = "alignment failed here";
    // Words intentionally empty — simulates aligner failure.
    std::vector<crispasr_segment> segs;
    segs.push_back(seg);

    auto p = make_sherpa_params();
    std::vector<float> dummy(96000, 0.0f);

    bool ok = crispasr_apply_diarize(dummy, dummy, false, 0, segs, p, nullptr, &cache);
    REQUIRE(ok);
    // Should NOT split — no words
    REQUIRE(segs.size() == 1);
    // Dominant speaker: [0,3] overlap=3s (spk0), [3,6] overlap=3s (spk1) → tie → spk0
    REQUIRE(segs[0].speaker.find("speaker 0") != std::string::npos);
}

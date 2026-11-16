// tests/test-diarize-slice-rewalk.cpp — issue #324 follow-up.
//
// The server transcribes every VAD/chunk slice first and diarizes the MERGED
// segment list afterwards, so it has to re-walk the slices and hand each one
// its own sub-range. `crispasr_apply_diarize` splits a segment at speaker-turn
// boundaries, which makes that sub-range GROW — and the original re-walk wrote
// back only the element count it started with, silently dropping every
// sub-segment past it. Reported symptom: with `diarize=true` (pyannote or
// foxnose) large parts of the transcript vanish, and precisely the parts around
// a speaker change.
//
// These tests pin the property the buggy version violated: the re-walk must
// preserve every segment the diarizer produces. The first three drive
// `crispasr_diarize_merged_by_slice` with synthetic callbacks; the last one
// wires in the REAL `crispasr_apply_diarize` (via a hand-built global sherpa
// cache) so the guard is anchored to production splitting behaviour rather
// than to a mock that merely imitates it.
//
// Pure CPU, no model load, no subprocess.
//
// Run: ctest -R test-diarize-slice-rewalk --output-on-failure

#include <catch2/catch_test_macros.hpp>

#include "crispasr_backend.h"
#include "crispasr_diarize_cli.h"
#include "crispasr_vad.h"
#include "whisper_params.h"

#include <string>
#include <utility>
#include <vector>

namespace {

crispasr_audio_slice make_slice(int64_t t0_cs, int64_t t1_cs) {
    crispasr_audio_slice s{};
    s.start = (int)(t0_cs * 160); // centiseconds → samples at 16 kHz
    s.end = (int)(t1_cs * 160);
    s.t0_cs = t0_cs;
    s.t1_cs = t1_cs;
    return s;
}

crispasr_segment make_seg(int64_t t0, int64_t t1, const std::string& text) {
    crispasr_segment s;
    s.t0 = t0;
    s.t1 = t1;
    s.text = text;
    return s;
}

// Split `text` on spaces into words spanning [t0,t1] evenly.
crispasr_segment make_worded_seg(int64_t t0, int64_t t1, const std::string& text) {
    crispasr_segment s = make_seg(t0, t1, text);
    std::vector<std::string> parts;
    std::string cur;
    for (char c : text) {
        if (c == ' ') {
            if (!cur.empty())
                parts.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty())
        parts.push_back(cur);

    const int64_t span = (t1 > t0 && !parts.empty()) ? (t1 - t0) / (int64_t)parts.size() : 0;
    for (size_t i = 0; i < parts.size(); i++) {
        crispasr_word w;
        w.text = parts[i];
        w.t0 = t0 + span * (int64_t)i;
        w.t1 = (i + 1 == parts.size()) ? t1 : t0 + span * (int64_t)(i + 1);
        s.words.push_back(w);
    }
    return s;
}

// Concatenate every segment's text, space-separated — the "did any transcript
// go missing" probe.
std::string joined_text(const std::vector<crispasr_segment>& segs) {
    std::string out;
    for (const auto& s : segs) {
        if (!out.empty() && !s.text.empty())
            out += ' ';
        out += s.text;
    }
    return out;
}

} // namespace

// ── #324: the regression itself — a splitting diarizer must not lose text ──

TEST_CASE("issue324: splitting diarizer keeps every sub-segment", "[diarize][unit][issue324]") {
    std::vector<crispasr_audio_slice> slices = {make_slice(0, 1000), make_slice(1000, 2000)};

    std::vector<crispasr_segment> segs = {
        make_seg(0, 500, "alpha"),
        make_seg(500, 1000, "bravo"),
        make_seg(1000, 1500, "charlie"),
        make_seg(1500, 2000, "delta"),
    };

    // Stand-in for the pyannote/foxnose/sherpa turn splitter: every segment
    // comes back as two half-length sub-segments.
    crispasr_diarize_merged_by_slice(segs, slices,
                                     [](const crispasr_audio_slice&, std::vector<crispasr_segment>& slice_segs) {
                                         std::vector<crispasr_segment> out;
                                         for (auto& s : slice_segs) {
                                             const int64_t mid = (s.t0 + s.t1) / 2;
                                             out.push_back(make_seg(s.t0, mid, s.text + "-a"));
                                             out.push_back(make_seg(mid, s.t1, s.text + "-b"));
                                         }
                                         slice_segs = std::move(out);
                                     });

    REQUIRE(segs.size() == 8);
    REQUIRE(joined_text(segs) == "alpha-a alpha-b bravo-a bravo-b charlie-a charlie-b delta-a delta-b");
}

TEST_CASE("issue324: each slice sees only the segments it owns", "[diarize][unit][issue324]") {
    std::vector<crispasr_audio_slice> slices = {make_slice(0, 1000), make_slice(1000, 2000), make_slice(2000, 3000)};

    std::vector<crispasr_segment> segs = {
        make_seg(0, 400, "one"),      make_seg(400, 900, "two"),    make_seg(1000, 1600, "three"),
        make_seg(2000, 2400, "four"), make_seg(2400, 3000, "five"),
    };

    std::vector<std::pair<int64_t, std::string>> seen;
    crispasr_diarize_merged_by_slice(segs, slices,
                                     [&](const crispasr_audio_slice& sl, std::vector<crispasr_segment>& slice_segs) {
                                         seen.push_back({sl.t0_cs, joined_text(slice_segs)});
                                     });

    REQUIRE(seen.size() == 3);
    REQUIRE(seen[0] == std::make_pair((int64_t)0, std::string("one two")));
    REQUIRE(seen[1] == std::make_pair((int64_t)1000, std::string("three")));
    REQUIRE(seen[2] == std::make_pair((int64_t)2000, std::string("four five")));
    // A pure inspection pass leaves the list untouched.
    REQUIRE(segs.size() == 5);
    REQUIRE(joined_text(segs) == "one two three four five");
}

TEST_CASE("issue324: a shrinking diarizer stays in bounds", "[diarize][unit][issue324]") {
    std::vector<crispasr_audio_slice> slices = {make_slice(0, 1000), make_slice(1000, 2000)};

    std::vector<crispasr_segment> segs = {
        make_seg(0, 400, "keep"),
        make_seg(400, 900, "drop"),
        make_seg(1000, 1500, "also-keep"),
    };

    // Dropping a segment used to leave the copy-back loop reading past the end
    // of the (now shorter) slice vector.
    crispasr_diarize_merged_by_slice(segs, slices,
                                     [](const crispasr_audio_slice&, std::vector<crispasr_segment>& slice_segs) {
                                         std::vector<crispasr_segment> out;
                                         for (auto& s : slice_segs)
                                             if (s.text != "drop")
                                                 out.push_back(std::move(s));
                                         slice_segs = std::move(out);
                                     });

    REQUIRE(segs.size() == 2);
    REQUIRE(joined_text(segs) == "keep also-keep");
}

TEST_CASE("issue324: relabel-only diarizer preserves order and text", "[diarize][unit][issue324]") {
    std::vector<crispasr_audio_slice> slices = {make_slice(0, 1000)};
    std::vector<crispasr_segment> segs = {make_seg(0, 500, "first"), make_seg(500, 1000, "second")};

    crispasr_diarize_merged_by_slice(segs, slices,
                                     [](const crispasr_audio_slice&, std::vector<crispasr_segment>& slice_segs) {
                                         for (auto& s : slice_segs)
                                             s.speaker = "(speaker 0) ";
                                     });

    REQUIRE(segs.size() == 2);
    REQUIRE(joined_text(segs) == "first second");
    REQUIRE(segs[0].speaker == "(speaker 0) ");
    REQUIRE(segs[1].speaker == "(speaker 0) ");
}

// ── The same property against the REAL diarizer ────────────────────────────
//
// A synthetic callback proves the re-walk contract; this proves the contract
// is the one production needs. The global-sherpa path is the cheapest real
// splitter to drive (no model, no subprocess — just a hand-built timeline).

TEST_CASE("issue324: real turn-splitting diarizer loses no words", "[diarize][unit][issue324]") {
    CrispasrSherpaCache cache;
    // Speaker changes at 5 s and 10 s — inside both slices' segments.
    cache.segments.push_back({0.0, 5.0, 0});
    cache.segments.push_back({5.0, 10.0, 1});
    cache.segments.push_back({10.0, 20.0, 0});

    whisper_params p{};
    p.diarize = true;
    p.diarize_method = "sherpa";
    p.no_prints = true;

    std::vector<float> audio(16000 * 20, 0.0f);
    std::vector<crispasr_audio_slice> slices = {make_slice(0, 1000), make_slice(1000, 2000)};

    // One segment per slice, each straddling a speaker turn.
    std::vector<crispasr_segment> segs = {
        make_worded_seg(0, 1000, "aa bb cc dd ee ff gg hh ii jj"),
        make_worded_seg(1000, 2000, "kk ll mm nn oo pp qq rr ss tt"),
    };
    const std::string before = joined_text(segs);

    crispasr_diarize_merged_by_slice(
        segs, slices, [&](const crispasr_audio_slice& sl, std::vector<crispasr_segment>& slice_segs) {
            std::vector<float> mono(audio.begin() + sl.start, audio.begin() + sl.end);
            crispasr_apply_diarize(mono, mono, /*is_stereo=*/false, sl.t0_cs, slice_segs, p, nullptr, &cache);
        });

    // The splitter really did split — otherwise this guard proves nothing.
    REQUIRE(segs.size() > 2);
    // …and not one word was dropped on the way back into the merged list.
    REQUIRE(joined_text(segs) == before);
}

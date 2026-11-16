// test-diarize-smooth.cpp — hermetic unit tests for temporal label smoothing
// (#324). No model, no audio, no network.
//
// Weight-free code the diff harness cannot see (HARD RULE #3b). Every case is
// a hand-built label sequence with a known correct answer, and the tuned
// constants are bracketed so retuning one fails a test (HARD RULE #2c).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/diarize_smooth.h"

#include <cmath>
#include <vector>

using namespace core_diarize_smooth;

namespace {
// Evenly spaced windows of `step` seconds each.
std::vector<Window> even_windows(size_t n, float step) {
    std::vector<Window> w;
    w.reserve(n);
    for (size_t i = 0; i < n; i++)
        w.push_back({(float)i * step, (float)(i + 1) * step});
    return w;
}
} // namespace

TEST_CASE("majority_label: unique winner, and -1 on a tie", "[unit][diarize_smooth]") {
    CHECK(majority_label({1, 1, 2}) == 1);
    CHECK(majority_label({3}) == 3);
    // A tie must NOT pick a winner — that is what stops the 3-window filter
    // inventing a label at a real speaker boundary.
    CHECK(majority_label({1, 2}) == -1);
    CHECK(majority_label({1, 1, 2, 2}) == -1);
    CHECK(majority_label({}) == -1);
}

TEST_CASE("smooth_window_labels: removes a single-window flicker", "[unit][diarize_smooth]") {
    // 0 0 1 0 0 -> the lone 1 is outvoted by its neighbours.
    std::vector<int> in = {0, 0, 1, 0, 0};
    auto out = smooth_window_labels(in);
    CHECK(out == std::vector<int>{0, 0, 0, 0, 0});
}

TEST_CASE("smooth_window_labels: preserves a real boundary", "[unit][diarize_smooth]") {
    // A clean 0->1 turn must survive; at the boundary the 3-window vote ties
    // and the original label is kept.
    std::vector<int> in = {0, 0, 0, 1, 1, 1};
    auto out = smooth_window_labels(in);
    CHECK(out == in);
}

TEST_CASE("smooth_window_labels: short sequences pass through", "[unit][diarize_smooth]") {
    CHECK(smooth_window_labels({1, 0}) == std::vector<int>{1, 0});
    CHECK(smooth_window_labels({}).empty());
}

TEST_CASE("speaker_centroids: unit-norm, one row per label, ascending", "[unit][diarize_smooth]") {
    // Two labels along +x and +y, with differing magnitudes so that a
    // magnitude-weighted mean would give a different answer than the
    // normalise-then-average the reference specifies.
    std::vector<float> emb = {5.0f, 0.0f, 1.0f, 0.0f, 0.0f, 9.0f, 0.0f, 2.0f};
    std::vector<int> lab = {7, 7, 3, 3};
    std::vector<int> values;
    auto c = speaker_centroids(emb.data(), 4, 2, lab, &values);
    REQUIRE(values == std::vector<int>{3, 7}); // ascending label order
    REQUIRE(c.size() == 4);
    CHECK(c[0] == Catch::Approx(0.0f).margin(1e-6)); // label 3 -> +y
    CHECK(c[1] == Catch::Approx(1.0f));
    CHECK(c[2] == Catch::Approx(1.0f)); // label 7 -> +x
    CHECK(c[3] == Catch::Approx(0.0f).margin(1e-6));
}

TEST_CASE("speaker_centroids: drops a label whose members cancel out", "[unit][diarize_smooth]") {
    // Antipodal members average to the zero vector: there is no direction to
    // report, so the label must be dropped rather than emitted as NaN.
    std::vector<float> emb = {1.0f, 0.0f, -1.0f, 0.0f};
    std::vector<int> lab = {0, 0};
    std::vector<int> values;
    auto c = speaker_centroids(emb.data(), 2, 2, lab, &values);
    CHECK(values.empty());
    CHECK(c.empty());
}

TEST_CASE("viterbi_smooth: follows the scores when the margin beats the penalty", "[unit][diarize_smooth]") {
    // Two frames, two labels; the second frame prefers label 1 by 1.0, far
    // above kSwitchPenalty, so the switch must be taken.
    std::vector<float> scores = {1.0f, 0.0f, 0.0f, 1.0f};
    auto p = viterbi_smooth(scores.data(), 2, 2);
    CHECK(p == std::vector<int>{0, 1});
}

TEST_CASE("viterbi_smooth: suppresses a flip that does not beat the penalty", "[unit][diarize_smooth]") {
    // Middle frame prefers label 1 by only 0.10 < kSwitchPenalty (0.18), and
    // switching there would cost the penalty twice. It must stay on label 0.
    std::vector<float> scores = {
        1.0f, 0.0f,  //
        0.0f, 0.10f, //
        1.0f, 0.0f,  //
    };
    auto p = viterbi_smooth(scores.data(), 3, 2);
    CHECK(p == std::vector<int>{0, 0, 0});

    // Bracket the constant: with a penalty of 0 the same evidence DOES flip.
    auto p0 = viterbi_smooth(scores.data(), 3, 2, 0.0f);
    CHECK(p0 == std::vector<int>{0, 1, 0});
}

TEST_CASE("viterbi_smooth: degenerate shapes", "[unit][diarize_smooth]") {
    std::vector<float> s = {1.0f};
    CHECK(viterbi_smooth(s.data(), 0, 2).empty());
    CHECK(viterbi_smooth(s.data(), 3, 1) == std::vector<int>{0, 0, 0});
}

TEST_CASE("collapse_short_islands: absorbs a short A-B-A island", "[unit][diarize_smooth]") {
    std::vector<int> in = {0, 0, 1, 0, 0};
    auto w = even_windows(in.size(), 0.5f); // island is 0.5 s < 1.2 s
    CHECK(collapse_short_islands(in, w) == std::vector<int>{0, 0, 0, 0, 0});
}

TEST_CASE("collapse_short_islands: keeps a sustained B run", "[unit][diarize_smooth]") {
    // Same A-B-A shape but B lasts 1.5 s > kMaxFragmentSeconds: a real turn.
    std::vector<int> in = {0, 1, 1, 1, 0};
    auto w = even_windows(in.size(), 0.5f);
    CHECK(collapse_short_islands(in, w) == in);
}

TEST_CASE("collapse_short_islands: A-B-C is never collapsed", "[unit][diarize_smooth]") {
    // Only a sandwich between the SAME label is flicker. A genuine three-way
    // change must survive however short the middle run is.
    std::vector<int> in = {0, 1, 2};
    auto w = even_windows(in.size(), 0.2f);
    CHECK(collapse_short_islands(in, w) == in);
}

TEST_CASE("restore_sustained_runs: puts back a long run Viterbi erased", "[unit][diarize_smooth]") {
    std::vector<int> original = {0, 1, 1, 1, 0};
    std::vector<int> smoothed = {0, 0, 0, 0, 0};  // Viterbi flattened the turn
    auto w = even_windows(original.size(), 0.5f); // the 1-run is 1.5 s > 1.2 s
    CHECK(restore_sustained_runs(original, smoothed, w) == original);
}

TEST_CASE("restore_sustained_runs: leaves short runs erased", "[unit][diarize_smooth]") {
    std::vector<int> original = {0, 1, 0};
    std::vector<int> smoothed = {0, 0, 0};
    auto w = even_windows(original.size(), 0.3f); // the 1-run is 0.3 s < 1.2 s
    CHECK(restore_sustained_runs(original, smoothed, w) == smoothed);
}

TEST_CASE("restore_sustained_runs: mismatched sizes are a no-op", "[unit][diarize_smooth]") {
    std::vector<int> a = {0, 1};
    std::vector<int> b = {0, 1, 0};
    CHECK(restore_sustained_runs(a, b, even_windows(3, 0.5f)) == b);
}

TEST_CASE("smooth_segment_temporal: pulls a flicker back onto its centroid", "[unit][diarize_smooth]") {
    // Five windows all genuinely +x (speaker 0); window 2 was mislabelled 1.
    // Scored against both centroids, Viterbi must restore it.
    const int n = 5, d = 2;
    std::vector<float> emb = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    std::vector<int> labels = {0, 0, 1, 0, 0};
    std::vector<float> centroids = {1, 0, 0, 1}; // label 0 -> +x, label 1 -> +y
    std::vector<int> values = {0, 1};
    auto out = smooth_segment_temporal(labels, emb.data(), n, d, values, centroids.data());
    CHECK(out == std::vector<int>{0, 0, 0, 0, 0});
}

TEST_CASE("smooth_segment_temporal: short input or single label passes through", "[unit][diarize_smooth]") {
    std::vector<float> emb = {1, 0, 1, 0};
    std::vector<float> cent = {1, 0};
    std::vector<int> one = {0};
    std::vector<int> labels2 = {0, 0};
    CHECK(smooth_segment_temporal(labels2, emb.data(), 2, 2, one, cent.data()) == labels2);
}

// test_tabcnn_live.cpp — TabCNN guitar tablature, via the session C ABI.
//
// Needs a real model: set CRISPASR_MODEL_TABCNN (see tests/env-live-tests.sh).
// Skips cleanly when unset.
//
// Exercises the SESSION path deliberately, not the runtime directly. That is the
// surface every binding and the server actually use, and the repo's recurring
// bug is a backend wired into the CLI but missing from the C ABI — so testing
// the runtime alone would pass while Python/Go/Dart got a null session.
//
// What is asserted here is the CONTRACT, not the model's musical accuracy:
// emission geometry, that the rows are normalised log-probabilities, that the
// metadata a decoder depends on is present and sane, and that silence produces
// silence. Accuracy is measured separately against EGSet12 ground truth
// (tablature F1 0.7732 vs the torch reference's 0.7708) — a synthetic tone in a
// unit test cannot stand in for that, and pretending otherwise would be the
// "self-vs-self cosine" failure this repo has been bitten by.

#include "crispasr_session.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kSampleRate = 22050;
constexpr int kStrings = 6;
constexpr int kClasses = 21;

// A plucked-ish E2 (82.41 Hz, the guitar's lowest open string) with harmonics
// and a decaying envelope, so the CQT sees something instrument-shaped rather
// than a single-bin sine.
std::vector<float> plucked_low_e(double seconds) {
    const int n = (int)(seconds * kSampleRate);
    std::vector<float> x((size_t)n, 0.0f);
    const double f0 = 82.41;
    for (int i = 0; i < n; i++) {
        const double t = (double)i / kSampleRate;
        const double env = std::exp(-1.5 * t);
        double v = 0.0;
        for (int h = 1; h <= 5; h++)
            v += std::sin(2.0 * M_PI * f0 * h * t) / (double)h;
        x[(size_t)i] = (float)(0.3 * env * v);
    }
    return x;
}

} // namespace

TEST_CASE("tabcnn emission scorer via the session ABI", "[integration][tabcnn]") {
    const char* model = std::getenv("CRISPASR_MODEL_TABCNN");
    if (!model || !*model)
        SKIP("CRISPASR_MODEL_TABCNN not set");

    crispasr_session* s = crispasr_session_open_explicit(model, "tabcnn", 2);
    REQUIRE(s != nullptr);

    // Metadata a decoder depends on. silent_class is read, never assumed: a
    // decoder that guesses it wrong emits confidently wrong tablature with no
    // error anywhere.
    const int silent = crispasr_session_tab_silent_class(s);
    CHECK(silent == kClasses - 1);
    const float period = crispasr_session_tab_frame_period(s);
    CHECK(period > 0.0f);
    CHECK(period < 0.1f); // 512/22050 = 23.2 ms
    // Standard tuning, low string first: E2 A2 D3 G3 B3 E4 = MIDI 40 45 50 55 59 64.
    const int expect_midi[kStrings] = {40, 45, 50, 55, 59, 64};
    for (int i = 0; i < kStrings; i++)
        CHECK(crispasr_session_tab_string_open_midi(s, i) == expect_midi[i]);
    // Out-of-range must not read past the array.
    CHECK(crispasr_session_tab_string_open_midi(s, -1) == -1);
    CHECK(crispasr_session_tab_string_open_midi(s, kStrings) == -1);

    const auto audio = plucked_low_e(2.0);
    const int n_frames = crispasr_session_tab(s, audio.data(), (int)audio.size(), kSampleRate);
    REQUIRE(n_frames > 0);
    CHECK(crispasr_session_tab_n_frames(s) == n_frames);

    int f = 0, strings = 0, classes = 0;
    const float* em = crispasr_session_tab_emissions(s, &f, &strings, &classes);
    REQUIRE(em != nullptr);
    CHECK(f == n_frames);
    CHECK(strings == kStrings);
    CHECK(classes == kClasses);

    // Every (frame, string) row must be a normalised distribution in log space.
    // This is the contract the whole design rests on -- a caller's DP sums these
    // as costs, so a row that does not sum to 1 silently biases every path
    // through it, and a +inf or NaN poisons the search outright.
    double worst_sum_err = 0.0;
    for (int t = 0; t < f; t++) {
        for (int st = 0; st < strings; st++) {
            const float* row = em + ((size_t)t * strings + st) * classes;
            double total = 0.0;
            for (int c = 0; c < classes; c++) {
                REQUIRE(std::isfinite(row[c]));
                CHECK(row[c] <= 1e-4f); // log-prob, so never positive
                total += std::exp((double)row[c]);
            }
            worst_sum_err = std::max(worst_sum_err, std::fabs(total - 1.0));
        }
    }
    INFO("worst |sum(exp(logp)) - 1| = " << worst_sum_err);
    CHECK(worst_sum_err < 1e-3);

    SECTION("silence is scored as unplayed on every string") {
        // Not an accuracy claim -- it is the one behaviour a caller can rely on
        // without a decoder, and it fails loudly if the front end is misconfigured
        // (a wrong fmin still RUNS and produces confident garbage; see
        // src/tabcnn.h).
        const std::vector<float> quiet((size_t)kSampleRate, 0.0f);
        const int qn = crispasr_session_tab(s, quiet.data(), (int)quiet.size(), kSampleRate);
        REQUIRE(qn > 0);
        int fq = 0, sq = 0, cq = 0;
        const float* qe = crispasr_session_tab_emissions(s, &fq, &sq, &cq);
        REQUIRE(qe != nullptr);
        int fretted = 0;
        for (int t = 0; t < fq; t++)
            for (int st = 0; st < sq; st++) {
                const float* row = qe + ((size_t)t * sq + st) * cq;
                int best = 0;
                for (int c = 1; c < cq; c++)
                    if (row[c] > row[best])
                        best = c;
                fretted += (best != silent);
            }
        INFO("fretted string-frames on digital silence: " << fretted << " of " << (fq * sq));
        CHECK(fretted == 0);
    }

    SECTION("bad arguments are rejected, not crashed") {
        CHECK(crispasr_session_tab(s, nullptr, 100, kSampleRate) < 0);
        CHECK(crispasr_session_tab(s, audio.data(), 0, kSampleRate) < 0);
        CHECK(crispasr_session_tab(s, audio.data(), (int)audio.size(), 0) < 0);
    }

    crispasr_session_close(s);
}

// BTC chord recognition integration test.
//
// Requires CRISPASR_MODEL_BTC_CHORDS pointing at a btc GGUF. SKIPs cleanly
// when unset.
//
// This drives the SESSION C-ABI, not btc_chords.h directly. The CLI adapter
// already has coverage via crispasr-diff; the session arm is the surface that
// silently rots when a backend is added to one dispatcher and not the others
// (see docs/multi-surface dispatch notes), so it is the one worth a live test.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_session.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kSampleRate = 22050;

// Three seconds of a C major triad (C4-E4-G4) then three of G major (G3-B3-D4),
// each with two harmonics so the CQT sees a plausible instrument spectrum
// rather than pure sine energy in a single bin.
std::vector<float> triad_pair() {
    const double c_major[3] = {261.63, 329.63, 392.00};
    const double g_major[3] = {196.00, 246.94, 293.66};
    std::vector<float> pcm((size_t)kSampleRate * 6, 0.0f);
    for (size_t i = 0; i < pcm.size(); i++) {
        const double t = (double)i / kSampleRate;
        const double* chord = (i < pcm.size() / 2) ? c_major : g_major;
        double v = 0.0;
        for (int n = 0; n < 3; n++)
            for (int h = 1; h <= 2; h++)
                v += std::sin(2.0 * M_PI * chord[n] * h * t) / (h * 3.0);
        pcm[i] = (float)(v * 0.3);
    }
    return pcm;
}

crispasr_session* open_session(const char* model) {
    return crispasr_session_open_explicit(model, "btc-chords", 2);
}

} // namespace

TEST_CASE("btc-chords session opens and reports a vocabulary", "[integration][btc-chords]") {
    const char* model = std::getenv("CRISPASR_MODEL_BTC_CHORDS");
    if (!model || !*model)
        SKIP("CRISPASR_MODEL_BTC_CHORDS not set");

    crispasr_session* s = open_session(model);
    REQUIRE(s != nullptr);
    // 25 (maj/min + N) or 170 (full quality set) — nothing else is a BTC head.
    const int vocab = crispasr_session_chords_vocab_size(s);
    REQUIRE((vocab == 25 || vocab == 170));
    crispasr_session_close(s);
}

TEST_CASE("btc-chords recognises a chord timeline", "[integration][btc-chords]") {
    const char* model = std::getenv("CRISPASR_MODEL_BTC_CHORDS");
    if (!model || !*model)
        SKIP("CRISPASR_MODEL_BTC_CHORDS not set");

    crispasr_session* s = open_session(model);
    REQUIRE(s != nullptr);

    const std::vector<float> pcm = triad_pair();
    const int n = crispasr_session_chords(s, pcm.data(), (int)pcm.size(), kSampleRate);
    REQUIRE(n > 0);
    REQUIRE(crispasr_session_chords_n_spans(s) == n);

    int n_flat = 0;
    const float* flat = crispasr_session_chords_spans(s, &n_flat);
    REQUIRE(flat != nullptr);
    REQUIRE(n_flat == n);

    double prev_end = -1.0;
    bool any_chord = false;
    for (int i = 0; i < n; i++) {
        const float start = flat[i * 4 + 0];
        const float end = flat[i * 4 + 1];
        const float label = flat[i * 4 + 2];
        const float conf = flat[i * 4 + 3];

        // Spans must be ordered, non-empty and contiguous — a timeline with
        // gaps or overlaps means the run-length merge is wrong.
        REQUIRE(end > start);
        REQUIRE(start >= prev_end - 1e-3);
        prev_end = end;

        REQUIRE(label >= 0.0f);
        REQUIRE(label < (float)crispasr_session_chords_vocab_size(s));
        REQUIRE(conf > 0.0f);
        REQUIRE(conf <= 1.0f);

        const char* name = crispasr_session_chords_span_name(s, i);
        REQUIRE(name != nullptr);
        REQUIRE(std::strlen(name) > 0);
        if (std::strcmp(name, "N") != 0)
            any_chord = true;
    }

    // The whole clip is voiced triads, so an all-"N" timeline means the front
    // end is feeding the model silence. That is exactly how the missing
    // librosa scale=True normalisation in core/cqt.h presented (every bin low
    // by sqrt(N_k)), and it is the reason this assertion exists.
    REQUIRE(any_chord);

    // Out-of-range name lookups return NULL rather than reading past the end.
    REQUIRE(crispasr_session_chords_span_name(s, n) == nullptr);
    REQUIRE(crispasr_session_chords_span_name(s, -1) == nullptr);

    crispasr_session_close(s);
}

TEST_CASE("btc-chords rejects bad arguments", "[integration][btc-chords]") {
    const char* model = std::getenv("CRISPASR_MODEL_BTC_CHORDS");
    if (!model || !*model)
        SKIP("CRISPASR_MODEL_BTC_CHORDS not set");

    crispasr_session* s = open_session(model);
    REQUIRE(s != nullptr);

    const std::vector<float> pcm(1024, 0.0f);
    REQUIRE(crispasr_session_chords(s, nullptr, 1024, kSampleRate) == -1);
    REQUIRE(crispasr_session_chords(s, pcm.data(), 0, kSampleRate) == -1);
    REQUIRE(crispasr_session_chords(s, pcm.data(), (int)pcm.size(), 0) == -1);
    REQUIRE(crispasr_session_chords(nullptr, pcm.data(), (int)pcm.size(), kSampleRate) == -1);

    // A session with no chord arm must report "no spans", not crash.
    REQUIRE(crispasr_session_chords_n_spans(nullptr) == 0);
    REQUIRE(crispasr_session_chords_spans(nullptr, nullptr) == nullptr);

    crispasr_session_close(s);
}

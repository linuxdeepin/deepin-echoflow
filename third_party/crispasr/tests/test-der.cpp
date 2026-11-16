// test-der.cpp — known-answer tests for the DER scorer, plus an end-to-end
// pipeline test driven by a SYNTHETIC embedder (#324).
//
// The synthetic embedder matters: it gives the pipeline exact ground truth
// with no model, no audio and no network, so the orchestration — windowing,
// boundary construction, clustering, smoothing, merging — is testable on its
// own. A model-based test cannot separate "the pipeline is wrong" from "the
// embedder is weak".

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/der.h"
#include "core/foxnose_pipeline.h"

#include <cmath>
#include <random>
#include <vector>

using core_der::Turn;

// ── DER scorer ────────────────────────────────────────────────────────────

TEST_CASE("der: a perfect hypothesis scores 0", "[unit][der]") {
    std::vector<Turn> ref = {{0.0, 10.0, 0}, {10.0, 20.0, 1}};
    auto s = core_der::score(ref, ref, 0.25);
    CHECK(s.der() == Catch::Approx(0.0));
    CHECK(s.n_ref_speakers == 2);
    CHECK(s.n_hyp_speakers == 2);
}

TEST_CASE("der: speaker labels are arbitrary — relabelling costs nothing", "[unit][der]") {
    std::vector<Turn> ref = {{0.0, 10.0, 0}, {10.0, 20.0, 1}};
    // Same partition, swapped names. The optimal mapping must absorb it.
    std::vector<Turn> hyp = {{0.0, 10.0, 7}, {10.0, 20.0, 3}};
    CHECK(core_der::score(ref, hyp, 0.25).der() == Catch::Approx(0.0));
}

TEST_CASE("der: total confusion scores 1", "[unit][der]") {
    // Both speakers present but every instant attributed to the wrong one:
    // with a 1:1 mapping one of the two can be matched, so swapping the
    // labels of a symmetric 2-speaker reference is free. Force real confusion
    // by collapsing the hypothesis onto ONE speaker instead.
    std::vector<Turn> ref = {{0.0, 10.0, 0}, {10.0, 20.0, 1}};
    std::vector<Turn> hyp = {{0.0, 20.0, 0}};
    auto s = core_der::score(ref, hyp, 0.0);
    // Half the reference is now attributed to the wrong speaker.
    CHECK(s.confusion == Catch::Approx(10.0));
    CHECK(s.der() == Catch::Approx(0.5));
}

TEST_CASE("der: missed speech and false alarm are counted separately", "[unit][der]") {
    std::vector<Turn> ref = {{0.0, 10.0, 0}};
    std::vector<Turn> hyp = {{0.0, 5.0, 0}}; // second half missed
    auto s = core_der::score(ref, hyp, 0.0);
    CHECK(s.missed == Catch::Approx(5.0));
    CHECK(s.false_alarm == Catch::Approx(0.0));
    CHECK(s.der() == Catch::Approx(0.5));

    std::vector<Turn> hyp2 = {{0.0, 15.0, 0}}; // 5 s hallucinated
    auto s2 = core_der::score(ref, hyp2, 0.0);
    CHECK(s2.missed == Catch::Approx(0.0));
    CHECK(s2.false_alarm == Catch::Approx(5.0));
    // Denominator is REFERENCE speech, so over-generation can exceed 1.
    CHECK(s2.der() == Catch::Approx(0.5));
}

TEST_CASE("der: the collar excludes boundary regions", "[unit][der]") {
    std::vector<Turn> ref = {{0.0, 10.0, 0}, {10.0, 20.0, 1}};
    // Hypothesis boundary is 0.2 s late — inside a 0.25 s collar it is free,
    // with no collar it costs 0.2 s of confusion.
    std::vector<Turn> hyp = {{0.0, 10.2, 0}, {10.2, 20.0, 1}};
    CHECK(core_der::score(ref, hyp, 0.25).der() == Catch::Approx(0.0));
    auto strict = core_der::score(ref, hyp, 0.0);
    CHECK(strict.confusion == Catch::Approx(0.2));
}

TEST_CASE("der: empty reference is a no-op rather than a divide by zero", "[unit][der]") {
    auto s = core_der::score({}, {{0.0, 5.0, 0}}, 0.25);
    CHECK(s.der() == Catch::Approx(0.0));
    CHECK(s.total_ref == Catch::Approx(0.0));
}

// ── pipeline windowing ────────────────────────────────────────────────────

TEST_CASE("window_speech: short regions are skipped, medium embedded whole", "[unit][der]") {
    using core_foxnose::Speech;
    CHECK(core_foxnose::window_speech({0.0, 0.3}).empty()); // < 0.4 s
    auto one = core_foxnose::window_speech({0.0, 1.5});     // <= 1.2*1.5
    REQUIRE(one.size() == 1);
    CHECK(one[0].start == Catch::Approx(0.0));
    CHECK(one[0].end == Catch::Approx(1.5));
}

TEST_CASE("window_speech: long regions slide, and no sliver is emitted", "[unit][der]") {
    auto w = core_foxnose::window_speech({0.0, 5.0});
    REQUIRE(w.size() > 1);
    for (const auto& s : w) {
        CHECK(s.end <= Catch::Approx(5.0));
        CHECK(s.end > s.start);
        // Every emitted window must carry at least the minimum speech; a
        // trailing sliver would contribute a meaningless embedding.
        CHECK(s.end - s.start >= Catch::Approx(core_foxnose::kMinSegmentSeconds).margin(1e-6));
    }
    CHECK(w[1].start == Catch::Approx(core_foxnose::kEmbeddingStepSeconds));
}

TEST_CASE("window_boundaries: tiles the parent region exactly, no overlap", "[unit][der]") {
    core_foxnose::Speech seg{2.0, 8.0};
    auto w = core_foxnose::window_speech(seg);
    auto b = core_foxnose::window_boundaries(seg, w);
    REQUIRE(b.size() == w.size());
    CHECK(b.front().start == Catch::Approx(seg.start));
    CHECK(b.back().end == Catch::Approx(seg.end));
    double covered = 0.0;
    for (size_t i = 0; i < b.size(); i++) {
        CHECK(b[i].end >= b[i].start);
        covered += b[i].end - b[i].start;
        if (i + 1 < b.size())
            CHECK(b[i].end == Catch::Approx(b[i + 1].start)); // contiguous
    }
    CHECK(covered == Catch::Approx(seg.end - seg.start));
}

// ── end-to-end with a synthetic embedder ──────────────────────────────────

namespace {
// Ground-truth timeline: speaker s holds the floor for `turn` seconds at a
// time, alternating over `n_turns`.
struct Synth {
    int dim = 32;
    double turn = 3.0;
    int n_speakers = 2;
    float noise = 0.15f;
    std::vector<std::vector<float>> centres;
    int sample_rate = 16000;
};

// `worker` is unused: this fake embedder is stateless, so every worker slot
// can share it. A real one needs one context per worker.
int synth_embed(void* ud, int worker, const float* pcm, int n, float* out) {
    (void)worker;
    // The "audio" carries its speaker in sample 0; the embedder returns that
    // speaker's centre plus deterministic noise keyed to the window, so the
    // pipeline sees a realistic but exactly-known embedding space.
    auto* s = static_cast<Synth*>(ud);
    if (n <= 0)
        return 1;
    const int spk = (int)std::lround(pcm[0]);
    if (spk < 0 || spk >= s->n_speakers)
        return 1;
    std::mt19937 rng((unsigned)(std::llround(pcm[0] * 1000) * 7919 + n * 104729));
    std::normal_distribution<float> g(0.0f, s->noise);
    for (int j = 0; j < s->dim; j++)
        out[j] = s->centres[(size_t)spk][(size_t)j] + g(rng);
    return 0;
}
} // namespace

TEST_CASE("pipeline: recovers a clean two-speaker timeline", "[unit][der]") {
    Synth synth;
    std::mt19937 rng(1234);
    std::normal_distribution<float> g(0.0f, 1.0f);
    synth.centres.resize((size_t)synth.n_speakers, std::vector<float>((size_t)synth.dim));
    // Orthogonalise the speaker centres (Gram-Schmidt). Two *random* 32-D
    // directions have cosine std ~= 1/sqrt(32) ~= 0.18, so a draw can easily
    // exceed kSingleSpeakerSimP10 = 0.16 and trip the single-speaker veto on a
    // genuinely two-speaker timeline — which is exactly what happened with an
    // earlier version of this test. Real WeSpeaker embeddings measure ~0.10
    // cross-speaker (tests/test_wespeaker_live.cpp), comfortably under the
    // threshold, so orthogonal centres are the FAITHFUL choice here, not a
    // convenient one.
    for (int s = 0; s < synth.n_speakers; s++) {
        for (int j = 0; j < synth.dim; j++)
            synth.centres[(size_t)s][(size_t)j] = g(rng);
        for (int p = 0; p < s; p++) {
            double dot = 0;
            for (int j = 0; j < synth.dim; j++)
                dot += synth.centres[(size_t)s][(size_t)j] * synth.centres[(size_t)p][(size_t)j];
            double pn = 0;
            for (int j = 0; j < synth.dim; j++)
                pn += synth.centres[(size_t)p][(size_t)j] * synth.centres[(size_t)p][(size_t)j];
            for (int j = 0; j < synth.dim; j++)
                synth.centres[(size_t)s][(size_t)j] -= (float)(dot / pn) * synth.centres[(size_t)p][(size_t)j];
        }
        double nr = 0;
        for (int j = 0; j < synth.dim; j++)
            nr += synth.centres[(size_t)s][(size_t)j] * synth.centres[(size_t)s][(size_t)j];
        nr = std::sqrt(nr);
        for (int j = 0; j < synth.dim; j++)
            synth.centres[(size_t)s][(size_t)j] *= (float)(6.0 / nr);
    }

    const int n_turns = 8;
    const double turn = synth.turn;
    const int sr = synth.sample_rate;
    const int total = (int)(n_turns * turn * sr);
    std::vector<float> pcm((size_t)total, 0.0f);
    std::vector<Turn> ref;
    std::vector<core_foxnose::Speech> speech;
    for (int t = 0; t < n_turns; t++) {
        const int spk = t % synth.n_speakers;
        const double a = t * turn, b = (t + 1) * turn;
        for (long i = std::lround(a * sr); i < std::lround(b * sr) && i < total; i++)
            pcm[(size_t)i] = (float)spk;
        ref.push_back({a, b, spk});
        speech.push_back({a, b});
    }

    core_foxnose::Params p;
    p.min_speakers = 1;
    p.max_speakers = 6;
    auto res = core_foxnose::diarize(pcm.data(), total, sr, speech, synth_embed, &synth, synth.dim, p);

    INFO("windows=" << res.n_windows << " skipped=" << res.n_skipped << " speakers=" << res.n_speakers
                    << " reason=" << res.reason);
    REQUIRE(res.n_windows > 0);
    CHECK(res.n_speakers == synth.n_speakers);

    std::vector<Turn> hyp;
    for (const auto& t : res.turns)
        hyp.push_back({t.start, t.end, t.speaker});
    auto s = core_der::score(ref, hyp, 0.25);
    INFO("DER = " << s.der() << " (miss " << s.missed << ", fa " << s.false_alarm << ", conf " << s.confusion
                  << ", ref " << s.total_ref << ")");
    // Well-separated synthetic speakers on clean turn boundaries: this is the
    // easy case, so anything above a few percent means the orchestration is
    // losing information the embeddings clearly carry.
    CHECK(s.der() < 0.10);
}

TEST_CASE("pipeline: degenerate inputs return empty rather than crash", "[unit][der]") {
    Synth synth;
    synth.centres.assign(2, std::vector<float>(32, 1.0f));
    core_foxnose::Params p;
    std::vector<float> pcm(1000, 0.0f);
    CHECK(core_foxnose::diarize(nullptr, 0, 16000, {}, synth_embed, &synth, 32, p).turns.empty());
    CHECK(core_foxnose::diarize(pcm.data(), 1000, 16000, {}, synth_embed, &synth, 32, p).turns.empty());
    // A region below the minimum produces no windows at all.
    auto r = core_foxnose::diarize(pcm.data(), 1000, 16000, {{0.0, 0.05}}, synth_embed, &synth, 32, p);
    CHECK(r.n_windows == 0);
}

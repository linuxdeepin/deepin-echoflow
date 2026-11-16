// test-vad-failover.cpp — "the VAD obviously failed" predicate (PLAN.md §W4).
//
// The asymmetry this encodes: a false positive costs a slow re-transcribe of
// silence; a false negative costs the user their whole transcript. So the
// must-NOT-fire cases below matter more than the must-fire ones — they are what
// stops the thresholds drifting loose enough to re-transcribe everything.

#include <catch2/catch_test_macros.hpp>

#include "core/vad_failover.h"

#include <vector>

using core_vad_failover::assess;
using core_vad_failover::Config;
using core_vad_failover::Span;

namespace {

// `n` evenly spaced speech spans of `dur` seconds each across `audio` seconds.
std::vector<Span> spread(int n, double dur, double audio) {
    std::vector<Span> v;
    for (int i = 0; i < n; i++) {
        const double t0 = audio * (double)i / (double)n;
        v.push_back({t0, t0 + dur});
    }
    return v;
}

} // namespace

TEST_CASE("a healthy long recording does not fail over", "[vad-failover]") {
    // 30 min, 200 segments, ~55% speech — an ordinary interview.
    const auto v = assess(spread(200, 5.0, 1800.0), 1800.0);
    REQUIRE_FALSE(v.failover);
    REQUIRE(v.segment_count == 200);
    REQUIRE(v.coverage > 0.5);
}

TEST_CASE("no speech at all on a long clip fails over", "[vad-failover]") {
    const auto v = assess({}, 900.0);
    REQUIRE(v.failover);
    REQUIRE_FALSE(v.reason.empty());
}

TEST_CASE("a trace of speech on a long clip fails over", "[vad-failover]") {
    // 20 min, 3 segments totalling 4.5 s = 0.375% — the reported failure mode.
    const auto v = assess({{10.0, 11.5}, {400.0, 401.5}, {900.0, 901.5}}, 1200.0);
    REQUIRE(v.failover);
    REQUIRE(v.coverage < 0.01);
}

TEST_CASE("short clips never fail over", "[vad-failover]") {
    // The SAME near-empty shape on a 60 s clip is plausible and must pass: a
    // minute of room tone with one word in it is a real recording.
    const auto v = assess({{10.0, 10.3}}, 60.0);
    REQUIRE_FALSE(v.failover);
    // And on 119 s, just under the floor.
    REQUIRE_FALSE(assess({{10.0, 10.3}}, 119.0).failover);
    // 121 s, just over, with the same trace of speech: now it fires.
    REQUIRE(assess({{10.0, 10.3}}, 121.0).failover);
}

TEST_CASE("a long CONTINUOUS monologue is not mistaken for a failure", "[vad-failover]") {
    // This is the case WhisperJAV's version gets wrong. Its rule is
    // `len(segments) <= 2 and duration >= 480` with no coverage condition, so a
    // 10-minute talk detected as ONE segment covering ~99% of the file trips it
    // and the whole thing is needlessly re-transcribed. Requiring low coverage
    // as well keeps the real signal and drops this accident.
    const auto v = assess({{2.0, 598.0}}, 600.0);
    REQUIRE_FALSE(v.failover);
    REQUIRE(v.segment_count == 1);
    REQUIRE(v.coverage > 0.9);

    // Two long segments with a break in the middle: also fine.
    REQUIRE_FALSE(assess({{2.0, 300.0}, {310.0, 598.0}}, 600.0).failover);
}

TEST_CASE("few segments AND low coverage on a very long clip fails over", "[vad-failover]") {
    // 15 min, 2 segments totalling 30 s = 3.3%: over the 1% coverage bar, so
    // the coverage signal alone would miss it, but 2 segments in 15 minutes
    // with 3% speech is not a recording.
    const auto v = assess({{5.0, 20.0}, {600.0, 615.0}}, 900.0);
    REQUIRE(v.failover);
    REQUIRE(v.coverage > 0.01); // NOT caught by the coverage rule
    REQUIRE(v.coverage < 0.10);
}

TEST_CASE("sparse-but-real speech on a long clip is preserved", "[vad-failover]") {
    // The genuine "long mostly-silent recording" use case: a 30-minute capture
    // with 90 s of speech spread over 30 segments = 5%. Above the 1% bar and
    // too many segments for the few-segment rule. Must NOT fail over — this is
    // exactly what a user runs VAD *for*.
    const auto v = assess(spread(30, 3.0, 1800.0), 1800.0);
    REQUIRE_FALSE(v.failover);
    REQUIRE(v.coverage > 0.01);
    REQUIRE(v.coverage < 0.10);
}

TEST_CASE("unknown duration never fails over", "[vad-failover]") {
    REQUIRE_FALSE(assess({}, 0.0).failover);
    REQUIRE_FALSE(assess({}, -1.0).failover);
}

TEST_CASE("degenerate spans do not inflate coverage", "[vad-failover]") {
    // Zero-length and inverted spans contribute nothing rather than negative
    // time, so a broken VAD emitting them still trips the coverage rule.
    std::vector<Span> v = {{10.0, 10.0}, {20.0, 15.0}, {30.0, 31.0}};
    const auto a = assess(v, 900.0);
    REQUIRE(a.speech_sec == 1.0);
    REQUIRE(a.failover);
}

TEST_CASE("thresholds are configurable", "[vad-failover]") {
    // Proof the predicate reads its config rather than hard-coding: the healthy
    // 30-minute interview flips to failover under an absurd coverage bar.
    Config strict;
    strict.min_coverage = 0.99;
    REQUIRE(assess(spread(200, 5.0, 1800.0), 1800.0, strict).failover);

    // ...and the true failure stops firing when the floor is raised past it.
    Config lax;
    lax.min_audio_sec = 7200.0;
    REQUIRE_FALSE(assess({}, 900.0, lax).failover);
}

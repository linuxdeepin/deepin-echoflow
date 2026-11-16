// test-audio-window.cpp — unit tests for the shared --offset-t / --duration
// window math (core/audio_window.h, #91). Pure arithmetic; no model, no audio.

#include "core/audio_window.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using core_audio_window::compute;
using core_audio_window::trim;

// 10 s @ 16 kHz = 160000 samples.
static constexpr int SR = 16000;
static constexpr int64_t TEN_S = 160000;

TEST_CASE("no offset/duration -> inactive window", "[unit][audio-window]") {
    auto w = compute(TEN_S, 0, 0, SR);
    REQUIRE_FALSE(w.active);
    REQUIRE_FALSE(w.past_end);

    // trim() is a no-op on an inactive window.
    std::vector<float> v(TEN_S, 1.0f);
    trim(v, w);
    REQUIRE(v.size() == (size_t)TEN_S);
}

TEST_CASE("offset only -> window from offset to end", "[unit][audio-window]") {
    auto w = compute(TEN_S, 5000, 0, SR); // skip first 5 s
    REQUIRE(w.active);
    REQUIRE_FALSE(w.past_end);
    REQUIRE(w.start == 80000);       // 5 s
    REQUIRE(w.len == TEN_S - 80000); // to the end

    std::vector<float> v(TEN_S, 1.0f);
    trim(v, w);
    REQUIRE(v.size() == 80000);
}

TEST_CASE("offset + duration -> bounded window", "[unit][audio-window]") {
    auto w = compute(TEN_S, 4000, 3000, SR); // 4 s .. 7 s
    REQUIRE(w.active);
    REQUIRE(w.start == 64000); // 4 s
    REQUIRE(w.len == 48000);   // 3 s

    std::vector<float> v(TEN_S, 1.0f);
    trim(v, w);
    REQUIRE(v.size() == 48000);
}

TEST_CASE("duration only -> window from start", "[unit][audio-window]") {
    auto w = compute(TEN_S, 0, 2000, SR); // first 2 s
    REQUIRE(w.active);
    REQUIRE(w.start == 0);
    REQUIRE(w.len == 32000);
}

TEST_CASE("duration past end is clamped to the tail", "[unit][audio-window]") {
    auto w = compute(TEN_S, 8000, 5000, SR); // 8 s + 5 s, but only 2 s left
    REQUIRE(w.active);
    REQUIRE(w.start == 128000);       // 8 s
    REQUIRE(w.len == TEN_S - 128000); // clamped to 2 s
    REQUIRE_FALSE(w.past_end);
}

TEST_CASE("offset at or past end -> past_end", "[unit][audio-window]") {
    REQUIRE(compute(TEN_S, 10000, 0, SR).past_end); // exactly at end
    REQUIRE(compute(TEN_S, 20000, 0, SR).past_end); // beyond end
    // still marked active (a window was requested), just empty.
    REQUIRE(compute(TEN_S, 20000, 0, SR).active);
}

TEST_CASE("empty / zero-rate buffer with a window request -> past_end", "[unit][audio-window]") {
    REQUIRE(compute(0, 1000, 0, SR).past_end);
    REQUIRE(compute(TEN_S, 1000, 0, 0).past_end);
}

TEST_CASE("trim re-clamps to a shorter channel buffer", "[unit][audio-window]") {
    // A window computed against the primary buffer must stay in-bounds when
    // applied to a stereo channel that happens to be shorter.
    auto w = compute(TEN_S, 5000, 2000, SR); // start 80000, len 32000
    std::vector<float> shortch(90000, 1.0f); // shorter than start+len
    trim(shortch, w);
    // start 80000 < 90000, so 10000 samples remain (90000 - 80000), not 32000.
    REQUIRE(shortch.size() == 10000);
}

TEST_CASE("trim on empty vector is safe", "[unit][audio-window]") {
    auto w = compute(TEN_S, 5000, 0, SR);
    std::vector<float> empty;
    trim(empty, w);
    REQUIRE(empty.empty());
}

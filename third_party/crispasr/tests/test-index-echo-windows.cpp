#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "core/index_echo_windows.h"

using Catch::Approx;
using core_index_echo::windows;

TEST_CASE("Index-Echo speech windows preserve padding and source offsets", "[unit][index-echo]") {
    auto out = windows({{1, 3}, {4, 6}}, 8);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0].start == Approx(0.7));
    REQUIRE(out[0].end == Approx(6.5));
    REQUIRE_FALSE(out[0].hard_cut);
}
TEST_CASE("Index-Echo long speech cuts cannot be merged across the hard boundary", "[unit][index-echo]") {
    auto out = windows({{0, 125}}, 126);
    REQUIRE(out.size() == 3);
    REQUIRE(out[0].start == 0);
    REQUIRE(out[0].end == 60);
    REQUIRE(out[1].start == 60);
    REQUIRE(out[1].end == 120);
    REQUIRE(out[2].start == 120);
    REQUIRE(out[2].end == Approx(125.5));
    REQUIRE(out[0].hard_cut);
    REQUIRE(out[1].hard_cut);
    REQUIRE_FALSE(out[2].hard_cut);
}
TEST_CASE("Index-Echo short final window folds only when the combined span fits", "[unit][index-echo]") {
    auto out = windows({{0, 55}, {58, 59.8}}, 60);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0].end == 60);
    auto separate = windows({{0, 55}, {62, 63}}, 64);
    REQUIRE(separate.size() == 2);
    REQUIRE(separate[1].start == Approx(61.7));
}
TEST_CASE("Index-Echo silence has no windows", "[unit][index-echo]") {
    REQUIRE(windows({}, 120).empty());
}

TEST_CASE("Index-Echo native VAD spans match released Silero timestamp fixtures", "[unit][index-echo]") {
    // Expected values captured from v6.2 get_speech_timestamps with infer.py's
    // 300 ms silence and return_seconds=True; the classifier is held fixed.
    std::vector<float> probabilities(43, 0.1f);
    std::fill(probabilities.begin() + 5, probabilities.begin() + 13, 0.9f);
    std::fill(probabilities.begin() + 24, probabilities.begin() + 32, 0.9f);
    auto spans = core_index_echo::speech_spans(probabilities.data(), probabilities.size(), 43 * 512 - 197);
    REQUIRE(spans.size() == 2);
    CHECK(spans[0].first == Approx(0.1));
    CHECK(spans[0].second == Approx(0.4));
    CHECK(spans[1].first == Approx(0.7));
    CHECK(spans[1].second == Approx(1.1));

    probabilities.assign(36, 0.1f);
    std::fill(probabilities.begin(), probabilities.begin() + 8, 0.9f);
    std::fill(probabilities.begin() + 17, probabilities.begin() + 25, 0.9f);
    spans = core_index_echo::speech_spans(probabilities.data(), probabilities.size(), 36 * 512);
    REQUIRE(spans.size() == 1);
    CHECK(spans[0].first == 0);
    CHECK(spans[0].second == Approx(0.8));

    probabilities.assign(20, 0.1f);
    std::fill(probabilities.begin() + 8, probabilities.end(), 0.9f);
    spans = core_index_echo::speech_spans(probabilities.data(), probabilities.size(), 20 * 512 - 197);
    REQUIRE(spans.size() == 1);
    CHECK(spans[0].first == Approx(0.2));
    CHECK(spans[0].second == Approx(0.6));

    probabilities.assign(18, 0.1f);
    std::fill(probabilities.begin(), probabilities.begin() + 7, 0.9f);
    CHECK(core_index_echo::speech_spans(probabilities.data(), probabilities.size(), 18 * 512).empty());
    probabilities.assign(100, 0.1f);
    CHECK(core_index_echo::speech_spans(probabilities.data(), probabilities.size(), 100 * 512).empty());
}

TEST_CASE("Index-Echo VAD rounding preserves Python binary-value ties", "[unit][index-echo]") {
    CHECK(core_index_echo::speech_seconds(4000) == Approx(0.2)); // round(0.25, 1)
    CHECK(core_index_echo::speech_seconds(5600) == Approx(0.3)); // round(0.35, 1)
    CHECK(core_index_echo::speech_seconds(8800) == Approx(0.6)); // round(0.55, 1)
}

TEST_CASE("Index-Echo window extraction matches Python ffmpeg millisecond formatting", "[unit][index-echo]") {
    CHECK(core_index_echo::window_samples(1.0005) == 16000);
    CHECK(core_index_echo::window_samples(1.2345) == 19744);
    CHECK(core_index_echo::window_samples(1.2355) == 19776);
    CHECK(core_index_echo::window_samples(0) == 0);
}

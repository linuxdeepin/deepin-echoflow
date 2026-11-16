#include "crispasr_lid.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

TEST_CASE("LID speech prefix compacts VAD slices", "[unit][lid]") {
    std::vector<float> pcm(100, 0.0f);
    std::fill(pcm.begin() + 10, pcm.begin() + 20, 1.0f);
    std::fill(pcm.begin() + 40, pcm.begin() + 50, 2.0f);
    const std::vector<crispasr_audio_slice> slices = {{10, 20, 0, 0}, {40, 50, 0, 0}};

    const auto got = crispasr_lid_speech_prefix(pcm, slices);
    REQUIRE(got.size() == 10 + 1600 + 10);
    CHECK(std::all_of(got.begin(), got.begin() + 10, [](float v) { return v == 1.0f; }));
    CHECK(std::all_of(got.begin() + 10, got.begin() + 1610, [](float v) { return v == 0.0f; }));
    CHECK(std::all_of(got.begin() + 1610, got.end(), [](float v) { return v == 2.0f; }));
}

TEST_CASE("LID speech prefix clamps slices and caps at 15 seconds", "[unit][lid]") {
    std::vector<float> pcm(16 * 16000, 3.0f);
    const std::vector<crispasr_audio_slice> slices = {{-10, 16 * 16000 + 10, 0, 0}};

    const auto got = crispasr_lid_speech_prefix(pcm, slices);
    REQUIRE(got.size() == 15 * 16000);
    CHECK(std::all_of(got.begin(), got.end(), [](float v) { return v == 3.0f; }));
}

TEST_CASE("LID speech prefix ignores empty slices", "[unit][lid]") {
    const std::vector<float> pcm(10, 1.0f);
    const std::vector<crispasr_audio_slice> slices = {{4, 4, 0, 0}, {20, 30, 0, 0}};
    CHECK(crispasr_lid_speech_prefix(pcm, slices).empty());
}

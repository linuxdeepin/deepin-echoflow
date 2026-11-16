// test-generation-health.cpp — unit tests for the generation-health regression gate.
// Tests the objective quality checks in core/generation_health.h without needing
// any model GGUF — pure string/numeric logic.

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "core/generation_health.h"

using namespace core_generation_health;

// ---- check_not_empty ----

TEST_CASE("gen-health: non-empty text passes", "[unit][gen-health]") {
    auto r = check_not_empty("Hello world");
    REQUIRE(r.pass);
}

TEST_CASE("gen-health: empty string fails", "[unit][gen-health]") {
    auto r = check_not_empty("");
    REQUIRE_FALSE(r.pass);
    REQUIRE(r.detail.find("empty") != std::string::npos);
}

TEST_CASE("gen-health: null fails", "[unit][gen-health]") {
    auto r = check_not_empty(nullptr);
    REQUIRE_FALSE(r.pass);
}

TEST_CASE("gen-health: whitespace-only fails", "[unit][gen-health]") {
    auto r = check_not_empty("   \n\t  ");
    REQUIRE_FALSE(r.pass);
    REQUIRE(r.detail.find("whitespace") != std::string::npos);
}

// ---- check_duration_plausibility ----

TEST_CASE("gen-health: plausible duration passes", "[unit][gen-health]") {
    // "And so my fellow Americans" ≈ 26 chars, 11s audio → ~2.4 chars/sec
    auto r = check_duration_plausibility("And so my fellow Americans", 11.0f);
    REQUIRE(r.pass);
}

TEST_CASE("gen-health: too-short transcript fails", "[unit][gen-health]") {
    // 2 chars for 60s audio → 0.03 chars/sec
    auto r = check_duration_plausibility("Hi", 60.0f);
    REQUIRE_FALSE(r.pass);
    REQUIRE(r.detail.find("too short") != std::string::npos);
}

TEST_CASE("gen-health: too-long transcript fails", "[unit][gen-health]") {
    // 5000 chars for 1s audio → 5000 chars/sec
    std::string long_text(5000, 'x');
    auto r = check_duration_plausibility(long_text.c_str(), 1.0f);
    REQUIRE_FALSE(r.pass);
    REQUIRE(r.detail.find("too long") != std::string::npos);
}

// ---- check_no_ngram_loop ----

TEST_CASE("gen-health: normal text has no loop", "[unit][gen-health]") {
    auto r = check_no_ngram_loop("And so my fellow Americans ask not what your country can do for you");
    REQUIRE(r.pass);
}

TEST_CASE("gen-health: repeated word triggers loop", "[unit][gen-health]") {
    auto r = check_no_ngram_loop("hey hey hey hey hey hey this is bad");
    REQUIRE_FALSE(r.pass);
    REQUIRE(r.detail.find("hey") != std::string::npos);
}

TEST_CASE("gen-health: short repetition is OK", "[unit][gen-health]") {
    // 3 repeats is under the default threshold of 4
    auto r = check_no_ngram_loop("yes yes yes no");
    REQUIRE(r.pass);
}

TEST_CASE("gen-health: custom repeat threshold", "[unit][gen-health]") {
    auto r = check_no_ngram_loop("ok ok ok", /*max_consecutive_repeats=*/2);
    REQUIRE_FALSE(r.pass);
}

// ---- check_not_truncated ----

TEST_CASE("gen-health: within max_tokens passes", "[unit][gen-health]") {
    auto r = check_not_truncated(42, 512);
    REQUIRE(r.pass);
}

TEST_CASE("gen-health: at max_tokens fails", "[unit][gen-health]") {
    auto r = check_not_truncated(512, 512);
    REQUIRE_FALSE(r.pass);
    REQUIRE(r.detail.find("truncated") != std::string::npos);
}

// ---- check_tts_duration ----

TEST_CASE("gen-health: plausible TTS duration passes", "[unit][gen-health]") {
    // "Hello world" = 2 words, 0.48s at 48kHz = 23040 samples → 0.24 sec/word
    auto r = check_tts_duration(23040, 48000, 2);
    REQUIRE(r.pass);
}

TEST_CASE("gen-health: TTS too short fails", "[unit][gen-health]") {
    // 2 words, 480 samples at 48kHz = 0.01s → 0.005 sec/word
    auto r = check_tts_duration(480, 48000, 2);
    REQUIRE_FALSE(r.pass);
    REQUIRE(r.detail.find("too short") != std::string::npos);
}

TEST_CASE("gen-health: TTS too long fails", "[unit][gen-health]") {
    // 1 word, 480000 samples at 48kHz = 10s → 10 sec/word
    auto r = check_tts_duration(480000, 48000, 1);
    REQUIRE_FALSE(r.pass);
    REQUIRE(r.detail.find("too long") != std::string::npos);
}

// ---- check_trailing_silence ----

// A 220 Hz tone (|amp| 0.3, RMS ~0.21 >> the 1e-3 silence floor) for `speech_s`,
// then `silence_s` of zeros — a synthetic "speech + trailing dead air" clip.
static std::vector<float> make_tone_then_silence(float speech_s, float silence_s, int sr) {
    std::vector<float> v;
    int ns = (int)(speech_s * sr), nz = (int)(silence_s * sr);
    v.reserve(ns + nz);
    for (int i = 0; i < ns; ++i)
        v.push_back(0.3f * std::sin(2.0f * 3.14159265f * 220.0f * (float)i / (float)sr));
    v.insert(v.end(), nz, 0.0f);
    return v;
}

TEST_CASE("gen-health: short trailing silence passes", "[unit][gen-health]") {
    auto pcm = make_tone_then_silence(1.0f, 0.5f, 24000); // 0.5 s tail < 2 s default
    auto r = check_trailing_silence(pcm.data(), (int)pcm.size(), 24000);
    REQUIRE(r.pass);
}

TEST_CASE("gen-health: long trailing silence fails", "[unit][gen-health]") {
    auto pcm = make_tone_then_silence(1.0f, 3.0f, 24000); // 3 s tail > 2 s default
    auto r = check_trailing_silence(pcm.data(), (int)pcm.size(), 24000);
    REQUIRE_FALSE(r.pass);
    REQUIRE(r.detail.find("trailing silence") != std::string::npos);
}

TEST_CASE("gen-health: all-silence audio fails trailing-silence", "[unit][gen-health]") {
    std::vector<float> pcm(24000 * 3, 0.0f); // 3 s of pure silence
    auto r = check_trailing_silence(pcm.data(), (int)pcm.size(), 24000);
    REQUIRE_FALSE(r.pass);
}

TEST_CASE("gen-health: trailing-silence null/zero is skipped", "[unit][gen-health]") {
    REQUIRE(check_trailing_silence(nullptr, 0, 24000).pass);
    std::vector<float> pcm(100, 0.3f);
    REQUIRE(check_trailing_silence(pcm.data(), (int)pcm.size(), /*sample_rate=*/0).pass);
}

// ---- multilingual (en/fr/de) robustness ----
// The char/sec and byte-wise word-loop heuristics are language-agnostic; these
// pin that a drift toward English-only tuning would fail.

TEST_CASE("gen-health: French transcript plausible duration passes", "[unit][gen-health]") {
    // ~41 chars over 4 s → ~10 chars/sec, inside the 1-50 band
    auto r = check_duration_plausibility("Bonjour tout le monde, comment allez-vous", 4.0f);
    REQUIRE(r.pass);
}

TEST_CASE("gen-health: German compound-word transcript passes", "[unit][gen-health]") {
    auto r = check_duration_plausibility("Die Donaudampfschifffahrtsgesellschaft faehrt heute", 4.0f);
    REQUIRE(r.pass);
}

TEST_CASE("gen-health: accented repeated word triggers loop (UTF-8)", "[unit][gen-health]") {
    // Multibyte accents compare fine byte-wise; 5 repeats > default threshold 4
    auto r = check_no_ngram_loop("café café café café café");
    REQUIRE_FALSE(r.pass);
}

TEST_CASE("gen-health: German repeated word triggers loop", "[unit][gen-health]") {
    auto r = check_no_ngram_loop("ja ja ja ja ja das ist schlecht");
    REQUIRE_FALSE(r.pass);
}

TEST_CASE("gen-health: mixed-language normal text has no loop", "[unit][gen-health]") {
    auto r = check_no_ngram_loop("Ich möchte un café s'il vous plaît bitte");
    REQUIRE(r.pass);
}

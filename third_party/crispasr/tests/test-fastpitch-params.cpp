// test-fastpitch-params.cpp — unit tests for fastpitch_tts_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "fastpitch_tts.h"

TEST_CASE("fastpitch_params: default values are sensible", "[unit][fastpitch]") {
    struct fastpitch_tts_params p = fastpitch_tts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). FastPitch's
// prosody knobs default to neutral (pace 1.0 = unchanged speed, pitch_shift 0.0 =
// no shift). Pin them so a drift can't silently alter every synthesis.
TEST_CASE("fastpitch_params: value knobs default to neutral", "[unit][fastpitch]") {
    struct fastpitch_tts_params p = fastpitch_tts_default_params();

    REQUIRE(p.pace == Catch::Approx(1.0f));        // 1.0 = unchanged duration
    REQUIRE(p.pitch_shift == Catch::Approx(0.0f)); // 0.0 = no pitch shift
    REQUIRE(p.speaker_id == 0);
}

TEST_CASE("fastpitch_init_from_file: null path returns nullptr", "[unit][fastpitch]") {
    struct fastpitch_tts_params p = fastpitch_tts_default_params();
    struct fastpitch_tts_context* ctx = fastpitch_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("fastpitch_init_from_file: empty path returns nullptr", "[unit][fastpitch]") {
    struct fastpitch_tts_params p = fastpitch_tts_default_params();
    struct fastpitch_tts_context* ctx = fastpitch_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("fastpitch_free: NULL context is a no-op", "[unit][fastpitch]") {
    fastpitch_tts_free(nullptr);
    SUCCEED("fastpitch_tts_free tolerated a NULL ctx.");
}

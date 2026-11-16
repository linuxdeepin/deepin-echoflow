// test-speecht5-params.cpp — unit tests for speecht5_tts_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "speecht5_tts.h"

TEST_CASE("speecht5_params: default values are sensible", "[unit][speecht5]") {
    struct speecht5_tts_params p = speecht5_tts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). SpeechT5's
// spectrogram stop-threshold (0.5 upstream) controls when generation ends —
// drifting it truncates or over-runs every utterance. Pin it plus the sentinels.
TEST_CASE("speecht5_params: value knobs match upstream defaults", "[unit][speecht5]") {
    struct speecht5_tts_params p = speecht5_tts_default_params();

    REQUIRE(p.threshold == Catch::Approx(0.5f)); // stop-token probability threshold
    REQUIRE(p.max_len == 0);                     // 0 = model default
    REQUIRE(p.seed == 0);
}

TEST_CASE("speecht5_init: null path returns nullptr", "[unit][speecht5]") {
    struct speecht5_tts_params p = speecht5_tts_default_params();
    struct speecht5_tts_context* ctx = speecht5_tts_init(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("speecht5_init: empty path returns nullptr", "[unit][speecht5]") {
    struct speecht5_tts_params p = speecht5_tts_default_params();
    struct speecht5_tts_context* ctx = speecht5_tts_init("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("speecht5_free: NULL context is a no-op", "[unit][speecht5]") {
    speecht5_tts_free(nullptr);
    SUCCEED("speecht5_tts_free tolerated a NULL ctx.");
}

// test-parlertts-params.cpp — unit tests for parler_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "parler_tts.h"

TEST_CASE("parler_tts_params: default values are sensible", "[unit][parler_tts]") {
    struct parler_tts_context_params p = parler_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). Parler ships
// upstream sampling temperature 1.0 with top_k disabled; pin them so a drift fails
// CI.
TEST_CASE("parler_tts_params: value knobs match upstream defaults", "[unit][parler_tts]") {
    struct parler_tts_context_params p = parler_tts_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(1.0f)); // upstream Parler default
    REQUIRE(p.top_k == 0);                         // 0 = disabled
    REQUIRE(p.seed == 0);
    REQUIRE(p.max_audio_tokens == 0); // 0 = model default
    REQUIRE(p.flash_attn == false);
}

TEST_CASE("parler_tts_init_from_file: null path returns nullptr", "[unit][parler_tts]") {
    struct parler_tts_context_params p = parler_tts_context_default_params();
    struct parler_tts_context* ctx = parler_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("parler_tts_init_from_file: empty path returns nullptr", "[unit][parler_tts]") {
    struct parler_tts_context_params p = parler_tts_context_default_params();
    struct parler_tts_context* ctx = parler_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("parler_tts_free: NULL context is a no-op", "[unit][parler_tts]") {
    parler_tts_free(nullptr);
    SUCCEED("parler_tts_free tolerated a NULL ctx.");
}

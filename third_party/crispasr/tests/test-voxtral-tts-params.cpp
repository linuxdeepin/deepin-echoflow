// test-voxtral-tts-params.cpp — unit tests for voxtral_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "voxtral_tts.h"

TEST_CASE("voxtral_tts_params: default values are sensible", "[unit][voxtral-tts]") {
    struct voxtral_tts_context_params p = voxtral_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    // 0 selects the built-in defaults (8 ODE steps, CFG alpha 1.2).
    REQUIRE(p.n_ode_steps >= 0);
    REQUIRE(p.cfg_alpha >= 0.0f);
}

TEST_CASE("voxtral_tts_sample_rate: 24 kHz", "[unit][voxtral-tts]") {
    REQUIRE(voxtral_tts_sample_rate() == 24000);
}

TEST_CASE("voxtral_tts_init_from_file: null path returns nullptr", "[unit][voxtral-tts]") {
    struct voxtral_tts_context_params p = voxtral_tts_context_default_params();
    struct voxtral_tts_context* ctx = voxtral_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxtral_tts_init_from_file: empty path returns nullptr", "[unit][voxtral-tts]") {
    struct voxtral_tts_context_params p = voxtral_tts_context_default_params();
    struct voxtral_tts_context* ctx = voxtral_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxtral_tts_free: NULL context is a no-op", "[unit][voxtral-tts]") {
    voxtral_tts_free(nullptr);
    SUCCEED("voxtral_tts_free tolerated a NULL ctx.");
}

TEST_CASE("voxtral_tts_set_seed / pcm_free: NULL ctx tolerated", "[unit][voxtral-tts]") {
    voxtral_tts_set_seed(nullptr, 42);
    voxtral_tts_pcm_free(nullptr);
    SUCCEED("NULL-guarded APIs tolerated a NULL ctx.");
}

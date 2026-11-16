// test-f5tts-params.cpp — unit tests for f5_tts_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "f5_tts.h"

TEST_CASE("f5tts_params: default values are sensible", "[unit][f5tts]") {
    struct f5_tts_params p = f5_tts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.speed > 0.0f);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). F5-TTS reads its
// sampler knobs from the GGUF (0 = "use the model-stored value"), so the library
// default deliberately ships the 0-sentinels; the upstream SWivid/F5-TTS values
// they resolve to are nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0.
// Pin the sentinels + the concrete speed/seed so a drift (e.g. someone hard-codes
// a non-zero override into the default) fails CI.
TEST_CASE("f5tts_params: value knobs match the shipped/upstream contract", "[unit][f5tts]") {
    struct f5_tts_params p = f5_tts_default_params();

    REQUIRE(p.speed == Catch::Approx(1.0f));        // upstream F5-TTS speed 1.0
    REQUIRE(p.seed == 42);                          // deterministic default noise seed
    REQUIRE(p.ode_steps == 0);                      // sentinel → GGUF nfe_step (upstream 32)
    REQUIRE(p.cfg_strength == Catch::Approx(0.0f)); // sentinel → GGUF cfg_strength (upstream 2.0)
    REQUIRE(p.sway_coef == Catch::Approx(0.0f));    // sentinel → GGUF sway coef (upstream -1.0)
    REQUIRE(p.use_gpu == false);
}

TEST_CASE("f5tts_init_from_file: null path returns nullptr", "[unit][f5tts]") {
    struct f5_tts_params p = f5_tts_default_params();
    struct f5_tts_context* ctx = f5_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("f5tts_init_from_file: empty path returns nullptr", "[unit][f5tts]") {
    struct f5_tts_params p = f5_tts_default_params();
    struct f5_tts_context* ctx = f5_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("f5tts_free: NULL context is a no-op", "[unit][f5tts]") {
    f5_tts_free(nullptr);
    SUCCEED("f5_tts_free tolerated a NULL ctx.");
}

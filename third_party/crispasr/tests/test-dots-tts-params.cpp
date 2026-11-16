// test-dots-tts-params.cpp — unit tests for dots_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dots_tts.h"

TEST_CASE("dots_tts_params: default values are sensible", "[unit][dots-tts]") {
    struct dots_tts_context_params p = dots_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature >= 0.0f);
    REQUIRE(p.seed > 0);          // 0 means "use default 42"; the default itself is concrete
    REQUIRE(p.max_patches > 0);   // bounded generation
    REQUIRE(p.ode_steps > 0);     // flow-matching needs at least one step
    REQUIRE(p.cfg_scale >= 1.0f); // CFG scale of 1 = no guidance; default amplifies
    REQUIRE(p.eos_threshold > 0.0f);
    REQUIRE(p.eos_threshold <= 1.0f);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). The "sensible"
// case above only bounds these knobs; a drift that stays in-range (ode_steps
// 16→8, cfg_scale 3.0→1.5) would slip through and silently change every
// synthesis. Pin the exact upstream dots.tts inference defaults (documented in
// dots_tts.h) so any change to a shipped default has to update this test on
// purpose.
TEST_CASE("dots_tts_params: value knobs match upstream inference defaults", "[unit][dots-tts]") {
    struct dots_tts_context_params p = dots_tts_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(0.7f));
    REQUIRE(p.seed == 42u);
    REQUIRE(p.max_patches == 256);                   // ≈40 s of audio patches
    REQUIRE(p.ode_steps == 16);                      // flow-matching ODE steps
    REQUIRE(p.cfg_scale == Catch::Approx(3.0f));     // classifier-free guidance scale
    REQUIRE(p.eos_threshold == Catch::Approx(0.8f)); // stop-prob threshold
}

TEST_CASE("dots_tts_init_from_file: null path returns nullptr", "[unit][dots-tts]") {
    struct dots_tts_context_params p = dots_tts_context_default_params();
    struct dots_tts_context* ctx = dots_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("dots_tts_init_from_file: empty path returns nullptr", "[unit][dots-tts]") {
    struct dots_tts_context_params p = dots_tts_context_default_params();
    struct dots_tts_context* ctx = dots_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("dots_tts setters tolerate a NULL context", "[unit][dots-tts]") {
    // Runtime setters must null-guard so wrapper bindings can call them
    // defensively before a context exists without crashing.
    dots_tts_set_n_threads(nullptr, 8);
    dots_tts_set_temperature(nullptr, 0.5f);
    dots_tts_set_seed(nullptr, 123);
    REQUIRE(dots_tts_set_vocoder_path(nullptr, "x.gguf") != 0);
    REQUIRE(dots_tts_set_speaker_path(nullptr, "x.gguf") != 0);
    REQUIRE(dots_tts_set_voice_prompt(nullptr, "x.wav") != 0);
    REQUIRE(dots_tts_set_speaker_pcm(nullptr, nullptr, 0) != 0);
    REQUIRE(dots_tts_set_voice_enabled(nullptr, 1) != 0);
    SUCCEED("dots_tts setters tolerated a NULL ctx.");
}

TEST_CASE("dots_tts_synthesize: NULL context returns nullptr", "[unit][dots-tts]") {
    int n = -1;
    float* pcm = dots_tts_synthesize(nullptr, "hello", &n);
    REQUIRE(pcm == nullptr);
}

TEST_CASE("dots_tts_free / pcm_free: NULL is a no-op", "[unit][dots-tts]") {
    dots_tts_pcm_free(nullptr);
    dots_tts_free(nullptr);
    SUCCEED("dots_tts_free / dots_tts_pcm_free tolerated NULL.");
}

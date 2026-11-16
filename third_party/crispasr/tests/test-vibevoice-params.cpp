// test-vibevoice-params.cpp — unit tests for vibevoice_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "vibevoice.h"

TEST_CASE("vibevoice_params: default values are sensible", "[unit][vibevoice]") {
    struct vibevoice_context_params p = vibevoice_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197 — a knob silently
// drifting from its upstream value is invisible to crispasr-diff, which pins a
// non-default deterministic mode). Pin the shipped value knobs exactly so any
// future change to the default has to update this test on purpose. Values are the
// documented contract in vibevoice.h + upstream microsoft/VibeVoice.
TEST_CASE("vibevoice_params: value knobs match the shipped/upstream contract", "[unit][vibevoice]") {
    struct vibevoice_context_params p = vibevoice_context_default_params();

    REQUIRE(p.tts_steps == 20);                  // DPM-Solver++ inference steps (header: default 20, min 4)
    REQUIRE(p.cfg_scale == Catch::Approx(0.0f)); // sentinel: 0 → model default (1.3 base) resolved downstream
    REQUIRE(p.max_new_tokens == 0);              // 0 = duration-scaled ASR budget
    REQUIRE(p.seed == 0u);                       // 0 = env/default noise seed
    REQUIRE(p.use_gpu == true);
    REQUIRE(p.flash_attn == true); // PLAN #89 plumbing — must stay on
}

TEST_CASE("vibevoice_params: ASR token budget scales with audio duration", "[unit][vibevoice]") {
    REQUIRE(vibevoice_resolve_max_new_tokens(0, 30 * 24000) == 512);
    REQUIRE(vibevoice_resolve_max_new_tokens(0, 142 * 24000 + 1) == 1144);
}

TEST_CASE("vibevoice_params: explicit ASR token budget overrides duration scaling", "[unit][vibevoice]") {
    REQUIRE(vibevoice_resolve_max_new_tokens(768, 300 * 24000) == 768);
}

TEST_CASE("vibevoice_init_from_file: null path returns nullptr", "[unit][vibevoice]") {
    struct vibevoice_context_params p = vibevoice_context_default_params();
    struct vibevoice_context* ctx = vibevoice_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("vibevoice_init_from_file: empty path returns nullptr", "[unit][vibevoice]") {
    struct vibevoice_context_params p = vibevoice_context_default_params();
    struct vibevoice_context* ctx = vibevoice_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("vibevoice_free: NULL context is a no-op", "[unit][vibevoice]") {
    vibevoice_free(nullptr);
    SUCCEED("vibevoice_free tolerated a NULL ctx.");
}

// test-cosyvoice3-params.cpp — unit tests for cosyvoice3_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "cosyvoice3_tts.h"

TEST_CASE("cosyvoice3_params: default values are sensible", "[unit][cosyvoice3]") {
    struct cosyvoice3_tts_context_params p = cosyvoice3_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature >= 0.0f);
    REQUIRE(p.max_tokens >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). Pin the exact
// upstream CosyVoice repetition-aware-sampling (RAS) defaults — these directly
// shape token selection, so a silent drift changes every synthesis.
TEST_CASE("cosyvoice3_params: RAS knobs match upstream defaults", "[unit][cosyvoice3]") {
    struct cosyvoice3_tts_context_params p = cosyvoice3_tts_context_default_params();

    REQUIRE(p.ras_top_k == 25);
    REQUIRE(p.ras_top_p == Catch::Approx(0.8f));
    REQUIRE(p.ras_win_size == 10);
    REQUIRE(p.ras_tau_r == Catch::Approx(0.1f));
    REQUIRE(p.temperature == Catch::Approx(0.0f)); // 0 = greedy/model default
    REQUIRE(p.seed == 0);                          // 0 → default 42 downstream
    REQUIRE(p.max_tokens == 0);                    // 0 = model default
}

TEST_CASE("cosyvoice3_init_from_file: null path returns nullptr", "[unit][cosyvoice3]") {
    struct cosyvoice3_tts_context_params p = cosyvoice3_tts_context_default_params();
    struct cosyvoice3_tts_context* ctx = cosyvoice3_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("cosyvoice3_init_from_file: empty path returns nullptr", "[unit][cosyvoice3]") {
    struct cosyvoice3_tts_context_params p = cosyvoice3_tts_context_default_params();
    struct cosyvoice3_tts_context* ctx = cosyvoice3_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("cosyvoice3_free: NULL context is a no-op", "[unit][cosyvoice3]") {
    cosyvoice3_tts_free(nullptr);
    SUCCEED("cosyvoice3_tts_free tolerated a NULL ctx.");
}

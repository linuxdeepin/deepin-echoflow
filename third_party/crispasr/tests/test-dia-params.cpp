// test-dia-params.cpp — unit tests for dia_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "dia_tts.h"
#include "dia_sampling.h"

TEST_CASE("dia_params: default values are sensible", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). Pin the shipped
// Dia sampling defaults (cfg_scale + nucleus/top-k gate) so a silent drift fails
// CI. Dia needs CFG for meaningful output, so cfg_scale is the load-bearing knob.
TEST_CASE("dia_params: value knobs match the shipped defaults", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(1.2f));
    REQUIRE(p.cfg_scale == Catch::Approx(3.0f)); // CFG required for coherent output
    REQUIRE(p.top_p == Catch::Approx(0.95f));
    REQUIRE(p.top_k == 45);
    REQUIRE(p.seed == 0);       // 0 = non-deterministic
    REQUIRE(p.max_tokens == 0); // 0 = model default
    REQUIRE(p.flash_attn == false);
}

TEST_CASE("dia_init_from_file: null path returns nullptr", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();
    struct dia_tts_context* ctx = dia_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("dia_init_from_file: empty path returns nullptr", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();
    struct dia_tts_context* ctx = dia_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("dia_free: NULL context is a no-op", "[unit][dia]") {
    dia_tts_free(nullptr);
    SUCCEED("dia_tts_free tolerated a NULL ctx.");
}

// These inputs expose the previous all-zero nucleus weights and missing
// threshold-crossing token. Expectations follow the pinned official sampler.
TEST_CASE("dia_sampling: a dominant token survives nucleus filtering", "[unit][dia]") {
    const float logits[] = {0.0f, -1.0f, 20.0f};
    std::mt19937 rng(42);
    for (int i = 0; i < 512; ++i)
        REQUIRE(dia_sample_token(logits, 3, 1.2f, 0.95f, 0, rng) == 2);
}

TEST_CASE("dia_sampling: nucleus includes its crossing token", "[unit][dia]") {
    const float logits[] = {std::log(0.6f), std::log(0.3f), std::log(0.1f)};
    std::mt19937 rng(42);
    int counts[3] = {};
    for (int i = 0; i < 12000; ++i)
        ++counts[dia_sample_token(logits, 3, 1.0f, 0.7f, 0, rng)];
    REQUIRE(counts[2] == 0);
    REQUIRE(counts[0] / 12000.0 == Catch::Approx(2.0 / 3.0).margin(0.02));
    REQUIRE(counts[1] / 12000.0 == Catch::Approx(1.0 / 3.0).margin(0.02));
}

TEST_CASE("dia_sampling: top-k precedes nucleus and greedy bypasses both", "[unit][dia]") {
    const float logits[] = {2.0f, 1.0f, 0.0f, -1.0f};
    std::mt19937 rng(42);
    int crossing = 0;
    for (int i = 0; i < 1024; ++i) {
        const auto token = dia_sample_token(logits, 4, 1.0f, 0.95f, 2, rng);
        REQUIRE(token < 2);
        crossing += token == 1;
    }
    REQUIRE(crossing > 0);
    REQUIRE(dia_sample_token(logits, 4, 0.0f, 0.1f, 1, rng) == 0);
}

TEST_CASE("dia_params: null generation-limit setter is safe", "[unit][dia]") {
    dia_tts_set_max_tokens(nullptr, 1024);
    SUCCEED();
}

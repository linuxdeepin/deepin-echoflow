// test-indextts-params.cpp — unit tests for indextts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "indextts.h"

TEST_CASE("indextts_params: default values are sensible", "[unit][indextts]") {
    struct indextts_context_params p = indextts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). Pin the exact
// upstream IndexTTS Python defaults (documented inline in indextts.cpp). The
// repetition_penalty=10.0 is flagged "critical for quality" — exactly the kind of
// load-bearing knob a silent drift must not touch unnoticed.
TEST_CASE("indextts_params: value knobs match upstream Python defaults", "[unit][indextts]") {
    struct indextts_context_params p = indextts_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(0.0f)); // 0 = greedy argmax
    REQUIRE(p.top_p == Catch::Approx(0.8f));
    REQUIRE(p.top_k == 30);
    REQUIRE(p.repetition_penalty == Catch::Approx(10.0f)); // critical for quality
    REQUIRE(p.max_mel_tokens == 600);
    REQUIRE(p.seed == 0);
}

TEST_CASE("indextts_init_from_file: null path returns nullptr", "[unit][indextts]") {
    struct indextts_context_params p = indextts_context_default_params();
    struct indextts_context* ctx = indextts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("indextts_init_from_file: empty path returns nullptr", "[unit][indextts]") {
    struct indextts_context_params p = indextts_context_default_params();
    struct indextts_context* ctx = indextts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("indextts_free: NULL context is a no-op", "[unit][indextts]") {
    indextts_free(nullptr);
    SUCCEED("indextts_free tolerated a NULL ctx.");
}

// test-voxtral4b-params.cpp — unit tests for voxtral4b_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "voxtral4b.h"

TEST_CASE("voxtral4b_params: default values are sensible", "[unit][voxtral4b]") {
    struct voxtral4b_context_params p = voxtral4b_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197 + PLAN #89).
// flash_attn must stay on by default; a silent removal (the #89 regression class)
// fails CI here.
TEST_CASE("voxtral4b_params: gpu/flash defaults are pinned", "[unit][voxtral4b]") {
    struct voxtral4b_context_params p = voxtral4b_context_default_params();
    REQUIRE(p.use_gpu == true);
    REQUIRE(p.flash_attn == true);
}

TEST_CASE("voxtral4b_init_from_file: null path returns nullptr", "[unit][voxtral4b]") {
    struct voxtral4b_context_params p = voxtral4b_context_default_params();
    struct voxtral4b_context* ctx = voxtral4b_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxtral4b_init_from_file: empty path returns nullptr", "[unit][voxtral4b]") {
    struct voxtral4b_context_params p = voxtral4b_context_default_params();
    struct voxtral4b_context* ctx = voxtral4b_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxtral4b_free: NULL context is a no-op", "[unit][voxtral4b]") {
    voxtral4b_free(nullptr);
    SUCCEED("voxtral4b_free tolerated a NULL ctx.");
}

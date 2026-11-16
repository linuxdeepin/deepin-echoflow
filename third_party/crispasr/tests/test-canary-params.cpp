// test-canary-params.cpp — unit tests for canary_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "canary.h"

TEST_CASE("canary_params: default values are sensible", "[unit][canary]") {
    struct canary_context_params p = canary_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197 + PLAN #89). Pin
// the shipped use_flash/use_gpu defaults so a silent flip fails CI.
TEST_CASE("canary_params: gpu/flash defaults are pinned", "[unit][canary]") {
    struct canary_context_params p = canary_context_default_params();
    REQUIRE(p.use_gpu == true);
    REQUIRE(p.use_flash == false);
}

TEST_CASE("canary_init_from_file: null path returns nullptr", "[unit][canary]") {
    struct canary_context_params p = canary_context_default_params();
    struct canary_context* ctx = canary_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("canary_init_from_file: empty path returns nullptr", "[unit][canary]") {
    struct canary_context_params p = canary_context_default_params();
    struct canary_context* ctx = canary_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("canary_free: NULL context is a no-op", "[unit][canary]") {
    canary_free(nullptr);
    SUCCEED("canary_free tolerated a NULL ctx.");
}

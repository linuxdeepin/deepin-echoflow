// test-canaryctc-params.cpp — unit tests for canary_ctc_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "canary_ctc.h"

TEST_CASE("canary_ctc_params: default values are sensible", "[unit][canary_ctc]") {
    struct canary_ctc_context_params p = canary_ctc_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("canary_ctc_init_from_file: null path returns nullptr", "[unit][canary_ctc]") {
    struct canary_ctc_context_params p = canary_ctc_context_default_params();
    struct canary_ctc_context* ctx = canary_ctc_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("canary_ctc_init_from_file: empty path returns nullptr", "[unit][canary_ctc]") {
    struct canary_ctc_context_params p = canary_ctc_context_default_params();
    struct canary_ctc_context* ctx = canary_ctc_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("canary_ctc_free: NULL context is a no-op", "[unit][canary_ctc]") {
    canary_ctc_free(nullptr);
    SUCCEED("canary_ctc_free tolerated a NULL ctx.");
}

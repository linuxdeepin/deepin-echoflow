// test-omniasr-params.cpp — unit tests for omniasr_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "omniasr.h"

TEST_CASE("omniasr_params: default values are sensible", "[unit][omniasr]") {
    struct omniasr_context_params p = omniasr_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("omniasr_init_from_file: null path returns nullptr", "[unit][omniasr]") {
    struct omniasr_context_params p = omniasr_context_default_params();
    struct omniasr_context* ctx = omniasr_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("omniasr_init_from_file: empty path returns nullptr", "[unit][omniasr]") {
    struct omniasr_context_params p = omniasr_context_default_params();
    struct omniasr_context* ctx = omniasr_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("omniasr_free: NULL context is a no-op", "[unit][omniasr]") {
    omniasr_free(nullptr);
    SUCCEED("omniasr_free tolerated a NULL ctx.");
}

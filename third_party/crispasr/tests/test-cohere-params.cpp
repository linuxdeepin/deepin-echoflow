// test-cohere-params.cpp — unit tests for cohere_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "cohere.h"

TEST_CASE("cohere_params: default values are sensible", "[unit][cohere]") {
    struct cohere_context_params p = cohere_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("cohere_init_from_file: null path returns nullptr", "[unit][cohere]") {
    struct cohere_context_params p = cohere_context_default_params();
    struct cohere_context* ctx = cohere_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("cohere_init_from_file: empty path returns nullptr", "[unit][cohere]") {
    struct cohere_context_params p = cohere_context_default_params();
    struct cohere_context* ctx = cohere_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("cohere_free: NULL context is a no-op", "[unit][cohere]") {
    cohere_free(nullptr);
    SUCCEED("cohere_free tolerated a NULL ctx.");
}

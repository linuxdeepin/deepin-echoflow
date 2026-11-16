// test-granitenle-params.cpp — unit tests for granite_nle_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "granite_nle.h"

TEST_CASE("granite_nle_params: default values are sensible", "[unit][granite_nle]") {
    struct granite_nle_context_params p = granite_nle_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("granite_nle_init_from_file: null path returns nullptr", "[unit][granite_nle]") {
    struct granite_nle_context_params p = granite_nle_context_default_params();
    struct granite_nle_context* ctx = granite_nle_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("granite_nle_init_from_file: empty path returns nullptr", "[unit][granite_nle]") {
    struct granite_nle_context_params p = granite_nle_context_default_params();
    struct granite_nle_context* ctx = granite_nle_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("granite_nle_free: NULL context is a no-op", "[unit][granite_nle]") {
    granite_nle_free(nullptr);
    SUCCEED("granite_nle_free tolerated a NULL ctx.");
}

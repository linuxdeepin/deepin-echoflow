// test-sensevoice-params.cpp — unit tests for sensevoice_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "sensevoice.h"

TEST_CASE("sensevoice_params: default values are sensible", "[unit][sensevoice]") {
    struct sensevoice_context_params p = sensevoice_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("sensevoice_init_from_file: null path returns nullptr", "[unit][sensevoice]") {
    struct sensevoice_context_params p = sensevoice_context_default_params();
    struct sensevoice_context* ctx = sensevoice_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("sensevoice_init_from_file: empty path returns nullptr", "[unit][sensevoice]") {
    struct sensevoice_context_params p = sensevoice_context_default_params();
    struct sensevoice_context* ctx = sensevoice_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("sensevoice_free: NULL context is a no-op", "[unit][sensevoice]") {
    sensevoice_free(nullptr);
    SUCCEED("sensevoice_free tolerated a NULL ctx.");
}

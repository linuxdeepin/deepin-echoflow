// test-openvoice2-params.cpp — unit tests for openvoice2_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "openvoice2.h"

TEST_CASE("openvoice2_params: default values are sensible", "[unit][openvoice2]") {
    struct openvoice2_context_params p = openvoice2_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("openvoice2_init_from_file: null path returns nullptr", "[unit][openvoice2]") {
    struct openvoice2_context_params p = openvoice2_context_default_params();
    struct openvoice2_context* ctx = openvoice2_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("openvoice2_init_from_file: empty path returns nullptr", "[unit][openvoice2]") {
    struct openvoice2_context_params p = openvoice2_context_default_params();
    struct openvoice2_context* ctx = openvoice2_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("openvoice2_free: NULL context is a no-op", "[unit][openvoice2]") {
    openvoice2_free(nullptr);
    SUCCEED("openvoice2_free tolerated a NULL ctx.");
}

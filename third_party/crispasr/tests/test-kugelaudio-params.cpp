// test-kugelaudio-params.cpp — unit tests for kugelaudio_context_params
// defaults and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "kugelaudio.h"

TEST_CASE("kugelaudio_context_params: default values are sensible", "[unit][kugelaudio]") {
    struct kugelaudio_context_params p = kugelaudio_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("kugelaudio_init_from_file: null path returns nullptr", "[unit][kugelaudio]") {
    struct kugelaudio_context_params p = kugelaudio_context_default_params();
    struct kugelaudio_context* ctx = kugelaudio_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kugelaudio_init_from_file: empty path returns nullptr", "[unit][kugelaudio]") {
    struct kugelaudio_context_params p = kugelaudio_context_default_params();
    struct kugelaudio_context* ctx = kugelaudio_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kugelaudio_free: NULL context is a no-op", "[unit][kugelaudio]") {
    kugelaudio_free(nullptr);
    SUCCEED("kugelaudio_free tolerated a NULL ctx.");
}

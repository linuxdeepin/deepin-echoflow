// test-firered-params.cpp — unit tests for firered_asr_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "firered_asr.h"

TEST_CASE("firered_asr_params: default values are sensible", "[unit][firered]") {
    struct firered_asr_context_params p = firered_asr_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("firered_asr_init_from_file: null path returns nullptr", "[unit][firered]") {
    struct firered_asr_context_params p = firered_asr_context_default_params();
    struct firered_asr_context* ctx = firered_asr_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("firered_asr_init_from_file: empty path returns nullptr", "[unit][firered]") {
    struct firered_asr_context_params p = firered_asr_context_default_params();
    struct firered_asr_context* ctx = firered_asr_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("firered_asr_free: NULL context is a no-op", "[unit][firered]") {
    firered_asr_free(nullptr);
    SUCCEED("firered_asr_free tolerated a NULL ctx.");
}

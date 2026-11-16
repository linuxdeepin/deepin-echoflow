// test-moonshine-streaming-params.cpp — unit tests for
// moonshine_streaming_context_params defaults and null-guard coverage.
// No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "moonshine_streaming.h"

TEST_CASE("moonshine_streaming_context_params: default values are sensible", "[unit][moonshine-streaming]") {
    struct moonshine_streaming_context_params p = moonshine_streaming_context_default_params();
    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("moonshine_streaming_init_from_file: null path returns nullptr", "[unit][moonshine-streaming]") {
    struct moonshine_streaming_context_params p = moonshine_streaming_context_default_params();
    struct moonshine_streaming_context* ctx = moonshine_streaming_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("moonshine_streaming_init_from_file: empty path returns nullptr", "[unit][moonshine-streaming]") {
    struct moonshine_streaming_context_params p = moonshine_streaming_context_default_params();
    struct moonshine_streaming_context* ctx = moonshine_streaming_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("moonshine_streaming_free: NULL context is a no-op", "[unit][moonshine-streaming]") {
    moonshine_streaming_free(nullptr);
    SUCCEED("moonshine_streaming_free tolerated a NULL ctx.");
}

// test-firered-asr-params.cpp — unit tests for firered_asr_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "firered_asr.h"

TEST_CASE("firered_asr_context_params: default values are sensible", "[unit][firered-asr]") {
    struct firered_asr_context_params p = firered_asr_context_default_params();
    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). FireRedASR
// decodes with beam search; beam_size is the load-bearing decode-policy knob, so
// pin the shipped default (3) to catch a silent drift.
TEST_CASE("firered_asr_context_params: decode knobs match the shipped defaults", "[unit][firered-asr]") {
    struct firered_asr_context_params p = firered_asr_context_default_params();
    REQUIRE(p.beam_size == 3);
    REQUIRE(p.use_gpu == true);
}

TEST_CASE("firered_asr_init_from_file: null path returns nullptr", "[unit][firered-asr]") {
    struct firered_asr_context_params p = firered_asr_context_default_params();
    struct firered_asr_context* ctx = firered_asr_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("firered_asr_init_from_file: empty path returns nullptr", "[unit][firered-asr]") {
    struct firered_asr_context_params p = firered_asr_context_default_params();
    struct firered_asr_context* ctx = firered_asr_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("firered_asr_free: NULL context is a no-op", "[unit][firered-asr]") {
    firered_asr_free(nullptr);
    SUCCEED("firered_asr_free tolerated a NULL ctx.");
}

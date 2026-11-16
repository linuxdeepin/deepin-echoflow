// test-orpheussnac-params.cpp — unit tests for snac_decoder_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "orpheus_snac.h"

TEST_CASE("snac_decoder_params: default values are sensible", "[unit][orpheus_snac]") {
    struct snac_decoder_params p = snac_decoder_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("snac_decoder_init_from_file: null path returns nullptr", "[unit][orpheus_snac]") {
    struct snac_decoder_params p = snac_decoder_default_params();
    struct snac_decoder_ctx* ctx = snac_decoder_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("snac_decoder_init_from_file: empty path returns nullptr", "[unit][orpheus_snac]") {
    struct snac_decoder_params p = snac_decoder_default_params();
    struct snac_decoder_ctx* ctx = snac_decoder_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("snac_decoder_free: NULL context is a no-op", "[unit][orpheus_snac]") {
    snac_decoder_free(nullptr);
    SUCCEED("snac_decoder_free tolerated a NULL ctx.");
}

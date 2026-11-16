// test-wespeaker-params.cpp — unit tests for wespeaker_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>

#include "wespeaker.h"

TEST_CASE("wespeaker_params: defaults are sensible", "[unit][wespeaker]") {
    struct wespeaker_context_params p = wespeaker_context_default_params();
    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Pin the shipped GPU default so a silent flip fails CI.
TEST_CASE("wespeaker_params: gpu default is pinned", "[unit][wespeaker]") {
    struct wespeaker_context_params p = wespeaker_context_default_params();
    REQUIRE(p.use_gpu == false);
}

TEST_CASE("wespeaker_init_from_file: bad paths return nullptr", "[unit][wespeaker]") {
    struct wespeaker_context_params p = wespeaker_context_default_params();
    REQUIRE(wespeaker_init_from_file(nullptr, p) == nullptr);
    REQUIRE(wespeaker_init_from_file("", p) == nullptr);
    REQUIRE(wespeaker_init_from_file("/nonexistent/nope.gguf", p) == nullptr);
}

TEST_CASE("wespeaker_free: NULL context is a no-op", "[unit][wespeaker]") {
    wespeaker_free(nullptr);
    SUCCEED("wespeaker_free tolerated a NULL ctx.");
}

TEST_CASE("wespeaker accessors: NULL context returns zero", "[unit][wespeaker]") {
    REQUIRE(wespeaker_embed_dim(nullptr) == 0);
    REQUIRE(wespeaker_sample_rate(nullptr) == 0);
    REQUIRE(wespeaker_n_mels(nullptr) == 0);
    REQUIRE(wespeaker_min_samples(nullptr) == 0);
}

TEST_CASE("wespeaker entry points: NULL context fails cleanly", "[unit][wespeaker]") {
    const float dummy[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float emb[4] = {0, 0, 0, 0};
    int a = 0, b = 0;
    REQUIRE(wespeaker_compute_fbank(nullptr, dummy, 8, &a, &b) == nullptr);
    REQUIRE(wespeaker_embed(nullptr, dummy, 8, emb) != 0);
    REQUIRE(wespeaker_embed_staged(nullptr, dummy, 8, nullptr, nullptr, emb) != 0);
}

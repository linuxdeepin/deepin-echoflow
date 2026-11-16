// test-melotts-params.cpp — unit tests for melotts_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "melotts.h"

TEST_CASE("melotts_params: default values are sensible", "[unit][melotts]") {
    struct melotts_params p = melotts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). MeloTTS reads
// its VITS knobs from the model (-1 = "use model default"), so the library ships
// the -1 sentinels; pin them so someone hard-coding a concrete override into the
// default (which would override every model's tuned value) fails CI.
TEST_CASE("melotts_params: knobs ship the -1 model-default sentinels", "[unit][melotts]") {
    struct melotts_params p = melotts_default_params();

    REQUIRE(p.noise_scale == Catch::Approx(-1.0f));
    REQUIRE(p.length_scale == Catch::Approx(-1.0f));
    REQUIRE(p.noise_w == Catch::Approx(-1.0f));
    REQUIRE(p.sdp_ratio == Catch::Approx(-1.0f));
    REQUIRE(p.speaker_id == 0);
    REQUIRE(p.seed == 0);
}

TEST_CASE("melotts_init_from_file: null path returns nullptr", "[unit][melotts]") {
    struct melotts_params p = melotts_default_params();
    struct melotts_context* ctx = melotts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("melotts_init_from_file: empty path returns nullptr", "[unit][melotts]") {
    struct melotts_params p = melotts_default_params();
    struct melotts_context* ctx = melotts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("melotts_free: NULL context is a no-op", "[unit][melotts]") {
    melotts_free(nullptr);
    SUCCEED("melotts_free tolerated a NULL ctx.");
}

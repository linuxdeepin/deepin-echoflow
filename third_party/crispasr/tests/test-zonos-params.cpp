// test-zonos-params.cpp — unit tests for zonos_tts_params defaults
// and null-guard coverage on public setters. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "zonos_tts.h"

TEST_CASE("zonos_params: default values are sensible", "[unit][zonos]") {
    struct zonos_tts_params p = zonos_tts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature >= 0.0f);
    REQUIRE(p.cfg_scale > 0.0f);
    REQUIRE(p.max_audio_tokens >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). The "sensible"
// case only bounds these; pin the exact shipped values. Zonos ships raw logits
// (temperature 1.0, equivalent to upstream's min_p=0.1) and REQUIRES cfg_scale>1
// for meaningful output, so cfg_scale=2.0 is load-bearing.
TEST_CASE("zonos_params: value knobs match the shipped defaults", "[unit][zonos]") {
    struct zonos_tts_params p = zonos_tts_default_params();

    REQUIRE(p.temperature == Catch::Approx(1.0f)); // raw logits (upstream min_p=0.1)
    REQUIRE(p.cfg_scale == Catch::Approx(2.0f));   // CFG required for coherent output
    REQUIRE(p.seed == 0);
    REQUIRE(p.max_audio_tokens == 0); // 0 = default (86*30=2580)
    REQUIRE(p.flash_attn == false);
}

TEST_CASE("zonos_init_from_file: null path returns nullptr", "[unit][zonos]") {
    struct zonos_tts_params p = zonos_tts_default_params();
    struct zonos_tts_context* ctx = zonos_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("zonos_init_from_file: empty path returns nullptr", "[unit][zonos]") {
    struct zonos_tts_params p = zonos_tts_default_params();
    struct zonos_tts_context* ctx = zonos_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("zonos setters: NULL context is a no-op", "[unit][zonos]") {
    zonos_tts_set_n_threads(nullptr, 4);
    zonos_tts_set_temperature(nullptr, 0.5f);
    zonos_tts_set_seed(nullptr, 42);
    zonos_tts_set_cfg_scale(nullptr, 2.0f);
    zonos_tts_set_pitch_std(nullptr, 20.0f);
    zonos_tts_set_speaking_rate(nullptr, 15.0f);
    zonos_tts_set_fmax(nullptr, 22050.0f);
    zonos_tts_set_codec_path(nullptr, "/dev/null");
    zonos_tts_set_language(nullptr, "en-us");
    zonos_tts_set_speaker_embedding(nullptr, nullptr, 0);
    zonos_tts_free(nullptr);
    SUCCEED("All zonos setters tolerated a NULL ctx.");
}

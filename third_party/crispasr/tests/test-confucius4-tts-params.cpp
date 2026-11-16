// test-confucius4-tts-params.cpp — unit tests for confucius4_tts_params
// defaults and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "confucius4_tts.h"

TEST_CASE("confucius4_tts_params: default values are sensible", "[unit][confucius4-tts]") {
    struct confucius4_tts_params p = confucius4_tts_default_params();

    REQUIRE(p.n_threads >= 0);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature > 0.0f); // greedy T2S degenerates into repeat loops
    REQUIRE(p.top_p > 0.0f);
    REQUIRE(p.top_p <= 1.0f);
    REQUIRE(p.top_k > 0);
    REQUIRE(p.repetition_penalty >= 1.0f);
    REQUIRE(p.ode_steps > 0);
    REQUIRE(p.cfg_rate > 0.0f); // reference always runs CFG
}

// Defaults-audit guard: pin the exact reference inference recipe
// (confuciustts/cli/inference.py generate() defaults) so any drift in a
// shipped default has to update this test on purpose.
TEST_CASE("confucius4_tts_params: knobs match the reference inference defaults", "[unit][confucius4-tts]") {
    struct confucius4_tts_params p = confucius4_tts_default_params();

    REQUIRE(p.temperature == Catch::Approx(0.8f));
    REQUIRE(p.top_p == Catch::Approx(0.8f));
    REQUIRE(p.top_k == 30);
    REQUIRE(p.repetition_penalty == Catch::Approx(10.0f));
    REQUIRE(p.ode_steps == 25); // S2A flow-matching steps
    REQUIRE(p.cfg_rate == Catch::Approx(0.7f));
    REQUIRE(p.max_semantic_tokens == 0); // 0 = reference max_length semantics
}

TEST_CASE("confucius4_tts_init_from_file: null/empty path returns nullptr", "[unit][confucius4-tts]") {
    struct confucius4_tts_params p = confucius4_tts_default_params();
    REQUIRE(confucius4_tts_init_from_file(nullptr, p) == nullptr);
    REQUIRE(confucius4_tts_init_from_file("", p) == nullptr);
}

TEST_CASE("confucius4_tts setters tolerate a NULL context", "[unit][confucius4-tts]") {
    REQUIRE(confucius4_tts_set_s2a_path(nullptr, "x.gguf") != 0);
    REQUIRE(confucius4_tts_set_vocoder_path(nullptr, "x.gguf") != 0);
    REQUIRE(confucius4_tts_set_voice_path(nullptr, "x.wav") != 0);
    REQUIRE(confucius4_tts_set_speaker(nullptr, nullptr, 0, nullptr) != 0);
    REQUIRE(confucius4_tts_set_conditioning(nullptr, nullptr, 0, nullptr, 0, nullptr, 0, 0) != 0);
    int n = -1;
    REQUIRE(confucius4_tts_synthesize(nullptr, "hi", "en", &n) == nullptr);
    confucius4_tts_pcm_free(nullptr);
    confucius4_tts_free(nullptr);
}

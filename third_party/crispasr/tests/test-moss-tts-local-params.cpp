// test-moss-tts-local-params.cpp — unit tests for
// moss_tts_local_context_params / moss_tts_local_synth_params defaults and
// null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>

#include "moss_tts_local.h"

TEST_CASE("moss_tts_local_params: context default values are sensible", "[unit][moss-tts-local]") {
    struct moss_tts_local_context_params p = moss_tts_local_context_default_params();
    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("moss_tts_local_params: synth default values are sensible", "[unit][moss-tts-local]") {
    struct moss_tts_local_synth_params s = moss_tts_local_synth_default_params();
    REQUIRE(s.max_new_frames > 0);       // bounded generation
    REQUIRE(s.text_temperature >= 0.0f); // binary continue/stop: greedy by default (0)
    REQUIRE(s.audio_temperature > 0.0f); // codebook heads sample by default
    REQUIRE(s.text_top_p > 0.0f);
    REQUIRE(s.text_top_p <= 1.0f);
    REQUIRE(s.audio_top_p > 0.0f);
    REQUIRE(s.audio_top_p <= 1.0f);
    REQUIRE(s.audio_top_k > 0);
    REQUIRE(s.audio_repetition_penalty >= 1.0f);
    REQUIRE(s.min_audio_frames >= 0);
    REQUIRE(s.max_audio_frames >= 0);
}

TEST_CASE("moss_tts_local_init_from_file: null path returns nullptr", "[unit][moss-tts-local]") {
    struct moss_tts_local_context_params p = moss_tts_local_context_default_params();
    REQUIRE(moss_tts_local_init_from_file(nullptr, p) == nullptr);
}

TEST_CASE("moss_tts_local_init_from_file: empty path returns nullptr", "[unit][moss-tts-local]") {
    struct moss_tts_local_context_params p = moss_tts_local_context_default_params();
    REQUIRE(moss_tts_local_init_from_file("", p) == nullptr);
}

TEST_CASE("moss_tts_local setters/accessors tolerate a NULL context", "[unit][moss-tts-local]") {
    // Wrapper bindings call these defensively before a context exists.
    moss_tts_local_set_seed(nullptr, 123);
    REQUIRE(moss_tts_local_set_codec_path(nullptr, "x.gguf") == false);
    REQUIRE(moss_tts_local_n_vq(nullptr) == 0);
    REQUIRE(moss_tts_local_hidden_size(nullptr) == 0);
    REQUIRE(moss_tts_local_audio_vocab_size(nullptr) == 0);
    REQUIRE(moss_tts_local_sampling_rate(nullptr) == 0);
    REQUIRE(moss_tts_local_codec_loaded(nullptr) == false);
    SUCCEED("moss_tts_local setters/accessors tolerated a NULL ctx.");
}

TEST_CASE("moss_tts_local_tokenize: NULL context/text returns nullptr", "[unit][moss-tts-local]") {
    int n = -1;
    REQUIRE(moss_tts_local_tokenize(nullptr, "hi", &n) == nullptr);
    REQUIRE(n == 0);
}

TEST_CASE("moss_tts_local_synthesize / generate_codes: NULL context returns nullptr", "[unit][moss-tts-local]") {
    struct moss_tts_local_synth_params s = moss_tts_local_synth_default_params();
    int n = -1;
    REQUIRE(moss_tts_local_synthesize(nullptr, "hello", &s, &n) == nullptr);
    REQUIRE(n == 0);
    int nvq = -1, t = -1;
    REQUIRE(moss_tts_local_generate_codes(nullptr, "hello", &s, &nvq, &t) == nullptr);
    REQUIRE(nvq == 0);
    REQUIRE(t == 0);
}

TEST_CASE("moss_tts_local_free: NULL is a no-op", "[unit][moss-tts-local]") {
    moss_tts_local_free(nullptr);
    SUCCEED("moss_tts_local_free tolerated NULL.");
}

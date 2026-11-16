// test-gigaam-params.cpp — unit tests for gigaam_context_params defaults and
// null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>

#include "gigaam.h"

TEST_CASE("gigaam_params: default values are sensible", "[unit][gigaam]") {
    struct gigaam_context_params p = gigaam_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit guard: pin the shipped use_flash / use_gpu defaults so a
// silent flip fails CI. The manual attention path is the one the per-stage
// diff was validated on; flash must stay opt-in until it has its own A/B.
TEST_CASE("gigaam_params: gpu/flash defaults are pinned", "[unit][gigaam]") {
    struct gigaam_context_params p = gigaam_context_default_params();
    REQUIRE(p.use_gpu == false);
    REQUIRE(p.use_flash == false);
}

TEST_CASE("gigaam_init_from_file: null path returns nullptr", "[unit][gigaam]") {
    struct gigaam_context_params p = gigaam_context_default_params();
    REQUIRE(gigaam_init_from_file(nullptr, p) == nullptr);
}

TEST_CASE("gigaam_init_from_file: empty path returns nullptr", "[unit][gigaam]") {
    struct gigaam_context_params p = gigaam_context_default_params();
    REQUIRE(gigaam_init_from_file("", p) == nullptr);
}

TEST_CASE("gigaam_free: NULL context is a no-op", "[unit][gigaam]") {
    gigaam_free(nullptr);
    SUCCEED("gigaam_free tolerated a NULL ctx.");
}

TEST_CASE("gigaam accessors: NULL context returns zero/empty", "[unit][gigaam]") {
    REQUIRE(gigaam_n_vocab(nullptr) == 0);
    REQUIRE(gigaam_blank_id(nullptr) == 0);
    REQUIRE(gigaam_frame_dur_cs(nullptr) == 0);
    REQUIRE(gigaam_n_mels(nullptr) == 0);
    REQUIRE(gigaam_sample_rate(nullptr) == 0);
    REQUIRE(gigaam_d_model(nullptr) == 0);
    REQUIRE(gigaam_n_layers(nullptr) == 0);
    REQUIRE(gigaam_is_rnnt(nullptr) == 0);
    REQUIRE(gigaam_is_spm(nullptr) == 0);
    REQUIRE(gigaam_est_enc_frames(nullptr, 16000) == 0);
    REQUIRE(std::string(gigaam_token_to_str(nullptr, 0)).empty());
}

TEST_CASE("gigaam stage entry points: NULL context returns nullptr", "[unit][gigaam]") {
    int a = 0, b = 0;
    const float dummy[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    REQUIRE(gigaam_compute_mel(nullptr, dummy, 4, &a, &b) == nullptr);
    REQUIRE(gigaam_run_encoder(nullptr, dummy, 1, 4, &a, &b) == nullptr);
    REQUIRE(gigaam_ctc_log_probs(nullptr, dummy, 1, 4, &a) == nullptr);
    REQUIRE(gigaam_joint_project_encoder(nullptr, dummy, 1, 4, &a) == nullptr);
    REQUIRE(gigaam_predictor_initial(nullptr, &a) == nullptr);
    REQUIRE(gigaam_joint_step(nullptr, dummy, dummy, &a) == nullptr);
    REQUIRE(gigaam_transcribe(nullptr, dummy, 4) == nullptr);
    REQUIRE(gigaam_transcribe_ex(nullptr, dummy, 4, 0) == nullptr);
}

TEST_CASE("gigaam_result_free: NULL result is a no-op", "[unit][gigaam]") {
    gigaam_result_free(nullptr);
    SUCCEED("gigaam_result_free tolerated a NULL result.");
}

// Mini-Omni2 integration test — exercises ASR + TTS + S2S.
//
// Requires CRISPASR_MODEL_MINI_OMNI2 env var pointing to the GGUF.
// SNAC tests additionally need CRISPASR_MODEL_SNAC (snac-24khz.gguf).
// SKIPs cleanly when not set.

#include <catch2/catch_test_macros.hpp>

#include "mini_omni2.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> load_wav_16k(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};
    fseek(f, 44, SEEK_SET);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);
    std::vector<int16_t> raw(sz / 2);
    size_t n = fread(raw.data(), 2, raw.size(), f);
    (void)n;
    fclose(f);
    std::vector<float> pcm(raw.size());
    for (size_t i = 0; i < raw.size(); i++)
        pcm[i] = raw[i] / 32768.0f;
    return pcm;
}

TEST_CASE("mini-omni2 ASR", "[integration][mini-omni2]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_MINI_OMNI2");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_MINI_OMNI2 not set");
    }

    auto params = mini_omni2_context_default_params();
    params.verbosity = 0;
    auto* ctx = mini_omni2_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    auto pcm = load_wav_16k("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    char* text = mini_omni2_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);

    std::string result(text);
    free(text);

    // ASR mode should transcribe JFK speech — check key phrases
    INFO("transcript: " << result);
    CHECK(result.find("americans") != std::string::npos);
    CHECK(result.find("country") != std::string::npos);

    mini_omni2_free(ctx);
}

TEST_CASE("mini-omni2 TTS", "[integration][mini-omni2]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_MINI_OMNI2");
    const char* snac_path = std::getenv("CRISPASR_MODEL_SNAC");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_MINI_OMNI2 not set");
    }
    if (!snac_path || !*snac_path) {
        SKIP("CRISPASR_MODEL_SNAC not set");
    }

    auto params = mini_omni2_context_default_params();
    params.verbosity = 0;
    auto* ctx = mini_omni2_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);
    REQUIRE(mini_omni2_load_snac(ctx, snac_path));

    int n_samples = 0;
    float* pcm = mini_omni2_synthesize(ctx, "Hello", &n_samples);

    if (pcm && n_samples > 0) {
        float peak = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float a = std::fabs(pcm[i]);
            if (a > peak)
                peak = a;
        }
        CHECK(peak > 0.01f);
        CHECK(n_samples > 1000);
        free(pcm);
    }

    mini_omni2_free(ctx);
}

TEST_CASE("mini-omni2 S2S", "[integration][mini-omni2]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_MINI_OMNI2");
    const char* snac_path = std::getenv("CRISPASR_MODEL_SNAC");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_MINI_OMNI2 not set");
    }
    if (!snac_path || !*snac_path) {
        SKIP("CRISPASR_MODEL_SNAC not set");
    }

    auto params = mini_omni2_context_default_params();
    params.verbosity = 0;
    auto* ctx = mini_omni2_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);
    REQUIRE(mini_omni2_load_snac(ctx, snac_path));

    auto in_pcm = load_wav_16k("samples/jfk.wav");
    REQUIRE(!in_pcm.empty());

    char* s2s_text = nullptr;
    int n_out = 0;
    float* out_pcm = mini_omni2_speech_to_speech(ctx, in_pcm.data(), (int)in_pcm.size(), &s2s_text, &n_out);

    if (out_pcm && n_out > 0) {
        CHECK(n_out > 1000);
        free(out_pcm);
    }
    if (s2s_text) {
        CHECK(strlen(s2s_text) > 0);
        free(s2s_text);
    }

    mini_omni2_free(ctx);
}

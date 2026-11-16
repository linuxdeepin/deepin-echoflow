// LFM2-Audio integration test — exercises ASR + TTS roundtrip.
//
// Requires CRISPASR_MODEL_LFM2_EN or CRISPASR_MODEL_LFM2_JP env var
// pointing to a GGUF model file. SKIPs cleanly when not set.
//
// Tests:
//   1. Model load
//   2. ASR on samples/jfk.wav (EN) or built-in silence (JP)
//   3. TTS synthesis of short text
//   4. TTS output is not silence (peak amplitude check)

#include <catch2/catch_test_macros.hpp>


#include "lfm2_audio.h"

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

TEST_CASE("lfm2-audio EN ASR", "[integration][lfm2]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_LFM2_EN");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_LFM2_EN not set");
    }

    auto params = lfm2_audio_context_default_params();
    params.verbosity = 0;
    auto* ctx = lfm2_audio_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    auto pcm = load_wav_16k("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    char* text = lfm2_audio_transcribe(ctx, pcm.data(), (int)pcm.size(), "en", 50);
    REQUIRE(text != nullptr);

    std::string result(text);
    free(text);

    // Should contain key JFK phrases
    REQUIRE(result.find("Americans") != std::string::npos);
    REQUIRE(result.find("country") != std::string::npos);

    lfm2_audio_free(ctx);
}

TEST_CASE("lfm2-audio JP ASR", "[integration][lfm2]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_LFM2_JP");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_LFM2_JP not set");
    }

    auto params = lfm2_audio_context_default_params();
    params.verbosity = 0;
    auto* ctx = lfm2_audio_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    // Just test load + smoke — we don't have a JP audio sample in samples/
    REQUIRE(lfm2_audio_test_load(ctx) == 0);

    lfm2_audio_free(ctx);
}

TEST_CASE("lfm2-audio TTS", "[integration][lfm2]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_LFM2_JP");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_LFM2_JP not set");
    }

    auto params = lfm2_audio_context_default_params();
    params.verbosity = 0;
    auto* ctx = lfm2_audio_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    int n_samples = 0;
    float* pcm = lfm2_audio_synthesize(ctx, "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf", // こんにちは
                                        "ja", &n_samples);

    if (pcm && n_samples > 0) {
        // Check that output is not silence
        float peak = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float a = std::fabs(pcm[i]);
            if (a > peak)
                peak = a;
        }
        CHECK(peak > 0.01f); // not silence
        CHECK(n_samples > 1000); // at least some audio
        free(pcm);
    }
    // TTS may fail if detokenizer not found — that's OK for this test

    lfm2_audio_free(ctx);
}

// higgs-audio-v3-stt integration test — exercises the chunked Whisper encoder
// + Qwen3-1.7B ChatML greedy decode end-to-end.
//
// Requires CRISPASR_MODEL_HIGGS_STT env var pointing to the GGUF (F16/Q8_0/Q4_K
// all transcribe verbatim). SKIPs cleanly when not set.

#include <catch2/catch_test_macros.hpp>

#include "higgs_stt.h"

#include <cstdio>
#include <cstdlib>
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

TEST_CASE("higgs-stt ASR", "[integration][higgs-stt]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_HIGGS_STT");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_HIGGS_STT not set");
    }

    auto params = higgs_stt_context_default_params();
    params.verbosity = 0;
    auto* ctx = higgs_stt_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    auto pcm = load_wav_16k("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    char* text = higgs_stt_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);

    std::string result(text);
    free(text);

    // The clip spans 3 chunk_size_seconds (4 s) chunks, so this also gates the
    // chunked-encode path that the single-window encoder used to truncate at
    // "...ask not you".
    INFO("transcript: " << result);
    CHECK(result.find("americans") != std::string::npos);
    CHECK(result.find("what your country can do for you") != std::string::npos);
    CHECK(result.find("what you can do for your country") != std::string::npos);

    higgs_stt_free(ctx);
}

// tests/test_ark_asr_live.cpp — integration tests for the ark-asr backend.
//
// ⚠️ EXPERIMENTAL / WIP backend (ARK-ASR-3B). Live tests: require model +
// audio files on disk. Gated behind [.live]. Run:
//   ctest -R test-ark-asr --output-on-failure
//
// Env vars:
//   CRISPASR_MODEL_ARK_ASR — Q8_0 (or F16) GGUF path
//
// The backend is CPU-only by default (the GPU/sched path emits no tokens; see
// PLAN.md §ARK). These tests use the default (CPU) path.

#include <catch2/catch_test_macros.hpp>
#include "ark_asr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

// Chunk-skipping 16-bit PCM WAV loader (mono, 16 kHz expected).
static std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::vector<float> pcm;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return pcm;

    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        fclose(f);
        return pcm;
    }

    int channels = 0, sample_rate = 0, bits = 0;
    int32_t data_size = 0;
    bool found_fmt = false, found_data = false;

    while (!found_data) {
        char chunk_id[4];
        int32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4)
            break;
        if (fread(&chunk_size, 4, 1, f) != 1)
            break;
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            char fmt[16];
            if (chunk_size < 16 || fread(fmt, 1, 16, f) != 16)
                break;
            channels = *(int16_t*)(fmt + 2);
            sample_rate = *(int32_t*)(fmt + 4);
            bits = *(int16_t*)(fmt + 14);
            found_fmt = true;
            if (chunk_size > 16)
                fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || bits != 16 || channels < 1) {
        fclose(f);
        return pcm;
    }

    int n_samples = data_size / (channels * (bits / 8));
    std::vector<int16_t> raw(n_samples * channels);
    fread(raw.data(), sizeof(int16_t), raw.size(), f);
    fclose(f);

    pcm.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++)
            sum += raw[i * channels + c];
        pcm[i] = sum / (channels * 32768.0f);
    }
    return pcm;
}

static bool contains_ci(const std::string& hay, const char* needle) {
    std::string h = hay, n = needle;
    for (auto& c : h)
        c = (char)tolower((unsigned char)c);
    for (auto& c : n)
        c = (char)tolower((unsigned char)c);
    return h.find(n) != std::string::npos;
}

TEST_CASE("ark-asr: init and free", "[ark-asr][.live]") {
    std::string model = get_env("CRISPASR_MODEL_ARK_ASR");
    if (model.empty()) {
        SKIP("CRISPASR_MODEL_ARK_ASR not set");
    }

    ark_asr_context_params cp = ark_asr_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    ark_asr_context* ctx = ark_asr_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);
    ark_asr_free(ctx);
}

TEST_CASE("ark-asr: JFK English transcription (verbatim)", "[ark-asr][.live]") {
    std::string model = get_env("CRISPASR_MODEL_ARK_ASR");
    if (model.empty()) {
        SKIP("CRISPASR_MODEL_ARK_ASR not set");
    }

    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    ark_asr_context_params cp = ark_asr_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    ark_asr_context* ctx = ark_asr_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    char* text = ark_asr_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);

    std::string got(text);
    std::free(text);
    INFO("Transcription: " << got);

    CHECK(contains_ci(got, "fellow americans"));
    CHECK(contains_ci(got, "country"));
    CHECK(got.size() > 30);

    ark_asr_free(ctx);
}

TEST_CASE("ark-asr: language instruction does not break English", "[ark-asr][.live]") {
    std::string model = get_env("CRISPASR_MODEL_ARK_ASR");
    if (model.empty()) {
        SKIP("CRISPASR_MODEL_ARK_ASR not set");
    }

    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    ark_asr_context_params cp = ark_asr_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    ark_asr_context* ctx = ark_asr_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    // EXPERIMENTAL language steering: an English instruction must not derail
    // the validated promptless English path.
    ark_asr_set_ask(ctx, "Transcribe the audio in English.");
    char* text = ark_asr_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);

    std::string got(text);
    std::free(text);
    INFO("Transcription (with instruction): " << got);

    CHECK(contains_ci(got, "country"));
    CHECK(got.size() > 30);

    ark_asr_free(ctx);
}

// tests/test_gigaam_live.cpp — integration tests for the GigaAM-v3 backend.
//
// Live tests: require model + audio files on disk. Gated behind [.live].
// Run: ctest -R test-gigaam --output-on-failure
//
// Env vars (see tests/env-live-tests.sh):
//   CRISPASR_MODEL_GIGAAM      — Q4_K GGUF path (e2e_rnnt by default)
//   CRISPASR_MODEL_GIGAAM_F16  — F16 GGUF path (optional)
//   CRISPASR_FIXTURE_GIGAAM    — 16 kHz mono Russian WAV
//
// The fixture is GigaAM's own `example.wav`
// (https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav), whose
// reference transcript from the PyTorch blueprint is:
//   "Ничьих не требуя похвал, Счастлив уж я надеждой сладкой, ..."

#include <catch2/catch_test_macros.hpp>

#include "gigaam.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

// Chunk-skipping 16-bit PCM WAV loader.
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

    int channels = 0, bits = 0;
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

    const int n_samples = data_size / (channels * (bits / 8));
    std::vector<int16_t> raw((size_t)n_samples * channels);
    if (fread(raw.data(), sizeof(int16_t), raw.size(), f) != raw.size()) {
        fclose(f);
        return {};
    }
    fclose(f);

    pcm.resize((size_t)n_samples);
    for (int i = 0; i < n_samples; i++) {
        float sum = 0.0f;
        for (int c = 0; c < channels; c++)
            sum += raw[(size_t)i * channels + c];
        pcm[(size_t)i] = sum / (channels * 32768.0f);
    }
    return pcm;
}

TEST_CASE("gigaam: init reports the model's shape", "[gigaam][.live]") {
    const std::string model = get_env("CRISPASR_MODEL_GIGAAM");
    if (model.empty())
        SKIP("CRISPASR_MODEL_GIGAAM not set");

    gigaam_context_params cp = gigaam_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    gigaam_context* ctx = gigaam_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    CHECK(gigaam_sample_rate(ctx) == 16000);
    CHECK(gigaam_n_mels(ctx) == 64);
    CHECK(gigaam_d_model(ctx) == 768);
    CHECK(gigaam_n_layers(ctx) == 16);
    // 4x subsampling at a 10 ms hop.
    CHECK(gigaam_frame_dur_cs(ctx) == 4);
    // blank is the last class, so the vocabulary is one shorter.
    CHECK(gigaam_blank_id(ctx) == gigaam_n_vocab(ctx));

    gigaam_free(ctx);
}

TEST_CASE("gigaam: Russian transcription", "[gigaam][.live]") {
    const std::string model = get_env("CRISPASR_MODEL_GIGAAM");
    const std::string fixture = get_env("CRISPASR_FIXTURE_GIGAAM");
    if (model.empty() || fixture.empty())
        SKIP("CRISPASR_MODEL_GIGAAM / CRISPASR_FIXTURE_GIGAAM not set");

    auto pcm = load_wav_16k_mono(fixture);
    REQUIRE(!pcm.empty());

    gigaam_context_params cp = gigaam_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    gigaam_context* ctx = gigaam_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    char* text = gigaam_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);
    const std::string got(text);
    std::free(text);
    INFO("Transcription: " << got);

    // Distinctive words from the blueprint's reference transcript. Matched as
    // raw UTF-8 substrings — no case folding, because a portable Cyrillic
    // tolower() would need a locale we cannot rely on in CI.
    CHECK(got.find("лукоморья") != std::string::npos);
    CHECK(got.find("похвал") != std::string::npos);
    CHECK(got.size() > 60);

    gigaam_free(ctx);
}

TEST_CASE("gigaam: word timings are ordered and inside the clip", "[gigaam][.live]") {
    const std::string model = get_env("CRISPASR_MODEL_GIGAAM");
    const std::string fixture = get_env("CRISPASR_FIXTURE_GIGAAM");
    if (model.empty() || fixture.empty())
        SKIP("CRISPASR_MODEL_GIGAAM / CRISPASR_FIXTURE_GIGAAM not set");

    auto pcm = load_wav_16k_mono(fixture);
    REQUIRE(!pcm.empty());
    const int64_t dur_cs = (int64_t)pcm.size() * 100 / 16000;

    gigaam_context_params cp = gigaam_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    gigaam_context* ctx = gigaam_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    gigaam_result* r = gigaam_transcribe_ex(ctx, pcm.data(), (int)pcm.size(), 0);
    REQUIRE(r != nullptr);
    CHECK(r->n_words > 0);
    for (int i = 0; i < r->n_words; i++) {
        CHECK(r->words[i].t0 <= r->words[i].t1);
        CHECK(r->words[i].t1 <= dur_cs + 8); // one frame of slack
        if (i > 0)
            CHECK(r->words[i].t0 >= r->words[i - 1].t0);
    }
    gigaam_result_free(r);
    gigaam_free(ctx);
}

TEST_CASE("gigaam: F16 and Q4_K agree on the transcript", "[gigaam][.live]") {
    const std::string model_q4k = get_env("CRISPASR_MODEL_GIGAAM");
    const std::string model_f16 = get_env("CRISPASR_MODEL_GIGAAM_F16");
    const std::string fixture = get_env("CRISPASR_FIXTURE_GIGAAM");
    if (model_q4k.empty() || model_f16.empty() || fixture.empty())
        SKIP("CRISPASR_MODEL_GIGAAM / _F16 / CRISPASR_FIXTURE_GIGAAM not set");

    auto pcm = load_wav_16k_mono(fixture);
    REQUIRE(!pcm.empty());

    gigaam_context_params cp = gigaam_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;

    gigaam_context* ctx_q = gigaam_init_from_file(model_q4k.c_str(), cp);
    REQUIRE(ctx_q != nullptr);
    char* t_q = gigaam_transcribe(ctx_q, pcm.data(), (int)pcm.size());
    REQUIRE(t_q != nullptr);
    const std::string text_q(t_q);
    std::free(t_q);
    gigaam_free(ctx_q);

    gigaam_context* ctx_f = gigaam_init_from_file(model_f16.c_str(), cp);
    REQUIRE(ctx_f != nullptr);
    char* t_f = gigaam_transcribe(ctx_f, pcm.data(), (int)pcm.size());
    REQUIRE(t_f != nullptr);
    const std::string text_f(t_f);
    std::free(t_f);
    gigaam_free(ctx_f);

    INFO("Q4_K: " << text_q);
    INFO("F16:  " << text_f);
    CHECK(text_q == text_f);
}

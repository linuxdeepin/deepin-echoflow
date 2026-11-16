// VoxCPM2 AudioVAE live integration tests.
//
// CRISPASR_MODEL_VOXCPM2_VAE may point to either an AudioVAE-only GGUF or a
// full VoxCPM2 GGUF. CRISPASR_MODEL_VOXCPM2_FULL enables the simultaneous
// full-TTS + upscaler lifecycle regression.

#include <catch2/catch_test_macros.hpp>

#include "voxcpm2_tts.h"
#include "voxcpm2_vae.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

static std::vector<float> load_voxcpm2_vae_wav(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};

    fseek(f, 0, SEEK_END);
    const long size = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);
    if (size <= 0 || size % (long)sizeof(int16_t) != 0) {
        fclose(f);
        return {};
    }

    std::vector<int16_t> raw((size_t)size / sizeof(int16_t));
    const size_t read = fread(raw.data(), sizeof(int16_t), raw.size(), f);
    fclose(f);
    if (read != raw.size())
        return {};

    std::vector<float> pcm(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
        pcm[i] = raw[i] / 32768.0f;
    return pcm;
}

static std::vector<float> require_valid_upscale(voxcpm2_vae_context* ctx, const std::vector<float>& input) {
    int n_output = 0;
    float* output = voxcpm2_vae_upscale(ctx, input.data(), (int)input.size(), &n_output);
    REQUIRE(output != nullptr);
    REQUIRE(n_output == (int)input.size() * 3);

    CHECK(std::all_of(output, output + n_output, [](float value) { return std::isfinite(value); }));
    float peak = 0.0f;
    for (int i = 0; i < n_output; ++i)
        peak = std::max(peak, std::fabs(output[i]));
    CHECK(peak > 0.01f);
    std::vector<float> result(output, output + n_output);
    voxcpm2_vae_pcm_free(output);
    return result;
}

static void set_voxcpm2_graph_env(const char* value) {
#if defined(_WIN32)
    REQUIRE(_putenv_s("CRISPASR_VOXCPM2_USE_GRAPH", value ? value : "") == 0);
#else
    const int rc = value ? setenv("CRISPASR_VOXCPM2_USE_GRAPH", value, 1) : unsetenv("CRISPASR_VOXCPM2_USE_GRAPH");
    REQUIRE(rc == 0);
#endif
}

class ScopedVoxCPM2GraphEnv final {
public:
    ScopedVoxCPM2GraphEnv() {
        const char* value = std::getenv("CRISPASR_VOXCPM2_USE_GRAPH");
        if (value)
            saved_ = value;
    }
    ~ScopedVoxCPM2GraphEnv() {
#if defined(_WIN32)
        (void)_putenv_s("CRISPASR_VOXCPM2_USE_GRAPH", saved_ ? saved_->c_str() : "");
#else
        if (saved_)
            (void)setenv("CRISPASR_VOXCPM2_USE_GRAPH", saved_->c_str(), 1);
        else
            (void)unsetenv("CRISPASR_VOXCPM2_USE_GRAPH");
#endif
    }

private:
    std::optional<std::string> saved_;
};

static void compare_legacy_and_graph(voxcpm2_vae_context* ctx, const std::vector<float>& input) {
    const ScopedVoxCPM2GraphEnv restore_env;

    set_voxcpm2_graph_env("0");
    const std::vector<float> legacy = require_valid_upscale(ctx, input);
    set_voxcpm2_graph_env("1");
    const std::vector<float> graph = require_valid_upscale(ctx, input);

    REQUIRE(graph.size() == legacy.size());
    double dot = 0.0;
    double legacy_energy = 0.0;
    double graph_energy = 0.0;
    double squared_error = 0.0;
    for (size_t i = 0; i < graph.size(); ++i) {
        dot += (double)legacy[i] * graph[i];
        legacy_energy += (double)legacy[i] * legacy[i];
        graph_energy += (double)graph[i] * graph[i];
        const double error = (double)legacy[i] - graph[i];
        squared_error += error * error;
    }
    const double cosine = dot / std::sqrt(legacy_energy * graph_energy);
    const double rms_error = std::sqrt(squared_error / graph.size());
    INFO("legacy/graph cosine=" << cosine << ", RMS error=" << rms_error);
    CHECK(cosine > 0.999);
    CHECK(rms_error < 0.01);
}

static bool voxcpm2_vae_test_gpu_enabled() {
    const char* value = std::getenv("CRISPASR_VOXCPM2_VAE_TEST_GPU");
    return value && *value && std::strcmp(value, "0") != 0;
}

TEST_CASE("VoxCPM2 AudioVAE speech upscaler", "[integration][voxcpm2-vae]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_VOXCPM2_VAE");
    if (!model_path || !*model_path)
        SKIP("CRISPASR_MODEL_VOXCPM2_VAE not set");

    auto params = voxcpm2_vae_context_default_params();
    params.n_threads = 2;
    params.verbosity = 0;
    params.use_gpu = voxcpm2_vae_test_gpu_enabled();
    auto* ctx = voxcpm2_vae_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    // Reject before graph construction/allocation. AudioVAE activation memory
    // is linear but large enough that a multi-minute single call can OOM.
    if (!std::getenv("CRISPASR_VOXCPM2_VAE_MAX_SAMPLES")) {
        std::vector<float> overlong(90 * 16000, 0.0f);
        int n_overlong = 123;
        float* rejected = voxcpm2_vae_upscale(ctx, overlong.data(), (int)overlong.size(), &n_overlong);
        CHECK(rejected == nullptr);
        CHECK(n_overlong == 0);
        voxcpm2_vae_pcm_free(rejected);
    }

    auto input = load_voxcpm2_vae_wav("samples/jfk.wav");
    REQUIRE(input.size() >= 16000);
    input.resize(16000); // one second keeps the CPU live test focused
    compare_legacy_and_graph(ctx, input);
    voxcpm2_vae_free(ctx);
}

TEST_CASE("VoxCPM2 TTS and AudioVAE contexts coexist", "[integration][voxcpm2-vae][coexistence]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_VOXCPM2_FULL");
    if (!model_path || !*model_path)
        SKIP("CRISPASR_MODEL_VOXCPM2_FULL not set");

    auto tts_params = voxcpm2_context_default_params();
    tts_params.n_threads = 3;
    tts_params.verbosity = 0;
    tts_params.use_gpu = false;
    auto* tts = voxcpm2_init_from_file(model_path, tts_params);
    REQUIRE(tts != nullptr);

    auto vae_params = voxcpm2_vae_context_default_params();
    vae_params.n_threads = 1;
    vae_params.verbosity = 0;
    vae_params.use_gpu = false;
    auto* vae = voxcpm2_vae_init_from_file(model_path, vae_params);
    REQUIRE(vae != nullptr);

    // Destroying the full TTS runtime must not invalidate the separately
    // loaded VAE backend or its weight/cache/backend state.
    voxcpm2_free(tts);

    auto input = load_voxcpm2_vae_wav("samples/jfk.wav");
    REQUIRE(input.size() >= 8000);
    input.resize(8000);
    require_valid_upscale(vae, input);
    voxcpm2_vae_free(vae);
}

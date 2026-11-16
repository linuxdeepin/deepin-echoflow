// tests/test_confucius4_tts_live.cpp — live integration test for the
// Confucius4-TTS backend.
//
// Requires CRISPASR_MODEL_CONFUCIUS4_T2S pointing at the T2S GGUF; the S2A /
// BigVGAN / w2v companions are auto-resolved as siblings (the same layout the
// registry downloads). Optionally CRISPASR_CONFUCIUS4_VOICE_WAV points at a
// reference wav for the zero-shot conditioning — without it the synthesis is
// unintelligible BY DESIGN (zero-shot cloner), so the unvoiced case only
// checks the pipeline runs and emits non-silent PCM.

#include "confucius4_tts.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>

namespace {

std::string sibling(const std::string& model, const char* name) {
    auto sep = model.find_last_of("/\\");
    std::string dir = sep == std::string::npos ? std::string(".") : model.substr(0, sep);
    std::string p = dir + "/" + name;
    if (FILE* f = fopen(p.c_str(), "rb")) {
        fclose(f);
        return p;
    }
    return {};
}

} // namespace

TEST_CASE("confucius4-tts: init from GGUF", "[confucius4-tts][live]") {
    const char* model = std::getenv("CRISPASR_MODEL_CONFUCIUS4_T2S");
    if (!model || !*model) {
        SKIP("CRISPASR_MODEL_CONFUCIUS4_T2S not set");
        return;
    }
    auto p = confucius4_tts_default_params();
    p.n_threads = 4;
    p.verbosity = 0;
    auto* ctx = confucius4_tts_init_from_file(model, p);
    REQUIRE(ctx != nullptr);
    REQUIRE(confucius4_tts_sample_rate(ctx) == 22050);
    confucius4_tts_free(ctx);
}

TEST_CASE("confucius4-tts: synthesize produces non-silent audio", "[confucius4-tts][live]") {
    const char* model = std::getenv("CRISPASR_MODEL_CONFUCIUS4_T2S");
    if (!model || !*model) {
        SKIP("CRISPASR_MODEL_CONFUCIUS4_T2S not set");
        return;
    }
    auto p = confucius4_tts_default_params();
    p.n_threads = 4;
    p.verbosity = 0;
    p.ode_steps = 4; // keep the live tier fast; quality is the kernel's job
    auto* ctx = confucius4_tts_init_from_file(model, p);
    REQUIRE(ctx != nullptr);

    std::string s2a = sibling(model, "confucius4-tts-s2a-q4_k.gguf");
    if (s2a.empty())
        s2a = sibling(model, "confucius4-tts-s2a-f16.gguf");
    if (s2a.empty()) {
        confucius4_tts_free(ctx);
        SKIP("no S2A sibling GGUF");
        return;
    }
    REQUIRE(confucius4_tts_set_s2a_path(ctx, s2a.c_str()) == 0);
    std::string voc = sibling(model, "confucius4-tts-bigvgan-22k-f16.gguf");
    if (!voc.empty())
        REQUIRE(confucius4_tts_set_vocoder_path(ctx, voc.c_str()) == 0);
    std::string w2v = sibling(model, "confucius4-tts-w2v-f16.gguf");
    if (!w2v.empty())
        confucius4_tts_set_w2v_path(ctx, w2v.c_str());
    if (const char* wav = std::getenv("CRISPASR_CONFUCIUS4_VOICE_WAV"); wav && *wav)
        REQUIRE(confucius4_tts_set_voice_path(ctx, wav) == 0);

    int n = 0;
    float* pcm = confucius4_tts_synthesize(ctx, "Hello there.", "en", &n);
    REQUIRE(pcm != nullptr);
    REQUIRE(n > 0);
    bool has_audio = false;
    for (int i = 0; i < n && !has_audio; i++)
        has_audio = pcm[i] > 0.001f || pcm[i] < -0.001f;
    REQUIRE(has_audio);
    confucius4_tts_pcm_free(pcm);
    confucius4_tts_free(ctx);
}

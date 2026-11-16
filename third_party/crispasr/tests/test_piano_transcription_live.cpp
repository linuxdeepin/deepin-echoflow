// Piano transcription integration test.
//
// Requires CRISPASR_MODEL_PIANO_TRANSCRIPTION env var pointing to the GGUF.
// SKIPs cleanly when not set.

#include <catch2/catch_test_macros.hpp>

#include "piano_transcription.h"

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

TEST_CASE("piano-transcription init", "[integration][piano-transcription]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_PIANO_TRANSCRIPTION");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_PIANO_TRANSCRIPTION not set");
    }

    auto params = piano_transcription_default_params();
    params.verbosity = 1;
    auto* ctx = piano_transcription_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);
    REQUIRE(piano_transcription_sample_rate(ctx) == 16000);
    piano_transcription_free(ctx);
}

TEST_CASE("piano-transcription mel spectrogram", "[integration][piano-transcription]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_PIANO_TRANSCRIPTION");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_PIANO_TRANSCRIPTION not set");
    }

    auto params = piano_transcription_default_params();
    params.verbosity = 0;
    auto* ctx = piano_transcription_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    // Generate 1 second of 440 Hz sine wave
    int sr = 16000;
    int n = sr;
    std::vector<float> pcm(n);
    for (int i = 0; i < n; i++) {
        pcm[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * i / sr);
    }

    int out_n = 0;
    float* mel = piano_transcription_mel_spectrogram(ctx, pcm.data(), n, &out_n);
    REQUIRE(mel != nullptr);
    REQUIRE(out_n > 0);
    // At 100 fps, 1 second should produce ~100 frames × 229 mels
    int T = out_n / 229;
    REQUIRE(T >= 95);
    REQUIRE(T <= 105);
    free(mel);

    piano_transcription_free(ctx);
}

TEST_CASE("piano-transcription transcribe", "[integration][piano-transcription]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_PIANO_TRANSCRIPTION");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_PIANO_TRANSCRIPTION not set");
    }

    auto params = piano_transcription_default_params();
    params.verbosity = 1;
    auto* ctx = piano_transcription_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    // Generate 2 seconds of 440 Hz + 880 Hz sine wave (A4 + A5 piano notes)
    int sr = 16000;
    int n = sr * 2;
    std::vector<float> pcm(n);
    for (int i = 0; i < n; i++) {
        pcm[i] = 0.3f * std::sin(2.0f * M_PI * 440.0f * i / sr) + 0.3f * std::sin(2.0f * M_PI * 880.0f * i / sr);
    }

    piano_transcription_result result = {};
    int rc = piano_transcription_transcribe(ctx, pcm.data(), n, &result);
    REQUIRE(rc == 0);
    // The model should detect *something* from these pure tones
    // (even if not perfect, since the model expects piano timbre)
    INFO("detected " << result.n_notes << " notes");
    // Just verify it doesn't crash and returns a valid result
    REQUIRE(result.n_notes >= 0);
    for (int i = 0; i < result.n_notes; i++) {
        REQUIRE(result.note_events[i].midi_note >= 21);
        REQUIRE(result.note_events[i].midi_note <= 108);
        REQUIRE(result.note_events[i].velocity >= 0);
        REQUIRE(result.note_events[i].velocity <= 127);
        REQUIRE(result.note_events[i].onset_time >= 0.0f);
    }

    piano_transcription_result_free(&result);
    piano_transcription_free(ctx);
}

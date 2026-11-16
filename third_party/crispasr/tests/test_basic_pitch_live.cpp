// Basic Pitch integration test (#250).
//
// Requires CRISPASR_MODEL_BASIC_PITCH env var pointing to the GGUF
// (basic-pitch-f32.gguf for the exact-notes expectations below; F16 can
// shift note ends by one frame at threshold boundaries). SKIPs cleanly
// when not set. Deeper parity (per-stage cosine vs the upstream ONNX)
// lives in the `crispasr-diff basic-pitch` harness with
// tools/reference_backends/basic_pitch.py as the oracle.

#include <catch2/catch_test_macros.hpp>

#include "basic_pitch.h"

#include <cmath>
#include <cstdlib>
#include <vector>

TEST_CASE("basic-pitch init", "[integration][basic-pitch]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_BASIC_PITCH");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_BASIC_PITCH not set");
    }

    auto params = basic_pitch_default_params();
    auto* ctx = basic_pitch_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);
    REQUIRE(basic_pitch_sample_rate(ctx) == 22050);
    basic_pitch_free(ctx);
}

TEST_CASE("basic-pitch detects a pure tone at the right pitch", "[integration][basic-pitch]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_BASIC_PITCH");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_BASIC_PITCH not set");
    }

    auto params = basic_pitch_default_params();
    auto* ctx = basic_pitch_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    // 2 s of A4 (440 Hz = MIDI 69) at the model's native 22050 Hz, with a
    // short fade in/out so the onset head sees a clean attack rather than a
    // click. A sine is in-domain enough for the note head; the assertion is
    // deliberately loose (69 +/- 1) — exactness is the diff harness's job.
    const int sr = 22050;
    const int n = 2 * sr;
    std::vector<float> pcm(n);
    for (int i = 0; i < n; i++) {
        float env = 1.0f;
        if (i < 1024)
            env = i / 1024.0f;
        if (n - i < 1024)
            env = (n - i) / 1024.0f;
        pcm[i] = 0.5f * env * std::sin(2.0f * (float)M_PI * 440.0f * i / sr);
    }

    basic_pitch_result res{};
    REQUIRE(basic_pitch_transcribe(ctx, pcm.data(), n, &res) == 0);
    REQUIRE(res.n_notes >= 1);

    // The dominant note (highest amplitude) must sit on/next to A4.
    int best = 0;
    for (int i = 1; i < res.n_notes; i++)
        if (res.notes[i].amplitude > res.notes[best].amplitude)
            best = i;
    REQUIRE(res.notes[best].midi_note >= 68);
    REQUIRE(res.notes[best].midi_note <= 70);
    REQUIRE(res.notes[best].end_time > res.notes[best].start_time);
    REQUIRE(res.notes[best].velocity >= 1);
    REQUIRE(res.notes[best].velocity <= 127);

    basic_pitch_result_free(&res);
    basic_pitch_free(ctx);
}

TEST_CASE("basic-pitch rejects null/empty input", "[integration][basic-pitch]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_BASIC_PITCH");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_BASIC_PITCH not set");
    }

    auto params = basic_pitch_default_params();
    auto* ctx = basic_pitch_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    basic_pitch_result res{};
    REQUIRE(basic_pitch_transcribe(ctx, nullptr, 22050, &res) != 0);
    REQUIRE(basic_pitch_transcribe(ctx, nullptr, 0, &res) != 0);
    basic_pitch_free(ctx);

    // Init failure paths must not crash either.
    REQUIRE(basic_pitch_init_from_file("/nonexistent/model.gguf", params) == nullptr);
    REQUIRE(basic_pitch_transcribe(nullptr, nullptr, 0, &res) != 0);
}

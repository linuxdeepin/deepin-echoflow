// test-crispasr-speaker-resample.cpp — unit tests for the pre-resample math
// inside crispasr_speaker_play().
//
// crispasr_speaker_play() converts app PCM (e.g. 24 kHz mono from a TTS
// backend) to device-native PCM (e.g. 96 kHz stereo on MacBook Air) via
// linear interpolation + per-channel mono expansion. This file replicates
// that exact math and verifies its invariants without opening any audio
// device.
//
// Invariants under test:
//   - Output length = round(n_app_frames * dev_rate / app_rate) * dev_ch
//   - DC signal preserved through resample
//   - Endpoints match src[0] and src[last]
//   - Stereo L/R mix-down averages channels correctly
//   - Identity transform (same rate, same channel count)

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Exact copy of the pre-resample logic from crispasr_speaker_play() so we
// can test it without an audio device (CRISPASR_SPEAKER_STUB_ONLY would
// compile out the implementation, and ma_device isn't available in a unit-test
// binary anyway).
static void speaker_resample(const float* pcm, int n_samples, int app_rate, int app_ch, int dev_rate, int dev_ch,
                             std::vector<float>& out) {
    const int n_app_frames = n_samples / app_ch;
    const int n_dev_frames = (int)((double)n_app_frames * dev_rate / app_rate + 0.5);
    out.resize((size_t)n_dev_frames * (size_t)dev_ch);
    const double ratio = (double)(n_app_frames - 1) / std::max(n_dev_frames - 1, 1);
    for (int fi = 0; fi < n_dev_frames; fi++) {
        double src_pos = fi * ratio;
        int i0 = (int)src_pos;
        int i1 = std::min(i0 + 1, n_app_frames - 1);
        float alpha = (float)(src_pos - (double)i0);
        // mono mix-down of app frame i0
        float mono = 0.0f;
        for (int c = 0; c < app_ch; c++)
            mono += pcm[i0 * app_ch + c];
        mono /= (float)app_ch;
        // mono mix-down of app frame i1
        float s1 = 0.0f;
        for (int c = 0; c < app_ch; c++)
            s1 += pcm[i1 * app_ch + c];
        s1 /= (float)app_ch;
        float sample = mono * (1.0f - alpha) + s1 * alpha;
        for (int c = 0; c < dev_ch; c++)
            out[(size_t)fi * (size_t)dev_ch + (size_t)c] = sample;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("speaker resample — 4x upsample mono→stereo (24k→96k)", "[unit][speaker]") {
    // Typical: TTS backend 24000 Hz mono, MacBook Air speakers 96000 Hz stereo
    float src[4] = {0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> out;
    speaker_resample(src, 4, 24000, 1, 96000, 2, out);

    // n_dev_frames = round(4 * 96000/24000) = 16; * 2 ch = 32 samples
    REQUIRE(out.size() == 32);
    // First frame both channels equal src[0]
    REQUIRE(out[0] == Catch::Approx(src[0]));
    REQUIRE(out[1] == Catch::Approx(src[0]));
    // Last frame both channels equal src[3]
    REQUIRE(out[30] == Catch::Approx(src[3]));
    REQUIRE(out[31] == Catch::Approx(src[3]));
}

TEST_CASE("speaker resample — identity (same rate, mono→mono)", "[unit][speaker]") {
    float src[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    std::vector<float> out;
    speaker_resample(src, 5, 16000, 1, 16000, 1, out);

    REQUIRE(out.size() == 5);
    for (int i = 0; i < 5; i++)
        REQUIRE(out[(size_t)i] == Catch::Approx(src[i]));
}

TEST_CASE("speaker resample — DC signal preserved", "[unit][speaker]") {
    // DC at 0.8 must survive upsample without drift
    std::vector<float> src(100, 0.8f);
    std::vector<float> out;
    speaker_resample(src.data(), 100, 24000, 1, 96000, 2, out);

    // n_dev_frames = round(100 * 96000/24000) = 400; * 2 = 800 samples
    REQUIRE(out.size() == 800);
    for (auto v : out)
        REQUIRE(v == Catch::Approx(0.8f));
}

TEST_CASE("speaker resample — stereo L+R mix-down averages correctly", "[unit][speaker]") {
    // Stereo: L=1.0, R=-1.0 should mix to 0.0 for all output frames
    float src[4] = {1.0f, -1.0f, 1.0f, -1.0f}; // 2 stereo frames
    std::vector<float> out;
    speaker_resample(src, 4, 44100, 2, 44100, 1, out);

    // Same rate, 2 app frames → 2 dev frames mono
    REQUIRE(out.size() == 2);
    REQUIRE(out[0] == Catch::Approx(0.0f));
    REQUIRE(out[1] == Catch::Approx(0.0f));
}

TEST_CASE("speaker resample — stereo mix-down asymmetric levels", "[unit][speaker]") {
    // L=0.8, R=0.4 → expected mono = 0.6
    float src[2] = {0.8f, 0.4f}; // 1 stereo frame
    std::vector<float> out;
    speaker_resample(src, 2, 48000, 2, 48000, 1, out);

    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == Catch::Approx(0.6f).epsilon(1e-5));
}

TEST_CASE("speaker resample — mono to multi-channel expansion", "[unit][speaker]") {
    // Mono signal expanded to 6 channels (surround) — all channels equal
    float src[3] = {0.3f, 0.6f, 0.9f};
    std::vector<float> out;
    speaker_resample(src, 3, 48000, 1, 48000, 6, out);

    REQUIRE(out.size() == 18); // 3 frames * 6 ch
    // Each group of 6 should all be equal
    for (int fi = 0; fi < 3; fi++) {
        float expected = src[fi];
        for (int c = 0; c < 6; c++)
            REQUIRE(out[(size_t)fi * 6 + c] == Catch::Approx(expected));
    }
}

TEST_CASE("speaker resample — 2x downsample endpoints match", "[unit][speaker]") {
    float src[6] = {0.1f, 0.5f, 0.9f, 0.7f, 0.3f, 0.2f};
    std::vector<float> out;
    speaker_resample(src, 6, 48000, 1, 24000, 1, out);

    // n_dev_frames = round(6 * 24000/48000) = 3
    REQUIRE(out.size() == 3);
    REQUIRE(out.front() == Catch::Approx(src[0]));
    REQUIRE(out.back() == Catch::Approx(src[5]));
}

TEST_CASE("speaker resample — single-frame input no crash", "[unit][speaker]") {
    float src[1] = {0.5f};
    std::vector<float> out;
    speaker_resample(src, 1, 24000, 1, 96000, 2, out);
    // n_dev_frames = round(1 * 4) = 4; * 2 = 8 samples
    REQUIRE(out.size() == 8);
    // Single-input → all outputs must equal src[0]
    for (auto v : out)
        REQUIRE(v == Catch::Approx(src[0]));
}

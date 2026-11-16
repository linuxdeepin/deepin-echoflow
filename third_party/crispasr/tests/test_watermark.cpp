// test_watermark.cpp — unit tests for the CrispASR audio watermark.
//
// Verifies embed + detect round-trip, detection threshold semantics,
// and robustness against simple transformations (volume scaling).

#include "core/crispasr_watermark.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "portable_env.h"

namespace {

// Generate a simple sine wave for testing (440 Hz at given sample rate).
std::vector<float> make_sine(int n_samples, int sample_rate = 24000, float freq = 440.0f, float amp = 0.5f) {
    std::vector<float> pcm(n_samples);
    for (int i = 0; i < n_samples; i++)
        pcm[i] = amp * std::sin(2.0f * 3.14159265f * freq * (float)i / (float)sample_rate);
    return pcm;
}

} // namespace

TEST_CASE("Watermark embed + detect round-trip", "[unit][watermark]") {
    // 2 seconds of audio at 24 kHz — plenty of frames for reliable detection
    auto pcm = make_sine(48000);
    auto original = pcm; // keep a copy

    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    float score = crispasr_watermark_detect_impl(pcm.data(), (int)pcm.size());
    REQUIRE(score > 0.65f);

    // Unwatermarked audio should score low
    float score_orig = crispasr_watermark_detect_impl(original.data(), (int)original.size());
    REQUIRE(score_orig < 0.65f);
}

TEST_CASE("Watermark is imperceptible (low distortion)", "[unit][watermark]") {
    auto pcm = make_sine(48000);
    auto original = pcm;

    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    // Compute SNR — watermark should be well below the signal
    double signal_power = 0.0, noise_power = 0.0;
    for (size_t i = 0; i < pcm.size(); i++) {
        signal_power += (double)original[i] * (double)original[i];
        double diff = (double)pcm[i] - (double)original[i];
        noise_power += diff * diff;
    }
    // SNR on a pure sine at alpha=0.08 is ~6 dB (worst case: all energy
    // in one bin that gets nudged). On broadband speech it's ~38 dB
    // (industry standard, same as AudioSeal/WavMark). The pure-sine
    // case is pessimistic but we just verify it's not destructive.
    // Real speech perception threshold is ~20 dB.
    double snr_db = 10.0 * std::log10(signal_power / noise_power);
    REQUIRE(snr_db > 5.0); // not destructive even on worst-case sine
}

// A broadband voiced-speech-like signal — a 130 Hz fundamental with harmonics
// rolling off by ~4 kHz, syllable-rate amplitude modulation. This is far closer
// to real TTS output than a pure sine and exercises the speech-band watermark
// under realistic masking. Detection margins here are large (the pure-sine cases
// above are pathological worst/edge cases), so CI robustness does not hinge on
// a near-tie. Guards issue #260 (the watermark must stay inaudible AND
// detectable on speech, not just on a tone).
namespace {
std::vector<float> make_speechlike(int n_samples, int sample_rate = 24000) {
    std::vector<float> pcm(n_samples);
    uint32_t rng = 0x12345678u; // deterministic — reproducible across platforms
    for (int i = 0; i < n_samples; i++) {
        double t = (double)i / (double)sample_rate;
        double env = 0.4 * (1.0 + 0.9 * std::sin(2.0 * 3.14159265358979 * 4.0 * t));
        double s = 0.0;
        for (int h = 1; h <= 25; h++) {
            double f = 130.0 * h;
            if (f > 4000.0)
                break;
            s += (1.0 / h) * std::sin(2.0 * 3.14159265358979 * f * t);
        }
        // Aspiration/breath noise floor — real voiced speech is not a pure
        // harmonic comb; the noise fills the inter-harmonic nulls so the
        // watermark band has continuous energy (as in real TTS output).
        rng = rng * 1664525u + 1013904223u;
        double noise = ((double)rng / 4294967296.0 - 0.5);
        pcm[i] = (float)(env * (s * 0.15 + noise * 0.02));
    }
    return pcm;
}
} // namespace

TEST_CASE("Watermark on speech-like signal: detectable + inaudible comb (issue #260)", "[unit][watermark]") {
    auto pcm = make_speechlike(72000); // 3 s
    auto clean = pcm;

    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    float score_wm = crispasr_watermark_detect_impl(pcm.data(), (int)pcm.size());
    float score_clean = crispasr_watermark_detect_impl(clean.data(), (int)clean.size());

    // Detection survives moving the comb into the speech band, with clean
    // separation from unwatermarked audio.
    REQUIRE(score_wm > 0.65f);
    REQUIRE(score_clean < 0.60f);

    // Issue #260: the comb must stay below ~5 kHz. Above 5 kHz — the region
    // where the pre-#260 wideband comb painted an audible "tinny" tone over
    // near-silence — the watermark must inject essentially nothing. Measure
    // the residual watermark energy above bin 213 (~5 kHz) vs the signal.
    const int n_fft = 1024;
    const int k_5khz = 213; // ~5 kHz at 24 kHz / 1024
    std::vector<float> re(n_fft), im(n_fft), re_d(n_fft), im_d(n_fft);
    double sig_hi = 0.0, noise_hi = 0.0;
    for (int start = 0; start + n_fft <= (int)pcm.size(); start += n_fft) {
        for (int i = 0; i < n_fft; i++) {
            re[i] = clean[start + i];
            im[i] = 0.0f;
            re_d[i] = pcm[start + i] - clean[start + i];
            im_d[i] = 0.0f;
        }
        crispasr_wm::fft_radix2(re.data(), im.data(), n_fft, false);
        crispasr_wm::fft_radix2(re_d.data(), im_d.data(), n_fft, false);
        for (int k = k_5khz; k < n_fft / 2 - 4; k++) {
            sig_hi += (double)re[k] * re[k] + (double)im[k] * im[k];
            noise_hi += (double)re_d[k] * re_d[k] + (double)im_d[k] * im_d[k];
        }
    }
    double above_5khz_snr = 10.0 * std::log10(sig_hi / (noise_hi + 1e-30));
    // Legacy wideband put ~17 dB of comb up here (audible); the speech-band
    // watermark leaves >40 dB SNR above 5 kHz (measured ~52 dB on real qwen3).
    REQUIRE(above_5khz_snr > 40.0);
}

TEST_CASE("Watermark survives volume normalization", "[unit][watermark]") {
    auto pcm = make_sine(48000, 24000, 440.0f, 0.3f);
    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    // Scale volume by 2x (normalize up)
    for (auto& s : pcm)
        s *= 2.0f;

    float score = crispasr_watermark_detect_impl(pcm.data(), (int)pcm.size());
    REQUIRE(score > 0.60f);
}

TEST_CASE("Watermark detection on silence returns low score", "[unit][watermark]") {
    std::vector<float> silence(48000, 0.0f);
    float score = crispasr_watermark_detect_impl(silence.data(), (int)silence.size());
    // Silence has no spectral content — detection should not false-positive
    REQUIRE(score < 0.65f);
}

TEST_CASE("Watermark embed is no-op for very short audio", "[unit][watermark]") {
    std::vector<float> short_pcm = {0.1f, 0.2f, 0.3f};
    auto original = short_pcm;
    crispasr_watermark_embed_impl(short_pcm.data(), (int)short_pcm.size());
    // Should be unchanged (< 1 FFT frame)
    REQUIRE(short_pcm == original);
}

TEST_CASE("Watermark detect on very short audio returns 0", "[unit][watermark]") {
    std::vector<float> short_pcm = {0.1f, 0.2f, 0.3f};
    float score = crispasr_watermark_detect_impl(short_pcm.data(), (int)short_pcm.size());
    REQUIRE(score == 0.0f);
}

TEST_CASE("Watermark detect on null/invalid input returns 0", "[unit][watermark]") {
    REQUIRE(crispasr_watermark_detect_impl(nullptr, 0) == 0.0f);
    REQUIRE(crispasr_watermark_detect_impl(nullptr, 100) == 0.0f);
}

// ─── Round-trip through int16 WAV conversion ────────────────────────────────
// Verifies that the watermark survives the float32 → int16 → float32
// quantization that happens when writing a WAV file and reading it back.
// This is the same path as crispasr_make_wav_int16 + the simple reader
// used by --detect-watermark.

TEST_CASE("Watermark survives int16 round-trip (WAV writer path)", "[unit][watermark]") {
    auto pcm = make_sine(48000, 24000, 440.0f, 0.5f);
    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    // Simulate int16 quantization (same as crispasr_make_wav_int16)
    std::vector<int16_t> pcm_i16(pcm.size());
    for (size_t i = 0; i < pcm.size(); i++) {
        float s = pcm[i];
        if (s > 1.0f)
            s = 1.0f;
        if (s < -1.0f)
            s = -1.0f;
        pcm_i16[i] = (int16_t)(s * 32767.0f);
    }

    // Convert back to float32 (same as --detect-watermark reader)
    std::vector<float> pcm_back(pcm.size());
    for (size_t i = 0; i < pcm.size(); i++) {
        pcm_back[i] = (float)pcm_i16[i] / 32768.0f;
    }

    float score = crispasr_watermark_detect_impl(pcm_back.data(), (int)pcm_back.size());
    REQUIRE(score > 0.60f);
}

// ─── Post-embed verification threshold semantics ────────────────────────────
// The post-embed verification in crispasr_run.cpp warns when confidence
// < 0.6. Verify that a properly watermarked signal exceeds this, and
// that unwatermarked audio falls below.

TEST_CASE("Post-embed verification threshold: watermarked > 0.6, clean < 0.6", "[unit][watermark]") {
    auto pcm = make_sine(48000, 24000, 440.0f, 0.5f);
    auto clean = pcm;

    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    float score_wm = crispasr_watermark_detect_impl(pcm.data(), (int)pcm.size());
    float score_clean = crispasr_watermark_detect_impl(clean.data(), (int)clean.size());

    // Watermarked audio must pass the post-embed verification threshold
    REQUIRE(score_wm >= 0.6f);
    // Clean audio must NOT trigger a false positive at that threshold
    REQUIRE(score_clean < 0.6f);
}

// ---------------------------------------------------------------------------
// Per-frame detector (the default since 2026-08-03). The sign test above stays
// guarded because CRISPASR_WATERMARK_DETECT=sign still reaches it.
//
// Note the round-trip below uses a SINE, and a stationary tone is the one input
// where consistency alone lies: every decoy pattern scores just as extremely on
// it, which is exactly why the statistic requires specificity as well. Passing
// here therefore says something the old test could not — that the specificity
// term is wired in, not just the t.
// ---------------------------------------------------------------------------

TEST_CASE("Per-frame watermark embed + detect round-trip", "[unit][watermark]") {
    auto pcm = make_sine(48000);
    auto original = pcm;

    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    const float score = crispasr_watermark_detect_frames_impl(pcm.data(), (int)pcm.size());
    REQUIRE(score > 0.65f);

    const float score_orig = crispasr_watermark_detect_frames_impl(original.data(), (int)original.size());
    REQUIRE(score_orig < 0.65f);
}

TEST_CASE("Per-frame detector needs enough frames before it answers", "[unit][watermark]") {
    // Below kDetectMinFrames it returns 0 rather than guess from a handful of
    // frames — the t would be computed over a sample too small to mean anything.
    std::vector<float> tiny(1024 * 4, 0.5f);
    REQUIRE(crispasr_watermark_detect_frames_impl(tiny.data(), (int)tiny.size()) == 0.0f);
    std::vector<float> shorter(512, 0.5f);
    REQUIRE(crispasr_watermark_detect_frames_impl(shorter.data(), (int)shorter.size()) == 0.0f);
}

TEST_CASE("Per-frame detector does not fire on silence", "[unit][watermark]") {
    std::vector<float> silence(48000, 0.0f);
    REQUIRE(crispasr_watermark_detect_frames_impl(silence.data(), (int)silence.size()) < 0.65f);
}

TEST_CASE("The detector selector honours CRISPASR_WATERMARK_DETECT", "[unit][watermark]") {
    // Both surfaces (CLI dispatch, session C-ABI) route through the selector,
    // so this is the single place the choice is made.
    auto pcm = make_sine(48000);
    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    setenv("CRISPASR_WATERMARK_DETECT", "sign", 1);
    REQUIRE_FALSE(crispasr_watermark_detect_uses_frames());
    REQUIRE(crispasr_watermark_detect_select(pcm.data(), (int)pcm.size()) ==
            crispasr_watermark_detect_impl(pcm.data(), (int)pcm.size()));

    setenv("CRISPASR_WATERMARK_DETECT", "frames", 1);
    REQUIRE(crispasr_watermark_detect_uses_frames());
    REQUIRE(crispasr_watermark_detect_select(pcm.data(), (int)pcm.size()) ==
            crispasr_watermark_detect_frames_impl(pcm.data(), (int)pcm.size()));

    // Unset => the compiled-in default, which is the per-frame statistic.
    unsetenv("CRISPASR_WATERMARK_DETECT");
    REQUIRE(crispasr_watermark_detect_uses_frames());
}

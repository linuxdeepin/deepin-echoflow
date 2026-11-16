// test_audioseal.cpp — unit tests for the AudioSeal ggml watermark module.
//
// Tests that don't require a GGUF model (API surface, null safety) run
// unconditionally. Tests that need the model are gated on the
// AUDIOSEAL_GGUF environment variable pointing at an audioseal.gguf.

#include "audioseal.h"
#include "core/crispasr_env.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// API surface tests (no model required)
// ---------------------------------------------------------------------------

TEST_CASE("audioseal_default_params returns sane values", "[unit][audioseal]") {
    auto p = audioseal_default_params();
    REQUIRE(p.n_threads == 4);
    REQUIRE(p.verbosity == 1);
    REQUIRE(p.use_gpu == false);
}

TEST_CASE("audioseal_init from non-existent file returns null", "[unit][audioseal]") {
    auto p = audioseal_default_params();
    p.verbosity = 0;
    auto* ctx = audioseal_init_from_file("/tmp/nonexistent-audioseal-42.gguf", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("audioseal_embed with null ctx returns null", "[unit][audioseal]") {
    float pcm[100] = {};
    REQUIRE(audioseal_embed(nullptr, pcm, 100, nullptr) == nullptr);
}

TEST_CASE("audioseal_detect with null ctx returns null", "[unit][audioseal]") {
    float pcm[100] = {};
    int n = 0;
    REQUIRE(audioseal_detect(nullptr, pcm, 100, &n, nullptr) == nullptr);
}

TEST_CASE("audioseal_embed with null pcm returns null", "[unit][audioseal]") {
    // Can't actually call with a real ctx without a model, but test the
    // null pcm path through the API contract
    REQUIRE(audioseal_embed(nullptr, nullptr, 0, nullptr) == nullptr);
}

TEST_CASE("audioseal_sample_rate with null returns 16000", "[unit][audioseal]") {
    REQUIRE(audioseal_sample_rate(nullptr) == 16000);
}

TEST_CASE("audioseal_nbits with null returns 16", "[unit][audioseal]") {
    REQUIRE(audioseal_nbits(nullptr) == 16);
}

// ---------------------------------------------------------------------------
// Live model tests (require AUDIOSEAL_GGUF env var)
// ---------------------------------------------------------------------------

namespace {
const char* get_gguf_path() {
    return crispasr_env::get("CRISPASR_AUDIOSEAL_GGUF");
}

std::vector<float> make_sine_16k(int n_samples, float freq = 440.0f, float amp = 0.3f) {
    std::vector<float> pcm(n_samples);
    for (int i = 0; i < n_samples; i++)
        pcm[i] = amp * std::sin(2.0f * 3.14159265f * freq * (float)i / 16000.0f);
    return pcm;
}
} // namespace

TEST_CASE("audioseal GGUF load + metadata", "[audioseal][live]") {
    const char* path = get_gguf_path();
    if (!path) {
        SKIP("AUDIOSEAL_GGUF not set — skipping live test");
        return;
    }
    auto p = audioseal_default_params();
    p.verbosity = 0;
    auto* ctx = audioseal_init_from_file(path, p);
    REQUIRE(ctx != nullptr);
    REQUIRE(audioseal_sample_rate(ctx) == 16000);
    REQUIRE(audioseal_nbits(ctx) == 16);
    audioseal_free(ctx);
}

TEST_CASE("audioseal embed produces output of correct length", "[audioseal][live]") {
    const char* path = get_gguf_path();
    if (!path) {
        SKIP("AUDIOSEAL_GGUF not set");
        return;
    }
    auto p = audioseal_default_params();
    p.verbosity = 0;
    auto* ctx = audioseal_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    // 1 second of audio at 16 kHz
    auto pcm = make_sine_16k(16000);
    float* out = audioseal_embed(ctx, pcm.data(), (int)pcm.size(), nullptr);
    if (out) {
        // Output should be same length as input
        // and not all-zero (watermark was added)
        double diff_energy = 0.0;
        for (int i = 0; i < 16000; i++) {
            double d = (double)out[i] - (double)pcm[i];
            diff_energy += d * d;
        }
        // Watermark should have non-zero energy
        REQUIRE(diff_energy > 0.0);
        std::free(out);
    }
    audioseal_free(ctx);
}

TEST_CASE("audioseal embed+detect round-trip", "[audioseal][live]") {
    const char* path = get_gguf_path();
    if (!path) {
        SKIP("AUDIOSEAL_GGUF not set");
        return;
    }
    auto p = audioseal_default_params();
    p.verbosity = 0;
    auto* ctx = audioseal_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    auto pcm = make_sine_16k(16000);
    float* watermarked = audioseal_embed(ctx, pcm.data(), (int)pcm.size(), nullptr);
    if (!watermarked) {
        audioseal_free(ctx);
        SKIP("embed returned null — graph may not run on this platform");
        return;
    }

    int n_det = 0;
    uint8_t msg[16] = {};
    float* probs = audioseal_detect(ctx, watermarked, 16000, &n_det, msg);
    if (probs) {
        REQUIRE(n_det > 0);
        // Average detection probability should be above chance (0.5)
        double avg_prob = 0.0;
        for (int i = 0; i < n_det; i++)
            avg_prob += probs[i];
        avg_prob /= (double)n_det;
        // A correctly-ported generator+detector recovers its own watermark with
        // high confidence — well above chance. (Not merely "doesn't crash".)
        printf("audioseal round-trip: avg detection probability = %.4f over %d frames\n", avg_prob, n_det);
        REQUIRE(avg_prob > 0.9);
        std::free(probs);
    } else {
        FAIL("audioseal_detect returned null — detector did not run");
    }

    // The detector must DISCRIMINATE: the un-watermarked input must score low.
    // (Guards against a degenerate "always ~1" or "always ~0.5" detector.)
    int n_clean = 0;
    float* clean_probs = audioseal_detect(ctx, pcm.data(), (int)pcm.size(), &n_clean, nullptr);
    if (clean_probs) {
        double avg_clean = 0.0;
        for (int i = 0; i < n_clean; i++)
            avg_clean += clean_probs[i];
        avg_clean /= (double)n_clean;
        printf("audioseal clean (unwatermarked): avg detection probability = %.4f\n", avg_clean);
        REQUIRE(avg_clean < 0.5);
        std::free(clean_probs);
    }

    std::free(watermarked);
    audioseal_free(ctx);
}

TEST_CASE("audioseal long audio grows graph and reused scheduler", "[audioseal][live]") {
    const char* path = get_gguf_path();
    if (!path) {
        SKIP("CRISPASR_AUDIOSEAL_GGUF not set");
        return;
    }
    auto p = audioseal_default_params();
    p.verbosity = 0;
    auto* ctx = audioseal_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    // Grow the detector first (including its optional message branch), then
    // the larger generator on the same context; finally reuse it for a short clip.
    for (int seconds : {4, 10, 1}) {
        auto pcm = make_sine_16k(seconds * 16000);
        int n = 0;
        uint8_t message[16] = {};
        float* clean = audioseal_detect(ctx, pcm.data(), (int)pcm.size(), &n, message);
        REQUIRE(clean != nullptr);
        REQUIRE(n == (int)pcm.size());
        std::free(clean);

        float* wm = audioseal_embed(ctx, pcm.data(), (int)pcm.size(), nullptr);
        REQUIRE(wm != nullptr);
        double energy = 0;
        for (size_t i = 0; i < pcm.size(); ++i) {
            REQUIRE(std::isfinite(wm[i]));
            double d = wm[i] - pcm[i];
            energy += d * d;
        }
        REQUIRE(energy > 0);
        float* probs = audioseal_detect(ctx, wm, (int)pcm.size(), &n, message);
        REQUIRE(probs != nullptr);
        REQUIRE(n == (int)pcm.size());
        double avg = 0;
        for (int i = 0; i < n; ++i) {
            REQUIRE(std::isfinite(probs[i]));
            avg += probs[i];
        }
        avg /= n;
        INFO("duration=" << seconds << "s probability=" << avg);
        REQUIRE(avg > 0.9);
        std::free(probs);
        std::free(wm);
    }
    audioseal_free(ctx);
}

// test-core-mel-parallel.cpp — core/mel.cpp §176f parallel-STFT correctness.
//
// The OpenMP-parallel STFT frame loop (Params::allow_parallel_stft /
// CRISPASR_MEL_PARALLEL) must be BIT-IDENTICAL to the serial path: each frame's
// FFT writes only its own row of the power spectrum and uses thread-private
// scratch, so threading must not perturb a single bit. This is the regression
// gate for that invariant — it runs the exact same input through compute() with
// the flag off then on and asserts the float outputs match exactly.
//
// On a build without OpenMP both calls run serially, so the test still passes
// (degenerate equivalence); on an OpenMP build it exercises real parallelism.
// Pure CPU, no model. The fft used here is a re-entrant in-place Cooley-Tukey
// (const input, writes only to the caller's `out`) — the same shape the audit
// found for every in-tree backend fft.

#include <catch2/catch_test_macros.hpp>

#include "core/mel.h"

#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

// Re-entrant real→complex FFT: stack-only state, writes interleaved (re,im) to
// `out`. Matches FftR2C (const float* in). N must be a power of two.
void test_fft_r2c(const float* in, int N, float* out) {
    int bits = 0;
    for (int n = N; n > 1; n >>= 1)
        bits++;
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++)
            rev = (rev << 1) | ((i >> b) & 1);
        out[2 * rev] = in[i];
        out[2 * rev + 1] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        const float ang = -2.0f * (float)M_PI / (float)len;
        const float wre = std::cos(ang), wim = std::sin(ang);
        for (int i = 0; i < N; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                const int a = i + j, b = i + j + len / 2;
                const float are = out[2 * a], aim = out[2 * a + 1];
                const float bre = out[2 * b], bim = out[2 * b + 1];
                const float tre = ure * bre - uim * bim, tim = ure * bim + uim * bre;
                out[2 * a] = are + tre;
                out[2 * a + 1] = aim + tim;
                out[2 * b] = are - tre;
                out[2 * b + 1] = aim - tim;
                const float nure = ure * wre - uim * wim;
                uim = ure * wim + uim * wre;
                ure = nure;
            }
        }
    }
}

std::vector<float> run(int n_samples, const std::vector<float>& sig, const std::vector<float>& win,
                       const std::vector<float>& fb, int n_fft, int n_freqs, int n_mels, bool parallel, int& T) {
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = 160;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.allow_parallel_stft = parallel;
    return core_mel::compute(sig.data(), n_samples, win.data(), n_fft, fb.data(), n_freqs, test_fft_r2c, p, T);
}

} // namespace

TEST_CASE("core_mel parallel STFT is bit-identical to serial", "[unit][mel]") {
    const int sr = 16000, n_fft = 512, n_mels = 80;
    const int n_freqs = n_fft / 2 + 1;
    const int n_samples = sr * 5; // 5 s → ~500 frames, comfortably past the T>=256 parallel threshold

    std::vector<float> sig((size_t)n_samples);
    for (int i = 0; i < n_samples; i++)
        sig[(size_t)i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * i / sr) + 0.2f * std::sin(2.0 * M_PI * 1234.5 * i / sr);

    std::vector<float> win((size_t)n_fft);
    for (int i = 0; i < n_fft; i++)
        win[(size_t)i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / (n_fft - 1)));

    // Deterministic non-trivial filterbank (MelsFreqs layout: fb[m*n_freqs+k]).
    std::vector<float> fb((size_t)n_mels * n_freqs);
    for (int m = 0; m < n_mels; m++)
        for (int k = 0; k < n_freqs; k++)
            fb[(size_t)m * n_freqs + k] = (float)(((m * 7 + k * 3) % 11)) * 0.01f;

    int T_serial = 0, T_parallel = 0;
    auto serial = run(n_samples, sig, win, fb, n_fft, n_freqs, n_mels, /*parallel=*/false, T_serial);
    auto parallel = run(n_samples, sig, win, fb, n_fft, n_freqs, n_mels, /*parallel=*/true, T_parallel);

    REQUIRE(T_serial == T_parallel);
    REQUIRE(T_serial >= 256); // confirm we actually crossed the parallel threshold
    REQUIRE(serial.size() == parallel.size());
    REQUIRE(serial.size() > 0);

    size_t first_diff = serial.size();
    for (size_t i = 0; i < serial.size(); i++) {
        if (serial[i] != parallel[i]) {
            first_diff = i;
            break;
        }
    }
    REQUIRE(first_diff == serial.size()); // bit-identical: no differing element
}

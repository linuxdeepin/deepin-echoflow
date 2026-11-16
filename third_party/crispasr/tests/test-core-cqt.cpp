// test-core-cqt.cpp — core/cqt.h constant-Q transform.
//
// Deterministic properties verified here (no model, no network, CI-safe):
//   * geometric bin spacing and Q-derived kernel lengths
//   * a pure tone peaks in the bin its frequency maps to
//   * octave-related tones land exactly bins_per_octave apart
//   * silence -> zeros; L1 normalisation makes bins comparable across lengths
//
//   * ABSOLUTE kernel scale (librosa scale=True) — see below
//   * linearity, superposition, and frame-count edge cases
//
// Cross-checking against librosa is a SEPARATE step and is not done here (CI has
// no numpy/librosa). Run `test-core-cqt --dump <file>` to write the magnitude
// matrix for the shared test signal, then score it with
// `python tools/cqt_librosa_parity.py <file>`.
//
// WHY THE ABSOLUTE-SCALE TESTS EXIST. This file once passed 726 assertions
// against a build whose every bin was low by sqrt(N_k) — up to 152x at the
// bottom octave — because the kernels were L1-normalised but never given
// librosa's scale=True factor. BTC then read its features as near-silence and
// emitted "no chord" for every frame.
//
// Every test here was scale-blind, and one was blind for a subtler reason
// worth recording. The cross-length test below compared peak magnitude two
// octaves apart and required ratio < 3. It passed either way — because the two
// builds sit on OPPOSITE sides of it:
//
//   L1 only (the bug)      -> response is FLAT across bins, ratio 1.0
//   L1 * sqrt(N) (librosa) -> response ~ sqrt(N_k),        ratio 2.0
//
// So "equal-amplitude tones give equal magnitude" — the property that test was
// named for — is what the BUGGY build satisfies. scale=True deliberately does
// not have it. A test can be measuring the right quantity, with a bound that
// admits both the correct and the broken answer, and be worse than no test at
// all because its name asserts the wrong law.
//
// The replacement asserts the EXACT law instead of a bound. For a sinusoid of
// amplitude A at a bin centre, the response is exactly (A/2) * sqrt(N_k) —
// measured to four decimals across the whole range (peak/sqrt(N) = 0.2500 for
// A = 0.5, every bin). And the kernel-norm identity below needs no signal at
// all: it is arithmetic on the coefficients, so no tolerance can hide in it.

#include "core/cqt.h"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

namespace {

constexpr int SR = 22050;
constexpr float FMIN = 32.703195662574829f; // C1
constexpr int HOP = 2048;

core_cqt::Params btc_params() {
    core_cqt::Params p;
    p.sample_rate = SR;
    p.fmin = FMIN;
    p.n_bins = 144;
    p.bins_per_octave = 24;
    p.hop_length = HOP;
    return p;
}

std::vector<float> tone(int n, double f, double amp = 0.5) {
    std::vector<float> x((size_t)n);
    for (int i = 0; i < n; i++)
        x[(size_t)i] = (float)(amp * std::sin(2.0 * M_PI * f * (double)i / (double)SR));
    return x;
}

// Must match tools/cqt_librosa_parity.py test_signal() exactly.
// `sr` defaults to the BTC rate so every existing caller is unchanged. It is a
// parameter because the TabCNN front end runs at 44.1 kHz: baking SR into the
// phase term would place the tones at half their intended frequency there, and
// a parity run against librosa would then be comparing two different signals
// while looking like a front-end disagreement.
std::vector<float> shared_test_signal(int n, int sr = SR) {
    std::vector<float> x((size_t)n, 0.0f);
    const double fs[3] = {130.8127826502993, 261.6255653005986, 523.2511306011972};
    const int third = n / 3;
    for (int i = 0; i < 3; i++) {
        const int lo = i * third;
        const int hi = (i < 2) ? (i + 1) * third : n;
        for (int s = lo; s < hi; s++)
            x[(size_t)s] = (float)(0.5 * std::sin(2.0 * M_PI * fs[i] * (double)s / (double)sr));
    }
    return x;
}

int expected_bin(const core_cqt::Params& p, double f) {
    return (int)std::lround((double)p.bins_per_octave * std::log2(f / (double)p.fmin));
}

int argmax_frame(const std::vector<float>& m, int t, int n_bins) {
    int best = 0;
    for (int k = 1; k < n_bins; k++)
        if (m[(size_t)t * (size_t)n_bins + (size_t)k] > m[(size_t)t * (size_t)n_bins + (size_t)best])
            best = k;
    return best;
}

} // namespace

TEST_CASE("cqt: bin frequencies are geometric", "[core][cqt]") {
    const auto p = btc_params();
    REQUIRE_THAT(core_cqt::bin_frequency(p, 0), Catch::Matchers::WithinRel(FMIN, 1e-6f));
    // One octave up = bins_per_octave bins.
    REQUIRE_THAT(core_cqt::bin_frequency(p, p.bins_per_octave), Catch::Matchers::WithinRel(2.0f * FMIN, 1e-5f));
    // Every adjacent pair has the same ratio.
    const float r = core_cqt::bin_frequency(p, 1) / core_cqt::bin_frequency(p, 0);
    for (int k = 1; k < 40; k++) {
        const float rk = core_cqt::bin_frequency(p, k + 1) / core_cqt::bin_frequency(p, k);
        REQUIRE_THAT(rk, Catch::Matchers::WithinRel(r, 1e-5f));
    }
}

TEST_CASE("cqt: kernel length shrinks with frequency, Q is constant", "[core][cqt]") {
    const auto p = btc_params();
    const int n0 = core_cqt::kernel_length(p, 0);
    const int n_oct = core_cqt::kernel_length(p, p.bins_per_octave);
    REQUIRE(n0 > 0);
    // An octave up halves the required window.
    REQUIRE(std::abs((double)n_oct - (double)n0 / 2.0) <= 2.0);
    // Lengths are monotonically non-increasing.
    for (int k = 1; k < p.n_bins; k++)
        REQUIRE(core_cqt::kernel_length(p, k) <= core_cqt::kernel_length(p, k - 1));
}

TEST_CASE("cqt: a pure tone peaks in its own bin", "[core][cqt]") {
    auto p = btc_params();
    // 5 octaves: the top test tone (C5, 523.25 Hz) maps to bin 96, so n_bins
    // must EXCEED 96 or argmax saturates at the last bin and the octave-
    // spacing check silently reads 23 instead of 24.
    p.n_bins = 120;
    const auto kernels = core_cqt::build_kernels(p);

    for (double f : {130.8127826502993, 261.6255653005986, 523.2511306011972}) {
        const auto x = tone(SR, f);
        std::vector<float> m;
        const int T = core_cqt::magnitude(p, kernels, x.data(), (int)x.size(), m);
        REQUIRE(T > 2);
        // Use a middle frame: the first/last are edge-padded.
        const int k = argmax_frame(m, T / 2, p.n_bins);
        INFO("f=" << f << " expected bin " << expected_bin(p, f) << " got " << k);
        REQUIRE(std::abs(k - expected_bin(p, f)) <= 1);
    }
}

TEST_CASE("cqt: octave-related tones are exactly bins_per_octave apart", "[core][cqt]") {
    auto p = btc_params();
    p.n_bins = 120; // must exceed bin 96 (C5) -- see the tone test above
    const auto kernels = core_cqt::build_kernels(p);

    std::vector<float> m1, m2;
    const auto x1 = tone(SR, 261.6255653005986);
    const auto x2 = tone(SR, 523.2511306011972);
    const int T1 = core_cqt::magnitude(p, kernels, x1.data(), (int)x1.size(), m1);
    const int T2 = core_cqt::magnitude(p, kernels, x2.data(), (int)x2.size(), m2);
    const int k1 = argmax_frame(m1, T1 / 2, p.n_bins);
    const int k2 = argmax_frame(m2, T2 / 2, p.n_bins);
    REQUIRE(k2 - k1 == p.bins_per_octave);
}

TEST_CASE("cqt: silence is zero, shape is right", "[core][cqt]") {
    auto p = btc_params();
    p.n_bins = 48;
    std::vector<float> x((size_t)SR, 0.0f);
    std::vector<float> m;
    const int T = core_cqt::magnitude(p, x.data(), (int)x.size(), m);
    REQUIRE(T == core_cqt::n_frames(p, (int)x.size()));
    REQUIRE(m.size() == (size_t)T * (size_t)p.n_bins);
    for (float v : m)
        REQUIRE(v == 0.0f);
}

TEST_CASE("cqt: response follows the exact sqrt(N_k) scale law", "[core][cqt]") {
    // Replaces a test called "L1 normalisation makes bins comparable across
    // kernel lengths", which asserted ratio < 3 between two octaves. That
    // admitted both the correct build (ratio 2.0) and the sqrt(N_k) bug
    // (ratio 1.0) -- and its NAME described the buggy behaviour. See the file
    // header.
    //
    // librosa scale=True gives a sinusoid of amplitude A at a bin centre a
    // response of exactly (A/2) * sqrt(N_k). The A/2 is the analytic kernel
    // picking up one side of the real sinusoid's +/-f pair.
    auto p = btc_params();
    const auto kernels = core_cqt::build_kernels(p);
    const double amp = 0.5;

    // Start at bin 24: below that the kernel is longer than the 1 s probe, so
    // the measurement is bounded by the probe rather than by the transform.
    for (int k = 24; k < p.n_bins; k += 6) {
        const double f = (double)core_cqt::bin_frequency(p, k);
        const auto x = tone(SR, f, amp);
        std::vector<float> m;
        const int T = core_cqt::magnitude(p, kernels, x.data(), (int)x.size(), m);
        REQUIRE(T > 2);
        const int t = T / 2;
        const double peak = m[(size_t)t * (size_t)p.n_bins + (size_t)argmax_frame(m, t, p.n_bins)];
        const double expect = (amp / 2.0) * std::sqrt((double)core_cqt::kernel_length(p, k));
        INFO("bin " << k << " f " << f << " peak " << peak << " expected " << expect);
        REQUIRE_THAT(peak, Catch::Matchers::WithinRel(expect, 0.02));
    }
}

TEST_CASE("cqt: two octaves apart differ by exactly sqrt(4) = 2", "[core][cqt]") {
    // The scale law stated as the ratio the old test got wrong. Under the
    // sqrt(N_k) bug this ratio is 1.0, so the assertion separates the two
    // builds cleanly instead of straddling them.
    auto p = btc_params();
    p.n_bins = 120;
    const auto kernels = core_cqt::build_kernels(p);
    std::vector<float> lo_m, hi_m;
    const auto lo = tone(SR, 130.8127826502993);
    const auto hi = tone(SR, 523.2511306011972); // two octaves up
    const int Tl = core_cqt::magnitude(p, kernels, lo.data(), (int)lo.size(), lo_m);
    const int Th = core_cqt::magnitude(p, kernels, hi.data(), (int)hi.size(), hi_m);
    const float pl = lo_m[(size_t)(Tl / 2) * (size_t)p.n_bins + (size_t)argmax_frame(lo_m, Tl / 2, p.n_bins)];
    const float ph = hi_m[(size_t)(Th / 2) * (size_t)p.n_bins + (size_t)argmax_frame(hi_m, Th / 2, p.n_bins)];
    REQUIRE(pl > 0.0f);
    REQUIRE(ph > 0.0f);
    INFO("low peak " << pl << " high peak " << ph << " ratio " << (pl / ph));
    REQUIRE_THAT((double)(pl / ph), Catch::Matchers::WithinRel(2.0, 0.02));
}

// ---------------------------------------------------------------------------
// Absolute scale. These are the tests the sqrt(N_k) bug could not have survived.
// ---------------------------------------------------------------------------

TEST_CASE("cqt: normalised kernel L1 norm is exactly sqrt(N) (librosa scale=True)", "[core][cqt]") {
    // THE regression guard. build_kernels scales each kernel by sqrt(N)/l1, so
    // afterwards its L1 norm must be sqrt(N) exactly. Dropping the sqrt(N)
    // leaves 1.0 instead — a factor of sqrt(N_k), which is 152 at bin 0. There
    // is no signal, no window leakage and no tolerance to argue about here:
    // it is arithmetic on the kernel coefficients.
    auto p = btc_params();
    const auto kernels = core_cqt::build_kernels(p);
    REQUIRE(kernels.size() == (size_t)p.n_bins);

    for (int k = 0; k < p.n_bins; k++) {
        const core_cqt::Kernel& kern = kernels[(size_t)k];
        const int N = core_cqt::kernel_length(p, k);
        REQUIRE(kern.length == N);
        REQUIRE(kern.re.size() == (size_t)N);
        REQUIRE(kern.im.size() == (size_t)N);

        double l1 = 0.0;
        for (int n = 0; n < N; n++)
            l1 += std::sqrt((double)kern.re[(size_t)n] * kern.re[(size_t)n] +
                            (double)kern.im[(size_t)n] * kern.im[(size_t)n]);

        INFO("bin " << k << " length " << N << " l1 " << l1 << " expected " << std::sqrt((double)N));
        REQUIRE_THAT(l1, Catch::Matchers::WithinRel(std::sqrt((double)N), 1e-4));
    }
}

TEST_CASE("cqt: l1_normalize=false leaves the raw kernel scale", "[core][cqt]") {
    // The flag must actually gate the normalisation, or the test above would
    // pass for a build that ignores it.
    auto p = btc_params();
    p.n_bins = 24;
    p.l1_normalize = false;
    const auto raw = core_cqt::build_kernels(p);
    p.l1_normalize = true;
    const auto normed = core_cqt::build_kernels(p);

    for (int k = 0; k < p.n_bins; k++) {
        double l1_raw = 0.0, l1_norm = 0.0;
        const int N = core_cqt::kernel_length(p, k);
        for (int n = 0; n < N; n++) {
            l1_raw += std::sqrt((double)raw[(size_t)k].re[(size_t)n] * raw[(size_t)k].re[(size_t)n] +
                                (double)raw[(size_t)k].im[(size_t)n] * raw[(size_t)k].im[(size_t)n]);
            l1_norm += std::sqrt((double)normed[(size_t)k].re[(size_t)n] * normed[(size_t)k].re[(size_t)n] +
                                 (double)normed[(size_t)k].im[(size_t)n] * normed[(size_t)k].im[(size_t)n]);
        }
        REQUIRE(l1_raw > 0.0);
        REQUIRE_THAT(l1_norm, Catch::Matchers::WithinRel(std::sqrt((double)N), 1e-4));
        // The raw Hann-windowed exponential has L1 ~ N/2, nowhere near sqrt(N).
        REQUIRE(l1_raw > 4.0 * l1_norm);
    }
}

TEST_CASE("cqt: peak/sqrt(N_k) is constant across the whole bin range", "[core][cqt]") {
    // The scale law as a flatness statement: dividing out sqrt(N_k) must leave
    // a constant. Under the bug this quantity varies as 1/sqrt(N_k) -- a 5.6x
    // spread over this range -- so the bound below excludes it by a wide
    // margin while staying loose enough for window-mainlobe effects.
    auto p = btc_params();
    const auto kernels = core_cqt::build_kernels(p);

    double lo = 1e30, hi = 0.0;
    for (int k = 24; k < p.n_bins; k += 12) {
        const double f = (double)core_cqt::bin_frequency(p, k);
        const auto x = tone(SR, f);
        std::vector<float> m;
        const int T = core_cqt::magnitude(p, kernels, x.data(), (int)x.size(), m);
        REQUIRE(T > 0);
        const int t = T / 2;
        const double peak = m[(size_t)t * (size_t)p.n_bins + (size_t)argmax_frame(m, t, p.n_bins)];
        const double norm = peak / std::sqrt((double)core_cqt::kernel_length(p, k));
        INFO("bin " << k << " f " << f << " peak/sqrtN " << norm);
        REQUIRE(norm > 0.0);
        lo = std::min(lo, norm);
        hi = std::max(hi, norm);
    }
    INFO("peak/sqrtN spread " << hi / lo);
    REQUIRE(hi / lo < 1.05);
}

// ---------------------------------------------------------------------------
// Linearity and edge cases.
// ---------------------------------------------------------------------------

TEST_CASE("cqt: magnitude is homogeneous in the input amplitude", "[core][cqt]") {
    // |CQT(a*x)| == a*|CQT(x)|. Catches a stray absolute offset or a
    // normalisation that depends on signal energy rather than the kernel.
    auto p = btc_params();
    const auto kernels = core_cqt::build_kernels(p);
    const auto x1 = tone(SR, 261.6255653005986, 0.25);
    const auto x2 = tone(SR, 261.6255653005986, 0.75); // exactly 3x
    std::vector<float> m1, m2;
    const int T1 = core_cqt::magnitude(p, kernels, x1.data(), (int)x1.size(), m1);
    const int T2 = core_cqt::magnitude(p, kernels, x2.data(), (int)x2.size(), m2);
    REQUIRE(T1 == T2);
    REQUIRE(m1.size() == m2.size());

    int checked = 0;
    for (size_t i = 0; i < m1.size(); i++) {
        if (m1[i] < 1e-4f)
            continue; // ratios of near-zero bins are numerical noise
        REQUIRE_THAT((double)m2[i], Catch::Matchers::WithinRel(3.0 * (double)m1[i], 1e-3));
        checked++;
    }
    REQUIRE(checked > 100);
}

TEST_CASE("cqt: a two-tone mix keeps both peaks", "[core][cqt]") {
    // Superposition at the level that matters for chords: a triad must not
    // collapse into one bin.
    auto p = btc_params();
    const auto kernels = core_cqt::build_kernels(p);
    const double f_lo = 261.6255653005986;  // C4
    const double f_hi = 391.99543598174927; // G4
    const auto a = tone(SR, f_lo, 0.4);
    const auto b = tone(SR, f_hi, 0.4);
    std::vector<float> mix(a.size());
    for (size_t i = 0; i < a.size(); i++)
        mix[i] = a[i] + b[i];

    std::vector<float> m;
    const int T = core_cqt::magnitude(p, kernels, mix.data(), (int)mix.size(), m);
    const int t = T / 2;
    const int k_lo = expected_bin(p, f_lo);
    const int k_hi = expected_bin(p, f_hi);
    REQUIRE(k_lo != k_hi);

    const float* frame = &m[(size_t)t * (size_t)p.n_bins];
    // Both must stand well clear of the median bin.
    std::vector<float> sorted(frame, frame + p.n_bins);
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted[(size_t)p.n_bins / 2];
    INFO("lo " << frame[k_lo] << " hi " << frame[k_hi] << " median " << median);
    REQUIRE(frame[k_lo] > 5.0f * median);
    REQUIRE(frame[k_hi] > 5.0f * median);
}

TEST_CASE("cqt: frame count and degenerate inputs", "[core][cqt]") {
    auto p = btc_params();

    // n_frames must agree with what magnitude() actually produces, for lengths
    // spanning "shorter than one hop" to "many hops".
    for (int n : {1, HOP - 1, HOP, HOP + 1, 4 * HOP, SR}) {
        std::vector<float> x((size_t)n, 0.1f);
        std::vector<float> m;
        const int T = core_cqt::magnitude(p, x.data(), n, m);
        INFO("n_samples " << n);
        REQUIRE(T == core_cqt::n_frames(p, n));
        REQUIRE(T >= 0);
        REQUIRE(m.size() == (size_t)T * (size_t)p.n_bins);
        for (float v : m)
            REQUIRE(std::isfinite(v));
    }

    // Zero-length input must not read anything or emit a frame.
    std::vector<float> empty_out;
    const int T0 = core_cqt::magnitude(p, nullptr, 0, empty_out);
    REQUIRE(T0 == core_cqt::n_frames(p, 0));
    REQUIRE(empty_out.size() == (size_t)T0 * (size_t)p.n_bins);
}

TEST_CASE("cqt: output is finite for a full-scale and a DC signal", "[core][cqt]") {
    // A DC input has no CQT bin (fmin is C1) but must not produce NaN, and
    // full-scale input must not overflow.
    auto p = btc_params();
    const auto kernels = core_cqt::build_kernels(p);
    for (float level : {1.0f, -1.0f}) {
        std::vector<float> dc((size_t)SR, level);
        std::vector<float> m;
        core_cqt::magnitude(p, kernels, dc.data(), (int)dc.size(), m);
        for (float v : m) {
            REQUIRE(std::isfinite(v));
            REQUIRE(v >= 0.0f);
        }
    }
}

// `test-core-cqt --dump <file>` writes the shared signal's magnitude matrix for
// tools/cqt_librosa_parity.py. Not part of the Catch2 run.
// TabCNN's front end, which is NOT BTC's. Read from the checkpoint and from
// amt_tools/datasets/GuitarSet.py — see tools/reference_backends/tabcnn.py.
// The two differ in ways that matter for reuse: hop 512 vs 2048 means 86 fps
// instead of 10.8, so `core/cqt.h`'s documented transition-frame divergence
// (per-frame correlation min 0.1136 at tone boundaries, from librosa's
// recursive per-octave downsampling giving each octave a different group
// delay) is far more consequential here. BTC scores 10-second chord segments
// where boundary latency is immaterial; a frame-level tablature model at 86 fps
// is exactly the case that comment says is the known gap.
core_cqt::Params tabcnn_params() {
    core_cqt::Params p;
    p.sample_rate = 22050;
    p.fmin = 32.703195662574829f; // C1 -- NOT the guitar low E; see below
    p.n_bins = 192;
    p.bins_per_octave = 24;
    p.hop_length = 512;
    return p;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--dump-tabcnn") == 0 && i + 1 < argc) {
            const auto p = tabcnn_params();
            const int T_target = 40;
            // Optional third arg: a raw float32 mono file already at
            // p.sample_rate. Synthetic tones are a harsh, unrepresentative
            // leakage test -- most bins carry only sidelobe energy -- so the
            // parity verdict has to be re-taken on real guitar audio before it
            // means anything. Python does the decode/resample; this stays a
            // dependency-free raw read.
            std::vector<float> x;
            if (i + 2 < argc) {
                FILE* in = std::fopen(argv[i + 2], "rb");
                if (!in) {
                    std::fprintf(stderr, "cannot open %s\n", argv[i + 2]);
                    return 1;
                }
                std::fseek(in, 0, SEEK_END);
                const long bytes = std::ftell(in);
                std::fseek(in, 0, SEEK_SET);
                x.resize((size_t)bytes / sizeof(float));
                if (std::fread(x.data(), sizeof(float), x.size(), in) != x.size()) {
                    std::fclose(in);
                    std::fprintf(stderr, "short read on %s\n", argv[i + 2]);
                    return 1;
                }
                std::fclose(in);
                std::fprintf(stderr, "read %zu samples from %s\n", x.size(), argv[i + 2]);
            } else {
                x = shared_test_signal(T_target * p.hop_length, p.sample_rate);
            }
            std::vector<float> m;
            const int T = core_cqt::magnitude(p, x.data(), (int)x.size(), m);
            FILE* f = std::fopen(argv[i + 1], "wb");
            if (!f)
                return 1;
            const int32_t hdr[2] = {(int32_t)T, (int32_t)p.n_bins};
            std::fwrite(hdr, sizeof(int32_t), 2, f);
            std::fwrite(m.data(), sizeof(float), m.size(), f);
            std::fclose(f);
            std::printf("wrote %s (%d frames x %d bins, TabCNN params)\n", argv[i + 1], T, p.n_bins);
            return 0;
        }
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            const auto p = btc_params();
            const int T_target = 40;
            const auto x = shared_test_signal(T_target * HOP);
            std::vector<float> m;
            const int T = core_cqt::magnitude(p, x.data(), (int)x.size(), m);
            FILE* f = std::fopen(argv[i + 1], "wb");
            if (!f)
                return 1;
            const int32_t hdr[2] = {(int32_t)T, (int32_t)p.n_bins};
            std::fwrite(hdr, sizeof(int32_t), 2, f);
            std::fwrite(m.data(), sizeof(float), m.size(), f);
            std::fclose(f);
            std::printf("wrote %s (%d frames x %d bins)\n", argv[i + 1], T, p.n_bins);
            return 0;
        }
    }
    return Catch::Session().run(argc, argv);
}

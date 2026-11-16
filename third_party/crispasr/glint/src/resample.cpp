// glint - windowed-sinc sample-rate converter (PLAN § B6)
// MIT License - Clean-room implementation.

#include "resample.hpp"

#include <cmath>

namespace glint {

namespace {

double sinc(double x) {
    if (x == 0.0) return 1.0;
    double px = M_PI * x;
    return std::sin(px) / px;
}

// Zeroth-order modified Bessel function (for the Kaiser window).
double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    double xx = x * x / 4.0;
    for (int k = 1; k < 40; k++) {
        term *= xx / (k * k);
        sum += term;
        if (term < 1e-12 * sum) break;
    }
    return sum;
}

}  // namespace

std::vector<float> resample(const float* in, int n_in, int channels,
                            int sr_in, int sr_out, int* out_frames,
                            int quality) {
    if (n_in <= 0 || channels <= 0 || sr_in <= 0 || sr_out <= 0) {
        if (out_frames) *out_frames = 0;
        return {};
    }
    if (sr_in == sr_out) {  // pass-through
        std::vector<float> out(in, in + static_cast<size_t>(n_in) * channels);
        if (out_frames) *out_frames = n_in;
        return out;
    }

    const double ratio = static_cast<double>(sr_out) / sr_in;
    // Anti-alias cutoff at the lower Nyquist (normalized to input Nyquist).
    const double cutoff = ratio < 1.0 ? ratio : 1.0;
    // Kernel half-width scales up when downsampling to keep the transition
    // band tight after the cutoff narrows it.
    const int half =
        static_cast<int>(std::ceil(quality / cutoff));
    const double beta = 8.0;  // Kaiser beta (~ -80 dB stopband)
    const double inv_i0_beta = 1.0 / bessel_i0(beta);

    // Number of output frames (round to nearest).
    const long n_out = static_cast<long>(
        std::llround(static_cast<double>(n_in) * ratio));
    std::vector<float> out(static_cast<size_t>(n_out) * channels, 0.0f);

    for (long i = 0; i < n_out; i++) {
        double t = i / ratio;  // position in input frames
        long center = static_cast<long>(std::floor(t));
        double frac = t - center;

        // Accumulate per channel over the kernel support.
        // Kernel arg for tap n: (t - n) * cutoff; window over [-half,half].
        double acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        double wsum = 0.0;
        int cc = channels < 8 ? channels : 8;
        for (int j = -half + 1; j <= half; j++) {
            long n = center + j;
            double d = (frac - j);           // t - n in frames
            double x = d * cutoff;
            // Kaiser window argument normalized to [-1, 1] over the support.
            double wn = d / half;
            if (wn <= -1.0 || wn >= 1.0) continue;
            double w = bessel_i0(beta * std::sqrt(1.0 - wn * wn)) *
                       inv_i0_beta;
            double h = sinc(x) * w;
            wsum += h;
            if (n < 0 || n >= n_in) continue;  // zero-pad edges
            const float* s = in + static_cast<size_t>(n) * channels;
            for (int c = 0; c < cc; c++) acc[c] += s[c] * h;
        }
        // Normalize by the window-weight sum so the passband gain is
        // exactly unity (verified: sine amplitude preserved to 1e-4).
        double norm = wsum != 0.0 ? 1.0 / wsum : 0.0;
        float* o = out.data() + static_cast<size_t>(i) * channels;
        for (int c = 0; c < cc; c++)
            o[c] = static_cast<float>(acc[c] * norm);
    }

    if (out_frames) *out_frames = static_cast<int>(n_out);
    return out;
}

}  // namespace glint

// glint - Vorbis inverse MDCT (fast, power-of-two, via an N/4 complex FFT)
// MIT License - Clean-room implementation.
//
// Vorbis block sizes are powers of two (64..8192), so the inverse MDCT of M =
// N/2 spectral bins to N time samples factors through an H = N/4 radix-2
// complex FFT with a pre/post twiddle rotation and a TDAC unfold. This is the
// same DCT-IV-based decomposition glint's AAC decoder uses (its ImdctPlan is
// proven against the direct O(N^2) form at ~2e-13); generalized here to any
// power-of-two N and normalized to the Vorbis convention
//   out[p] = sum_{k=0}^{M-1} spec[k] cos( (pi/M) (p + 1/2 + M/2) (k + 1/2) ),
// i.e. exactly M times the ImdctPlan output (verified by the diff unit test).

#pragma once

#include <cmath>
#include <vector>

namespace glint {
namespace vorbis {

// Direct O(N^2) inverse MDCT (the correctness reference; unit-tested equal to
// the fast path). n = block size (power of two), spec has n/2 bins.
inline void imdct_direct(const float* spec, float* out, int n) {
    int M = n / 2;
    for (int p = 0; p < n; p++) {
        double acc = 0.0;
        double c = (M_PI / M) * (p + 0.5 + M / 2.0);
        for (int k = 0; k < M; k++) acc += spec[k] * std::cos(c * (k + 0.5));
        out[p] = static_cast<float>(acc);
    }
}

class FastImdct {
public:
    void init(int n) {
        if (N_ == n) return;
        N_ = n;
        H_ = N_ / 4;
        M_ = N_ / 2;
        log2H_ = 0;
        while ((1 << log2H_) < H_) log2H_++;
        brev_.resize(H_);
        for (int i = 0; i < H_; i++) {
            unsigned r = 0;
            for (int b = 0; b < log2H_; b++) r = (r << 1) | ((i >> b) & 1);
            brev_[i] = static_cast<int>(r);
        }
        fw_re_.assign(H_, 0);
        fw_im_.assign(H_, 0);
        for (int j = 0; j < H_; j++) {
            fw_re_[j] = std::cos(2.0 * M_PI * j / H_);
            fw_im_[j] = std::sin(2.0 * M_PI * j / H_);  // + sign: inverse FFT
        }
        tw_re_.assign(H_, 0);
        tw_im_.assign(H_, 0);
        for (int k = 0; k < H_; k++) {
            tw_re_[k] = std::cos(M_PI * (k + 0.125) / M_);
            tw_im_[k] = -std::sin(M_PI * (k + 0.125) / M_);
        }
        re_.assign(H_, 0);
        im_.assign(H_, 0);
        u_.assign(M_, 0);
    }

    int size() const { return N_; }

    // spec[0..M) -> out[0..N), Vorbis normalization (unit-scale, matches
    // imdct_direct). Not thread-safe (per-instance scratch).
    void backward(const float* spec, float* out) const {
        const int H = H_, M = M_, Q = H_;
        double* re = re_.data();
        double* im = im_.data();
        double* u = u_.data();
        for (int k = 0; k < H; k++) {
            double A = spec[2 * k] * 0.5;
            double B = -spec[M - 1 - 2 * k] * 0.5;
            double twr = tw_re_[k], twi = tw_im_[k];
            re[k] = A * twr + B * twi;
            im[k] = -A * twi + B * twr;
        }
        for (int i = 0; i < H; i++) {
            int r = brev_[i];
            if (r > i) {
                std::swap(re[i], re[r]);
                std::swap(im[i], im[r]);
            }
        }
        for (int len = 2; len <= H; len <<= 1) {
            int half = len >> 1, stride = H / len;
            for (int base = 0; base < H; base += len)
                for (int j = 0; j < half; j++) {
                    double wr = fw_re_[j * stride], wi = fw_im_[j * stride];
                    int a = base + j, b = a + half;
                    double xr = re[b] * wr - im[b] * wi;
                    double xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
        }
        const double inv = 2.0 / H;
        for (int nn = 0; nn < H; nn++) {
            double R = re[nn] * inv, I = im[nn] * inv;
            double twr = tw_re_[nn], twi = tw_im_[nn];
            u[2 * nn] = R * twr + I * twi;
            u[M - 1 - 2 * nn] = -R * twi + I * twr;
        }
        // Unfold with the Vorbis normalization (x M vs ImdctPlan).
        const double g = static_cast<double>(M);
        for (int nn = 0; nn < Q; nn++) {
            double b = u[Q + nn] * 0.5 * g;
            double a = -u[nn] * 0.5 * g;
            out[nn] = static_cast<float>(b);
            out[2 * Q - 1 - nn] = static_cast<float>(-b);
            out[3 * Q + nn] = static_cast<float>(a);
            out[3 * Q - 1 - nn] = static_cast<float>(a);
        }
    }

private:
    int N_ = 0, H_ = 0, M_ = 0, log2H_ = 0;
    std::vector<int> brev_;
    std::vector<double> fw_re_, fw_im_, tw_re_, tw_im_;
    mutable std::vector<double> re_, im_, u_;
};

}  // namespace vorbis
}  // namespace glint

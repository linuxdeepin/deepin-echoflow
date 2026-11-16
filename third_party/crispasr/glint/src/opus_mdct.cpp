// glint - Opus/CELT inverse MDCT (PLAN.md § O1)
// MIT License - Clean-room implementation
//
// See opus_mdct.hpp for the full contract. The pipeline mirrors the
// reference decomposition (an S/2-point complex FFT does the heavy
// lifting for an S-bin inverse MDCT) but is derived independently:
//
//   pre-rotate:   z_i = (X[2i] + j*X[S-1-2i]) * e^{-j theta_i},
//                 theta_i = 2 pi (i + 1/8) / N,   i = 0 .. S/2 - 1
//   FFT:          Z = DFT_{S/2}(z)                (forward, unscaled)
//   post-rotate:  W_i = Z_i * e^{-j theta_i}
//                 u[2i] = Im(W_i),  u[S-1-2i] = -Re(W_i)
//   mirror/TDAC:  window the first `overlap` output samples against the
//                 previous block's tail (contract in the header).
//
// which computes u[j] = sum_k X[k] cos(pi/S (j + S + 1/2)(k + 1/2)) —
// the middle half of the time-aliased 2S-point inverse MDCT, unscaled.
// tools/opus_mdct_crosscheck.cpp proves this equivalence against the
// direct O(S^2) formula (< 1e-9) and against libopus (float, < 2e-4).

#include "opus_mdct.hpp"

#include <cassert>
#include <cmath>

namespace glint {
namespace opus {

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

double mdct_window(int i, int overlap) {
    double s = std::sin(0.5 * kPi * (i + 0.5) / overlap);
    return std::sin(0.5 * kPi * s * s);
}

void mdct_window_fill(double* w, int overlap) {
    for (int i = 0; i < overlap; i++) w[i] = mdct_window(i, overlap);
}

// ---------------------------------------------------------------------------
// Mixed-radix FFT
// ---------------------------------------------------------------------------

void CeltImdct::MixedFft::init(int size) {
    n = size;
    tw.resize(static_cast<size_t>(n));
    for (int k = 0; k < n; k++) {
        double a = 2.0 * kPi * k / n;
        tw[static_cast<size_t>(k)] = {std::cos(a), -std::sin(a)};
    }
    // The recursion only handles radices 2/3/4/5; reject anything else here
    // rather than deep in recurse().
    int m = n;
    while (m % 2 == 0) m /= 2;
    while (m % 3 == 0) m /= 3;
    while (m % 5 == 0) m /= 5;
    assert(m == 1 && "FFT size must factor into 2/3/5");
    (void)m;
}

void CeltImdct::MixedFft::run(const Cpx* x, Cpx* y) const {
    recurse(x, y, n, 1);
}

// Cooley-Tukey DIT, natural order. Decimate the input by radix r: the r
// sub-DFTs of x[q*stride :: r*stride] land in y[q*m .. (q+1)*m), then the
// combine stage rebuilds X[k + s*m] = sum_q Y_q[k] W^{qk} W_r^{qs} in place
// (each k only touches the slots {y[q*m + k]}, so a length-r temp suffices).
void CeltImdct::MixedFft::recurse(const Cpx* x, Cpx* y, int len,
                                  int stride) const {
    if (len == 1) {
        y[0] = x[0];
        return;
    }
    int r = (len % 4 == 0) ? 4 : (len % 2 == 0) ? 2 : (len % 3 == 0) ? 3 : 5;
    const int m = len / r;
    for (int q = 0; q < r; q++)
        recurse(x + static_cast<size_t>(q) * stride, y + q * m, m, stride * r);

    const size_t step = static_cast<size_t>(n / len);  // tw step for 1/len
    Cpx t[5];
    for (int k = 0; k < m; k++) {
        for (int q = 0; q < r; q++) {
            Cpx v = y[q * m + k];
            Cpx w = tw[static_cast<size_t>(q) * k * step % n];
            t[q] = {v.re * w.re - v.im * w.im, v.re * w.im + v.im * w.re};
        }
        for (int s = 0; s < r; s++) {
            // W_r^{qs} = tw[q*s*m*step mod n] (m*step = n/r)
            double ar = t[0].re, ai = t[0].im;
            for (int q = 1; q < r; q++) {
                Cpx w = tw[static_cast<size_t>(q) * s * m * step % n];
                ar += t[q].re * w.re - t[q].im * w.im;
                ai += t[q].re * w.im + t[q].im * w.re;
            }
            y[s * m + k] = {ar, ai};
        }
    }
}

// ---------------------------------------------------------------------------
// CeltImdct
// ---------------------------------------------------------------------------

CeltImdct::CeltImdct(int n, int maxshift) : n_(n), maxshift_(maxshift) {
    assert(n > 0 && maxshift >= 0 && (n >> maxshift) % 4 == 0);
    trig_.resize(static_cast<size_t>(maxshift) + 1);
    fft_.resize(static_cast<size_t>(maxshift) + 1);
    for (int shift = 0; shift <= maxshift; shift++) {
        const int N = n >> shift;
        auto& t = trig_[static_cast<size_t>(shift)];
        t.resize(static_cast<size_t>(N / 2));
        for (int i = 0; i < N / 2; i++)
            t[static_cast<size_t>(i)] = std::cos(2.0 * kPi * (i + 0.125) / N);
        fft_[static_cast<size_t>(shift)].init(N / 4);
    }
    scratch_.resize(static_cast<size_t>(n / 4) * 2);
}

void CeltImdct::backward(const double* in, double* out, const double* window,
                         int overlap, int shift, int stride) const {
    assert(shift >= 0 && shift <= maxshift_);
    const int N = n_ >> shift;
    const int N2 = N >> 1;  // spectral bins S
    const int N4 = N >> 2;
    assert(overlap >= 0 && overlap % 2 == 0 && overlap <= N2);
    const double* t = trig_[static_cast<size_t>(shift)].data();
    Cpx* z = scratch_.data();       // pre-rotated input
    Cpx* Z = scratch_.data() + N4;  // FFT output

    // Pre-rotate. t[i] = cos(theta_i), t[N4+i] = -sin(theta_i), so
    // z_i = (X[2i] + j*X[S-1-2i]) e^{-j theta_i}. All input reads happen
    // here, before any output write (in-place contract, see header).
    for (int i = 0; i < N4; i++) {
        double a = in[static_cast<size_t>(stride) * (2 * i)];
        double b = in[static_cast<size_t>(stride) * (N2 - 1 - 2 * i)];
        z[i] = {a * t[i] - b * t[N4 + i], b * t[i] + a * t[N4 + i]};
    }

    fft_[static_cast<size_t>(shift)].run(z, Z);

    // Post-rotate: W_i = Z_i e^{-j theta_i}; u[2i] = Im(W_i),
    // u[S-1-2i] = -Re(W_i), stored at out[overlap/2 + .].
    // (t[N4+i] = -sin(theta_i), so Im(W) = im*t[i] + re*t[N4+i] and
    //  -Re(W) = im*t[N4+i] - re*t[i].)
    double* u = out + overlap / 2;
    for (int i = 0; i < N4; i++) {
        double re = Z[i].re, im = Z[i].im;
        u[2 * i] = im * t[i] + re * t[N4 + i];           //  Im(W_i)
        u[N2 - 1 - 2 * i] = im * t[N4 + i] - re * t[i];  // -Re(W_i)
    }

    // Mirror/TDAC: cross-fade the previous block's raw tail (old
    // out[0..ov/2)) with this block's u[0..ov/2) under the window.
    for (int i = 0; i < overlap / 2; i++) {
        double prev = out[i];
        double cur = out[overlap - 1 - i];  // == u[overlap/2 - 1 - i]
        out[i] = window[overlap - 1 - i] * prev - window[i] * cur;
        out[overlap - 1 - i] = window[i] * prev + window[overlap - 1 - i] * cur;
    }
}

void CeltImdct::forward(const double* in, double* out, const double* window,
                        int overlap, int shift, int stride) const {
    assert(shift >= 0 && shift <= maxshift_);
    const int N = n_ >> shift;
    const int N2 = N >> 1;  // spectral bins S
    const int N4 = N >> 2;
    const int ov = overlap;
    assert(ov >= 0 && ov % 2 == 0 && ov <= N2);
    const double* t = trig_[static_cast<size_t>(shift)].data();
    Cpx* z = scratch_.data();       // folded + pre-rotated input
    Cpx* Z = scratch_.data() + N4;  // FFT output
    const double scale = 4.0 / N;   // the reference FFT's 1/nfft, nfft = N/4

    // Windowed input read: W(m) * in[m], with W rising over [0, ov),
    // flat 1 over [ov, S), falling over [S, S+ov), and 0 outside the
    // buffer (= the closed form's implicit zero padding to 2S samples).
    auto wx = [&](int m) -> double {
        if (m < 0 || m >= N2 + ov) return 0.0;
        double v = in[m];
        if (m < ov) return v * window[m];
        if (m >= N2) return v * window[N2 + ov - 1 - m];
        return v;
    };

    // Fold the (conceptually 2S-point) windowed frame down to S real
    // values = S/2 complex pairs using the MDCT kernel's symmetries
    // (kernel phase m + S - ov/2 + 1/2 in buffer coordinates; antiperiod
    // S in the conceptual frame index), then pre-rotate by e^{-j theta_i},
    // theta_i = 2 pi (i + 1/8) / N (t[i] = cos, t[N4+i] = -sin), scaled.
    // All input reads happen here, before any output write.
    for (int i = 0; i < N4; i++) {
        double im = wx(ov / 2 + 2 * i) - wx(ov / 2 - 1 - 2 * i) +
                    wx(N + ov / 2 - 1 - 2 * i);
        double re = wx(N2 - 1 + ov / 2 - 2 * i) + wx(N2 + ov / 2 + 2 * i) -
                    wx(ov / 2 + 2 * i - N2);
        z[i] = {scale * (re * t[i] - im * t[N4 + i]),
                scale * (im * t[i] + re * t[N4 + i])};
    }

    fft_[static_cast<size_t>(shift)].run(z, Z);

    // Post-rotate: W_i = Z_i e^{-j theta_i}; X[2i] = -Re(W_i),
    // X[S-1-2i] = Im(W_i)  (t[N4+i] = -sin(theta_i), so
    // -Re(W) = im*t[N4+i] - re*t[i] and Im(W) = re*t[N4+i] + im*t[i]).
    for (int i = 0; i < N4; i++) {
        double re = Z[i].re, im = Z[i].im;
        out[static_cast<size_t>(stride) * (2 * i)] = im * t[N4 + i] - re * t[i];
        out[static_cast<size_t>(stride) * (N2 - 1 - 2 * i)] =
            re * t[N4 + i] + im * t[i];
    }
}

}  // namespace opus
}  // namespace glint

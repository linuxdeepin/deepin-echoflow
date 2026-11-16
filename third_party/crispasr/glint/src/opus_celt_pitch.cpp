// CELT pitch estimation + comb filter (shared) — see header
// MIT License - Clean-room implementation

#include "opus_celt_pitch.hpp"

#include <cmath>
#include <cstring>

namespace glint {
namespace opus {

namespace pitch {
namespace detail {
constexpr int kMaxPeriod = 1024;
// Buffer bounds covering both callers (PLC: len 1328/max_pitch 620;
// prefilter: len up to 960 / max_pitch 979), full-rate units.
constexpr int kPlcSearchLen = 1408;
constexpr int kPlcMaxPitch = 1024;
inline int imax(int a, int b) { return a > b ? a : b; }
// Autocorrelation ac[0..lag] of x[0..n), with an optional analysis window
// applied to `overlap` samples at both ends. n <= kMaxPeriod.
void celt_autocorr_impl(const double* x, double* ac, const double* window,
                   int overlap, int lag, int n) {
    double xx[kMaxPeriod];
    const double* xptr = x;
    if (overlap != 0) {
        for (int i = 0; i < n; i++) xx[i] = x[i];
        for (int i = 0; i < overlap; i++) {
            xx[i] = x[i] * window[i];
            xx[n - i - 1] = x[n - i - 1] * window[i];
        }
        xptr = xx;
    }
    for (int k = 0; k <= lag; k++) {
        double d = 0;
        for (int i = k; i < n; i++) d += xptr[i] * xptr[i - k];
        ac[k] = d;
    }
}

// Levinson-Durbin recursion, with the reference's early exit once the
// prediction error drops 30 dB below ac[0].
void celt_lpc(double* lpc, const double* ac, int p) {
    for (int i = 0; i < p; i++) lpc[i] = 0;
    double error = ac[0];
    if (ac[0] > 1e-10) {
        for (int i = 0; i < p; i++) {
            double rr = 0;  // this iteration's reflection coefficient
            for (int j = 0; j < i; j++) rr += lpc[j] * ac[i - j];
            rr += ac[i + 1];
            double r = -rr / error;
            lpc[i] = r;
            for (int j = 0; j < (i + 1) >> 1; j++) {
                double tmp1 = lpc[j];
                double tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + r * tmp2;
                lpc[i - 1 - j] = tmp2 + r * tmp1;
            }
            error = error - r * r * error;
            if (error <= 0.001 * ac[0]) break;
        }
    }
}

// Whitening FIR y[i] = x[i] + sum_k num[k-1]*x[i-k]; x carries `ord`
// history samples before x[0]. x and y must not alias.
void celt_fir(const double* x, const double* num, double* y, int n,
              int ord) {
    for (int i = 0; i < n; i++) {
        double sum = x[i];
        for (int j = 0; j < ord; j++) sum += num[j] * x[i - 1 - j];
        y[i] = sum;
    }
}

// Synthesis IIR y[i] = x[i] - sum_k den[k-1]*y[i-k]. mem[k] holds y[-1-k]
// on entry and the last `ord` outputs on exit. In-place safe (x == y).
void celt_iir(const double* x, const double* den, double* y, int n, int ord,
              double* mem) {
    for (int i = 0; i < n; i++) {
        double sum = x[i];
        for (int j = 0; j < ord; j++) {
            int k = i - 1 - j;
            sum -= den[j] * (k >= 0 ? y[k] : mem[-k - 1]);
        }
        y[i] = sum;
    }
    for (int i = 0; i < ord; i++) mem[i] = y[n - 1 - i];
}

// In-place 5-tap all-zero filter (taps run against the pre-filter input).
void celt_fir5(double* x, const double* num, int n) {
    double mem0 = 0, mem1 = 0, mem2 = 0, mem3 = 0, mem4 = 0;
    for (int i = 0; i < n; i++) {
        double sum = x[i] + num[0] * mem0 + num[1] * mem1 + num[2] * mem2 +
                     num[3] * mem3 + num[4] * mem4;
        mem4 = mem3;
        mem3 = mem2;
        mem2 = mem1;
        mem1 = mem0;
        mem0 = x[i];
        x[i] = sum;
    }
}

// 2x downsample (channel-summed) + adaptive whitening pre-filter over the
// decode history. len is the input length; writes x_lp[0 .. len/2).
void pitch_downsample(const double* const* x, double* x_lp, int len, int C) {
    for (int i = 1; i < len >> 1; i++)
        x_lp[i] = 0.25 * x[0][2 * i - 1] + 0.25 * x[0][2 * i + 1] +
                  0.5 * x[0][2 * i];
    x_lp[0] = 0.25 * x[0][1] + 0.5 * x[0][0];
    if (C == 2) {
        for (int i = 1; i < len >> 1; i++)
            x_lp[i] += 0.25 * x[1][2 * i - 1] + 0.25 * x[1][2 * i + 1] +
                       0.5 * x[1][2 * i];
        x_lp[0] += 0.25 * x[1][1] + 0.5 * x[1][0];
    }
    double ac[5];
    celt_autocorr(x_lp, ac, nullptr, 0, 4, len >> 1);
    ac[0] *= 1.0001;  // -40 dB noise floor
    for (int i = 1; i <= 4; i++)  // lag windowing (stabilizes the recursion)
        ac[i] -= ac[i] * (0.008 * i) * (0.008 * i);
    double lpc[4];
    celt_lpc(lpc, ac, 4);
    double tmp = 1.0;
    for (int i = 0; i < 4; i++) {
        tmp *= 0.9;
        lpc[i] *= tmp;
    }
    // Add a zero at 0.8 (mild high-pass emphasis for the correlator).
    const double c1 = 0.8;
    double lpc2[5];
    lpc2[0] = lpc[0] + 0.8;
    lpc2[1] = lpc[1] + c1 * lpc[0];
    lpc2[2] = lpc[2] + c1 * lpc[1];
    lpc2[3] = lpc[3] + c1 * lpc[2];
    lpc2[4] = c1 * lpc[3];
    celt_fir5(x_lp, lpc2, len >> 1);
}

// Track the two best normalized-correlation peaks. Keeps the reference's
// 1e-12 rescale before squaring (overflow guard; harmless in double).
void find_best_pitch(const double* xcorr, const double* y, int len,
                     int max_pitch, int* best_pitch) {
    double Syy = 1;
    double best_num[2] = { -1, -1 };
    double best_den[2] = { 0, 0 };
    best_pitch[0] = 0;
    best_pitch[1] = 1;
    for (int j = 0; j < len; j++) Syy += y[j] * y[j];
    for (int i = 0; i < max_pitch; i++) {
        if (xcorr[i] > 0) {
            double xcorr16 = xcorr[i] * 1e-12;
            double num = xcorr16 * xcorr16;
            if (num * best_den[1] > best_num[1] * Syy) {
                if (num * best_den[0] > best_num[0] * Syy) {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = Syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = Syy;
                    best_pitch[1] = i;
                }
            }
        }
        Syy += y[i + len] * y[i + len] - y[i] * y[i];
        Syy = Syy > 1 ? Syy : 1;
    }
}

double inner_prod(const double* a, const double* b, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// Correlation pitch search: coarse pass at 4x decimation, refinement of the
// two best candidates at 2x, then pseudo-interpolation. x_lp/y are already
// 2x-decimated; len and max_pitch count pre-decimation samples
// (len <= kPlcSearchLen, max_pitch <= kPlcMaxPitch).
void pitch_search(const double* x_lp, const double* y, int len, int max_pitch,
                  int* pitch) {
    int lag = len + max_pitch;
    double x_lp4[kPlcSearchLen >> 2];
    double y_lp4[(kPlcSearchLen + kPlcMaxPitch) >> 2];
    double xcorr[kPlcMaxPitch >> 1];
    int best_pitch[2] = { 0, 0 };
    for (int j = 0; j < len >> 2; j++) x_lp4[j] = x_lp[2 * j];
    for (int j = 0; j < lag >> 2; j++) y_lp4[j] = y[2 * j];
    for (int i = 0; i < max_pitch >> 2; i++)
        xcorr[i] = inner_prod(x_lp4, y_lp4 + i, len >> 2);
    find_best_pitch(xcorr, y_lp4, len >> 2, max_pitch >> 2, best_pitch);
    for (int i = 0; i < max_pitch >> 1; i++) {
        xcorr[i] = 0;
        int d0 = i - 2 * best_pitch[0];
        int d1 = i - 2 * best_pitch[1];
        if ((d0 > 2 || d0 < -2) && (d1 > 2 || d1 < -2)) continue;
        double sum = inner_prod(x_lp, y + i, len >> 1);
        xcorr[i] = sum < -1 ? -1 : sum;
    }
    find_best_pitch(xcorr, y, len >> 1, max_pitch >> 1, best_pitch);
    int offset = 0;
    if (best_pitch[0] > 0 && best_pitch[0] < (max_pitch >> 1) - 1) {
        double a = xcorr[best_pitch[0] - 1];
        double b = xcorr[best_pitch[0]];
        double c = xcorr[best_pitch[0] + 1];
        if (c - a > 0.7 * (b - a))
            offset = 1;
        else if (a - c > 0.7 * (b - c))
            offset = -1;
    }
    *pitch = 2 * best_pitch[0] - offset;
}

// Unit-norm rescale (reference renormalise_vector, float path).
}  // namespace detail

// Public thin wrappers over the detail bodies.
void celt_autocorr(const double* x, double* ac, const double* window,
                   int overlap, int lag, int n) {
    detail::celt_autocorr_impl(x, ac, window, overlap, lag, n);
}
void celt_lpc(double* lpc, const double* ac, int p) {
    detail::celt_lpc(lpc, ac, p);
}
void pitch_downsample(const double* const* x, double* x_lp, int len,
                      int C) {
    detail::pitch_downsample(x, x_lp, len, C);
}
void pitch_search(const double* x_lp, const double* y, int len,
                  int max_pitch, int* pitch) {
    detail::pitch_search(x_lp, y, len, max_pitch, pitch);
}
}  // namespace pitch

namespace pitch {

namespace {
inline double compute_pitch_gain(double xy, double xx, double yy) {
    return xy / std::sqrt(1.0 + xx * yy);
}
const int kSecondCheck[16] = { 0, 0, 3, 2, 3, 2, 5, 2,
                               3, 2, 3, 2, 5, 2, 3, 2 };
}  // namespace

double remove_doubling(double* x, int maxperiod, int minperiod, int n,
                       int* t0_, int prev_period, double prev_gain) {
    int minperiod0 = minperiod;
    maxperiod /= 2;
    minperiod /= 2;
    *t0_ /= 2;
    prev_period /= 2;
    n /= 2;
    x += maxperiod;
    if (*t0_ >= maxperiod) *t0_ = maxperiod - 1;

    int t0 = *t0_;
    int t = t0;
    double yy_lookup[1024 / 2 + 1];
    double xx = 0, xy = 0;
    for (int i = 0; i < n; i++) {
        xx += x[i] * x[i];
        xy += x[i] * x[i - t0];
    }
    yy_lookup[0] = xx;
    double yy = xx;
    for (int i = 1; i <= maxperiod; i++) {
        yy = yy + x[-i] * x[-i] - x[n - i] * x[n - i];
        yy_lookup[i] = yy > 0 ? yy : 0;
    }
    yy = yy_lookup[t0];
    double best_xy = xy, best_yy = yy;
    double g0 = compute_pitch_gain(xy, xx, yy);
    double g = g0;
    // Probe T0/k (and a second correlate) for the true fundamental.
    for (int k = 2; k <= 15; k++) {
        int t1 = (2 * t0 + k) / (2 * k);
        if (t1 < minperiod) break;
        int t1b;
        if (k == 2)
            t1b = t1 + t0 > maxperiod ? t0 : t0 + t1;
        else
            t1b = (2 * kSecondCheck[k] * t0 + k) / (2 * k);
        double xy1 = 0, xy2 = 0;
        for (int i = 0; i < n; i++) {
            xy1 += x[i] * x[i - t1];
            xy2 += x[i] * x[i - t1b];
        }
        double xyk = 0.5 * (xy1 + xy2);
        double yyk = 0.5 * (yy_lookup[t1] + yy_lookup[t1b]);
        double g1 = compute_pitch_gain(xyk, xx, yyk);
        double cont;
        if (t1 - prev_period <= 1 && t1 - prev_period >= -1)
            cont = prev_gain;
        else if (t1 - prev_period <= 2 && t1 - prev_period >= -2 &&
                 5 * k * k < t0)
            cont = 0.5 * prev_gain;
        else
            cont = 0;
        double thresh = 0.7 * g0 - cont;
        if (thresh < 0.3) thresh = 0.3;
        // Bias against very short periods (short-term correlation traps).
        if (t1 < 3 * minperiod) {
            thresh = 0.85 * g0 - cont;
            if (thresh < 0.4) thresh = 0.4;
        } else if (t1 < 2 * minperiod) {
            thresh = 0.9 * g0 - cont;
            if (thresh < 0.5) thresh = 0.5;
        }
        if (g1 > thresh) {
            best_xy = xyk;
            best_yy = yyk;
            t = t1;
            g = g1;
        }
    }
    if (best_xy < 0) best_xy = 0;
    double pg = best_yy <= best_xy ? 1.0 : best_xy / (best_yy + 1);
    double xcorr[3];
    for (int k = 0; k < 3; k++) {
        double s = 0;
        for (int i = 0; i < n; i++) s += x[i] * x[i - (t + k - 1)];
        xcorr[k] = s;
    }
    int offset;
    if (xcorr[2] - xcorr[0] > 0.7 * (xcorr[1] - xcorr[0]))
        offset = 1;
    else if (xcorr[0] - xcorr[2] > 0.7 * (xcorr[1] - xcorr[2]))
        offset = -1;
    else
        offset = 0;
    if (pg > g) pg = g;
    *t0_ = 2 * t + offset;
    if (*t0_ < minperiod0) *t0_ = minperiod0;
    return pg;
}

}  // namespace pitch

namespace {
constexpr int kCombFilterMinPeriod = 15;
inline int imax(int a, int b) { return a > b ? a : b; }
}

// Pitch postfilter: 5-tap comb at period T, cross-faded over the overlap
// from the previous frame's filter parameters.
void comb_filter_shared(double* y, double* x, int t0, int t1, int n, double g0,
                 double g1, int tapset0, int tapset1, const double* window,
                 int overlap) {
    static const double kGains[3][3] = {
        { 0.3066406250, 0.2170410156, 0.1296386719 },
        { 0.4638671875, 0.2680664062, 0.0 },
        { 0.7998046875, 0.1000976562, 0.0 },
    };
    if (g0 == 0 && g1 == 0) {
        if (x != y) std::memmove(y, x, n * sizeof(double));
        return;
    }
    t0 = imax(t0, kCombFilterMinPeriod);
    t1 = imax(t1, kCombFilterMinPeriod);
    double g00 = g0 * kGains[tapset0][0];
    double g01 = g0 * kGains[tapset0][1];
    double g02 = g0 * kGains[tapset0][2];
    double g10 = g1 * kGains[tapset1][0];
    double g11 = g1 * kGains[tapset1][1];
    double g12 = g1 * kGains[tapset1][2];
    double x1 = x[-t1 + 1];
    double x2 = x[-t1];
    double x3 = x[-t1 - 1];
    double x4 = x[-t1 - 2];
    if (g0 == g1 && t0 == t1 && tapset0 == tapset1) overlap = 0;
    int i;
    for (i = 0; i < overlap; i++) {
        double x0 = x[i - t1 + 2];
        double f = window[i] * window[i];
        y[i] = x[i] + (1 - f) * g00 * x[i - t0] +
               (1 - f) * g01 * (x[i - t0 + 1] + x[i - t0 - 1]) +
               (1 - f) * g02 * (x[i - t0 + 2] + x[i - t0 - 2]) +
               f * g10 * x2 + f * g11 * (x1 + x3) + f * g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
    if (g1 == 0) {
        if (x != y)
            std::memmove(y + overlap, x + overlap,
                         (n - overlap) * sizeof(double));
        return;
    }
    // Constant-parameter tail.
    x1 = x[i - t1 + 1];
    x2 = x[i - t1];
    x3 = x[i - t1 - 1];
    x4 = x[i - t1 - 2];
    for (; i < n; i++) {
        double x0 = x[i - t1 + 2];
        y[i] = x[i] + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

}  // namespace opus
}  // namespace glint

// glint - AAC forward MDCT (all four window sequences)
// MIT License - Clean-room implementation

#include "aac_mdct.hpp"

#include <cmath>
#include <cstdint>

namespace glint {
namespace aac {

namespace {

constexpr double kPi = 3.14159265358979323846;

#ifdef GLINT_AAC_INT

// ---------------------------------------------------------------------------
// Integer MDCT (no-FPU hot path). Fixed scaling chain:
//   z = x * win                (Q30 window, >>18)        -> Q12
//   fold                                                  Q13
//   pre-twiddle (Q30, >>30)                               Q13
//   FFT with (v+1)>>1 per stage (H=512: 9, H=64: 6)       /H
//   post-twiddle (Q30, >>30), out = P<<1
// Long:  spec_int = spec_true * 2^(12-9) = Q3 (the ISO 2x is part of the
// fold/DCT-IV algebra, not an extra bit)
// Short: spec_int = spec_true * 2^(12-6) = Q6 -> >>3 -> Q3
// kSpecFracBits == 3. Worst-case |spec_true| ~8.5e7, *8 = 6.8e8 < 2^31.
// KNOWN LIMIT: integer M/S halving loses half an LSB on odd l+-r sums —
// the decoder's l=m+s cannot recover it. Proven by the L==R experiment
// (identical channels -> INT == float exactly; real stereo pays ~1 dB NMR
// at -q speed). Counter-intuitively Q4 measured WORSE than Q3 overall, so
// Q3 stands; a per-frame block exponent is the real fix if ever needed.
// ---------------------------------------------------------------------------

template <int H, int LOG2H>
struct FftSetInt {
    int32_t tw_re[H];       // Q30: e^{-i pi (n + 1/8) / (2H)}
    int32_t tw_im[H];
    int32_t fft_re[H / 2];  // Q30: e^{-i 2pi j / H}
    int32_t fft_im[H / 2];

    FftSetInt() {
        const int M = 2 * H;
        const double q = 1073741824.0;  // 2^30
        for (int n = 0; n < H; n++) {
            double a = kPi * (n + 0.125) / M;
            tw_re[n] = static_cast<int32_t>(std::lround(std::cos(a) * q));
            tw_im[n] = static_cast<int32_t>(std::lround(-std::sin(a) * q));
        }
        for (int j = 0; j < H / 2; j++) {
            double a = 2.0 * kPi * j / H;
            fft_re[j] = static_cast<int32_t>(std::lround(std::cos(a) * q));
            fft_im[j] = static_cast<int32_t>(std::lround(-std::sin(a) * q));
        }
    }

    // Radix-2 DIT with a rounded >>1 after every stage (output = FFT/H).
    void fft(int32_t* re, int32_t* im) const {
        for (int i = 0; i < H; i++) {
            unsigned r = 0;
            for (int b = 0; b < LOG2H; b++) r = (r << 1) | ((i >> b) & 1);
            if (static_cast<int>(r) > i) {
                int32_t t = re[i]; re[i] = re[r]; re[r] = t;
                t = im[i]; im[i] = im[r]; im[r] = t;
            }
        }
        for (int len = 2; len <= H; len <<= 1) {
            int half = len >> 1;
            int stride = H / len;
            for (int base = 0; base < H; base += len) {
                for (int j = 0; j < half; j++) {
                    int64_t wr = fft_re[j * stride];
                    int64_t wi = fft_im[j * stride];
                    int a = base + j, b = a + half;
                    int32_t xr = static_cast<int32_t>((re[b] * wr - im[b] * wi) >> 30);
                    int32_t xi = static_cast<int32_t>((re[b] * wi + im[b] * wr) >> 30);
                    int32_t ar = re[a], ai = im[a];
                    re[a] = (ar + xr + 1) >> 1;
                    im[a] = (ai + xi + 1) >> 1;
                    re[b] = (ar - xr + 1) >> 1;
                    im[b] = (ai - xi + 1) >> 1;
                }
            }
        }
    }
};

struct WindowsInt {
    int32_t long_h[1024];   // Q30 sine half (symmetric)
    int32_t short_h[128];
    WindowsInt() {
        const double q = 1073741824.0;
        for (int n = 0; n < 1024; n++) {
            long_h[n] = static_cast<int32_t>(std::lround(std::sin(kPi / 2048 * (n + 0.5)) * q));
        }
        for (int n = 0; n < 128; n++) {
            short_h[n] = static_cast<int32_t>(std::lround(std::sin(kPi / 256 * (n + 0.5)) * q));
        }
    }
    int32_t long_w(int n) const { return long_h[n < 1024 ? n : 2047 - n]; }
    int32_t short_w(int n) const { return short_h[n < 128 ? n : 255 - n]; }
};

// Startup-time globals (not function-local statics): the table builders use
// double trig once, and must not be inlined into the integer hot paths.
const WindowsInt g_windows_int{};
const FftSetInt<512, 9> g_iset1024{};
const FftSetInt<64, 6> g_iset128{};
const WindowsInt& windows_int() { return g_windows_int; }
const FftSetInt<512, 9>& iset1024() { return g_iset1024; }
const FftSetInt<64, 6>& iset128() { return g_iset128; }

inline int32_t win_mul(int32_t x, int32_t w_q30) {
    // x is int16-range PCM; result Q12
    return static_cast<int32_t>((static_cast<int64_t>(x) * w_q30) >> 18);
}

// z[N] Q12 -> spec (M = N/2 coefficients) at Q(12 - LOG2H + 1)
template <int H, int LOG2H>
void mdct_core_int(const int32_t* z, const FftSetInt<H, LOG2H>& s, int32_t* spec,
                   int out_rshift) {
    const int M = 2 * H;
    const int Q = H;
    int32_t u[2 * H];
    for (int n = 0; n < Q; n++) {
        u[n] = -z[3 * Q - 1 - n] - z[3 * Q + n];
        u[Q + n] = z[n] - z[2 * Q - 1 - n];
    }
    int32_t re[H], im[H];
    for (int n = 0; n < H; n++) {
        int64_t vr = u[2 * n];
        int64_t vi = u[M - 1 - 2 * n];
        re[n] = static_cast<int32_t>((vr * s.tw_re[n] - vi * s.tw_im[n]) >> 30);
        im[n] = static_cast<int32_t>((vr * s.tw_im[n] + vi * s.tw_re[n]) >> 30);
    }
    s.fft(re, im);
    const int rs = out_rshift;
    const int32_t rnd = rs > 0 ? (1 << (rs - 1)) : 0;
    for (int k = 0; k < H; k++) {
        int64_t pr = (static_cast<int64_t>(re[k]) * s.tw_re[k] -
                      static_cast<int64_t>(im[k]) * s.tw_im[k]) >> 30;
        int64_t pi_ = (static_cast<int64_t>(re[k]) * s.tw_im[k] +
                       static_cast<int64_t>(im[k]) * s.tw_re[k]) >> 30;
        spec[2 * k] = static_cast<int32_t>((pr * 2 + rnd) >> rs);
        spec[M - 1 - 2 * k] = static_cast<int32_t>((pi_ * -2 + rnd) >> rs);
    }
}

#endif  // GLINT_AAC_INT

// FFT + twiddle set for one transform size (H-point complex FFT computing a
// 2H-point DCT-IV, i.e. a 4H-point MDCT window length... see mdct_core).
template <int H, int LOG2H>
struct FftSet {
#ifdef GLINT_SMALL_BUFFERS
    using T = float;   // table storage; arithmetic stays double
#else
    using T = double;
#endif
    T tw_re[H];        // DCT-IV pre/post twiddle e^{-i pi (n + 1/8) / (2H)}
    T tw_im[H];
    T fft_re[H / 2];   // e^{-i 2pi j / H}
    T fft_im[H / 2];

    FftSet() {
        const int M = 2 * H;  // DCT-IV size
        for (int n = 0; n < H; n++) {
            double a = kPi * (n + 0.125) / M;
            tw_re[n] = static_cast<T>(std::cos(a));
            tw_im[n] = static_cast<T>(-std::sin(a));
        }
        for (int j = 0; j < H / 2; j++) {
            double a = 2.0 * kPi * j / H;
            fft_re[j] = static_cast<T>(std::cos(a));
            fft_im[j] = static_cast<T>(-std::sin(a));
        }
    }

    void fft(double* re, double* im) const {
        for (int i = 0; i < H; i++) {
            unsigned r = 0;
            for (int b = 0; b < LOG2H; b++) r = (r << 1) | ((i >> b) & 1);
            if (static_cast<int>(r) > i) {
                double t = re[i]; re[i] = re[r]; re[r] = t;
                t = im[i]; im[i] = im[r]; im[r] = t;
            }
        }
        for (int len = 2; len <= H; len <<= 1) {
            int half = len >> 1;
            int stride = H / len;
            for (int base = 0; base < H; base += len) {
                for (int j = 0; j < half; j++) {
                    double wr = fft_re[j * stride];
                    double wi = fft_im[j * stride];
                    int a = base + j, b = a + half;
                    double xr = re[b] * wr - im[b] * wi;
                    double xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
        }
    }
};

struct Windows {
#ifdef GLINT_SMALL_BUFFERS
    using T = float;
#else
    using T = double;
#endif
    // Sine windows are symmetric (w[n] == w[N-1-n]); store the first half.
    T long_h[1024];
    T short_h[128];
    Windows() {
        for (int n = 0; n < 1024; n++) long_h[n] = static_cast<T>(std::sin(kPi / 2048 * (n + 0.5)));
        for (int n = 0; n < 128; n++) short_h[n] = static_cast<T>(std::sin(kPi / 256 * (n + 0.5)));
    }
    double long_w(int n) const { return long_h[n < 1024 ? n : 2047 - n]; }
    double short_w(int n) const { return short_h[n < 128 ? n : 255 - n]; }
};

const Windows& windows() { static const Windows w; return w; }
const FftSet<512, 9>& set1024() { static const FftSet<512, 9> s; return s; }
const FftSet<64, 6>& set128() { static const FftSet<64, 6> s; return s; }

// MDCT core: windowed input z[N] -> M = N/2 coefficients, via fold + DCT-IV.
//   u[n]     = -z[3Q-1-n] - z[3Q+n]   n = 0..Q-1   (Q = N/4)
//   u[Q+n]   =  z[n]      - z[2Q-1-n]
//   spec = 2 * DCT-IV(u)
template <int H, int LOG2H>
void mdct_core(const double* z, const FftSet<H, LOG2H>& s, SpecT* spec) {
    const int M = 2 * H;
    const int Q = H;  // N/4 == M/2 == H
    double u[2 * H];
    for (int n = 0; n < Q; n++) {
        u[n] = -z[3 * Q - 1 - n] - z[3 * Q + n];
        u[Q + n] = z[n] - z[2 * Q - 1 - n];
    }
    double re[H], im[H];
    for (int n = 0; n < H; n++) {
        double vr = u[2 * n];
        double vi = u[M - 1 - 2 * n];
        re[n] = vr * s.tw_re[n] - vi * s.tw_im[n];
        im[n] = vr * s.tw_im[n] + vi * s.tw_re[n];
    }
    s.fft(re, im);
    for (int k = 0; k < H; k++) {
        double pr = re[k] * s.tw_re[k] - im[k] * s.tw_im[k];
        double pi_ = re[k] * s.tw_im[k] + im[k] * s.tw_re[k];
        spec[2 * k] = static_cast<SpecT>(2.0 * pr);
        spec[M - 1 - 2 * k] = static_cast<SpecT>(-2.0 * pi_);
    }
}

}  // namespace

#ifdef GLINT_AAC_INT

void aac_mdct_frame(int window_sequence, const PcmT* prev, const PcmT* cur,
                    SpecT* spec) {
    const WindowsInt& w = windows_int();
    int32_t x[2048];
    for (int n = 0; n < 1024; n++) {
        x[n] = prev[n];
        x[1024 + n] = cur[n];
    }

    if (window_sequence == kSeqShort) {
        // Eight 256-point MDCTs over the centre region [448, 1600).
        int32_t z[256];
        for (int win = 0; win < 8; win++) {
            const int32_t* base = x + 448 + 128 * win;
            for (int n = 0; n < 256; n++) z[n] = win_mul(base[n], w.short_w(n));
            // short lands at Q6; >>3 normalizes to Q3 like the long transform
            mdct_core_int(z, iset128(), spec + 128 * win, 3);
        }
        return;
    }

    int32_t z[2048];
    switch (window_sequence) {
        case kSeqStart:
            for (int n = 0; n < 1024; n++) z[n] = win_mul(x[n], w.long_w(n));
            for (int n = 1024; n < 1472; n++) z[n] = x[n] * 4096;
            for (int n = 0; n < 128; n++) z[1472 + n] = win_mul(x[1472 + n], w.short_w(128 + n));
            for (int n = 1600; n < 2048; n++) z[n] = 0;
            break;
        case kSeqStop:
            for (int n = 0; n < 448; n++) z[n] = 0;
            for (int n = 0; n < 128; n++) z[448 + n] = win_mul(x[448 + n], w.short_w(n));
            for (int n = 576; n < 1024; n++) z[n] = x[n] * 4096;
            for (int n = 1024; n < 2048; n++) z[n] = win_mul(x[n], w.long_w(n));
            break;
        default:  // kSeqLong
            for (int n = 0; n < 2048; n++) z[n] = win_mul(x[n], w.long_w(n));
            break;
    }
    mdct_core_int(z, iset1024(), spec, 0);
}

#else  // !GLINT_AAC_INT

void aac_mdct_frame(int window_sequence, const PcmT* prev, const PcmT* cur,
                    SpecT* spec) {
    const Windows& w = windows();
    double x[2048];
    for (int n = 0; n < 1024; n++) {
        x[n] = static_cast<double>(prev[n]);
        x[1024 + n] = static_cast<double>(cur[n]);
    }

    if (window_sequence == kSeqShort) {
        // Eight 256-point MDCTs over the centre region [448, 1600).
        double z[256];
        for (int win = 0; win < 8; win++) {
            const double* base = x + 448 + 128 * win;
            for (int n = 0; n < 256; n++) z[n] = w.short_w(n) * base[n];
            mdct_core(z, set128(), spec + 128 * win);
        }
        return;
    }

    double z[2048];
    switch (window_sequence) {
        case kSeqStart:
            // long rise | 448 ones | short fall | 448 zeros
            for (int n = 0; n < 1024; n++) z[n] = w.long_w(n) * x[n];
            for (int n = 1024; n < 1472; n++) z[n] = x[n];
            for (int n = 0; n < 128; n++) z[1472 + n] = w.short_w(128 + n) * x[1472 + n];
            for (int n = 1600; n < 2048; n++) z[n] = 0.0;
            break;
        case kSeqStop:
            // 448 zeros | short rise | ones | long fall
            for (int n = 0; n < 448; n++) z[n] = 0.0;
            for (int n = 0; n < 128; n++) z[448 + n] = w.short_w(n) * x[448 + n];
            for (int n = 576; n < 1024; n++) z[n] = x[n];
            for (int n = 1024; n < 2048; n++) z[n] = w.long_w(n) * x[n];
            break;
        default:  // kSeqLong
            for (int n = 0; n < 2048; n++) z[n] = w.long_w(n) * x[n];
            break;
    }
    mdct_core(z, set1024(), spec);
}

#endif  // GLINT_AAC_INT

}  // namespace aac
}  // namespace glint

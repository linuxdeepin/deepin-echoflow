// glint - MDCT with sine window and aliasing reduction
// MIT License - Clean-room implementation

#include "mdct.hpp"
#include "tables.hpp"
#include <cstring>
#include <cmath>

#include "simd.hpp"
#if defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#ifdef GLINT_FIXED_POINT
#include "fixedpoint.hpp"
#endif

namespace glint {

#if !defined(GLINT_FIXED_POINT) || defined(GLINT_BOTH_PATHS)
// === Double-precision path ===

// Fused window * cosine * (1/288) table: eliminates separate windowing
// loop and per-output division from the MDCT inner loop.
static double mdct_wincos_d[36][18];
// Same, for the start (block_type 1) and stop (block_type 3) transition
// windows. These granules are rare (one per long<->short switch), so they
// use the scalar loop — no transposed SIMD copies.
static double mdct_wincos_start_d[36][18];
static double mdct_wincos_stop_d[36][18];
#if defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__) || (defined(__ARM_NEON) && defined(__aarch64__))
static double mdct_wincos_d_t[18][36];  // transposed for SIMD contiguous access
#endif
static double alias_cs_d[8], alias_ca_d[8];
static bool mdct_init = false;

static void init_mdct_d() {
    if (mdct_init) return;
    constexpr double PI = 3.14159265358979323846;
    for (int n = 0; n < 36; n++) {
        double win = std::sin(PI / 36.0 * (n + 0.5));
        // ISO 11172-3 transition windows: start keeps the long rise, holds 1,
        // then falls with the short half-window; stop is the mirror image.
        double win_start, win_stop;
        if (n < 18)      win_start = win;
        else if (n < 24) win_start = 1.0;
        else if (n < 30) win_start = std::sin(PI / 12.0 * (n - 18 + 0.5));
        else             win_start = 0.0;
        if (n < 6)       win_stop = 0.0;
        else if (n < 12) win_stop = std::sin(PI / 12.0 * (n - 6 + 0.5));
        else if (n < 18) win_stop = 1.0;
        else             win_stop = win;
        for (int k = 0; k < 18; k++) {
            double c = std::cos(PI / 72.0 * (2.0*n + 19.0) * (2.0*k + 1.0)) / 288.0;
            mdct_wincos_d[n][k] = win * c;
            mdct_wincos_start_d[n][k] = win_start * c;
            mdct_wincos_stop_d[n][k] = win_stop * c;
        }
    }
#if defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__) || (defined(__ARM_NEON) && defined(__aarch64__))
    for (int n = 0; n < 36; n++)
        for (int k = 0; k < 18; k++)
            mdct_wincos_d_t[k][n] = mdct_wincos_d[n][k];
#endif
    static constexpr double c[8] = {-0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037};
    for (int i = 0; i < 8; i++) {
        alias_cs_d[i] = 1.0 / std::sqrt(1.0 + c[i]*c[i]);
        alias_ca_d[i] = c[i] / std::sqrt(1.0 + c[i]*c[i]);
    }
    mdct_init = true;
}

MDCT::MDCT() { reset(); init_mdct_d(); }
void MDCT::reset() { std::memset(prev_, 0, sizeof(prev_)); }

void MDCT::process(const double subband[32][18], double mdct_out[32][18]) {
    for (int sb = 0; sb < 32; sb++) {
        double x[36];
        for (int n = 0; n < 18; n++) x[n] = prev_[sb][n];
        for (int n = 0; n < 18; n++) x[n + 18] = subband[sb][n];

        // Fused MDCT: window * cosine * normalization pre-baked into table
        for (int k = 0; k < 18; k++) {
#if defined(__AVX2__) || defined(__AVX__)
            __m256d vsum0 = _mm256_setzero_pd();
            __m256d vsum1 = _mm256_setzero_pd();
            for (int n = 0; n < 32; n += 8) {
                __m256d vx0 = _mm256_loadu_pd(&x[n]);
                __m256d vc0 = _mm256_loadu_pd(&mdct_wincos_d_t[k][n]);
                vsum0 = _mm256_add_pd(vsum0, _mm256_mul_pd(vx0, vc0));
                __m256d vx1 = _mm256_loadu_pd(&x[n + 4]);
                __m256d vc1 = _mm256_loadu_pd(&mdct_wincos_d_t[k][n + 4]);
                vsum1 = _mm256_add_pd(vsum1, _mm256_mul_pd(vx1, vc1));
            }
            __m256d vx_tail = _mm256_loadu_pd(&x[32]);
            __m256d vc_tail = _mm256_loadu_pd(&mdct_wincos_d_t[k][32]);
            vsum0 = _mm256_add_pd(vsum0, _mm256_mul_pd(vx_tail, vc_tail));
            vsum0 = _mm256_add_pd(vsum0, vsum1);
            __m128d lo = _mm256_castpd256_pd128(vsum0);
            __m128d hi = _mm256_extractf128_pd(vsum0, 1);
            lo = _mm_add_pd(lo, hi);
            double hsum = _mm_cvtsd_f64(lo) + _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));
            mdct_out[sb][k] = hsum;
#elif defined(__SSE2__)
            __m128d vsum0 = _mm_setzero_pd();
            __m128d vsum1 = _mm_setzero_pd();
            for (int n = 0; n < 36; n += 4) {
                __m128d vx0 = _mm_loadu_pd(&x[n]);
                __m128d vc0 = _mm_loadu_pd(&mdct_wincos_d_t[k][n]);
                vsum0 = _mm_add_pd(vsum0, _mm_mul_pd(vx0, vc0));
                __m128d vx1 = _mm_loadu_pd(&x[n + 2]);
                __m128d vc1 = _mm_loadu_pd(&mdct_wincos_d_t[k][n + 2]);
                vsum1 = _mm_add_pd(vsum1, _mm_mul_pd(vx1, vc1));
            }
            vsum0 = _mm_add_pd(vsum0, vsum1);
            double tmp[2]; _mm_storeu_pd(tmp, vsum0);
            mdct_out[sb][k] = tmp[0] + tmp[1];
#else
#if defined(__ARM_NEON) && defined(__aarch64__)
            if (g_simd_level == GLINT_SIMD_NEON) {
                float64x2_t vsum0 = vdupq_n_f64(0.0);
                float64x2_t vsum1 = vdupq_n_f64(0.0);
                for (int n = 0; n < 36; n += 4) {
                    vsum0 = vfmaq_f64(vsum0, vld1q_f64(&x[n]), vld1q_f64(&mdct_wincos_d_t[k][n]));
                    vsum1 = vfmaq_f64(vsum1, vld1q_f64(&x[n+2]), vld1q_f64(&mdct_wincos_d_t[k][n+2]));
                }
                vsum0 = vaddq_f64(vsum0, vsum1);
                double hsum = vgetq_lane_f64(vsum0, 0) + vgetq_lane_f64(vsum0, 1);
                mdct_out[sb][k] = hsum;
            } else
#endif
            {
                double sum = 0.0;

#ifndef _MSC_VER
#pragma GCC unroll 6
#endif
                for (int n = 0; n < 36; n++)
                    sum += x[n] * mdct_wincos_d[n][k];
                mdct_out[sb][k] = sum;
            }
#endif
        }

        for (int n = 0; n < 18; n++) prev_[sb][n] = subband[sb][n];
    }
}

void MDCT::process_strided(const double subband_out[32][36], int slot_offset,
                           double mdct_out[32][18], int block_type) {
    // Read directly from subband_out with stride 36, applying frequency
    // inversion inline: negate odd subbands at odd time slots.
    const double (*wincos)[18] =
        (block_type == 1) ? mdct_wincos_start_d :
        (block_type == 3) ? mdct_wincos_stop_d : mdct_wincos_d;
    const bool fast = (block_type == 0);
    for (int sb = 0; sb < 32; sb++) {
        double x[36];
        bool invert = (sb & 1);
        for (int n = 0; n < 18; n++) x[n] = prev_[sb][n];
        if (invert) {
            for (int ts = 0; ts < 18; ts++)
                x[18 + ts] = (ts & 1) ? -subband_out[sb][slot_offset + ts]
                                       :  subband_out[sb][slot_offset + ts];
        } else {
            for (int ts = 0; ts < 18; ts++)
                x[18 + ts] = subband_out[sb][slot_offset + ts];
        }

        // Fused MDCT (same as process()); transition windows use the
        // scalar path (rare granules).
        if (!fast) {
            for (int k = 0; k < 18; k++) {
                double sum = 0.0;
                for (int n = 0; n < 36; n++)
                    sum += x[n] * wincos[n][k];
                mdct_out[sb][k] = sum;
            }
        } else
        for (int k = 0; k < 18; k++) {
#if defined(__AVX2__) || defined(__AVX__)
            __m256d vsum0 = _mm256_setzero_pd();
            __m256d vsum1 = _mm256_setzero_pd();
            for (int n = 0; n < 32; n += 8) {
                vsum0 = _mm256_add_pd(vsum0, _mm256_mul_pd(_mm256_loadu_pd(&x[n]), _mm256_loadu_pd(&mdct_wincos_d_t[k][n])));
                vsum1 = _mm256_add_pd(vsum1, _mm256_mul_pd(_mm256_loadu_pd(&x[n+4]), _mm256_loadu_pd(&mdct_wincos_d_t[k][n+4])));
            }
            __m256d vxt = _mm256_loadu_pd(&x[32]);
            __m256d vct = _mm256_loadu_pd(&mdct_wincos_d_t[k][32]);
            vsum0 = _mm256_add_pd(vsum0, _mm256_mul_pd(vxt, vct));
            vsum0 = _mm256_add_pd(vsum0, vsum1);
            __m128d lo = _mm256_castpd256_pd128(vsum0);
            __m128d hi = _mm256_extractf128_pd(vsum0, 1);
            lo = _mm_add_pd(lo, hi);
            mdct_out[sb][k] = _mm_cvtsd_f64(lo) + _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));
#elif defined(__SSE2__)
            __m128d vsum0 = _mm_setzero_pd(), vsum1 = _mm_setzero_pd();
            for (int n = 0; n < 36; n += 4) {
                vsum0 = _mm_add_pd(vsum0, _mm_mul_pd(_mm_loadu_pd(&x[n]), _mm_loadu_pd(&mdct_wincos_d_t[k][n])));
                vsum1 = _mm_add_pd(vsum1, _mm_mul_pd(_mm_loadu_pd(&x[n+2]), _mm_loadu_pd(&mdct_wincos_d_t[k][n+2])));
            }
            vsum0 = _mm_add_pd(vsum0, vsum1);
            double tmp[2]; _mm_storeu_pd(tmp, vsum0);
            mdct_out[sb][k] = tmp[0] + tmp[1];
#else
            double sum = 0.0;
            for (int n = 0; n < 36; n++)
                sum += x[n] * mdct_wincos_d[n][k];
            mdct_out[sb][k] = sum;
#endif
        }

        // Store prev for next granule (with freq inversion applied)
        if (invert) {
            for (int ts = 0; ts < 18; ts++)
                prev_[sb][ts] = (ts & 1) ? -subband_out[sb][slot_offset + ts]
                                          :  subband_out[sb][slot_offset + ts];
        } else {
            for (int ts = 0; ts < 18; ts++)
                prev_[sb][ts] = subband_out[sb][slot_offset + ts];
        }
    }
}

// Short-block MDCT: 12-point transform, 3 windows per subband
// ISO 11172-3: X[win][k] = sum_{n=0}^{11} z[n] * cos(pi/24 * (2n+7) * (2k+1))
// Window: sin(pi/12 * (n + 0.5))

static double short_win_d[12];
static double short_cos_d[12][6];
static bool short_mdct_init = false;

static void init_short_mdct_d() {
    if (short_mdct_init) return;
    constexpr double PI = 3.14159265358979323846;
    for (int n = 0; n < 12; n++) {
        short_win_d[n] = std::sin(PI / 12.0 * (n + 0.5));
        for (int k = 0; k < 6; k++)
            short_cos_d[n][k] = std::cos(PI / 24.0 * (2.0*n + 7.0) * (2.0*k + 1.0));
    }
    short_mdct_init = true;
}

void MDCT::process_short(const double subband[32][18], double mdct_out[32][3][6]) {
    init_short_mdct_d();

    for (int sb = 0; sb < 32; sb++) {
        // For short blocks, we have 18 samples from prev + 18 from current = 36
        // These are divided into 3 overlapping windows of 12 samples each:
        //   Window 0: samples [0..11]  (6 from prev, 6 from current)
        //   Window 1: samples [6..17]  (from current, offset by 6)
        //   Window 2: samples [12..23] (from current, offset by 12)
        // But we only have prev[0..17] and current subband[0..17].
        // Mapping: prev_[sb][0..17] are previous granule's 18 samples
        //          subband[sb][0..17] are current granule's 18 samples
        // Full 36-sample buffer: x[0..17] = prev, x[18..35] = current
        // Short windows are centered in the long block window:
        //   Window 0: x[6..17]   (samples from prev)
        //   Window 1: x[12..23]  (6 from prev + 6 from current)
        //   Window 2: x[18..29]  (samples from current)

        double x[36];
        for (int n = 0; n < 18; n++) x[n] = prev_[sb][n];
        for (int n = 0; n < 18; n++) x[n + 18] = subband[sb][n];

        for (int win = 0; win < 3; win++) {
            int offset = 6 + win * 6;  // start of 12-sample window within 36-sample block

            // Apply window and compute MDCT
            double z[12];
            for (int n = 0; n < 12; n++)
                z[n] = x[offset + n] * short_win_d[n];

            for (int k = 0; k < 6; k++) {
                double sum = 0.0;
                for (int n = 0; n < 12; n++)
                    sum += z[n] * short_cos_d[n][k];
                mdct_out[sb][win][k] = sum / 96.0;  // normalization for 12-point
            }
        }

        // Save current samples for next call's overlap
        for (int n = 0; n < 18; n++) prev_[sb][n] = subband[sb][n];
    }
}

void alias_reduce_d(double mdct_out[32][18]) {
    for (int sb = 0; sb < 31; sb++) {

#ifndef _MSC_VER
#pragma GCC unroll 8
#endif
        for (int i = 0; i < 8; i++) {
            double a = mdct_out[sb][17 - i];
            double b = mdct_out[sb + 1][i];
            mdct_out[sb][17 - i]  = a * alias_cs_d[i] + b * alias_ca_d[i];
            mdct_out[sb + 1][i]   = b * alias_cs_d[i] - a * alias_ca_d[i];
        }
    }
}
#endif // double-precision path

#ifdef GLINT_FIXED_POINT

// === Fixed-point (Q24) path ===

#if defined(__SSE2__) || defined(_M_X64)
// Transposed double-precision MDCT cosine table for SSE2 fixed-point path.
static double mdct_cos_d_fp[18][36];
static bool mdct_cos_d_fp_init = false;

static void init_mdct_cos_d_fp() {
    if (mdct_cos_d_fp_init) return;
    constexpr double PI = 3.14159265358979323846;
    for (int n = 0; n < 36; n++)
        for (int k = 0; k < 18; k++)
            mdct_cos_d_fp[k][n] = std::cos(PI / 72.0 * (2.0*n + 19.0) * (2.0*k + 1.0));
    mdct_cos_d_fp_init = true;
}
#endif

MDCT_FP::MDCT_FP() {
    reset();
    tables::init_tables();
#if defined(__SSE2__) || defined(_M_X64)
    init_mdct_cos_d_fp();
#endif
}
void MDCT_FP::reset() { std::memset(prev_, 0, sizeof(prev_)); }

void MDCT_FP::process(const int32_t subband[32][18], int32_t mdct_out[32][18]) {
    for (int sb = 0; sb < 32; sb++) {
        int32_t x[36];
        for (int n = 0; n < 18; n++) x[n] = prev_[sb][n];
        for (int n = 0; n < 18; n++) x[n + 18] = subband[sb][n];

        for (int n = 0; n < 36; n++)
            x[n] = static_cast<int32_t>((static_cast<int64_t>(x[n]) * tables::mdct_window[n]) >> 31);

        // MDCT cosine accumulation: x is Q24, mdct_cos is Q31.
        // Full product Q24*Q31 = Q55, sum of 36 terms can overflow int64.
        // Pre-shift x >> 2 to Q22: Q22*Q31 = Q53, sum of 36 fits int64.
        // Divide by 288, then >> 29 to get Q24.
        for (int k = 0; k < 18; k++) {
            int64_t sum = 0;

#ifndef _MSC_VER
#pragma GCC unroll 6
#endif
            for (int n = 0; n < 36; n++)
                sum += static_cast<int64_t>(x[n] >> 2) * tables::mdct_cos[n][k];
            mdct_out[sb][k] = static_cast<int32_t>((sum / 288) >> 29);
        }

        for (int n = 0; n < 18; n++) prev_[sb][n] = subband[sb][n];
    }
}

void alias_reduce_fp(int32_t mdct_out[32][18]) {
    for (int sb = 0; sb < 31; sb++) {

#ifndef _MSC_VER
#pragma GCC unroll 8
#endif
        for (int i = 0; i < 8; i++) {
            int32_t a = mdct_out[sb][17 - i];
            int32_t b = mdct_out[sb + 1][i];
            int32_t cs = tables::alias_cs[i];
            int32_t ca = tables::alias_ca[i];
            mdct_out[sb][17 - i] = static_cast<int32_t>((static_cast<int64_t>(a) * cs + static_cast<int64_t>(b) * ca) >> 31);
            mdct_out[sb + 1][i]  = static_cast<int32_t>((static_cast<int64_t>(b) * cs - static_cast<int64_t>(a) * ca) >> 31);
        }
    }
}

// Transition/short window tables for the fixed path (double precision —
// see the class comment). Separate copies from the double-path statics so
// pure-fixed builds have them too.
#ifdef GLINT_SMALL_BUFFERS
static float fp_wincos_start[36][18];   // rare transition granules: float
static float fp_wincos_stop[36][18];    // precision is ample, half the RAM
#else
static double fp_wincos_start[36][18];
static double fp_wincos_stop[36][18];
#endif
static double fp_short_win[12];
static double fp_short_cos[12][6];
static bool fp_win_tables_init = false;

static void init_fp_win_tables() {
    if (fp_win_tables_init) return;
    constexpr double PI = 3.14159265358979323846;
    for (int n = 0; n < 36; n++) {
        double win_start, win_stop;
        double win = std::sin(PI / 36.0 * (n + 0.5));
        if (n < 18)      win_start = win;
        else if (n < 24) win_start = 1.0;
        else if (n < 30) win_start = std::sin(PI / 12.0 * (n - 18 + 0.5));
        else             win_start = 0.0;
        if (n < 6)       win_stop = 0.0;
        else if (n < 12) win_stop = std::sin(PI / 12.0 * (n - 6 + 0.5));
        else if (n < 18) win_stop = 1.0;
        else             win_stop = win;
        for (int k = 0; k < 18; k++) {
            double c = std::cos(PI / 72.0 * (2.0*n + 19.0) * (2.0*k + 1.0)) / 288.0;
            fp_wincos_start[n][k] = win_start * c;
            fp_wincos_stop[n][k] = win_stop * c;
        }
    }
    for (int n = 0; n < 12; n++) {
        fp_short_win[n] = std::sin(PI / 12.0 * (n + 0.5));
        for (int k = 0; k < 6; k++)
            fp_short_cos[n][k] = std::cos(PI / 24.0 * (2.0*n + 7.0) * (2.0*k + 1.0));
    }
    fp_win_tables_init = true;
}

void MDCT_FP::process_short_and_convert(const int32_t subband[32][18],
                                        double mdct_out[32][3][6]) {
    init_fp_win_tables();
    const double kQ = 1.0 / 16777216.0;
    for (int sb = 0; sb < 32; sb++) {
        double x[36];
        for (int n = 0; n < 18; n++) x[n] = prev_[sb][n] * kQ;
        for (int n = 0; n < 18; n++) x[n + 18] = subband[sb][n] * kQ;
        for (int win = 0; win < 3; win++) {
            int offset = 6 + win * 6;
            double z[12];
            for (int n = 0; n < 12; n++)
                z[n] = x[offset + n] * fp_short_win[n];
            for (int k = 0; k < 6; k++) {
                double sum = 0.0;
                for (int n = 0; n < 12; n++)
                    sum += z[n] * fp_short_cos[n][k];
                mdct_out[sb][win][k] = sum / 96.0;
            }
        }
        for (int n = 0; n < 18; n++) prev_[sb][n] = subband[sb][n];
    }
}

void MDCT_FP::process_and_convert(const int32_t subband[32][18], double mdct_flat[576],
                                  int block_type) {
    // Fused MDCT + alias reduction + Q24->double conversion.
    // Outputs flat double[576] directly for the quantizer.
    double mdct_d[32][18];

    if (block_type == 1 || block_type == 3) {
        // Transition windows: rare granules, double-precision scalar path.
        init_fp_win_tables();
#ifdef GLINT_SMALL_BUFFERS
        const float (*wincos)[18] = (block_type == 1) ? fp_wincos_start
                                                      : fp_wincos_stop;
#else
        const double (*wincos)[18] = (block_type == 1) ? fp_wincos_start
                                                       : fp_wincos_stop;
#endif
        const double kQ = 1.0 / 16777216.0;
        for (int sb = 0; sb < 32; sb++) {
            double x[36];
            for (int n = 0; n < 18; n++) x[n] = prev_[sb][n] * kQ;
            for (int n = 0; n < 18; n++) x[n + 18] = subband[sb][n] * kQ;
            for (int k = 0; k < 18; k++) {
                double sum = 0.0;
                for (int n = 0; n < 36; n++)
                    sum += x[n] * wincos[n][k];
                mdct_d[sb][k] = sum;
            }
            for (int n = 0; n < 18; n++) prev_[sb][n] = subband[sb][n];
        }
    } else {


    // Pre-compute double MDCT window
    static double mdct_win_d_fp[36];
    static bool win_init = false;
    if (!win_init) {
        constexpr double PI = 3.14159265358979323846;
        for (int n = 0; n < 36; n++)
            mdct_win_d_fp[n] = std::sin(PI / 36.0 * (n + 0.5));
        win_init = true;
    }
    const double kQ24ToDouble = 1.0 / 16777216.0;

    for (int sb = 0; sb < 32; sb++) {
#if defined(__SSE2__) || defined(_M_X64)
        // Convert Q24 int32 to double and apply window in one pass.
        double x[36];
        const __m128d vq24 = _mm_set1_pd(kQ24ToDouble);
        for (int n = 0; n < 18; n += 2) {
            __m128i vi = _mm_loadl_epi64((__m128i*)&prev_[sb][n]);
            __m128d vd = _mm_mul_pd(_mm_cvtepi32_pd(vi), vq24);
            __m128d vw = _mm_loadu_pd(&mdct_win_d_fp[n]);
            _mm_storeu_pd(&x[n], _mm_mul_pd(vd, vw));
        }
        for (int n = 0; n < 18; n += 2) {
            __m128i vi = _mm_loadl_epi64((__m128i*)&subband[sb][n]);
            __m128d vd = _mm_mul_pd(_mm_cvtepi32_pd(vi), vq24);
            __m128d vw = _mm_loadu_pd(&mdct_win_d_fp[18 + n]);
            _mm_storeu_pd(&x[18 + n], _mm_mul_pd(vd, vw));
        }

        // SIMD double-precision MDCT cosine transform
        for (int k = 0; k < 18; k++) {
#if defined(__AVX2__) || defined(__AVX__)
            __m256d vsum0 = _mm256_setzero_pd();
            __m256d vsum1 = _mm256_setzero_pd();
            for (int n = 0; n < 32; n += 8) {
                vsum0 = _mm256_add_pd(vsum0, _mm256_mul_pd(
                    _mm256_loadu_pd(&x[n]), _mm256_loadu_pd(&mdct_cos_d_fp[k][n])));
                vsum1 = _mm256_add_pd(vsum1, _mm256_mul_pd(
                    _mm256_loadu_pd(&x[n + 4]), _mm256_loadu_pd(&mdct_cos_d_fp[k][n + 4])));
            }
            // Remaining 4 elements (n=32..35)
            vsum0 = _mm256_add_pd(vsum0, _mm256_mul_pd(
                _mm256_loadu_pd(&x[32]), _mm256_loadu_pd(&mdct_cos_d_fp[k][32])));
            vsum0 = _mm256_add_pd(vsum0, vsum1);
            __m128d lo = _mm256_castpd256_pd128(vsum0);
            __m128d hi = _mm256_extractf128_pd(vsum0, 1);
            lo = _mm_add_pd(lo, hi);
            double hsum = _mm_cvtsd_f64(lo) + _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));
            mdct_d[sb][k] = hsum / 288.0;
#else
            __m128d vsum0 = _mm_setzero_pd();
            __m128d vsum1 = _mm_setzero_pd();
            for (int n = 0; n < 36; n += 4) {
                vsum0 = _mm_add_pd(vsum0, _mm_mul_pd(
                    _mm_loadu_pd(&x[n]), _mm_loadu_pd(&mdct_cos_d_fp[k][n])));
                vsum1 = _mm_add_pd(vsum1, _mm_mul_pd(
                    _mm_loadu_pd(&x[n + 2]), _mm_loadu_pd(&mdct_cos_d_fp[k][n + 2])));
            }
            vsum0 = _mm_add_pd(vsum0, vsum1);
            double tmp[2]; _mm_storeu_pd(tmp, vsum0);
            mdct_d[sb][k] = (tmp[0] + tmp[1]) / 288.0;
#endif
        }
#else
        // Scalar fallback: integer windowing + convert to double
        int32_t xi[36];
        for (int n = 0; n < 18; n++) xi[n] = prev_[sb][n];
        for (int n = 0; n < 18; n++) xi[n + 18] = subband[sb][n];
        for (int n = 0; n < 36; n++)
            xi[n] = static_cast<int32_t>((static_cast<int64_t>(xi[n]) * tables::mdct_window[n]) >> 31);
        for (int k = 0; k < 18; k++) {
            int64_t sum = 0;
            for (int n = 0; n < 36; n++)
                sum += static_cast<int64_t>(xi[n] >> 2) * tables::mdct_cos[n][k];
            mdct_d[sb][k] = static_cast<double>((sum / 288) >> 29) / 16777216.0;
        }
#endif

        for (int n = 0; n < 18; n++) prev_[sb][n] = subband[sb][n];
    }
    }  // end long-path branch

    // Alias reduction in double
    static double ar_cs[8], ar_ca[8];
    static bool ar_init = false;
    if (!ar_init) {
        static constexpr double c[8] = {-0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037};
        for (int i = 0; i < 8; i++) {
            ar_cs[i] = 1.0 / std::sqrt(1.0 + c[i]*c[i]);
            ar_ca[i] = c[i] / std::sqrt(1.0 + c[i]*c[i]);
        }
        ar_init = true;
    }
    for (int sb = 0; sb < 31; sb++) {
        for (int i = 0; i < 8; i++) {
            double a = mdct_d[sb][17 - i];
            double b = mdct_d[sb + 1][i];
            mdct_d[sb][17 - i] = a * ar_cs[i] + b * ar_ca[i];
            mdct_d[sb + 1][i]  = b * ar_cs[i] - a * ar_ca[i];
        }
    }

    // Flatten to 576-element array
    for (int sb = 0; sb < 32; sb++)
        for (int k = 0; k < 18; k++)
            mdct_flat[sb * 18 + k] = mdct_d[sb][k];
}

#ifdef GLINT_MP3_INT
// Q31 fused window*cos/288 table for the integer long-block MDCT.
static int32_t mdct_wincos_q31[36][18];

// Built by a global ctor at startup: the double trig must not end up
// inlined into the integer hot path (see tools/check_nofpu.sh).
static struct WincosQ31Init {
    WincosQ31Init() {
        constexpr double PI = 3.14159265358979323846;
        for (int n = 0; n < 36; n++) {
            double w = std::sin(PI / 36.0 * (n + 0.5));
            for (int k = 0; k < 18; k++) {
                double v = w * std::cos(PI / 72.0 * (2.0 * n + 19.0) * (2.0 * k + 1.0)) / 288.0;
                mdct_wincos_q31[n][k] =
                    static_cast<int32_t>(std::lround(v * 2147483648.0));
            }
        }
    }
} g_wincos_q31_init;

void MDCT_FP::process_int(const int32_t subband[32][18], int32_t mdct_flat[576]) {
    int32_t out[32][18];
    for (int sb = 0; sb < 32; sb++) {
        for (int k = 0; k < 18; k++) {
            int64_t acc = 0;
            for (int n = 0; n < 18; n++) {
                acc += static_cast<int64_t>(prev_[sb][n]) * mdct_wincos_q31[n][k];
            }
            for (int n = 0; n < 18; n++) {
                acc += static_cast<int64_t>(subband[sb][n]) * mdct_wincos_q31[n + 18][k];
            }
            out[sb][k] = static_cast<int32_t>(acc >> 31);
        }
        for (int n = 0; n < 18; n++) prev_[sb][n] = subband[sb][n];
    }
    alias_reduce_fp(out);
    for (int sb = 0; sb < 32; sb++)
        for (int k = 0; k < 18; k++)
            mdct_flat[sb * 18 + k] = out[sb][k];
}
#endif // GLINT_MP3_INT

#endif // GLINT_FIXED_POINT

} // namespace glint

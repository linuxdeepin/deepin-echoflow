// glint - Subband analysis filter
// MIT License - Clean-room implementation

#include "subband.hpp"
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

SubbandAnalysis::SubbandAnalysis() { reset(); }

void SubbandAnalysis::reset() {
    std::memset(window_buf_, 0, sizeof(window_buf_));
    window_offset_ = 0;
}

void SubbandAnalysis::process_slot(const double* samples, double subband_out[kNumSubbands]) {
    window_offset_ = (window_offset_ - 32) & 0x1FF;
    for (int i = 0; i < 32; i++)
        window_buf_[(window_offset_ + 31 - i) & 0x1FF] = samples[i];

    double z[64];
    for (int j = 0; j < 64; j++) {
        double sum = 0.0;

#ifndef _MSC_VER
#pragma GCC unroll 8
#endif
        for (int p = 0; p < 8; p++) {
            int buf_idx = (window_offset_ + j + 64 * p) & 0x1FF;
            sum += window_buf_[buf_idx] * tables::analysis_window_d[j + 64 * p];
        }
        z[j] = sum;
    }

    for (int i = 0; i < 32; i++) {
#if defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__) || defined(_M_X64)
        if (g_simd_level == GLINT_SIMD_AVX) {
#if defined(__AVX2__) || defined(__AVX__)
            __m256d vsum0 = _mm256_setzero_pd();
            __m256d vsum1 = _mm256_setzero_pd();
            for (int k = 0; k < 64; k += 8) {
                vsum0 = _mm256_add_pd(vsum0, _mm256_mul_pd(_mm256_loadu_pd(&z[k]), _mm256_loadu_pd(&tables::subband_matrix_d[i][k])));
                vsum1 = _mm256_add_pd(vsum1, _mm256_mul_pd(_mm256_loadu_pd(&z[k+4]), _mm256_loadu_pd(&tables::subband_matrix_d[i][k+4])));
            }
            vsum0 = _mm256_add_pd(vsum0, vsum1);
            __m128d lo = _mm256_castpd256_pd128(vsum0);
            __m128d hi = _mm256_extractf128_pd(vsum0, 1);
            lo = _mm_add_pd(lo, hi);
            subband_out[i] = _mm_cvtsd_f64(lo) + _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));
            continue;
#endif
        }
        if (g_simd_level == GLINT_SIMD_SSE2 || g_simd_level == GLINT_SIMD_AVX) {
            __m128d vsum0 = _mm_setzero_pd();
            __m128d vsum1 = _mm_setzero_pd();
            for (int k = 0; k < 64; k += 4) {
                vsum0 = _mm_add_pd(vsum0, _mm_mul_pd(_mm_loadu_pd(&z[k]), _mm_loadu_pd(&tables::subband_matrix_d[i][k])));
                vsum1 = _mm_add_pd(vsum1, _mm_mul_pd(_mm_loadu_pd(&z[k+2]), _mm_loadu_pd(&tables::subband_matrix_d[i][k+2])));
            }
            vsum0 = _mm_add_pd(vsum0, vsum1);
            double tmp[2]; _mm_storeu_pd(tmp, vsum0);
            subband_out[i] = tmp[0] + tmp[1];
            continue;
        }
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
        if (g_simd_level == GLINT_SIMD_NEON) {
            float64x2_t vsum0 = vdupq_n_f64(0.0);
            float64x2_t vsum1 = vdupq_n_f64(0.0);
            for (int k = 0; k < 64; k += 4) {
                float64x2_t vz0 = vld1q_f64(&z[k]);
                float64x2_t vm0 = vld1q_f64(&tables::subband_matrix_d[i][k]);
                vsum0 = vfmaq_f64(vsum0, vz0, vm0);
                float64x2_t vz1 = vld1q_f64(&z[k+2]);
                float64x2_t vm1 = vld1q_f64(&tables::subband_matrix_d[i][k+2]);
                vsum1 = vfmaq_f64(vsum1, vz1, vm1);
            }
            vsum0 = vaddq_f64(vsum0, vsum1);
            subband_out[i] = vgetq_lane_f64(vsum0, 0) + vgetq_lane_f64(vsum0, 1);
            continue;
        }
#endif
        // Scalar fallback
        double sum = 0.0;
        for (int k = 0; k < 64; k++)
            sum += z[k] * tables::subband_matrix_d[i][k];
        subband_out[i] = sum;
    }
}

void SubbandAnalysis::analyze(const int16_t* pcm, double out[kNumSubbands][kTimeSlots], int num_slots) {
    for (int ts = 0; ts < num_slots; ts++) {
        double samples[32];
        for (int i = 0; i < 32; i++)
            samples[i] = pcm[ts * 32 + i] / 32768.0;
        double slot_out[kNumSubbands];
        process_slot(samples, slot_out);
        for (int sb = 0; sb < kNumSubbands; sb++)
            out[sb][ts] = slot_out[sb];
    }
}

void SubbandAnalysis::analyze_float(const float* pcm, double out[kNumSubbands][kTimeSlots], int num_slots) {
    for (int ts = 0; ts < num_slots; ts++) {
        double samples[32];
        for (int i = 0; i < 32; i++)
            samples[i] = static_cast<double>(pcm[ts * 32 + i]);  // float [-1,1] directly
        double slot_out[kNumSubbands];
        process_slot(samples, slot_out);
        for (int sb = 0; sb < kNumSubbands; sb++)
            out[sb][ts] = slot_out[sb];
    }
}
#endif // double-precision path

#ifdef GLINT_FIXED_POINT

// === Fixed-point (Q24) path ===
//
// Q format chain:
//   PCM int16 -> int32 Q15 (sign-extend, no shift)
//   Window buffer: int32 Q15
//   Windowed sum: Q15 * Q30 = Q45 in int64. Sum of 8 products.
//   Matrixing: pre-shift z >> 24 to Q21, then Q21 * Q31 = Q52 in int64.
//     Sum of 64 terms fits int64 (worst case ~2.6e18 < 9.2e18).
//   Output: Q52 >> 28 = Q24.

SubbandAnalysisFP::SubbandAnalysisFP() { reset(); }

void SubbandAnalysisFP::reset() {
    std::memset(window_buf_d_, 0, sizeof(window_buf_d_));
    window_offset_ = 0;
}

void SubbandAnalysisFP::process_slot(const int16_t* samples, int32_t subband_out[kNumSubbands]) {
    window_offset_ = (window_offset_ - 32) & 0x1FF;
    for (int i = 0; i < 32; i++)
        window_buf_d_[(window_offset_ + 31 - i) & 0x1FF] = static_cast<double>(samples[i]);

    // Scale factor: int16 input (not divided by 32768 like double path),
    // so result is 32768x larger. Q24 = result * 2^24/32768 = result * 512.
    const double kQ24Scale = 512.0;

    // Windowed partial sums: buffer is double, window coefficients are double.
    // Identical to double path except for the Q24 scale at the end.
    double z[64];
    for (int j = 0; j < 64; j++) {
        double sum = 0.0;

#ifndef _MSC_VER
#pragma GCC unroll 8
#endif
        for (int p = 0; p < 8; p++) {
            int buf_idx = (window_offset_ + j + 64 * p) & 0x1FF;
            sum += window_buf_d_[buf_idx] * tables::analysis_window_d[j + 64 * p];
        }
        z[j] = sum;
    }

    // Matrixing with SIMD double, then convert to Q24.
    for (int i = 0; i < 32; i++) {
#if defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__) || defined(_M_X64)
        if (g_simd_level == GLINT_SIMD_AVX) {
#if defined(__AVX2__) || defined(__AVX__)
            __m256d vsum0 = _mm256_setzero_pd();
            __m256d vsum1 = _mm256_setzero_pd();
            for (int k = 0; k < 64; k += 8) {
                vsum0 = _mm256_add_pd(vsum0, _mm256_mul_pd(
                    _mm256_loadu_pd(&z[k]),
                    _mm256_loadu_pd(&tables::subband_matrix_d[i][k])));
                vsum1 = _mm256_add_pd(vsum1, _mm256_mul_pd(
                    _mm256_loadu_pd(&z[k + 4]),
                    _mm256_loadu_pd(&tables::subband_matrix_d[i][k + 4])));
            }
            vsum0 = _mm256_add_pd(vsum0, vsum1);
            __m128d lo = _mm256_castpd256_pd128(vsum0);
            __m128d hi = _mm256_extractf128_pd(vsum0, 1);
            lo = _mm_add_pd(lo, hi);
            subband_out[i] = static_cast<int32_t>(
                (_mm_cvtsd_f64(lo) + _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo))) * kQ24Scale);
            continue;
#endif
        }
        if (g_simd_level == GLINT_SIMD_SSE2 || g_simd_level == GLINT_SIMD_AVX) {
            __m128d vsum0 = _mm_setzero_pd();
            __m128d vsum1 = _mm_setzero_pd();
            for (int k = 0; k < 64; k += 4) {
                vsum0 = _mm_add_pd(vsum0, _mm_mul_pd(
                    _mm_loadu_pd(&z[k]),
                    _mm_loadu_pd(&tables::subband_matrix_d[i][k])));
                vsum1 = _mm_add_pd(vsum1, _mm_mul_pd(
                    _mm_loadu_pd(&z[k + 2]),
                    _mm_loadu_pd(&tables::subband_matrix_d[i][k + 2])));
            }
            vsum0 = _mm_add_pd(vsum0, vsum1);
            double tmp[2]; _mm_storeu_pd(tmp, vsum0);
            subband_out[i] = static_cast<int32_t>((tmp[0] + tmp[1]) * kQ24Scale);
            continue;
        }
#endif
        // Scalar fallback
        double sum = 0.0;
        for (int k = 0; k < 64; k++)
            sum += z[k] * tables::subband_matrix_d[i][k];
        subband_out[i] = static_cast<int32_t>(sum * kQ24Scale);
    }
}

void SubbandAnalysisFP::analyze(const int16_t* pcm, int32_t out[kNumSubbands][kTimeSlots], int num_slots) {
    for (int ts = 0; ts < num_slots; ts++) {
        int32_t slot_out[kNumSubbands];
        process_slot(&pcm[ts * 32], slot_out);
        for (int sb = 0; sb < kNumSubbands; sb++)
            out[sb][ts] = slot_out[sb];
    }
}

#endif // GLINT_FIXED_POINT

} // namespace glint

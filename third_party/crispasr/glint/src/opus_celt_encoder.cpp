// CELT frame encoder — RFC 6716 section 4.3 (encoder side)
// MIT License - Clean-room implementation

#include "opus_celt_encoder.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "opus_celt_enc_bands.hpp"
#include "opus_celt_pitch.hpp"
#include "opus_celt_enc_energy.hpp"
#include "opus_celt_rate.hpp"
#include "opus_celt_tables.hpp"

namespace glint {
namespace opus {

namespace {
using celt::kEBands;
using celt::kNbEBands;
constexpr float kPreemphCoef = 0.85f;
constexpr int kSpreadNormal = 2;
constexpr int kCombFilterMinPeriodEnc = 15;

// tf_select_table (RFC 6716): per-LM effective tf resolution for
// [4*isTransient + 2*tf_select + tf_res]. Shared by tf_analysis,
// tf_encode and the post-encode remap that quant_all_bands consumes.
const signed char kTfSelectTable[4][8] = {
    { 0, -1, 0, -1, 0, -1, 0, -1 },
    { 0, -1, 0, -2, 1, 0, 1, -1 },
    { 0, -2, 0, -3, 2, 0, 1, -1 },
    { 0, -2, 0, -3, 3, 0, 1, -1 },
};

// Forward/backward-masking transient detector (reference float path).
// Pure encoder policy: the decision changes quality, never validity.
// Also produces tf_estimate (transient strength, feeds the tf_analysis
// bias) and tf_chan (the channel that tripped the metric).
int transient_analysis(const double* in, int len, int C,
                       double* tf_estimate, int* tf_chan) {
    // 6*64/x lookup, reference-trained.
    static const uint8_t kInvTable[128] = {
        255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25,
        23,  22,  21,  20,  19, 18, 17, 16, 16, 15, 15, 14, 13, 13, 12, 12,
        12,  12,  11,  11,  11, 10, 10, 10, 9,  9,  9,  9,  9,  9,  8,  8,
        8,   8,   8,   7,   7,  7,  7,  7,  7,  6,  6,  6,  6,  6,  6,  6,
        6,   6,   6,   6,   6,  6,  6,  6,  6,  5,  5,  5,  5,  5,  5,  5,
        5,   5,   5,   5,   5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
        4,   4,   4,   4,   4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3,  3,
        3,   3,   3,   3,   3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,
    };
    static thread_local double tmp[2 * (960 + 120)];
    int len2 = len / 2;
    int32_t mask_metric = 0;
    for (int c = 0; c < C; c++) {
        // High-pass (1 - 2z^-1 + z^-2)/(1 - z^-1 + .5z^-2).
        double mem0 = 0, mem1 = 0;
        for (int i = 0; i < len; i++) {
            double x = in[c * (960 + 120) + i];
            double y = mem0 + x;
            mem0 = mem1 + y - 2 * x;
            mem1 = x - 0.5 * y;
            tmp[i] = y;
        }
        for (int i = 0; i < 12; i++) tmp[i] = 0;

        // Forward pass: post-echo threshold (6.7 dB/ms decay).
        double mean = 0, mem = 0;
        const double fwd = 0.0625;
        for (int i = 0; i < len2; i++) {
            double x2 = tmp[2 * i] * tmp[2 * i] +
                        tmp[2 * i + 1] * tmp[2 * i + 1];
            mean += x2;
            mem = x2 + (1.0 - fwd) * mem;
            tmp[i] = fwd * mem;
        }
        // Backward pass: pre-echo threshold (13.9 dB/ms).
        mem = 0;
        double max_e = 0;
        for (int i = len2 - 1; i >= 0; i--) {
            mem = tmp[i] + 0.875 * mem;
            tmp[i] = 0.125 * mem;
            if (tmp[i] > max_e) max_e = tmp[i];
        }
        // Bitrate-normalized temporal noise-to-mask ratio: harmonic mean
        // of the threshold vs the (geometric-mean) frame energy.
        mean = std::sqrt(mean * max_e * 0.5 * len2);
        double norm = len2 / (1e-15 + mean);
        int32_t unmask = 0;
        for (int i = 12; i < len2 - 5; i += 4) {
            double v = std::floor(64.0 * norm * (tmp[i] + 1e-15));
            int id = v < 0 ? 0 : (v > 127 ? 127 : static_cast<int>(v));
            unmask += kInvTable[id];
        }
        unmask = 64 * unmask * 4 / (6 * (len2 - 17));
        if (unmask > mask_metric) {
            *tf_chan = c;
            mask_metric = unmask;
        }
    }
    double tf_max =
        std::max(0.0, std::sqrt(27.0 * mask_metric) - 42.0);
    *tf_estimate = std::sqrt(
        std::max(0.0, 0.0069 * std::min(163.0, tf_max) - 0.139));
    return mask_metric > 200;
}

inline int imin(int a, int b) { return a < b ? a : b; }
inline int imax(int a, int b) { return a > b ? a : b; }

// Local float Haar butterfly (the enc_bands copy is TU-private).
void haar1f(float* x, int n0, int stride) {
    n0 >>= 1;
    for (int i = 0; i < stride; i++) {
        for (int j = 0; j < n0; j++) {
            float tmp1 = 0.70710678f * x[stride * 2 * j + i];
            float tmp2 = 0.70710678f * x[stride * (2 * j + 1) + i];
            x[stride * 2 * j + i] = tmp1 + tmp2;
            x[stride * (2 * j + 1) + i] = tmp1 - tmp2;
        }
    }
}

// L1 norm with a bias toward frequency resolution when in doubt.
double l1_metric(const float* tmp, int n, int lm, double bias) {
    double l1 = 0;
    for (int i = 0; i < n; i++) l1 += std::fabs(tmp[i]);
    return l1 + lm * bias * l1;
}

// Per-band time-frequency resolution analysis (reference tf_analysis):
// L1-sparsity of each band under successive Haar merges/splits picks a
// per-band metric; a Viterbi pass with switching cost lambda smooths
// the change bits; tf_select picks the better of the two table halves.
// Policy-only: any tf_res is a valid stream.
int tf_analysis(int len, int is_transient, int* tf_res, int lambda,
                const float* X, int n0, int lm, double tf_estimate,
                int tf_chan, const int* importance) {
    // 176 = widest band (22) << max LM (3).
    float tmp[176], tmp_1[176];
    int metric[kNbEBands], path0[kNbEBands], path1[kNbEBands];
    double bias = 0.04 * std::max(-0.25, 0.5 - tf_estimate);

    for (int i = 0; i < len; i++) {
        int N = (kEBands[i + 1] - kEBands[i]) << lm;
        // Band too narrow to be split down to LM=-1.
        int narrow = (kEBands[i + 1] - kEBands[i]) == 1;
        std::memcpy(tmp, &X[tf_chan * n0 + (kEBands[i] << lm)],
                    N * sizeof(float));
        double L1 = l1_metric(tmp, N, is_transient ? lm : 0, bias);
        double best_L1 = L1;
        int best_level = 0;
        // The -1 case (extra merge) for transients.
        if (is_transient && !narrow) {
            std::memcpy(tmp_1, tmp, N * sizeof(float));
            haar1f(tmp_1, N >> lm, 1 << lm);
            L1 = l1_metric(tmp_1, N, lm + 1, bias);
            if (L1 < best_L1) {
                best_L1 = L1;
                best_level = -1;
            }
        }
        for (int k = 0; k < lm + !(is_transient || narrow); k++) {
            int B = is_transient ? lm - k - 1 : k + 1;
            haar1f(tmp, N >> k, 1 << k);
            L1 = l1_metric(tmp, N, B, bias);
            if (L1 < best_L1) {
                best_L1 = L1;
                best_level = k + 1;
            }
        }
        // Q1 metric so narrow bands can sit on the half-way point.
        metric[i] = is_transient ? 2 * best_level : -2 * best_level;
        if (narrow && (metric[i] == 0 || metric[i] == -2 * lm))
            metric[i] -= 1;
    }

    // Try both tf_select settings.
    int selcost[2];
    for (int sel = 0; sel < 2; sel++) {
        int cost0 =
            importance[0] *
            std::abs(metric[0] -
                     2 * kTfSelectTable[lm][4 * is_transient + 2 * sel]);
        int cost1 =
            importance[0] *
                std::abs(
                    metric[0] -
                    2 * kTfSelectTable[lm][4 * is_transient + 2 * sel + 1]) +
            (is_transient ? 0 : lambda);
        for (int i = 1; i < len; i++) {
            int curr0 = imin(cost0, cost1 + lambda);
            int curr1 = imin(cost0 + lambda, cost1);
            cost0 = curr0 +
                    importance[i] *
                        std::abs(metric[i] -
                                 2 * kTfSelectTable[lm]
                                                   [4 * is_transient +
                                                    2 * sel]);
            cost1 = curr1 +
                    importance[i] *
                        std::abs(metric[i] -
                                 2 * kTfSelectTable[lm]
                                                   [4 * is_transient +
                                                    2 * sel + 1]);
        }
        selcost[sel] = imin(cost0, cost1);
    }
    // Conservative: tf_select=1 only for transients (reference policy).
    int tf_select = (selcost[1] < selcost[0] && is_transient) ? 1 : 0;

    // Viterbi forward pass.
    int cost0 =
        importance[0] *
        std::abs(metric[0] -
                 2 * kTfSelectTable[lm][4 * is_transient + 2 * tf_select]);
    int cost1 =
        importance[0] *
            std::abs(metric[0] -
                     2 * kTfSelectTable[lm]
                                       [4 * is_transient + 2 * tf_select +
                                        1]) +
        (is_transient ? 0 : lambda);
    for (int i = 1; i < len; i++) {
        int from0 = cost0, from1 = cost1 + lambda;
        int curr0;
        if (from0 < from1) {
            curr0 = from0;
            path0[i] = 0;
        } else {
            curr0 = from1;
            path0[i] = 1;
        }
        from0 = cost0 + lambda;
        from1 = cost1;
        int curr1;
        if (from0 < from1) {
            curr1 = from0;
            path1[i] = 0;
        } else {
            curr1 = from1;
            path1[i] = 1;
        }
        cost0 = curr0 +
                importance[i] *
                    std::abs(metric[i] -
                             2 * kTfSelectTable[lm][4 * is_transient +
                                                    2 * tf_select]);
        cost1 = curr1 +
                importance[i] *
                    std::abs(metric[i] -
                             2 * kTfSelectTable[lm][4 * is_transient +
                                                    2 * tf_select + 1]);
    }
    tf_res[len - 1] = cost0 < cost1 ? 0 : 1;
    // Backward pass.
    for (int i = len - 2; i >= 0; i--)
        tf_res[i] = tf_res[i + 1] == 1 ? path1[i + 1] : path0[i + 1];
    return tf_select;
}

float median_of_5f(const float* x) {
    float t0, t1, t2 = x[2], t3, t4;
    if (x[0] > x[1]) {
        t0 = x[1];
        t1 = x[0];
    } else {
        t0 = x[0];
        t1 = x[1];
    }
    if (x[3] > x[4]) {
        t3 = x[4];
        t4 = x[3];
    } else {
        t3 = x[3];
        t4 = x[4];
    }
    if (t0 > t3) {
        std::swap(t0, t3);
        std::swap(t1, t4);
    }
    if (t2 > t1) return t1 < t3 ? std::min(t2, t3) : std::min(t4, t1);
    return t2 < t3 ? std::min(t1, t3) : std::min(t2, t4);
}

float median_of_3f(const float* x) {
    float t0, t1, t2 = x[2];
    if (x[0] > x[1]) {
        t0 = x[1];
        t1 = x[0];
    } else {
        t0 = x[0];
        t1 = x[1];
    }
    if (t1 < t2) return t1;
    return t0 < t2 ? t2 : t0;
}

// Dynamic-allocation analysis (reference float path, CBR, no surround /
// tonality analyzer / lfe). Produces per-band boost counts in offsets[]
// (quanta flags for the dynalloc wire loop), importance[] for
// tf_analysis, and spread_weight[] for the spread decision. The
// follower tracks a smoothed spectral floor; boosts go to bands that
// poke above it (tonal peaks the plain allocation starves).
// Returns maxDepth (largest band log-energy above the noise floor, for
// the VBR depth floor); *tot_boost_out = analysis-side boost estimate
// in eighth-bits (feeds compute_vbr; the wire loop re-accounts its own).
float dynalloc_analysis(const float* band_log_e, const float* band_log_e2,
                        const float* old_band_e, int end, int C,
                        int* offsets, int lsb_depth, int is_transient,
                        int lm, int effective_bytes, int* importance,
                        int* spread_weight, int32_t* tot_boost_out,
                        const AnalysisInfo& ainfo) {
    float follower[2 * kNbEBands], noise_floor[kNbEBands];
    float band_log_e3[kNbEBands];
    std::memset(offsets, 0, kNbEBands * sizeof(int));
    float max_depth = -31.9f;
    for (int i = 0; i < end; i++) {
        // eMeans, bit depth, band width and the preemphasis tilt
        // (~square of the bark band id) shape the noise floor.
        noise_floor[i] =
            0.0625f * celt::kLogN[i] + 0.5f + (9 - lsb_depth) -
            static_cast<float>(celt::kEMeans[i]) +
            0.0062f * (i + 5) * (i + 5);
    }
    for (int c = 0; c < C; c++)
        for (int i = 0; i < end; i++)
            max_depth = std::max(
                max_depth, band_log_e[c * kNbEBands + i] - noise_floor[i]);
    {
        // Simple masking model for the spreading decision only.
        float mask[kNbEBands], sig[kNbEBands];
        for (int i = 0; i < end; i++)
            mask[i] = band_log_e[i] - noise_floor[i];
        if (C == 2)
            for (int i = 0; i < end; i++)
                mask[i] = std::max(
                    mask[i], band_log_e[kNbEBands + i] - noise_floor[i]);
        std::memcpy(sig, mask, end * sizeof(float));
        for (int i = 1; i < end; i++)
            mask[i] = std::max(mask[i], mask[i - 1] - 2.0f);
        for (int i = end - 2; i >= 0; i--)
            mask[i] = std::max(mask[i], mask[i + 1] - 3.0f);
        for (int i = 0; i < end; i++) {
            float smr =
                sig[i] - std::max(std::max(0.0f, max_depth - 12.0f),
                                  mask[i]);
            int shift = imin(
                5, imax(0, -static_cast<int>(std::floor(0.5f + smr))));
            spread_weight[i] = 32 >> shift;
        }
    }
    *tot_boost_out = 0;
    // Feature floor: 24 kb/s at 20 ms up to 96 kb/s at 2.5 ms.
    if (effective_bytes >= 30 + 5 * lm) {
        int last = 0;
        for (int c = 0; c < C; c++) {
            std::memcpy(band_log_e3, &band_log_e2[c * kNbEBands],
                        end * sizeof(float));
            if (lm == 0) {
                // One-bin bands: energy too noisy, widen with history.
                for (int i = 0; i < imin(8, end); i++)
                    band_log_e3[i] =
                        std::max(band_log_e2[c * kNbEBands + i],
                                 old_band_e[c * kNbEBands + i]);
            }
            float* f = &follower[c * kNbEBands];
            f[0] = band_log_e3[0];
            for (int i = 1; i < end; i++) {
                // The last band >=.5 dB above its neighbor bounds the
                // backward pass (bandlimited-signal guard).
                if (band_log_e3[i] > band_log_e3[i - 1] + 0.5f) last = i;
                f[i] = std::min(f[i - 1] + 1.5f, band_log_e3[i]);
            }
            for (int i = last - 1; i >= 0; i--)
                f[i] = std::min(
                    f[i], std::min(f[i + 1] + 2.0f, band_log_e3[i]));
            // Median filter (offset 1 dB) against spurious triggers.
            const float offset = 1.0f;
            for (int i = 2; i < end - 2; i++)
                f[i] = std::max(f[i],
                                median_of_5f(&band_log_e3[i - 2]) - offset);
            float tmp = median_of_3f(&band_log_e3[0]) - offset;
            f[0] = std::max(f[0], tmp);
            f[1] = std::max(f[1], tmp);
            tmp = median_of_3f(&band_log_e3[end - 3]) - offset;
            f[end - 2] = std::max(f[end - 2], tmp);
            f[end - 1] = std::max(f[end - 1], tmp);
            for (int i = 0; i < end; i++)
                f[i] = std::max(f[i], noise_floor[i]);
        }
        if (C == 2) {
            for (int i = 0; i < end; i++) {
                // 24 dB (4 log2) cross-talk between channels.
                follower[kNbEBands + i] = std::max(
                    follower[kNbEBands + i], follower[i] - 4.0f);
                follower[i] = std::max(follower[i],
                                       follower[kNbEBands + i] - 4.0f);
                follower[i] =
                    0.5f *
                    (std::max(0.0f, band_log_e[i] - follower[i]) +
                     std::max(0.0f,
                              band_log_e[kNbEBands + i] -
                                  follower[kNbEBands + i]));
            }
        } else {
            for (int i = 0; i < end; i++)
                follower[i] =
                    std::max(0.0f, band_log_e[i] - follower[i]);
        }
        for (int i = 0; i < end; i++)
            importance[i] = static_cast<int>(std::floor(
                0.5f + 13.0f * std::exp2(std::min(follower[i], 4.0f))));
        // CBR non-transient frames: halve the dynalloc contribution.
        if (!is_transient)
            for (int i = 0; i < end; i++) follower[i] *= 0.5f;
        for (int i = 0; i < end; i++) {
            if (i < 8) follower[i] *= 2;
            if (i >= 12) follower[i] *= 0.5f;
        }
        if (ainfo.valid) {
            // Boost bands whose neighbours would leak into/out of them.
            for (int i = 0; i < imin(kLeakBands, end); i++)
                follower[i] += (1.0f / 64) * ainfo.leak_boost[i];
        }
        int32_t tot_boost = 0;
        for (int i = 0; i < end; i++) {
            follower[i] = std::min(follower[i], 4.0f);
            int width = C * (kEBands[i + 1] - kEBands[i]) << lm;
            int boost, boost_bits;
            if (width < 6) {
                boost = static_cast<int>(follower[i]);
                boost_bits = (boost * width) << kBitRes;
            } else if (width > 48) {
                boost = static_cast<int>(follower[i] * 8);
                boost_bits = ((boost * width) << kBitRes) / 8;
            } else {
                boost = static_cast<int>(follower[i] * width / 6);
                boost_bits = (boost * 6) << kBitRes;
            }
            // CBR: dynalloc limited to 2/3 of the frame's bits.
            if ((tot_boost + boost_bits) >> kBitRes >> 3 >
                2 * effective_bytes / 3) {
                int32_t cap = (2 * effective_bytes / 3) << kBitRes << 3;
                offsets[i] = cap - tot_boost;
                tot_boost = cap;
                break;
            }
            offsets[i] = boost;
            tot_boost += boost_bits;
        }
        *tot_boost_out = tot_boost;
    } else {
        for (int i = 0; i < end; i++) importance[i] = 13;
    }
    return max_depth;
}

// Stepped hysteresis: which threshold bucket val falls in, sticky
// around the previous decision (reference bands.c).
int hysteresis_decision(float val, const float* thresholds,
                        const float* hysteresis, int N, int prev) {
    int i;
    for (i = 0; i < N; i++)
        if (val < thresholds[i]) break;
    if (i > prev && val < thresholds[prev] + hysteresis[prev]) i = prev;
    if (i < prev && val > thresholds[prev - 1] - hysteresis[prev - 1])
        i = prev;
    return i;
}

// L1-entropy model of L/R vs M/S over the first 13 bands: decide
// dual_stereo (code channels separately) when L/R is sparser than M/S
// after accounting for the theta signalling cost. Reference
// stereo_analysis; policy-only (the allocator wire-codes the flag).
int stereo_analysis(const float* X, int lm, int n0) {
    double sum_lr = 1e-15, sum_ms = 1e-15;
    for (int i = 0; i < 13; i++) {
        for (int j = kEBands[i] << lm; j < kEBands[i + 1] << lm; j++) {
            double L = X[j], R = X[n0 + j];
            sum_lr += std::fabs(L) + std::fabs(R);
            sum_ms += std::fabs(L + R) + std::fabs(L - R);
        }
    }
    sum_ms *= 0.707107;
    int thetas = 13;
    if (lm <= 1) thetas -= 8;  // no thetas for low bands at LM<=1
    return ((kEBands[13] << (lm + 1)) + thetas) * sum_ms >
           (kEBands[13] << (lm + 1)) * sum_lr;
}

// Spreading (rotation) decision from band-coefficient sparsity: a rough
// CDF of |x|^2*N per band, tonality-weighted by spread_weight, with a
// recursive average and hysteresis vs the previous decision. Also
// tracks the HF sparsity average that drives the prefilter's tapset.
// Reference bands.c spreading_decision; policy-only.
int spreading_decision(const float* X, int* average, int last_decision,
                       int* hf_average, int* tapset_decision, int update_hf,
                       int end, int C, int m, const int* spread_weight,
                       int n0) {
    int sum = 0, nb_bands = 0, hf_sum = 0;
    if (m * (kEBands[end] - kEBands[end - 1]) <= 8) return 0;  // NONE
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < end; i++) {
            const float* x = X + m * kEBands[i] + c * n0;
            int N = m * (kEBands[i + 1] - kEBands[i]);
            if (N <= 8) continue;
            int tcount[3] = { 0, 0, 0 };
            for (int j = 0; j < N; j++) {
                float x2N = x[j] * x[j] * N;
                if (x2N < 0.25f) tcount[0]++;
                if (x2N < 0.0625f) tcount[1]++;
                if (x2N < 0.015625f) tcount[2]++;
            }
            // Only the last four bands (8 kHz up) feed the HF average.
            if (i > kNbEBands - 4)
                hf_sum += 32 * (tcount[1] + tcount[0]) / N;
            int tmp = (2 * tcount[2] >= N) + (2 * tcount[1] >= N) +
                      (2 * tcount[0] >= N);
            sum += tmp * spread_weight[i];
            nb_bands += spread_weight[i];
        }
    }
    if (update_hf) {
        if (hf_sum) hf_sum /= C * (4 - kNbEBands + end);
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;
        if (*tapset_decision == 2)
            hf_sum += 4;
        else if (*tapset_decision == 0)
            hf_sum -= 4;
        if (hf_sum > 22)
            *tapset_decision = 2;
        else if (hf_sum > 18)
            *tapset_decision = 1;
        else
            *tapset_decision = 0;
    }
    sum = (sum << 8) / nb_bands;
    sum = (sum + *average) >> 1;
    *average = sum;
    sum = (3 * sum + (((3 - last_decision) << 7) + 64) + 2) >> 2;
    if (sum < 80) return 3;   // AGGRESSIVE
    if (sum < 256) return 2;  // NORMAL
    if (sum < 384) return 1;  // LIGHT
    return 0;                 // NONE
}

// Allocation-trim analysis (reference float path, no surround / no
// tonality analyzer): base 5 shaded by the low-band stereo correlation
// (correlated stereo -> spend low), the spectral tilt of the band log
// energies (dark spectra -> trim up), and the transient strength.
// Policy-only: the chosen trim is wire-coded and the decoder follows.
int alloc_trim_analysis(const float* X, const float* band_log_e, int end,
                        int lm, int C, int n0, double tf_estimate,
                        int intensity, int32_t equiv_rate,
                        double* stereo_saving, const AnalysisInfo& ainfo) {
    double trim = 5.0;
    if (equiv_rate < 64000) {
        trim = 4.0;
    } else if (equiv_rate < 80000) {
        int32_t frac = (equiv_rate - 64000) >> 10;
        trim = 4.0 + (1.0 / 16.0) * frac;
    }
    if (C == 2) {
        // Inter-channel correlation of the normalized low bands.
        double sum = 0;
        for (int i = 0; i < 8; i++) {
            double partial = 0;
            for (int j = kEBands[i] << lm; j < kEBands[i + 1] << lm; j++)
                partial += static_cast<double>(X[j]) * X[n0 + j];
            sum += partial;
        }
        sum = std::min(1.0, std::fabs(sum * (1.0 / 8)));
        double min_xc = sum;
        for (int i = 8; i < intensity; i++) {
            double partial = 0;
            for (int j = kEBands[i] << lm; j < kEBands[i + 1] << lm; j++)
                partial += static_cast<double>(X[j]) * X[n0 + j];
            min_xc = std::min(min_xc, std::fabs(partial));
        }
        min_xc = std::min(1.0, min_xc);
        double log_xc = std::log2(1.001 - sum * sum);
        // Mid/side savings estimate from the min correlation (feeds the
        // VBR target; persists across frames with a slow release).
        double log_xc2 =
            std::max(0.5 * log_xc, std::log2(1.001 - min_xc * min_xc));
        trim += std::max(-4.0, 0.75 * log_xc);
        *stereo_saving = std::min(*stereo_saving + 0.25, -0.5 * log_xc2);
    }
    // Spectral tilt of the band log energies.
    double diff = 0;
    for (int c = 0; c < C; c++)
        for (int i = 0; i < end - 1; i++)
            diff += band_log_e[i + c * kNbEBands] * (2 + 2 * i - end);
    diff /= C * (end - 1);
    trim -= std::max(-2.0, std::min(2.0, (diff + 1.0) / 6));
    trim -= 2 * tf_estimate;
    if (ainfo.valid)
        trim -= std::max(-2.0f, std::min(2.0f,
                                         2.0f * (ainfo.tonality_slope +
                                                 0.05f)));
    int trim_index = static_cast<int>(std::floor(0.5 + trim));
    return imax(0, imin(10, trim_index));
}

// Per-frame VBR target in eighth-bits (reference compute_vbr, float
// path, no tonality analyzer / surround / lfe — the analyzer-gated
// activity/tonality/pitch_change boosts don't apply). Policy-only: the
// chosen packet size is what the container carries.
int32_t compute_vbr(int32_t base_target, int lm, int32_t equiv_rate,
                    int last_coded_bands, int C, int intensity,
                    double stereo_saving, int32_t tot_boost,
                    double tf_estimate, float max_depth,
                    float temporal_vbr, const AnalysisInfo& ainfo,
                    int pitch_change) {
    int coded_bands = last_coded_bands ? last_coded_bands : kNbEBands;
    int coded_bins = kEBands[coded_bands] << lm;
    if (C == 2)
        coded_bins += kEBands[imin(intensity, coded_bands)] << lm;

    int32_t target = base_target;
    // Low-activity frames need fewer bits.
    if (ainfo.valid && ainfo.activity < 0.4f)
        target -= static_cast<int32_t>((coded_bins << kBitRes) *
                                       (0.4f - ainfo.activity));
    if (C == 2) {
        // Bits we can save when the signal is (nearly) mono.
        int coded_stereo_bands = imin(intensity, coded_bands);
        int coded_stereo_dof =
            (kEBands[coded_stereo_bands] << lm) - coded_stereo_bands;
        double max_frac = 0.8 * coded_stereo_dof / coded_bins;
        double ss = std::min(stereo_saving, 1.0);
        target -= static_cast<int32_t>(std::min(
            max_frac * target,
            (ss - 0.1) * (coded_stereo_dof << kBitRes)));
    }
    // Dynalloc boost, minus its calibration average.
    target += tot_boost - (19 << lm);
    // Transient boost, compensating for the average.
    const double tf_calibration = 0.044;
    target += static_cast<int32_t>((tf_estimate - tf_calibration) *
                                   target);
    // Tonality boost (compensating for the average), plus a pitch-change
    // kick: highly tonal content needs the extra rate most.
    if (ainfo.valid) {
        float tonal = std::max(0.0f, ainfo.tonality - 0.15f) - 0.12f;
        int32_t tonal_target =
            target + static_cast<int32_t>((coded_bins << kBitRes) *
                                          1.2f * tonal);
        if (pitch_change)
            tonal_target += static_cast<int32_t>(
                (coded_bins << kBitRes) * 0.8f);
        target = tonal_target;
    }
    // Don't bury the signal: cap the rate by the depth over the noise
    // floor (no point coding below the quantization floor).
    {
        int bins = kEBands[kNbEBands - 2] << lm;
        int32_t floor_depth = static_cast<int32_t>(
            (C * bins << kBitRes) * max_depth);
        floor_depth = imax(floor_depth, target >> 2);
        target = imin(target, floor_depth);
    }
    // Temporal VBR: below 96 kb/s, louder-than-average frames get more.
    if (tf_estimate < 0.2) {
        double amount =
            0.0000031 * imax(0, imin(32000, 96000 - equiv_rate));
        double tvbr_factor = temporal_vbr * amount;
        target += static_cast<int32_t>(tvbr_factor * target);
    }
    // Never more than double the base rate.
    return imin(2 * base_target, target);
}
}  // namespace

void CeltEncoder::init(int channels) {
    channels_ = channels;
    final_range_ = 0;
    std::memset(in_mem_, 0, sizeof(in_mem_));
    std::memset(prefilter_mem_, 0, sizeof(prefilter_mem_));
    std::memset(old_ebands_, 0, sizeof(old_ebands_));
    std::memset(energy_error_, 0, sizeof(energy_error_));
    preemph_mem_[0] = preemph_mem_[1] = 0;
    delayed_intra_ = 1.0f;
    last_coded_bands_ = 0;
    prefilter_period_ = 0;
    prefilter_gain_ = 0;
    prefilter_tapset_ = 0;
    consec_transient_ = 0;
    tonal_average_ = 256;
    hf_average_ = 0;
    tapset_decision_ = 0;
    spread_decision_ = kSpreadNormal;
    intensity_ = 0;
    stereo_saving_ = 0;
    spec_avg_ = 0;
    analysis_.init();
    mdct_window_fill(window_, kOverlap);
}

int CeltEncoder::encode_frame(const float* pcm, int frame_size,
                              uint8_t* out, int nbytes) {
    const int C = channels_;
    const int start = 0;
    const int end = kNbEBands;
    int lm;
    for (lm = 0; lm <= 3; lm++)
        if (120 << lm == frame_size) break;
    if (lm > 3 || nbytes < 2 || nbytes > 1275) return -1;
    const int m = 1 << lm;
    const int n = frame_size;

    // VBR: nbytes is the CAP; analysis budgets key on the target rate.
    int32_t vbr_rate = 0;
    int effective_bytes = nbytes;
    if (vbr_bitrate_ > 0) {
        nbytes = imin(nbytes, 1275 >> (3 - lm));
        const int32_t den = 48000 >> kBitRes;
        vbr_rate = (vbr_bitrate_ * n + (den >> 1)) / den;
        effective_bytes = vbr_rate >> (3 + kBitRes);
    }

    RangeEncoder enc;
    enc.init(out, static_cast<uint32_t>(nbytes));
    const int32_t total_bits = nbytes * 8;

    // Tonality analysis on the raw input (policy: VBR/trim/dynalloc).
    analysis_.feed(pcm, frame_size, C);
    const AnalysisInfo& ainfo = analysis_.info();

    // Pre-emphasis (reference signal scale), raw samples after the
    // filtered overlap history.
    static thread_local double in[2][kOverlap + kMaxFrame];
    for (int c = 0; c < C; c++) {
        double* inp = in[c] + kOverlap;
        float mem = preemph_mem_[c];
        for (int i = 0; i < n; i++) {
            float x = pcm[C * i + c] * 32768.0f;
            inp[i] = x - mem;
            mem = kPreemphCoef * x;
        }
        preemph_mem_[c] = mem;
    }

    // ---- Pitch prefilter (the decoder re-adds it as the postfilter) ----
    int pf_on = 0, pitch_index = kCombFilterMinPeriodEnc, qg = 0;
    double gain1 = 0;
    // Tapset from last frame's HF sparsity analysis (one-frame lag by
    // design — the analysis runs after the prefilter).
    const int new_tapset = tapset_decision_;
    {
        // Unfiltered history + this frame's raw samples.
        static thread_local double pre[2][kCombMaxPeriod + kMaxFrame];
        for (int c = 0; c < C; c++) {
            std::memcpy(pre[c], prefilter_mem_[c],
                        kCombMaxPeriod * sizeof(double));
            std::memcpy(pre[c] + kCombMaxPeriod, in[c] + kOverlap,
                        n * sizeof(double));
        }
        const bool enabled = effective_bytes > 12 * C && start == 0;
        if (enabled) {
            double pitch_buf[(kCombMaxPeriod + kMaxFrame) / 2];
            const double* chans[2] = { pre[0], pre[1] };
            pitch::pitch_downsample(chans, pitch_buf, kCombMaxPeriod + n,
                                    C);
            // Skip the shortest 1.5 octaves (short-term correlation traps).
            pitch::pitch_search(pitch_buf + (kCombMaxPeriod >> 1),
                                pitch_buf, n,
                                kCombMaxPeriod -
                                    3 * kCombFilterMinPeriodEnc,
                                &pitch_index);
            pitch_index = kCombMaxPeriod - pitch_index;
            gain1 = pitch::remove_doubling(
                pitch_buf, kCombMaxPeriod, kCombFilterMinPeriodEnc, n,
                &pitch_index, prefilter_period_, prefilter_gain_);
            if (pitch_index > kCombMaxPeriod - 2)
                pitch_index = kCombMaxPeriod - 2;
            gain1 = 0.7 * gain1;
        }
        // Rate/continuity-adjusted enabling threshold.
        double pf_threshold = 0.2;
        int dp = pitch_index - prefilter_period_;
        if ((dp < 0 ? -dp : dp) * 10 > pitch_index) pf_threshold += 0.2;
        if (nbytes < 25) pf_threshold += 0.1;
        if (nbytes < 35) pf_threshold += 0.1;
        if (prefilter_gain_ > 0.4) pf_threshold -= 0.1;
        if (prefilter_gain_ > 0.55) pf_threshold -= 0.1;
        if (pf_threshold < 0.2) pf_threshold = 0.2;
        if (gain1 < pf_threshold) {
            gain1 = 0;
            pf_on = 0;
            qg = 0;
        } else {
            double dg = gain1 - prefilter_gain_;
            if ((dg < 0 ? -dg : dg) < 0.1) gain1 = prefilter_gain_;
            qg = static_cast<int>(std::floor(0.5 + gain1 * 32.0 / 3.0)) - 1;
            qg = qg < 0 ? 0 : (qg > 7 ? 7 : qg);
            gain1 = 0.09375 * (qg + 1);
            pf_on = 1;
        }
        // Apply: fade the previous frame's filter into this one over the
        // window, writing the FILTERED frame after the filtered history.
        for (int c = 0; c < C; c++) {
            prefilter_period_ =
                prefilter_period_ > kCombFilterMinPeriodEnc
                    ? prefilter_period_
                    : kCombFilterMinPeriodEnc;
            std::memcpy(in[c], in_mem_[c], kOverlap * sizeof(double));
            comb_filter_shared(in[c] + kOverlap, pre[c] + kCombMaxPeriod,
                               prefilter_period_, pitch_index, n,
                               -prefilter_gain_, -gain1,
                               prefilter_tapset_, new_tapset, window_,
                               kOverlap);
            std::memcpy(in_mem_[c], in[c] + n, kOverlap * sizeof(double));
            // Unfiltered history for the next frame's analysis.
            if (n >= kCombMaxPeriod) {
                std::memcpy(prefilter_mem_[c], pre[c] + n,
                            kCombMaxPeriod * sizeof(double));
            } else {
                std::memmove(prefilter_mem_[c], prefilter_mem_[c] + n,
                             (kCombMaxPeriod - n) * sizeof(double));
                std::memcpy(prefilter_mem_[c] + kCombMaxPeriod - n,
                            pre[c] + kCombMaxPeriod, n * sizeof(double));
            }
        }
    }

    // A pitch jump on strongly-voiced tonal content (new note) earns a
    // VBR kick; only consumed when the analysis is valid.
    int pitch_change = 0;
    if ((gain1 > 0.4 || prefilter_gain_ > 0.4) &&
        (!ainfo.valid || ainfo.tonality > 0.3f) &&
        (pitch_index > 1.26 * prefilter_period_ ||
         pitch_index < 0.79 * prefilter_period_))
        pitch_change = 1;

    // Transient detection on the (prefiltered) input incl. history.
    // Gate on the bit budget BEFORE the MDCT: the flag on the wire, the
    // transform interleave and the decoder's tf_res derivation must all
    // agree, so is_transient may never change after this point.
    double tf_estimate = 0;
    int tf_chan = 0;
    int is_transient =
        lm > 0 ? transient_analysis(&in[0][0], n + kOverlap, C,
                                    &tf_estimate, &tf_chan)
               : 0;
    if (!(lm > 0 && static_cast<int32_t>(enc.tell()) + 3 <= total_bits))
        is_transient = 0;
    if (std::getenv("GLINT_DBG_TRANSIENT"))
        std::fprintf(stderr, "T%d", is_transient);
    const int short_blocks = is_transient ? m : 0;

    // Forward MDCT: one long transform, or M interleaved short blocks.
    static thread_local float X[2 * kMaxFrame];
    {
        double freq_d[kMaxFrame];
        int B = is_transient ? m : 1;
        int nb = is_transient ? 120 : n;
        int shift = is_transient ? 3 : 3 - lm;
        for (int c = 0; c < C; c++) {
            for (int b = 0; b < B; b++)
                mdct_.forward(in[c] + b * nb, &freq_d[b], window_,
                              kOverlap, shift, B);
            for (int k = 0; k < n; k++)
                X[c * n + k] = static_cast<float>(freq_d[k]);
        }
    }

    // Band analysis. band_log_e2 = LONG-window log energies (+lm/2 dB)
    // even on transient frames — dynalloc's follower needs the stable
    // long-window view (the reference's secondMdct at complexity >= 8).
    float band_e[2 * kNbEBands], band_log_e[2 * kNbEBands];
    float band_log_e2[2 * kNbEBands];
    float x_norm[2 * kMaxFrame];
    compute_band_energies(X, band_e, end, C, lm);
    amp2Log2(end, end, band_e, band_log_e, C);
    if (is_transient) {
        static thread_local float x_long[2 * kMaxFrame];
        double freq_d[kMaxFrame];
        float band_e2[2 * kNbEBands];
        for (int c = 0; c < C; c++) {
            mdct_.forward(in[c], freq_d, window_, kOverlap, 3 - lm, 1);
            for (int k = 0; k < n; k++)
                x_long[c * n + k] = static_cast<float>(freq_d[k]);
        }
        compute_band_energies(x_long, band_e2, end, C, lm);
        amp2Log2(end, end, band_e2, band_log_e2, C);
        for (int i = 0; i < C * kNbEBands; i++)
            band_log_e2[i] += 0.5f * lm;
    } else {
        std::memcpy(band_log_e2, band_log_e, sizeof(band_log_e2));
    }
    normalise_bands(X, x_norm, band_e, end, C, m);

    // Temporal VBR: how loud this frame's band envelope is vs the
    // running average (louder-than-usual frames earn extra VBR bits).
    float temporal_vbr = 0;
    {
        float follow = -10.0f, frame_avg = 0;
        float offset = short_blocks ? 0.5f * lm : 0.0f;
        for (int i = start; i < end; i++) {
            follow = std::max(follow - 1.0f, band_log_e[i] - offset);
            if (C == 2)
                follow = std::max(follow,
                                  band_log_e[i + kNbEBands] - offset);
            frame_avg += follow;
        }
        frame_avg /= end - start;
        temporal_vbr = frame_avg - spec_avg_;
        temporal_vbr = std::min(3.0f, std::max(-1.5f, temporal_vbr));
        spec_avg_ += 0.02f * temporal_vbr;
    }

    // Dynalloc analysis: per-band boost counts (wire-coded below),
    // importance for tf_analysis, spread weights for the spreading
    // decision, maxDepth + analysis-side boost total for VBR.
    int offsets[kNbEBands], importance[kNbEBands];
    int spread_weight[kNbEBands];
    int32_t tot_boost_analysis = 0;
    float max_depth = dynalloc_analysis(
        band_log_e, band_log_e2, old_ebands_, end, C, offsets,
        /*lsb_depth=*/16, is_transient, lm, effective_bytes, importance,
        spread_weight, &tot_boost_analysis, ainfo);

    // Per-band TF resolution (quality item 3). Disabled at very small
    // frames, like the reference.
    int tf_res[kNbEBands];
    int tf_select = 0;
    if (effective_bytes >= 15 * C) {
        int lambda = imax(80, 20480 / effective_bytes + 2);
        tf_select = tf_analysis(end, is_transient, tf_res, lambda, x_norm,
                                n, lm, tf_estimate, tf_chan, importance);
    } else {
        for (int i = 0; i < end; i++) tf_res[i] = is_transient;
    }

    // ---- Symbol sequence (mirrors the conformant decoder) ----
    int32_t tell = static_cast<int32_t>(enc.tell());
    if (tell == 1) enc.enc_bit_logp(0, 15);  // not silence
    if (start == 0 && tell + 16 <= total_bits) {
        enc.enc_bit_logp(pf_on, 1);
        if (pf_on) {
            pitch_index += 1;
            int octave = ec::ilog(static_cast<uint32_t>(pitch_index)) - 5;
            enc.enc_uint(static_cast<uint32_t>(octave), 6);
            enc.enc_bits(
                static_cast<uint32_t>(pitch_index - (16 << octave)),
                static_cast<unsigned>(4 + octave));
            pitch_index -= 1;
            enc.enc_bits(static_cast<uint32_t>(qg), 3);
            enc.enc_icdf(new_tapset, celt::kTapsetIcdf, 2);
        }
    }
    // Filter state rotation (must mirror what the decoder's postfilter
    // state does).
    prefilter_period_ = pitch_index;
    prefilter_gain_ = gain1;
    prefilter_tapset_ = new_tapset;
    tell = static_cast<int32_t>(enc.tell());
    if (lm > 0 && tell + 3 <= total_bits) {
        enc.enc_bit_logp(is_transient, 3);
        tell = static_cast<int32_t>(enc.tell());
    }

    // Coarse energy (two-pass intra/inter decided inside).
    quant_coarse_energy(start, end, end, band_log_e, old_ebands_,
                        static_cast<uint32_t>(total_bits), energy_error_,
                        enc, C, lm, nbytes, /*force_intra=*/0,
                        &delayed_intra_, /*two_pass=*/1, /*loss_rate=*/0,
                        /*lfe=*/0);

    // tf_encode: XOR-delta-code the per-band change bits where the
    // budget allows, then the select bit when it would actually change
    // the decode. After encoding, tf_res is remapped through the select
    // table EXACTLY as the decoder's tf_decode does — quant_all_bands
    // must see the same per-band tf resolution the decoder will derive,
    // or the band symbols desync (valid wire, garbage audio).
    {
        uint32_t budget = static_cast<uint32_t>(total_bits);
        uint32_t tf_tell = enc.tell();
        int logp = is_transient ? 2 : 4;
        uint32_t tf_select_rsv = lm > 0 && tf_tell + logp + 1 <= budget;
        budget -= tf_select_rsv;
        int curr = 0, tf_changed = 0;
        for (int i = start; i < end; i++) {
            if (tf_tell + logp <= budget) {
                enc.enc_bit_logp(tf_res[i] ^ curr,
                                 static_cast<unsigned>(logp));
                tf_tell = enc.tell();
                curr = tf_res[i];
                tf_changed |= curr;
            } else {
                tf_res[i] = curr;
            }
            logp = is_transient ? 4 : 5;
        }
        if (tf_select_rsv &&
            kTfSelectTable[lm][4 * is_transient + tf_changed] !=
                kTfSelectTable[lm][4 * is_transient + 2 + tf_changed])
            enc.enc_bit_logp(static_cast<unsigned>(tf_select), 1);
        else
            tf_select = 0;
        for (int i = start; i < end; i++)
            tf_res[i] = kTfSelectTable[lm][4 * is_transient +
                                           2 * tf_select + tf_res[i]];
    }

    tell = static_cast<int32_t>(enc.tell());
    if (tell + 4 <= total_bits) {
        if (short_blocks || nbytes < 10 * C) {
            spread_decision_ = kSpreadNormal;
        } else {
            spread_decision_ = spreading_decision(
                x_norm, &tonal_average_, spread_decision_, &hf_average_,
                &tapset_decision_, pf_on && !short_blocks, end, C, m,
                spread_weight, n);
        }
        enc.enc_icdf(spread_decision_, celt::kSpreadIcdf, 5);
    }
    // If the symbol wasn't coded, the decoder assumes NORMAL — the
    // bands must be rotated with what the decoder will derotate with.
    int spread =
        tell + 4 <= total_bits ? spread_decision_ : kSpreadNormal;

    int cap[kNbEBands];
    init_caps(cap, lm, C);

    // Dynalloc wire loop: one flag chain per band; each accepted flag
    // buys `quanta` eighth-bits of boost. offsets[] comes in as the
    // analysis' flag counts and leaves as the wire-coded boost in
    // eighth-bits — exactly what the decoder derives and what
    // compute_allocation consumes.
    int32_t total_boost = 0;
    {
        int dynalloc_logp = 6;
        int32_t total_bits_q3 = total_bits << kBitRes;
        int32_t tellf = static_cast<int32_t>(enc.tell_frac());
        for (int i = start; i < end; i++) {
            int width = C * (kEBands[i + 1] - kEBands[i]) << lm;
            // 6 bits per quantum, capped at 1 bit/sample, floored at
            // 1/8 bit/sample.
            int quanta = imin(width << kBitRes, imax(6 << kBitRes, width));
            int dynalloc_loop_logp = dynalloc_logp;
            int boost = 0;
            int j;
            for (j = 0;
                 tellf + (dynalloc_loop_logp << kBitRes) <
                     total_bits_q3 - total_boost &&
                 boost < cap[i];
                 j++) {
                int flag = j < offsets[i];
                enc.enc_bit_logp(flag,
                                 static_cast<unsigned>(dynalloc_loop_logp));
                tellf = static_cast<int32_t>(enc.tell_frac());
                if (!flag) break;
                boost += quanta;
                total_boost += quanta;
                dynalloc_loop_logp = 1;
            }
            if (j) dynalloc_logp = imax(2, dynalloc_logp - 1);
            offsets[i] = boost;
        }
    }

    // equiv_rate: 20ms-equivalent bits/s (shorter frames discount their
    // fixed per-frame overhead, like the reference).
    int32_t equiv_rate =
        ((static_cast<int32_t>(nbytes) * 8 * 50) >> (3 - lm)) -
        (40 * C + 20) * ((400 >> lm) - 50);
    if (vbr_bitrate_ > 0)
        equiv_rate = imin(equiv_rate,
                          vbr_bitrate_ - (40 * C + 20) * ((400 >> lm) - 50));

    // Intensity + dual-stereo decisions (the allocator wire-codes both).
    int dual_stereo = 0;
    if (C == 2) {
        static const float kIntensityThresholds[21] = {
            1,  2,  3,  4,  5,  6,  7,  8,  16, 24, 36,
            44, 50, 56, 62, 67, 72, 79, 88, 106, 134
        };
        static const float kIntensityHisteresis[21] = {
            1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 8
        };
        // Always M/S for 2.5 ms frames (no analysis at LM=0).
        if (lm != 0) dual_stereo = stereo_analysis(x_norm, lm, n);
        intensity_ = hysteresis_decision(
            static_cast<float>(equiv_rate / 1000), kIntensityThresholds,
            kIntensityHisteresis, 21, intensity_);
        intensity_ = imin(end, imax(start, intensity_));
    }
    int intensity = C == 2 ? intensity_ : end;

    // Trim: analyzed when the budget (net of dynalloc boosts) allows
    // the symbol, else the decoder-side default 5.
    int alloc_trim = 5;
    if (static_cast<int32_t>(enc.tell_frac()) + (6 << kBitRes) <=
        (total_bits << kBitRes) - total_boost) {
        alloc_trim = alloc_trim_analysis(x_norm, band_log_e, end, lm, C,
                                         n, tf_estimate, intensity,
                                         equiv_rate, &stereo_saving_,
                                         ainfo);
        enc.enc_icdf(alloc_trim, celt::kTrimIcdf, 7);
    }

    // Unconstrained VBR: pick this packet's size from the analysis and
    // shrink the coder — raw bits move to the new buffer end; symbols
    // written so far are unaffected. Everything downstream budgets
    // against the new nbytes, exactly like the decoder will.
    if (vbr_rate > 0) {
        int32_t base_target = vbr_rate - ((40 * C + 20) << kBitRes);
        int32_t target = compute_vbr(
            base_target, lm, equiv_rate, last_coded_bands_, C, intensity,
            stereo_saving_, tot_boost_analysis, tf_estimate, max_depth,
            temporal_vbr, ainfo, pitch_change);
        int32_t tellf = static_cast<int32_t>(enc.tell_frac());
        target += tellf;
        // Never shrink below what's already written (+2-byte margin so
        // no decoder bust-prevention logic has triggered).
        int32_t min_allowed =
            ((tellf + total_boost + (1 << (kBitRes + 3)) - 1) >>
             (kBitRes + 3)) +
            2;
        int navail = (target + (1 << (kBitRes + 2))) >> (kBitRes + 3);
        navail = imax(min_allowed, navail);
        navail = imin(nbytes, navail);
        nbytes = navail;
        enc.shrink(static_cast<uint32_t>(nbytes));
    }

    int32_t bits = ((static_cast<int32_t>(nbytes) * 8) << kBitRes) -
                   static_cast<int32_t>(enc.tell_frac()) - 1;
    int anti_collapse_rsv =
        is_transient && lm >= 2 && bits >= ((lm + 2) << kBitRes)
            ? 1 << kBitRes
            : 0;
    bits -= anti_collapse_rsv;

    int pulses[kNbEBands], fine_quant[kNbEBands], fine_priority[kNbEBands];
    int32_t balance = 0;
    int coded_bands = compute_allocation_enc(
        start, end, offsets, cap, alloc_trim, &intensity, &dual_stereo,
        bits, &balance, pulses, fine_quant, fine_priority, C, lm, enc,
        last_coded_bands_, end - 1);
    last_coded_bands_ = coded_bands;

    quant_fine_energy(start, end, old_ebands_, energy_error_, fine_quant,
                      enc, C);

    uint8_t collapse_masks[2 * kNbEBands];
    uint32_t seed = 0;
    quant_all_bands_enc(start, end, x_norm, C == 2 ? x_norm + n : nullptr,
                        collapse_masks, band_e, pulses, short_blocks,
                        spread, dual_stereo, intensity, tf_res,
                        ((static_cast<int32_t>(nbytes) * 8) << kBitRes) -
                            anti_collapse_rsv,
                        balance, enc, lm, coded_bands, &seed);

    if (anti_collapse_rsv > 0)
        enc.enc_bits(consec_transient_ < 2 ? 1u : 0u, 1);

    quant_energy_finalise(start, end, old_ebands_, energy_error_,
                          fine_quant, fine_priority,
                          nbytes * 8 - static_cast<int>(enc.tell()), enc,
                          C);

    if (is_transient)
        consec_transient_++;
    else
        consec_transient_ = 0;

    final_range_ = enc.range();
    enc.done();
    if (enc.error()) return -2;
    (void)imax;
    (void)imin;
    return nbytes;
}

}  // namespace opus
}  // namespace glint

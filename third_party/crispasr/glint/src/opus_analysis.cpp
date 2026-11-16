// glint - Opus encoder tonality analysis (PLAN § O4, tonality item)
// MIT License - Clean-room implementation

#include "opus_analysis.hpp"

#include <cmath>
#include <cstring>

namespace glint {
namespace opus {

namespace {

// Analysis band edges in 50 Hz FFT bins (24 kHz / 480), reference tbands.
const int kTBands[19] = { 4,  8,  12, 16, 20,  24,  28,  32,  40, 48,
                          56, 64, 80, 96, 112, 136, 160, 192, 240 };

constexpr float kLeakageOffset = 2.5f;
constexpr float kLeakageSlope = 2.0f;
constexpr int kNbTonalSkipBands = 9;
constexpr int kAnalysisCountMax = 10000;

inline int imin(int a, int b) { return a < b ? a : b; }

// Reference celt/mathops.h fast_atan2f (float build) — the tonality
// metric was tuned with this approximation, so keep it.
float fast_atan2f(float y, float x) {
    constexpr float cA = 0.43157974f;
    constexpr float cB = 0.67848403f;
    constexpr float cC = 0.08595542f;
    const float cE = static_cast<float>(M_PI / 2);
    float x2 = x * x, y2 = y * y;
    if (x2 + y2 < 1e-18f) return 0;
    if (x2 < y2) {
        float den = (y2 + cB * x2) * (y2 + cC * x2);
        return -x * y * (y2 + cA * x2) / den + (y < 0 ? -cE : cE);
    }
    float den = (x2 + cB * y2) * (x2 + cC * y2);
    return x * y * (x2 + cA * y2) / den + (y < 0 ? -cE : cE) -
           (x * y < 0 ? -cE : cE);
}

// silk_resampler_down2_hp, float variant local to analysis.c: two
// all-pass sections, 2:1 decimation (the HP energy output is only used
// by the bandwidth detector, which is not ported).
void down2(float* S, float* out, const float* in, int in_len) {
    int len2 = in_len / 2;
    for (int k = 0; k < len2; k++) {
        float in32 = in[2 * k];
        float Y = in32 - S[0];
        float X = 0.6074371f * Y;
        float out32 = S[0] + X;
        S[0] = in32 + X;

        in32 = in[2 * k + 1];
        Y = in32 - S[1];
        X = 0.15063f * Y;
        out32 = out32 + S[1] + X;
        S[1] = in32 + X;

        out[k] = 0.5f * out32;
    }
}

}  // namespace

TonalityAnalyzer::TonalityAnalyzer() {
    fft_.init(480);
    // analysis_window[i] = sin^2(pi (i+1) / 480) — matches the reference
    // table to its printed precision.
    for (int i = 0; i < 240; i++) {
        double s = std::sin(M_PI * (i + 1) / 480.0);
        window_[i] = static_cast<float>(s * s);
    }
    init();
}

void TonalityAnalyzer::init() {
    std::memset(inmem_, 0, sizeof(inmem_));
    std::memset(resamp_state_, 0, sizeof(resamp_state_));
    std::memset(angle_, 0, sizeof(angle_));
    std::memset(d_angle_, 0, sizeof(d_angle_));
    std::memset(d2_angle_, 0, sizeof(d2_angle_));
    std::memset(e_, 0, sizeof(e_));
    std::memset(log_e_, 0, sizeof(log_e_));
    std::memset(low_e_, 0, sizeof(low_e_));
    std::memset(high_e_, 0, sizeof(high_e_));
    std::memset(prev_band_tonality_, 0, sizeof(prev_band_tonality_));
    prev_tonality_ = 0;
    count_ = 0;
    e_count_ = 0;
    mem_fill_ = 240;
    info_ = AnalysisInfo();
}

void TonalityAnalyzer::feed(const float* pcm, int frame_size, int C) {
    // Downmix to mono (mean of channels) at 48 kHz, then 2:1 all-pass
    // decimate to 24 kHz, straight into the analysis buffer.
    static thread_local float mono[960];
    static thread_local float half[480];
    int pos = 0;
    while (pos < frame_size) {
        int chunk = imin(frame_size - pos, 2 * (kBufSize - mem_fill_));
        chunk &= ~1;  // resampler consumes pairs
        if (chunk <= 0) break;
        for (int i = 0; i < chunk; i++) {
            float s = 0;
            for (int c = 0; c < C; c++) s += pcm[(pos + i) * C + c];
            mono[i] = s / C;
        }
        down2(resamp_state_, half, mono, chunk);
        std::memcpy(inmem_ + mem_fill_, half, (chunk / 2) * sizeof(float));
        mem_fill_ += chunk / 2;
        pos += chunk;
        if (mem_fill_ >= kBufSize) {
            process();
            // Keep the last 240 samples as the next frame's head.
            std::memmove(inmem_, inmem_ + kBufSize - 240,
                         240 * sizeof(float));
            mem_fill_ = 240;
        }
    }
}

void TonalityAnalyzer::process() {
    constexpr int N = 480, N2 = 240;
    const float pi4 = static_cast<float>(M_PI * M_PI * M_PI * M_PI);
    CeltImdct::Cpx in[N], out[N];
    float tonality[N2], tonality2[N2], noisiness[N2];

    // Two overlapping 480-sample windows packed as real/imag: the "2"
    // spectrum is 240 samples (10 ms) later than the "1" spectrum, giving
    // a phase-prediction with and without extra delay.
    for (int i = 0; i < N2; i++) {
        float w = window_[i];
        in[i].re = w * inmem_[i];
        in[i].im = w * inmem_[N2 + i];
        in[N - i - 1].re = w * inmem_[N - i - 1];
        in[N - i - 1].im = w * inmem_[N + N2 - i - 1];
    }
    fft_.run(in, out);
    // Reference kiss_fft applies 1/nfft in its float forward transform.
    for (int i = 0; i < N; i++) {
        out[i].re /= N;
        out[i].im /= N;
    }
    if (std::isnan(out[0].re)) {
        info_.valid = 0;
        return;
    }

    for (int i = 1; i < N2; i++) {
        float X1r = static_cast<float>(out[i].re + out[N - i].re);
        float X1i = static_cast<float>(out[i].im - out[N - i].im);
        float X2r = static_cast<float>(out[i].im + out[N - i].im);
        float X2i = static_cast<float>(out[N - i].re - out[i].re);

        float angle = (0.5f / static_cast<float>(M_PI)) *
                      fast_atan2f(X1i, X1r);
        float d_angle = angle - angle_[i];
        float d2_ang = d_angle - d_angle_[i];

        float angle2 = (0.5f / static_cast<float>(M_PI)) *
                       fast_atan2f(X2i, X2r);
        float d_angle2 = angle2 - angle;
        float d2_angle2 = d_angle2 - d_angle;

        float mod1 = d2_ang - std::nearbyintf(d2_ang);
        noisiness[i] = std::fabs(mod1);
        mod1 *= mod1;
        mod1 *= mod1;
        float mod2 = d2_angle2 - std::nearbyintf(d2_angle2);
        noisiness[i] += std::fabs(mod2);
        mod2 *= mod2;
        mod2 *= mod2;

        float avg_mod = 0.25f * (d2_angle_[i] + mod1 + 2 * mod2);
        // Two-frame-delayed (reliable) and instant (less reliable)
        // phase-predictability estimates.
        tonality[i] = 1.f / (1.f + 40.f * 16.f * pi4 * avg_mod) - 0.015f;
        tonality2[i] = 1.f / (1.f + 40.f * 16.f * pi4 * mod2) - 0.015f;

        angle_[i] = angle2;
        d_angle_[i] = d_angle2;
        d2_angle_[i] = mod2;
    }
    for (int i = 2; i < N2 - 1; i++) {
        float tt = std::min(tonality2[i],
                            std::max(tonality2[i - 1], tonality2[i + 1]));
        tonality[i] = 0.9f * std::max(tonality[i], tt - 0.1f);
    }

    if (!count_) {
        for (int b = 0; b < kNbTBands; b++) {
            low_e_[b] = 1e10f;
            high_e_[b] = -1e10f;
        }
    }

    float band_log2[kNbTBands + 1];
    {
        // First band is special because of DC.
        float X1r = 2 * static_cast<float>(out[0].re);
        float X2r = 2 * static_cast<float>(out[0].im);
        float E = X1r * X1r + X2r * X2r;
        for (int i = 1; i < 4; i++) {
            float binE = static_cast<float>(
                out[i].re * out[i].re + out[N - i].re * out[N - i].re +
                out[i].im * out[i].im + out[N - i].im * out[N - i].im);
            E += binE;
        }
        band_log2[0] = 0.5f * 1.442695f * std::log(E + 1e-10f);
    }

    float frame_tonality = 0, max_frame_tonality = 0;
    float frame_noisiness = 0, frame_stationarity = 0;
    float relative_e = 0, slope = 0;
    float band_tonality[kNbTBands];
    float logE[kNbTBands];
    for (int b = 0; b < kNbTBands; b++) {
        float E = 0, tE = 0, nE = 0;
        for (int i = kTBands[b]; i < kTBands[b + 1]; i++) {
            float binE = static_cast<float>(
                out[i].re * out[i].re + out[N - i].re * out[N - i].re +
                out[i].im * out[i].im + out[N - i].im * out[N - i].im);
            E += binE;
            tE += binE * std::max(0.0f, tonality[i]);
            nE += binE * 2.f * (0.5f - noisiness[i]);
        }
        if (!(E < 1e9f) || std::isnan(E)) {
            info_.valid = 0;
            return;
        }

        e_[e_count_][b] = E;
        frame_noisiness += nE / (1e-15f + E);
        logE[b] = std::log(E + 1e-10f);
        band_log2[b + 1] = 0.5f * 1.442695f * logE[b];
        log_e_[e_count_][b] = logE[b];
        if (count_ == 0) high_e_[b] = low_e_[b] = logE[b];
        if (high_e_[b] > low_e_[b] + 7.5f) {
            if (high_e_[b] - logE[b] > logE[b] - low_e_[b])
                high_e_[b] -= 0.01f;
            else
                low_e_[b] += 0.01f;
        }
        if (logE[b] > high_e_[b]) {
            high_e_[b] = logE[b];
            low_e_[b] = std::max(high_e_[b] - 15, low_e_[b]);
        } else if (logE[b] < low_e_[b]) {
            low_e_[b] = logE[b];
            high_e_[b] = std::min(low_e_[b] + 15, high_e_[b]);
        }
        relative_e += (logE[b] - low_e_[b]) /
                      (1e-5f + (high_e_[b] - low_e_[b]));

        float L1 = 0, L2 = 0;
        for (int i = 0; i < kNbFrames; i++) {
            L1 += std::sqrt(e_[i][b]);
            L2 += e_[i][b];
        }
        float stationarity =
            std::min(0.99f, L1 / std::sqrt(1e-15f + kNbFrames * L2));
        stationarity *= stationarity;
        stationarity *= stationarity;
        frame_stationarity += stationarity;
        band_tonality[b] = std::max(tE / (1e-15f + E),
                                    stationarity * prev_band_tonality_[b]);
        frame_tonality += band_tonality[b];
        if (b >= kNbTBands - kNbTonalSkipBands)
            frame_tonality -=
                band_tonality[b - kNbTBands + kNbTonalSkipBands];
        max_frame_tonality =
            std::max(max_frame_tonality,
                     (1.f + 0.03f * (b - kNbTBands)) * frame_tonality);
        slope += band_tonality[b] * (b - 8);
        prev_band_tonality_[b] = band_tonality[b];
    }

    // Leakage ladders -> per-band boost against analysis/synthesis leak.
    float leakage_from[kNbTBands + 1], leakage_to[kNbTBands + 1];
    leakage_from[0] = band_log2[0];
    leakage_to[0] = band_log2[0] - kLeakageOffset;
    for (int b = 1; b < kNbTBands + 1; b++) {
        float leak_slope =
            kLeakageSlope * (kTBands[b] - kTBands[b - 1]) / 4.0f;
        leakage_from[b] =
            std::min(leakage_from[b - 1] + leak_slope, band_log2[b]);
        leakage_to[b] = std::max(leakage_to[b - 1] - leak_slope,
                                 band_log2[b] - kLeakageOffset);
    }
    for (int b = kNbTBands - 2; b >= 0; b--) {
        float leak_slope =
            kLeakageSlope * (kTBands[b + 1] - kTBands[b]) / 4.0f;
        leakage_from[b] =
            std::min(leakage_from[b + 1] + leak_slope, leakage_from[b]);
        leakage_to[b] = std::max(leakage_to[b + 1] - leak_slope,
                                 leakage_to[b]);
    }
    int b;
    for (b = 0; b < kNbTBands + 1; b++) {
        float boost =
            std::max(0.0f, leakage_to[b] - band_log2[b]) +
            std::max(0.0f,
                     band_log2[b] - (leakage_from[b] + kLeakageOffset));
        info_.leak_boost[b] = imin(
            255, static_cast<int>(std::floor(0.5f + 64.f * boost)));
    }
    for (; b < kLeakBands; b++) info_.leak_boost[b] = 0;

    frame_stationarity /= kNbTBands;
    relative_e /= kNbTBands;
    if (count_ < 10) relative_e = 0.5f;
    frame_noisiness /= kNbTBands;
    info_.activity = frame_noisiness + (1 - frame_noisiness) * relative_e;

    frame_tonality = max_frame_tonality / (kNbTBands - kNbTonalSkipBands);
    frame_tonality = std::max(frame_tonality, prev_tonality_ * 0.8f);
    prev_tonality_ = frame_tonality;
    info_.tonality = frame_tonality;
    info_.tonality_slope = slope / (8 * 8);

    e_count_ = (e_count_ + 1) % kNbFrames;
    count_ = imin(count_ + 1, kAnalysisCountMax);
    info_.valid = 1;
}

}  // namespace opus
}  // namespace glint

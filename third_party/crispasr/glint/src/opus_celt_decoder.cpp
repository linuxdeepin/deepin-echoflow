// CELT frame decoder — RFC 6716 section 4.3
// MIT License - Clean-room implementation

#include "opus_celt_decoder.hpp"

#include <cmath>
#include <cstring>

#include "opus_celt_bands.hpp"
#include "opus_celt_energy.hpp"
#include "opus_celt_rate.hpp"
#include "opus_celt_tables.hpp"

namespace glint {
namespace opus {

namespace {

using celt::kEBands;
using celt::kNbEBands;

constexpr int kCombFilterMinPeriod = 15;
constexpr double kPreemphCoef = 0.85;  // 48 kHz mode
constexpr double kVerySmall = 1e-30;
constexpr int kSpreadNormal = 2;

inline int imin(int a, int b) { return a < b ? a : b; }
inline int imax(int a, int b) { return a > b ? a : b; }

// TF resolution change per (LM; transient, tf_select, per-band bit).
const signed char kTfSelectTable[4][8] = {
    { 0, -1, 0, -1, 0, -1, 0, -1 },
    { 0, -1, 0, -2, 1, 0, 1, -1 },
    { 0, -2, 0, -3, 2, 0, 1, -1 },
    { 0, -2, 0, -3, 3, 0, 1, -1 },
};

void tf_decode(int start, int end, int is_transient, int* tf_res, int lm,
               RangeDecoder& dec) {
    uint32_t budget = dec.storage_bytes() * 8;
    uint32_t tell = dec.tell();
    int logp = is_transient ? 2 : 4;
    // One reserved bit switches between the two halves of the table.
    int tf_select_rsv = lm > 0 && tell + logp + 1 <= budget;
    budget -= static_cast<uint32_t>(tf_select_rsv);
    int tf_changed = 0;
    int curr = 0;
    for (int i = start; i < end; i++) {
        if (tell + logp <= budget) {
            curr ^= dec.dec_bit_logp(static_cast<unsigned>(logp));
            tell = dec.tell();
            tf_changed |= curr;
        }
        tf_res[i] = curr;
        logp = is_transient ? 4 : 5;
    }
    int tf_select = 0;
    if (tf_select_rsv &&
        kTfSelectTable[lm][4 * is_transient + 0 + tf_changed] !=
            kTfSelectTable[lm][4 * is_transient + 2 + tf_changed]) {
        tf_select = dec.dec_bit_logp(1);
    }
    for (int i = start; i < end; i++)
        tf_res[i] =
            kTfSelectTable[lm][4 * is_transient + 2 * tf_select + tf_res[i]];
}

// Expand normalized bands back to signal scale: gain = 2^(logE + eMeans).
void denormalise_bands(const double* X, double* freq, const double* band_log_e,
                       int start, int end, int m, int n, int silence,
                       int downsample) {
    int bound = m * kEBands[end];
    // Non-48k output: everything above the output Nyquist is dropped
    // BEFORE synthesis, so the decimation in deemphasis cannot alias.
    if (downsample != 1 && bound > n / downsample) bound = n / downsample;
    if (silence) {
        bound = 0;
        start = end = 0;
    }
    double* f = freq;
    const double* x = X + m * kEBands[start];
    for (int i = 0; i < m * kEBands[start]; i++) *f++ = 0;
    for (int i = start; i < end; i++) {
        int j = m * kEBands[i];
        int band_end = m * kEBands[i + 1];
        double lg = band_log_e[i] + celt::kEMeans[i];
        double g = std::exp2(lg < 32.0 ? lg : 32.0);
        do {
            *f++ = *x++ * g;
        } while (++j < band_end);
    }
    std::memset(freq + bound, 0,
                static_cast<size_t>(n - bound) * sizeof(double));
}

// Pitch postfilter: 5-tap comb at period T, cross-faded over the overlap
// from the previous frame's filter parameters.
void comb_filter(double* y, double* x, int t0, int t1, int n, double g0,
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

// ---------------------------------------------------------------------------
// Packet-loss concealment support: the float paths of the reference pitch
// estimator (celt/pitch.c) and LPC kit (celt/celt_lpc.c). In the float build
// every fixed-point macro reduces to plain arithmetic, so these are direct
// double-precision ports; all decisions they feed (pitch lag, interpolation
// offset, energy gates) are continuous, not wire-coupled.

constexpr int kMaxPeriod = 1024;      // extrapolation history (MAX_PERIOD)
constexpr int kPlcPitchLagMax = 720;  // widest PLC pitch lag (66.67 Hz)
constexpr int kPlcPitchLagMin = 100;  // narrowest PLC pitch lag (480 Hz)
// pitch_search buffer bounds for the PLC call (len = decode buffer minus
// the max lag, both in pre-decimation samples).
constexpr int kPlcSearchLen = 2048 - kPlcPitchLagMax;
constexpr int kPlcMaxPitch = kPlcPitchLagMax - kPlcPitchLagMin;

// Autocorrelation ac[0..lag] of x[0..n), with an optional analysis window
// applied to `overlap` samples at both ends. n <= kMaxPeriod.
void celt_autocorr(const double* x, double* ac, const double* window,
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
void plc_renormalise(double* x, int n) {
    double e = 1e-15;
    for (int i = 0; i < n; i++) e += x[i] * x[i];
    double g = 1.0 / std::sqrt(e);
    for (int i = 0; i < n; i++) x[i] *= g;
}

}  // namespace

void CeltDecoder::init(int channels, int32_t fs) {
    channels_ = channels;
    switch (fs) {
    case 48000: downsample_ = 1; break;
    case 24000: downsample_ = 2; break;
    case 16000: downsample_ = 3; break;
    case 12000: downsample_ = 4; break;
    case 8000: downsample_ = 6; break;
    default: downsample_ = 1; break;
    }
    rng_ = 0;
    std::memset(decode_mem_, 0, sizeof(decode_mem_));
    std::memset(old_ebands_, 0, sizeof(old_ebands_));
    // Reference reset: only oldLogE/oldLogE2 start at -28; backgroundLogE is
    // cleared to ZERO (it only feeds the noise-PLC decay floor).
    std::memset(background_log_e_, 0, sizeof(background_log_e_));
    for (int i = 0; i < 2 * kNbEBands; i++)
        old_log_e_[i] = old_log_e2_[i] = -28.0;
    preemph_mem_[0] = preemph_mem_[1] = 0;
    postfilter_period_ = postfilter_period_old_ = 0;
    postfilter_gain_ = postfilter_gain_old_ = 0;
    postfilter_tapset_ = postfilter_tapset_old_ = 0;
    loss_duration_ = 0;
    skip_plc_ = 1;  // reference reset: no pitch PLC before the first frame
    prefilter_and_fold_ = 0;
    last_pitch_index_ = 0;
    plc_start_ = 0;
    plc_end_ = kNbEBands;
    std::memset(plc_lpc_, 0, sizeof(plc_lpc_));
    mdct_window_fill(window_, kOverlap);
}

void CeltDecoder::deemphasis(float* pcm, int n) {
    // The de-emphasis filter always runs at 48 kHz; non-48k output keeps
    // every downsample-th sample (band-limited in denormalise_bands).
    const int ds = downsample_;
    for (int c = 0; c < channels_; c++) {
        double* x = decode_mem_[c] + kDecodeBufferSize - n;
        double mem = preemph_mem_[c];
        for (int j = 0; j < n; j++) {
            double tmp = x[j] + kVerySmall + mem;
            mem = kPreemphCoef * tmp;
            if (j % ds == 0)
                pcm[(j / ds) * channels_ + c] =
                    static_cast<float>(tmp * (1.0 / 32768.0));
        }
        preemph_mem_[c] = mem;
    }
}

void CeltDecoder::prefilter_and_fold(int n) {
    double etmp[kOverlap];
    for (int c = 0; c < channels_; c++) {
        double* mem = decode_mem_[c];
        // Apply the pre-filter (negated postfilter) to the MDCT overlap for
        // the next frame: the decoder re-applies the postfilter after the
        // MDCT overlap, so the concealed tail must be "un-postfiltered".
        comb_filter(etmp, mem + kDecodeBufferSize - n, postfilter_period_old_,
                    postfilter_period_, kOverlap, -postfilter_gain_old_,
                    -postfilter_gain_, postfilter_tapset_old_,
                    postfilter_tapset_, nullptr, 0);
        // Simulate TDAC on the concealed audio so it blends with the next
        // frame's MDCT.
        for (int i = 0; i < kOverlap / 2; i++)
            mem[kDecodeBufferSize - n + i] =
                window_[i] * etmp[kOverlap - 1 - i] +
                window_[kOverlap - 1 - i] * etmp[i];
    }
}

int CeltDecoder::plc_pitch_search() const {
    double lp_pitch_buf[kDecodeBufferSize >> 1];
    const double* x[2] = { decode_mem_[0], decode_mem_[1] };
    pitch_downsample(x, lp_pitch_buf, kDecodeBufferSize, channels_);
    int pitch_index = 0;
    pitch_search(lp_pitch_buf + (kPlcPitchLagMax >> 1), lp_pitch_buf,
                 kDecodeBufferSize - kPlcPitchLagMax,
                 kPlcPitchLagMax - kPlcPitchLagMin, &pitch_index);
    return kPlcPitchLagMax - pitch_index;
}

int CeltDecoder::decode_lost(float* pcm, int frame_size, int start_band,
                             int end_band) {
    const int C = channels_;
    int lm;
    for (lm = 0; lm <= 3; lm++)
        if (120 << lm == frame_size) break;
    if (lm > 3) return -1;
    const int n = frame_size;
    if (start_band >= 0) plc_start_ = start_band;
    if (end_band >= 0) plc_end_ = end_band;
    const int start = plc_start_;
    const int end = plc_end_;
    const int loss_duration = loss_duration_;

    // Pitch PLC needs two consecutive good frames of history and gives up
    // after 100 ms of loss; hybrid frames (start != 0) always use noise.
    int noise_based = loss_duration >= 40 || start != 0 || skip_plc_;
    if (noise_based) {
        // Noise-based PLC/CNG: per-band normalized noise at the last band
        // energies, decaying toward the background noise floor.
        int effend = imax(start, imin(end, kNbEBands));
        for (int c = 0; c < C; c++)
            std::memmove(decode_mem_[c], decode_mem_[c] + n,
                         (kDecodeBufferSize - n + kOverlap) * sizeof(double));
        if (prefilter_and_fold_) prefilter_and_fold(n);

        double decay = loss_duration == 0 ? 1.5 : 0.5;
        for (int c = 0; c < C; c++) {
            for (int i = start; i < end; i++) {
                double floor_e = background_log_e_[c * kNbEBands + i];
                double e = old_ebands_[c * kNbEBands + i] - decay;
                old_ebands_[c * kNbEBands + i] = e > floor_e ? e : floor_e;
            }
        }
        double X[2 * kMaxFrame];
        uint32_t seed = rng_;
        for (int c = 0; c < C; c++) {
            for (int i = start; i < effend; i++) {
                int boffs = n * c + (kEBands[i] << lm);
                int blen = (kEBands[i + 1] - kEBands[i]) << lm;
                for (int j = 0; j < blen; j++) {
                    seed = celt_lcg_rand(seed);
                    // Arithmetic >>20 like the reference (floor, not trunc).
                    X[boffs + j] = static_cast<double>(
                        static_cast<int32_t>(seed) >> 20);
                }
                plc_renormalise(X + boffs, blen);
            }
        }
        rng_ = seed;
        synthesis(X, C, C, 0, lm, 0, effend, start);
        prefilter_and_fold_ = 0;
        skip_plc_ = 1;  // regular PLC only after two consecutive packets
    } else {
        // Pitch-based PLC: extrapolate the excitation with the estimated
        // pitch period, decaying, then re-apply the synthesis filter.
        double fade = 1.0;
        int pitch_index;
        if (loss_duration == 0) {
            last_pitch_index_ = pitch_index = plc_pitch_search();
        } else {
            pitch_index = last_pitch_index_;
            fade = 0.8;
        }
        // Excitation for two pitch periods (to detect decaying signals),
        // capped at the available history.
        int exc_length = imin(2 * pitch_index, kMaxPeriod);

        static thread_local double exc_buf[kMaxPeriod + kPlcLpcOrder];
        static thread_local double fir_tmp[kMaxPeriod];
        double* exc = exc_buf + kPlcLpcOrder;
        for (int c = 0; c < C; c++) {
            double* buf = decode_mem_[c];
            for (int i = 0; i < kMaxPeriod + kPlcLpcOrder; i++)
                exc_buf[i] =
                    buf[kDecodeBufferSize - kMaxPeriod - kPlcLpcOrder + i];

            if (loss_duration == 0) {
                // LPC over the last kMaxPeriod samples before the first
                // loss, so we can work in the excitation-filter domain.
                double ac[kPlcLpcOrder + 1];
                celt_autocorr(exc, ac, window_, kOverlap, kPlcLpcOrder,
                              kMaxPeriod);
                ac[0] *= 1.0001;  // -40 dB noise floor
                for (int i = 1; i <= kPlcLpcOrder; i++)  // lag windowing
                    ac[i] -= ac[i] * (0.008 * 0.008) * i * i;
                celt_lpc(plc_lpc_[c], ac, kPlcLpcOrder);
            }
            // Whiten the last exc_length samples (celt_fir can't run in
            // place; the LPC history sits just before the region).
            celt_fir(exc + kMaxPeriod - exc_length, plc_lpc_[c], fir_tmp,
                     exc_length, kPlcLpcOrder);
            std::memcpy(exc + kMaxPeriod - exc_length, fir_tmp,
                        static_cast<size_t>(exc_length) * sizeof(double));

            // How fast is the waveform decaying? Avoid adding energy when
            // concealing inside a decaying segment.
            double decay;
            {
                double e1 = 1, e2 = 1;
                int decay_length = exc_length >> 1;
                for (int i = 0; i < decay_length; i++) {
                    double e = exc[kMaxPeriod - decay_length + i];
                    e1 += e * e;
                    e = exc[kMaxPeriod - 2 * decay_length + i];
                    e2 += e * e;
                }
                e1 = e1 < e2 ? e1 : e2;
                decay = std::sqrt(e1 / e2);
            }

            // Shift the history left one frame (the overlap tail past the
            // buffer end is about to be overwritten below).
            std::memmove(buf, buf + n,
                         (kDecodeBufferSize - n) * sizeof(double));

            // Extrapolate with period pitch_index, one extra `decay` per
            // period (plus `fade` when this is not the first loss), for a
            // whole MDCT window incl. overlap/2 on both sides.
            int extrapolation_offset = kMaxPeriod - pitch_index;
            int extrapolation_len = n + kOverlap;
            double attenuation = fade * decay;
            double s1 = 0;
            for (int i = 0, j = 0; i < extrapolation_len; i++, j++) {
                if (j >= pitch_index) {
                    j -= pitch_index;
                    attenuation *= decay;
                }
                buf[kDecodeBufferSize - n + i] =
                    attenuation * exc[extrapolation_offset + j];
                // Energy of the previously decoded signal whose excitation
                // we're copying.
                double tmp = buf[kDecodeBufferSize - kMaxPeriod - n +
                                 extrapolation_offset + j];
                s1 += tmp * tmp;
            }
            {
                // Seed the synthesis filter with the last decoded samples
                // for a continuous signal, and leave the excitation domain.
                double lpc_mem[kPlcLpcOrder];
                for (int i = 0; i < kPlcLpcOrder; i++)
                    lpc_mem[i] = buf[kDecodeBufferSize - n - 1 - i];
                celt_iir(buf + kDecodeBufferSize - n, plc_lpc_[c],
                         buf + kDecodeBufferSize - n, extrapolation_len,
                         kPlcLpcOrder, lpc_mem);
            }

            // If the synthesis energy exceeds the source signal's,
            // attenuate (or zero an "explosion"; also catches NaN).
            double s2 = 0;
            for (int i = 0; i < extrapolation_len; i++) {
                double tmp = buf[kDecodeBufferSize - n + i];
                s2 += tmp * tmp;
            }
            if (!(s1 > 0.2 * s2)) {
                for (int i = 0; i < extrapolation_len; i++)
                    buf[kDecodeBufferSize - n + i] = 0;
            } else if (s1 < s2) {
                double ratio = std::sqrt((s1 + 1) / (s2 + 1));
                for (int i = 0; i < kOverlap; i++) {
                    double g = 1.0 - window_[i] * (1.0 - ratio);
                    buf[kDecodeBufferSize - n + i] *= g;
                }
                for (int i = kOverlap; i < extrapolation_len; i++)
                    buf[kDecodeBufferSize - n + i] *= ratio;
            }
        }
        prefilter_and_fold_ = 1;
    }

    // Saturate against wrap-around (10000 = 25 s of loss).
    loss_duration_ = imin(10000, loss_duration + (1 << lm));
    deemphasis(pcm, n);
    return frame_size;
}

void CeltDecoder::synthesis(const double* X, int CC, int C,
                            int is_transient, int lm, int silence,
                            int effend, int start) {
    int n = 120 << lm;
    int b, nb, shift;
    if (is_transient) {
        b = 1 << lm;
        nb = 120;
        shift = 3;  // maxLM
    } else {
        b = 1;
        nb = 120 << lm;
        shift = 3 - lm;
    }
    double freq[kMaxFrame];
    if (CC == 2 && C == 1) {
        // Mono stream to stereo output: same signal into both channels.
        // The IMDCT destroys its input, so stage a copy inside channel 1's
        // output region (in-place safe: input == output + overlap/2).
        denormalise_bands(X, freq, old_ebands_, start, effend, 1 << lm, n,
                          silence, downsample_);
        double* out0 = decode_mem_[0] + kDecodeBufferSize - n;
        double* out1 = decode_mem_[1] + kDecodeBufferSize - n;
        double* freq2 = out1 + kOverlap / 2;
        std::memcpy(freq2, freq, n * sizeof(double));
        for (int blk = 0; blk < b; blk++)
            imdct_.backward(&freq2[blk], out0 + nb * blk, window_, kOverlap,
                            shift, b);
        for (int blk = 0; blk < b; blk++)
            imdct_.backward(&freq[blk], out1 + nb * blk, window_, kOverlap,
                            shift, b);
    } else if (CC == 1 && C == 2) {
        // Stereo stream to mono output: average the denormalised spectra.
        double* out0 = decode_mem_[0] + kDecodeBufferSize - n;
        double* freq2 = out0 + kOverlap / 2;
        denormalise_bands(X, freq, old_ebands_, start, effend, 1 << lm, n,
                          silence, downsample_);
        denormalise_bands(X + n, freq2, old_ebands_ + kNbEBands, start,
                          effend, 1 << lm, n, silence, downsample_);
        for (int i = 0; i < n; i++) freq[i] = 0.5 * freq[i] + 0.5 * freq2[i];
        for (int blk = 0; blk < b; blk++)
            imdct_.backward(&freq[blk], out0 + nb * blk, window_, kOverlap,
                            shift, b);
    } else {
        for (int c = 0; c < CC; c++) {
            double* out_syn = decode_mem_[c] + kDecodeBufferSize - n;
            denormalise_bands(X + c * n, freq, old_ebands_ + c * kNbEBands,
                              start, effend, 1 << lm, n, silence,
                              downsample_);
            for (int blk = 0; blk < b; blk++)
                imdct_.backward(&freq[blk], out_syn + nb * blk, window_,
                                kOverlap, shift, b);
        }
    }
}

int CeltDecoder::decode_frame(RangeDecoder& dec, uint32_t payload_bytes,
                              float* pcm, int frame_size,
                              int stream_channels, int end_band,
                              int start_band) {
    const int CC = channels_;       // decoder output channels
    const int C = stream_channels;  // channels coded in the stream
    const int start = start_band;
    const int end = end_band;
    const int effend = end;

    int lm;
    for (lm = 0; lm <= 3; lm++)
        if (120 << lm == frame_size) break;
    if (lm > 3 || payload_bytes < 1 || payload_bytes > 1275) return -1;
    const int m = 1 << lm;
    const int n = frame_size;
    const int32_t len = static_cast<int32_t>(payload_bytes);

    plc_start_ = start;  // a following decode_lost conceals with this range
    plc_end_ = end;
    // Pitch-based PLC needs at least two consecutively received packets.
    if (loss_duration_ == 0) skip_plc_ = 0;

    double* old_ebands = old_ebands_;
    if (C == 1) {
        for (int i = 0; i < kNbEBands; i++)
            old_ebands[i] = old_ebands[i] > old_ebands[kNbEBands + i]
                                ? old_ebands[i]
                                : old_ebands[kNbEBands + i];
    }

    int32_t total_bits = len * 8;
    int32_t tell = static_cast<int32_t>(dec.tell());

    // Silence flag (only meaningful as the first symbol).
    int silence;
    if (tell >= total_bits)
        silence = 1;
    else if (tell == 1)
        silence = dec.dec_bit_logp(15);
    else
        silence = 0;
    if (silence) {
        tell = total_bits;
        dec.set_tell(static_cast<uint32_t>(tell));
    }

    // Postfilter parameters.
    double postfilter_gain = 0;
    int postfilter_pitch = 0;
    int postfilter_tapset = 0;
    if (start == 0 && tell + 16 <= total_bits) {
        if (dec.dec_bit_logp(1)) {
            int octave = static_cast<int>(dec.dec_uint(6));
            postfilter_pitch =
                (16 << octave) +
                static_cast<int>(dec.dec_bits(4 + octave)) - 1;
            int qg = static_cast<int>(dec.dec_bits(3));
            if (static_cast<int32_t>(dec.tell()) + 2 <= total_bits)
                postfilter_tapset = dec.dec_icdf(celt::kTapsetIcdf, 2);
            postfilter_gain = 0.09375 * (qg + 1);
        }
        tell = static_cast<int32_t>(dec.tell());
    }

    int is_transient = 0;
    if (lm > 0 && tell + 3 <= total_bits) {
        is_transient = dec.dec_bit_logp(3);
        tell = static_cast<int32_t>(dec.tell());
    }
    int short_blocks = is_transient ? m : 0;

    int intra_ener = tell + 3 <= total_bits ? dec.dec_bit_logp(3) : 0;

    // Recovering from loss: make the energy prediction safe to reduce the
    // risk of loud artifacts from stale history.
    if (!intra_ener && loss_duration_ != 0) {
        // Shorter frames have more natural fluctuation -- play it safe.
        double safety = lm == 0 ? 1.5 : lm == 1 ? 0.5 : 0.0;
        int missing = imin(10, loss_duration_ >> lm);
        for (int c = 0; c < 2; c++) {
            for (int i = start; i < end; i++) {
                double e0 = old_ebands[c * kNbEBands + i];
                double e1 = old_log_e_[c * kNbEBands + i];
                double e2 = old_log_e2_[c * kNbEBands + i];
                if (e0 < (e1 > e2 ? e1 : e2)) {
                    // Energy already going down: continue the trend.
                    double slope = e1 - e0 > 0.5 * (e2 - e0)
                                       ? e1 - e0
                                       : 0.5 * (e2 - e0);
                    e0 -= slope > 0 ? (1 + missing) * slope : 0.0;
                    old_ebands[c * kNbEBands + i] = e0 > -20.0 ? e0 : -20.0;
                } else {
                    // Otherwise take the min of the last frames.
                    double e = e0 < e1 ? e0 : e1;
                    old_ebands[c * kNbEBands + i] = e < e2 ? e : e2;
                }
                old_ebands[c * kNbEBands + i] -= safety;
            }
        }
    }

    unquant_coarse_energy(start, end, old_ebands, intra_ener != 0, dec, C,
                          lm);

    int tf_res[kNbEBands];
    tf_decode(start, end, is_transient, tf_res, lm, dec);

    tell = static_cast<int32_t>(dec.tell());
    int spread = kSpreadNormal;
    if (tell + 4 <= total_bits) spread = dec.dec_icdf(celt::kSpreadIcdf, 5);

    int cap[kNbEBands];
    init_caps(cap, lm, C);

    // Dynalloc: per-band boosts, cost halves after the first boost.
    int offsets[kNbEBands];
    int dynalloc_logp = 6;
    total_bits <<= kBitRes;
    int32_t tellf = static_cast<int32_t>(dec.tell_frac());
    for (int i = start; i < end; i++) {
        int width = C * (kEBands[i + 1] - kEBands[i]) << lm;
        // 6 bits/step, within [1/8 bit, 1 bit] per sample.
        int quanta = imin(width << kBitRes, imax(6 << kBitRes, width));
        int dynalloc_loop_logp = dynalloc_logp;
        int boost = 0;
        while (tellf + (dynalloc_loop_logp << kBitRes) < total_bits &&
               boost < cap[i]) {
            int flag = dec.dec_bit_logp(
                static_cast<unsigned>(dynalloc_loop_logp));
            tellf = static_cast<int32_t>(dec.tell_frac());
            if (!flag) break;
            boost += quanta;
            total_bits -= quanta;
            dynalloc_loop_logp = 1;
        }
        offsets[i] = boost;
        if (boost > 0) dynalloc_logp = imax(2, dynalloc_logp - 1);
    }

    int alloc_trim = tellf + (6 << kBitRes) <= total_bits
                         ? dec.dec_icdf(celt::kTrimIcdf, 7)
                         : 5;

    int32_t bits = (len * 8 << kBitRes) -
                   static_cast<int32_t>(dec.tell_frac()) - 1;
    int anti_collapse_rsv =
        is_transient && lm >= 2 && bits >= (lm + 2) << kBitRes ? 1 << kBitRes
                                                               : 0;
    bits -= anti_collapse_rsv;

    int pulses[kNbEBands], fine_quant[kNbEBands], fine_priority[kNbEBands];
    int intensity = 0, dual_stereo = 0;
    int32_t balance = 0;
    int coded_bands = compute_allocation_dec(
        start, end, offsets, cap, alloc_trim, &intensity, &dual_stereo,
        bits, &balance, pulses, fine_quant, fine_priority, C, lm, dec);

    unquant_fine_energy(start, end, old_ebands, fine_quant, dec, C);

    for (int c = 0; c < CC; c++)
        std::memmove(decode_mem_[c], decode_mem_[c] + n,
                     (kDecodeBufferSize - n + kOverlap) * sizeof(double));

    uint8_t collapse_masks[2 * kNbEBands];
    static thread_local double X[2 * kMaxFrame];
    quant_all_bands_dec(start, end, X, C == 2 ? X + n : nullptr,
                        collapse_masks, pulses, short_blocks, spread,
                        dual_stereo, intensity, tf_res,
                        len * (8 << kBitRes) - anti_collapse_rsv, balance,
                        dec, lm, coded_bands, &rng_,
                        C == 1 /* mono decoders disable inversion */);

    int anti_collapse_on = 0;
    if (anti_collapse_rsv > 0)
        anti_collapse_on = static_cast<int>(dec.dec_bits(1));

    unquant_energy_finalise(start, end, old_ebands, fine_quant,
                            fine_priority,
                            len * 8 - static_cast<int>(dec.tell()), dec, C);

    if (anti_collapse_on)
        anti_collapse(X, collapse_masks, lm, C, n, start, end, old_ebands,
                      old_log_e_, old_log_e2_, pulses, rng_);

    if (silence) {
        for (int i = 0; i < 2 * kNbEBands; i++) old_ebands[i] = -28.0;
    }

    // First good frame after loss: fold the concealed tail into the MDCT
    // overlap so it blends with this frame's synthesis.
    if (prefilter_and_fold_) prefilter_and_fold(n);

    synthesis(X, CC, C, is_transient, lm, silence, effend, start);

    // Postfilter: previous frame's parameters over the first short-MDCT,
    // then a cross-fade to this frame's parameters.
    for (int c = 0; c < CC; c++) {
        double* out_syn = decode_mem_[c] + kDecodeBufferSize - n;
        postfilter_period_ = imax(postfilter_period_, kCombFilterMinPeriod);
        postfilter_period_old_ =
            imax(postfilter_period_old_, kCombFilterMinPeriod);
        comb_filter(out_syn, out_syn, postfilter_period_old_,
                    postfilter_period_, 120, postfilter_gain_old_,
                    postfilter_gain_, postfilter_tapset_old_,
                    postfilter_tapset_, window_, kOverlap);
        if (lm != 0)
            comb_filter(out_syn + 120, out_syn + 120, postfilter_period_,
                        postfilter_pitch, n - 120, postfilter_gain_,
                        postfilter_gain, postfilter_tapset_,
                        postfilter_tapset, window_, kOverlap);
    }
    postfilter_period_old_ = postfilter_period_;
    postfilter_gain_old_ = postfilter_gain_;
    postfilter_tapset_old_ = postfilter_tapset_;
    postfilter_period_ = postfilter_pitch;
    postfilter_gain_ = postfilter_gain;
    postfilter_tapset_ = postfilter_tapset;
    if (lm != 0) {
        postfilter_period_old_ = postfilter_period_;
        postfilter_gain_old_ = postfilter_gain_;
        postfilter_tapset_old_ = postfilter_tapset_;
    }

    if (C == 1)
        std::memcpy(&old_ebands[kNbEBands], old_ebands,
                    kNbEBands * sizeof(double));

    // Energy history rotation (anti-collapse depends on two frames back).
    if (!is_transient) {
        std::memcpy(old_log_e2_, old_log_e_, sizeof(old_log_e2_));
        std::memcpy(old_log_e_, old_ebands, sizeof(old_log_e_));
    } else {
        for (int i = 0; i < 2 * kNbEBands; i++)
            old_log_e_[i] =
                old_log_e_[i] < old_ebands[i] ? old_log_e_[i] : old_ebands[i];
    }
    // Noise floor may rise at most 2.4 dB/s; after loss (or DTX) the missed
    // frames' weight all lands on this update packet.
    double max_bg_incr = imin(160, loss_duration_ + m) * 0.001;
    for (int i = 0; i < 2 * kNbEBands; i++)
        background_log_e_[i] =
            background_log_e_[i] + max_bg_incr < old_ebands[i]
                ? background_log_e_[i] + max_bg_incr
                : old_ebands[i];
    for (int c = 0; c < 2; c++) {
        for (int i = 0; i < start; i++) {
            old_ebands[c * kNbEBands + i] = 0;
            old_log_e_[c * kNbEBands + i] =
                old_log_e2_[c * kNbEBands + i] = -28.0;
        }
        for (int i = end; i < kNbEBands; i++) {
            old_ebands[c * kNbEBands + i] = 0;
            old_log_e_[c * kNbEBands + i] =
                old_log_e2_[c * kNbEBands + i] = -28.0;
        }
    }
    rng_ = dec.range();

    // De-emphasis (1-pole) and PCM output at ±1.0.
    deemphasis(pcm, n);
    loss_duration_ = 0;
    prefilter_and_fold_ = 0;

    if (static_cast<int32_t>(dec.tell()) > 8 * len) return -3;
    return frame_size;
}

}  // namespace opus
}  // namespace glint

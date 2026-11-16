// glint - AAC temporal noise shaping (TNS), long windows
// MIT License - Clean-room implementation

#include "aac_tns.hpp"
#include "aac_coder.hpp"
#include "aac_tables.hpp"

#include <cmath>
#include <cstring>

namespace glint {
namespace aac {

using namespace aac_tables;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kGainThreshold = 4.0;  // selective: only strongly predictable frames
constexpr double kTnsMinFreqHz = 2500.0;

// 4-bit arcsin coefficient (de)quantization per ISO (coef_res = 4,
// coef_compress = 0). Validated against vo-aacenc's Q31 tnsCoeff4 map by
// tools/gen_aac_tables.py.
constexpr double kIqPos = (8.0 - 0.5) / (kPi / 2.0);
constexpr double kIqNeg = (8.0 + 0.5) / (kPi / 2.0);

int quant_coef(double k) {
    if (k > 0.999) k = 0.999;
    if (k < -0.999) k = -0.999;
    double a = std::asin(k);
    int idx = static_cast<int>(std::lround(a * (k >= 0 ? kIqPos : kIqNeg)));
    if (idx > 7) idx = 7;
    if (idx < -8) idx = -8;
    return idx;
}

double dequant_coef(int idx) {
    return std::sin(idx / (idx >= 0 ? kIqPos : kIqNeg));
}

}  // namespace

void aac_tns_analyze(SpecT* spec, const AacBandLayout& L, int sr_index,
                     AacTnsFilter* f) {
    f->active = 0;
    if (L.window_sequence == 2) return;  // long-family only

    // Region: [start_band, min(max_sfb, tns_max_bands)), start above ~1 kHz.
    const uint16_t* swb = kSwbOffsetLong[sr_index];
    int top_band = L.max_sfb;
    if (top_band > kTnsMaxBandsLong[sr_index]) top_band = kTnsMaxBandsLong[sr_index];
    double hz_per_line = 0.5 * kSampleRates[sr_index] / 1024.0;
    int min_line = static_cast<int>(kTnsMinFreqHz / hz_per_line);
    int start_band = 0;
    while (start_band < top_band && swb[start_band] < min_line) start_band++;
    if (top_band - start_band < 4) return;

    const int start = swb[start_band];
    const int end = swb[top_band];
    const int n = end - start;
    if (n < 64) return;

    // Autocorrelation over the region.
    double r[kTnsMaxOrder + 1];
#ifdef GLINT_AAC_INT
    // Integer accumulation (>>8 pre-shift keeps ~600-term sums of Q3^2
    // products inside int64); the 13 lag totals then go to double for the
    // tiny Levinson recursion — per-frame scalars, not per-coefficient work.
    for (int lag = 0; lag <= kTnsMaxOrder; lag++) {
        int64_t acc = 0;
        for (int i = start + lag; i < end; i++) {
            acc += static_cast<int64_t>(spec[i]) * spec[i - lag];
        }
        r[lag] = static_cast<double>(acc);
    }
#else
    for (int lag = 0; lag <= kTnsMaxOrder; lag++) {
        double acc = 0.0;
        for (int i = start + lag; i < end; i++) acc += spec[i] * spec[i - lag];
        r[lag] = acc;
    }
#endif
    if (r[0] <= 0.0) return;
    r[0] *= 1.0 + 1e-9;  // regularize

    // Levinson-Durbin: reflection coefficients + prediction error.
    double a[kTnsMaxOrder + 1] = {1.0};
    double refl[kTnsMaxOrder];
    double err = r[0];
    int order = 0;
    for (int m = 1; m <= 8; m++) {
        double acc = r[m];
        for (int j = 1; j < m; j++) acc += a[j] * r[m - j];
        double k = -acc / err;
        if (k >= 1.0 || k <= -1.0) break;
        double tmp[kTnsMaxOrder + 1];
        std::memcpy(tmp, a, sizeof(tmp));
        for (int j = 1; j < m; j++) a[j] = tmp[j] + k * tmp[m - j];
        a[m] = k;
        refl[m - 1] = k;
        err *= 1.0 - k * k;
        order = m;
        if (err <= 0.0) break;
    }
    if (order == 0) return;
    double gain = r[0] / err;
    if (gain < kGainThreshold) return;

    // Quantize the reflection coefficients; trim the insignificant tail.
    // WIRE SIGN: the ISO map is +sin(idx/iqfac) (vo-aacenc's tnsCoeff4);
    // ffmpeg stores the map PRE-NEGATED and negates again inside its
    // reflection->LPC conversion, so the decoder's effective reflection
    // coefficient is +sin(idx/iqfac) — transmit k as-is, no negation.
    int idx[kTnsMaxOrder];
    for (int m = 0; m < order; m++) idx[m] = quant_coef(refl[m]);
    while (order > 0 && idx[order - 1] == 0) order--;
    if (order == 0) return;

    // Rebuild the LPC polynomial from the QUANTIZED reflection coefficients —
    // the decoder inverts exactly what we transmit, not the ideal filter.
    double aq[kTnsMaxOrder + 1] = {1.0};
    for (int m = 1; m <= order; m++) {
        double k = dequant_coef(idx[m - 1]);
        double tmp[kTnsMaxOrder + 1];
        std::memcpy(tmp, aq, sizeof(tmp));
        for (int j = 1; j < m; j++) aq[j] = tmp[j] + k * tmp[m - j];
        aq[m] = k;
    }

    // Forward FIR over the region: y[i] = x[i] + sum_j aq[j] * x[i-j],
    // history zero before the region start (matches the decoder's zero
    // initial filter state).
#ifdef GLINT_AAC_INT
    // Q24 filter coefficients, int64 accumulation, saturate to int32.
    int32_t aq_i[kTnsMaxOrder + 1];
    for (int m = 1; m <= order; m++) {
        aq_i[m] = static_cast<int32_t>(std::lround(aq[m] * 16777216.0));
    }
    int32_t hist[kTnsMaxOrder] = {0};
    for (int i = start; i < end; i++) {
        int32_t x = spec[i];
        int64_t y = static_cast<int64_t>(x) << 24;
        int hmax = i - start < order ? i - start : order;
        for (int j = 1; j <= hmax; j++) {
            y += static_cast<int64_t>(aq_i[j]) * hist[j - 1];
        }
        for (int j = order - 1; j > 0; j--) hist[j] = hist[j - 1];
        hist[0] = x;
        y >>= 24;
        if (y > INT32_MAX) y = INT32_MAX;
        if (y < INT32_MIN) y = INT32_MIN;
        spec[i] = static_cast<int32_t>(y);
    }
#else
    double hist[kTnsMaxOrder] = {0};
    for (int i = start; i < end; i++) {
        double x = spec[i];
        double y = x;
        int hmax = i - start < order ? i - start : order;
        for (int j = 1; j <= hmax; j++) y += aq[j] * hist[j - 1];
        for (int j = order - 1; j > 0; j--) hist[j] = hist[j - 1];
        hist[0] = x;
        spec[i] = y;
    }
#endif

    f->active = 1;
    // The decoder counts the filter region down from num_swb (the FULL sfb
    // count for the sample rate), NOT from max_sfb; it then clips the line
    // range to min(max_sfb, tns_max_bands). length = num_swb - start_band
    // makes its region exactly [start_band, top_band).
    f->length = static_cast<uint8_t>(kNumSwbLong[sr_index] - start_band);
    f->order = static_cast<uint8_t>(order);
    for (int m = 0; m < order; m++) f->coef_idx[m] = static_cast<int8_t>(idx[m]);
}

}  // namespace aac
}  // namespace glint

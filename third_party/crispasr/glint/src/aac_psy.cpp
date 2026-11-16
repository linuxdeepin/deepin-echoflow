// glint - AAC psychoacoustic band masks
// MIT License - Clean-room implementation

#include "aac_psy.hpp"
#include "aac_coder.hpp"
#include "intmath.hpp"
#include "aac_tables.hpp"
#include "aac_coder.hpp"
#include "intmath.hpp"

#include <cmath>

namespace glint {
namespace aac {

using namespace aac_tables;

namespace {

struct MaskModel {
#ifdef GLINT_SMALL_BUFFERS
    using T = float;   // table storage; arithmetic stays double
#else
    using T = double;
#endif
    int nb;
    T spread[kMaxSfb][kMaxSfb];  // energy-domain Schroeder spreading
    T ath_rel[kMaxSfb];          // 10^((ath_db - ath_min - 96)/10)
};

// Single-slot cache keyed by sample rate (embedded-friendly; matches the
// GLINT_SMALL_BUFFERS pattern used by the MP3 mask models).
const MaskModel* get_model(int sr_index) {
    static MaskModel model;
    static int slot_sr = -1;
    if (slot_sr == sr_index) return &model;

    const uint16_t* swb = kSwbOffsetLong[sr_index];
    const int nb = kNumSwbLong[sr_index];
    const double sr = kSampleRates[sr_index];

    double z[kMaxSfb];
    double ath_db[kMaxSfb];
    double ath_min = 1e30;
    for (int b = 0; b < nb; b++) {
        double fc = 0.5 * (swb[b] + swb[b + 1]) * (sr / 2.0) / 1024.0;
        if (fc < 20.0) fc = 20.0;
        z[b] = 13.0 * std::atan(0.00076 * fc) +
               3.5 * std::atan((fc / 7500.0) * (fc / 7500.0));
        double khz = fc / 1000.0;
        ath_db[b] = 3.64 * std::pow(khz, -0.8) -
                    6.5 * std::exp(-0.6 * (khz - 3.3) * (khz - 3.3)) +
                    1e-3 * khz * khz * khz * khz;
        if (ath_db[b] < ath_min) ath_min = ath_db[b];
    }
    for (int b = 0; b < nb; b++) {
        model.ath_rel[b] = static_cast<MaskModel::T>(
            std::pow(10.0, (ath_db[b] - ath_min - 96.0) / 10.0));
        for (int j = 0; j < nb; j++) {
            double dz = z[b] - z[j];
            double s_db = 15.81 + 7.5 * (dz + 0.474) -
                          17.5 * std::sqrt(1.0 + (dz + 0.474) * (dz + 0.474));
            model.spread[b][j] = static_cast<MaskModel::T>(std::pow(10.0, s_db / 10.0));
        }
    }
    model.nb = nb;
    slot_sr = sr_index;
    return &model;
}

}  // namespace

double aac_compute_masks(const SpecT* spec, int sr_index, int max_sfb,
                         double emax_ref, double* mask, bool tonal) {
    const MaskModel* m = get_model(sr_index);
    const uint16_t* swb = kSwbOffsetLong[sr_index];
    static const double kOffset = std::pow(10.0, -14.0 / 10.0);

    double e[kMaxSfb];
    double off[kMaxSfb];
    double emax = 0.0;
    const int nb = max_sfb < m->nb ? max_sfb : m->nb;
    for (int b = 0; b < nb; b++) {
#ifdef GLINT_AAC_INT
        // Integer per-coefficient work (the no-FPU hot path): energies in
        // int64 (>>8 keeps 96-line band sums of Q3^2 products in range),
        // geometric mean via an integer log2. Per-BAND math stays double —
        // ~50 float ops per band per frame is per-frame-scale work.
        // No pre-shift: Parseval bounds the whole frame's Q3^2 energy at
        // ~2^59, safely inside int64 — a >>8 zeroed every |v|<16 coefficient
        // and cost ~1 dB NMR at 256k (quiet-band energies vanished).
        int64_t iacc = 0;
        int64_t lg2q16 = 0;
        for (int i = swb[b]; i < swb[b + 1]; i++) {
            int64_t v = spec[i];
            iacc += v * v;
            if (tonal) {
                uint32_t a2 = static_cast<uint32_t>(v < 0 ? -v : v);
                // log2 of |S| (Q16); zero coefficients count as 2^-24
                int32_t l;
                if (a2 == 0) {
                    l = -(24 << 16);
                } else {
                    int msb = 31 - intmath::clz32(a2);
                    uint32_t u = a2 << (31 - msb);
                    // 2-term linear approx of log2 mantissa: frac ~ (u-2^31)/2^31
                    uint32_t frac = (u >> 15) & 0xFFFF;
                    l = (msb << 16) + static_cast<int32_t>(frac);
                }
                lg2q16 += 2 * static_cast<int64_t>(l);  // log2(v^2)
            }
        }
        double acc = static_cast<double>(iacc);
        e[b] = acc;
        if (acc > emax) emax = acc;
        if (tonal) {
            int n = swb[b + 1] - swb[b];
            double gm_log2 = static_cast<double>(lg2q16) / (65536.0 * n);
            double am_log2 = acc > 0 ? std::log2(acc / n) : -100.0;
            double sfm_db = 3.0102999566 * (gm_log2 - am_log2);
            double a = sfm_db / -20.0;
            if (a > 1.0) a = 1.0;
            if (a < 0.0) a = 0.0;
            off[b] = std::pow(10.0, -(6.0 + 12.0 * a) / 10.0);
        }
#else
        double acc = 0.0, lg = 0.0;
        for (int i = swb[b]; i < swb[b + 1]; i++) {
            double p = spec[i] * spec[i];
            acc += p;
            if (tonal) lg += std::log(p + 1e-30);
        }
        e[b] = acc;
        if (acc > emax) emax = acc;
        if (tonal) {
            // Spectral flatness -> tonality alpha in [0,1]; tonal maskers
            // mask less: per-masker offset -(6 + 12*alpha) dB.
            int n = swb[b + 1] - swb[b];
            double gm = std::exp(lg / n);
            double sfm_db = 10.0 * std::log10(gm / (acc / n + 1e-30) + 1e-30);
            double a = sfm_db / -20.0;
            if (a > 1.0) a = 1.0;
            if (a < 0.0) a = 0.0;
            off[b] = std::pow(10.0, -(6.0 + 12.0 * a) / 10.0);
        }
#endif
    }
    double ref = emax_ref > emax ? emax_ref : emax;
    for (int b = 0; b < nb; b++) {
        double acc = 0.0;
        if (tonal) {
            for (int j = 0; j < nb; j++) acc += e[j] * off[j] * m->spread[b][j];
        } else {
            for (int j = 0; j < nb; j++) acc += e[j] * m->spread[b][j];
            acc *= kOffset;
        }
        double floor = ref * m->ath_rel[b];
        mask[b] = acc > floor ? acc : floor;
    }
    return emax;
}

namespace {

struct ShortMaskModel {
#ifdef GLINT_SMALL_BUFFERS
    using T = float;
#else
    using T = double;
#endif
    int nb;
    T spread[16][16];
    T ath_rel[16];
};

const ShortMaskModel* get_short_model(int sr_index) {
    static ShortMaskModel model;
    static int slot_sr = -1;
    if (slot_sr == sr_index) return &model;

    const uint16_t* swb = kSwbOffsetShort[sr_index];
    const int nb = kNumSwbShort[sr_index];
    const double sr = kSampleRates[sr_index];

    double z[16];
    double ath_db[16];
    double ath_min = 1e30;
    for (int b = 0; b < nb; b++) {
        double fc = 0.5 * (swb[b] + swb[b + 1]) * (sr / 2.0) / 128.0;
        if (fc < 20.0) fc = 20.0;
        z[b] = 13.0 * std::atan(0.00076 * fc) +
               3.5 * std::atan((fc / 7500.0) * (fc / 7500.0));
        double khz = fc / 1000.0;
        ath_db[b] = 3.64 * std::pow(khz, -0.8) -
                    6.5 * std::exp(-0.6 * (khz - 3.3) * (khz - 3.3)) +
                    1e-3 * khz * khz * khz * khz;
        if (ath_db[b] < ath_min) ath_min = ath_db[b];
    }
    for (int b = 0; b < nb; b++) {
        model.ath_rel[b] = static_cast<ShortMaskModel::T>(
            std::pow(10.0, (ath_db[b] - ath_min - 96.0) / 10.0));
        for (int j = 0; j < nb; j++) {
            double dz = z[b] - z[j];
            double s_db = 15.81 + 7.5 * (dz + 0.474) -
                          17.5 * std::sqrt(1.0 + (dz + 0.474) * (dz + 0.474));
            model.spread[b][j] = static_cast<ShortMaskModel::T>(
                std::pow(10.0, s_db / 10.0));
        }
    }
    model.nb = nb;
    slot_sr = sr_index;
    return &model;
}

}  // namespace

double aac_compute_masks_short(const SpecT* spec, const AacBandLayout& L,
                               int sr_index, double emax_ref, double* mask) {
    const ShortMaskModel* m = get_short_model(sr_index);
    static const double kOffset = std::pow(10.0, -14.0 / 10.0);
    const int per = L.max_sfb;  // sfbs per group

    double emax = 0.0;
    double e[kMaxSfb];
    for (int b = 0; b < L.num_bands; b++) {
#ifdef GLINT_AAC_INT
        int64_t acc = 0;
        for (int i = L.offset[b]; i < L.offset[b + 1]; i++) {
            int64_t v = spec[i];
            acc += v * v;
        }
        e[b] = static_cast<double>(acc);
#else
        double acc = 0.0;
        for (int i = L.offset[b]; i < L.offset[b + 1]; i++) {
            double v = spec[i];
            acc += v * v;
        }
        e[b] = acc;
#endif
        // normalize by group length so the masks live in a per-window
        // energy domain (group widths differ)
        e[b] /= L.group_len[L.group_of_band[b]];
        if (e[b] > emax) emax = e[b];
    }
    double ref = emax_ref > emax ? emax_ref : emax;
    for (int g = 0, b0 = 0; g < L.num_groups; g++, b0 += per) {
        for (int b = 0; b < per && b < m->nb; b++) {
            double acc = 0.0;
            for (int j = 0; j < per && j < m->nb; j++) {
                acc += e[b0 + j] * m->spread[b][j];
            }
            double floor = ref * m->ath_rel[b];
            double v = acc * kOffset;
            // back to the band (group-width) domain for the allocator
            mask[b0 + b] = (v > floor ? v : floor) *
                           L.group_len[L.group_of_band[b0 + b]];
        }
    }
    return emax;
}

}  // namespace aac
}  // namespace glint

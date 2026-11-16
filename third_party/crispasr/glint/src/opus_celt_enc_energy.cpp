// CELT energy-envelope encoding — RFC 6716 § 4.3.2
// MIT License - Clean-room implementation
//
// See the header for the float32 precision contract: every arithmetic
// step below that feeds a coded decision is written in float, in the same
// association order as the float reference build.

#include "opus_celt_enc_energy.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include "opus_celt_energy.hpp"  // kMaxFineBits
#include "opus_celt_tables.hpp"
#include "opus_laplace.hpp"

namespace glint {
namespace opus {

namespace {

// log2 as the float reference computes it: double log, one final rounding
// to float. Inputs are >= sqrt(1e-27) (compute_band_energies' epsilon), so
// no denormal/zero handling is needed.
inline float celt_log2f(float x) {
    return static_cast<float>(1.442695040888963387 *
                              std::log(static_cast<double>(x)));
}

// Ordered scalar sum of squares (celt_inner_prod with x == y): the
// accumulation order is part of the float contract.
inline float sum_squares(const float* x, int n) {
    float xy = 0;
    for (int i = 0; i < n; i++) xy = xy + x[i] * x[i];
    return xy;
}

// Mean-squared log-energy change, saturated at 200; drives the
// delayed-intra state (an estimate of how much damage a lost inter frame
// would do).
float loss_distortion(const float* e_bands, const float* old_ebands,
                      int start, int end, int len, int channels) {
    float dist = 0;
    for (int c = 0; c < channels; c++) {
        for (int i = start; i < end; i++) {
            float d = e_bands[i + c * len] - old_ebands[i + c * len];
            dist = dist + d * d;
        }
    }
    return 200 < dist ? 200 : dist;
}

// One coarse pass (intra or inter). Writes symbols to enc, updates
// old_ebands/error/prev in place, and returns the "badness": the total
// amount the coded qi were clamped away from the ideal qi by budget
// exhaustion (the two-pass decision metric). lfe frames report 0.
int quant_coarse_energy_impl(int start, int end, const float* e_bands,
                             float* old_ebands, int32_t budget, int32_t tell,
                             const uint8_t* prob_model, float* error,
                             RangeEncoder& enc, int channels, int lm,
                             int intra, float max_decay, int lfe) {
    int badness = 0;
    float prev[2] = {0, 0};
    float coef, beta;

    if (tell + 3 <= budget) enc.enc_bit_logp(intra, 3);
    if (intra) {
        coef = 0;
        beta = static_cast<float>(celt::kBetaIntra);
    } else {
        beta = static_cast<float>(celt::kBetaCoef[lm]);
        coef = static_cast<float>(celt::kPredCoef[lm]);
    }

    for (int i = start; i < end; i++) {
        for (int c = 0; c < channels; c++) {
            const int idx = i + c * celt::kNbEBands;
            float x = e_bands[idx];
            float oldE = -9.0f > old_ebands[idx] ? -9.0f : old_ebands[idx];
            float f = x - coef * oldE - prev[c];
            // Round to nearest; the .5f offset in float32 is load-bearing.
            int qi = static_cast<int>(std::floor(.5f + f));
            float decay_bound =
                (-28.0f > old_ebands[idx] ? -28.0f : old_ebands[idx]) -
                max_decay;
            // Prevent the energy from decaying faster than max_decay
            // (e.g. one-bin bands would otherwise swing wildly).
            if (qi < 0 && x < decay_bound) {
                qi += static_cast<int>(decay_bound - x);
                if (qi > 0) qi = 0;
            }
            int qi0 = qi;
            // Running out of budget: assume something safe for the bands
            // still to come (they cost >= 3*C bits each in the worst
            // fallback), squeezing qi toward the cheap symbols.
            tell = static_cast<int32_t>(enc.tell());
            int bits_left = budget - tell - 3 * channels * (end - i);
            if (i != start && bits_left < 30) {
                if (bits_left < 24 && qi > 1) qi = 1;
                if (bits_left < 16 && qi < -1) qi = -1;
            }
            if (lfe && i >= 2 && qi > 0) qi = 0;
            if (budget - tell >= 15) {
                int pi = 2 * (i < 20 ? i : 20);
                // laplace_encode may clamp an extreme qi; the returned
                // value is what the decoder will reconstruct.
                qi = laplace_encode(enc, qi,
                                    static_cast<unsigned>(prob_model[pi])
                                        << 7,
                                    static_cast<int>(prob_model[pi + 1])
                                        << 6);
            } else if (budget - tell >= 2) {
                if (qi > 1) qi = 1;
                if (qi < -1) qi = -1;
                enc.enc_icdf((2 * qi) ^ -(qi < 0 ? 1 : 0),
                             celt::kSmallEnergyIcdf, 2);
            } else if (budget - tell >= 1) {
                if (qi > 0) qi = 0;
                enc.enc_bit_logp(-qi, 1);
            } else {
                qi = -1;
            }
            error[idx] = f - static_cast<float>(qi);
            badness += qi >= qi0 ? qi - qi0 : qi0 - qi;
            float q = static_cast<float>(qi);

            float tmp = coef * oldE + prev[c] + q;
            old_ebands[idx] = tmp;
            prev[c] = prev[c] + q - beta * q;
        }
    }
    return lfe ? 0 : badness;
}

}  // namespace

void compute_band_energies(const float* x, float* band_e, int end,
                           int channels, int lm) {
    const int n = celt::kShortMdctSize << lm;
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < end; i++) {
            float sum =
                1e-27f + sum_squares(&x[c * n + (celt::kEBands[i] << lm)],
                                     (celt::kEBands[i + 1] -
                                      celt::kEBands[i])
                                         << lm);
            // Double sqrt with one rounding to float, as the reference's
            // celt_sqrt does in the float build.
            band_e[i + c * celt::kNbEBands] = static_cast<float>(
                std::sqrt(static_cast<double>(sum)));
        }
    }
}

void amp2Log2(int eff_end, int end, const float* band_e, float* band_log_e,
              int channels) {
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < eff_end; i++) {
            band_log_e[i + c * celt::kNbEBands] =
                celt_log2f(band_e[i + c * celt::kNbEBands]) -
                static_cast<float>(celt::kEMeans[i]);
        }
        for (int i = eff_end; i < end; i++)
            band_log_e[c * celt::kNbEBands + i] = -14.0f;
    }
}

void normalise_bands(const float* freq, float* x_norm, const float* band_e,
                     int end, int channels, int m) {
    const int n = m * celt::kShortMdctSize;
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < end; i++) {
            float g = 1.f / (1e-27f + band_e[i + c * celt::kNbEBands]);
            for (int j = m * celt::kEBands[i]; j < m * celt::kEBands[i + 1];
                 j++)
                x_norm[j + c * n] = freq[j + c * n] * g;
        }
    }
}

void quant_coarse_energy(int start, int end, int eff_end,
                         const float* e_bands, float* old_ebands,
                         uint32_t budget, float* error, RangeEncoder& enc,
                         int channels, int lm, int nb_available_bytes,
                         int force_intra, float* delayed_intra, int two_pass,
                         int loss_rate, int lfe) {
    const int nb = celt::kNbEBands;
    // Force intra when packet loss is expected and the delayed-intra
    // account says the accumulated inter-frame drift risk is high enough
    // to be worth the extra bits (only when a one-pass encode was asked;
    // two_pass evaluates both streams anyway).
    int intra =
        force_intra ||
        (!two_pass &&
         *delayed_intra > 2 * channels * (end - start) &&
         nb_available_bytes > (end - start) * channels);
    int32_t intra_bias = static_cast<int32_t>(
        (budget * *delayed_intra * loss_rate) / (channels * 512));
    float new_distortion =
        loss_distortion(e_bands, old_ebands, start, eff_end, nb, channels);

    uint32_t tell = enc.tell();
    if (tell + 3 > budget) two_pass = intra = 0;

    float max_decay = 16.f;
    if (end - start > 10) {
        float cap = .125f * nb_available_bytes;
        max_decay = max_decay < cap ? max_decay : cap;
    }
    if (lfe) max_decay = 3.f;

    RangeEncoder enc_start_state = enc;

    float old_ebands_intra[2 * celt::kNbEBands];
    float error_intra[2 * celt::kNbEBands];
    std::memcpy(old_ebands_intra, old_ebands,
                channels * nb * sizeof(float));

    int badness1 = 0;
    if (two_pass || intra) {
        badness1 = quant_coarse_energy_impl(
            start, end, e_bands, old_ebands_intra,
            static_cast<int32_t>(budget), static_cast<int32_t>(tell),
            celt::kEProbModel[lm][1], error_intra, enc, channels, lm, 1,
            max_decay, lfe);
    }

    if (!intra) {
        int32_t tell_intra = static_cast<int32_t>(enc.tell_frac());
        RangeEncoder enc_intra_state = enc;

        uint32_t nstart_bytes = enc_start_state.range_bytes();
        uint32_t nintra_bytes = enc_intra_state.range_bytes();
        uint32_t save_bytes = nintra_bytes - nstart_bytes;
        uint8_t* intra_buf = enc_intra_state.buffer() + nstart_bytes;
        std::vector<uint8_t> intra_bits(intra_buf, intra_buf + save_bytes);

        enc = enc_start_state;
        int badness2 = quant_coarse_energy_impl(
            start, end, e_bands, old_ebands, static_cast<int32_t>(budget),
            static_cast<int32_t>(tell), celt::kEProbModel[lm][0], error,
            enc, channels, lm, 0, max_decay, lfe);

        // Keep intra when it clamped less, or on a badness tie when its
        // (bias-adjusted) size is no worse.
        if (two_pass &&
            (badness1 < badness2 ||
             (badness1 == badness2 &&
              static_cast<int32_t>(enc.tell_frac()) + intra_bias >
                  tell_intra))) {
            enc = enc_intra_state;
            if (save_bytes)
                std::memcpy(intra_buf, intra_bits.data(), save_bytes);
            std::memcpy(old_ebands, old_ebands_intra,
                        channels * nb * sizeof(float));
            std::memcpy(error, error_intra,
                        channels * nb * sizeof(float));
            intra = 1;
        }
    } else {
        std::memcpy(old_ebands, old_ebands_intra,
                    channels * nb * sizeof(float));
        std::memcpy(error, error_intra, channels * nb * sizeof(float));
    }

    if (intra) {
        *delayed_intra = new_distortion;
    } else {
        float pc = static_cast<float>(celt::kPredCoef[lm]);
        *delayed_intra = (pc * pc) * *delayed_intra + new_distortion;
    }
}

void quant_fine_energy(int start, int end, float* old_ebands, float* error,
                       const int* fine_quant, RangeEncoder& enc,
                       int channels) {
    for (int i = start; i < end; i++) {
        int16_t frac = static_cast<int16_t>(1 << fine_quant[i]);
        if (fine_quant[i] <= 0) continue;
        for (int c = 0; c < channels; c++) {
            const int idx = i + c * celt::kNbEBands;
            int q2 =
                static_cast<int>(std::floor((error[idx] + .5f) * frac));
            if (q2 > frac - 1) q2 = frac - 1;
            if (q2 < 0) q2 = 0;
            enc.enc_bits(static_cast<uint32_t>(q2),
                         static_cast<unsigned>(fine_quant[i]));
            float offset =
                (q2 + .5f) * (1 << (14 - fine_quant[i])) * (1.f / 16384) -
                .5f;
            old_ebands[idx] += offset;
            error[idx] -= offset;
        }
    }
}

void quant_energy_finalise(int start, int end, float* old_ebands,
                           float* error, const int* fine_quant,
                           const int* fine_priority, int bits_left,
                           RangeEncoder& enc, int channels) {
    for (int prio = 0; prio < 2; prio++) {
        for (int i = start; i < end && bits_left >= channels; i++) {
            if (fine_quant[i] >= kMaxFineBits || fine_priority[i] != prio)
                continue;
            for (int c = 0; c < channels; c++) {
                const int idx = i + c * celt::kNbEBands;
                int q2 = error[idx] < 0 ? 0 : 1;
                enc.enc_bits(static_cast<uint32_t>(q2), 1);
                float offset = (q2 - .5f) *
                               (1 << (14 - fine_quant[i] - 1)) *
                               (1.f / 16384);
                old_ebands[idx] += offset;
                error[idx] -= offset;
                bits_left--;
            }
        }
    }
}

}  // namespace opus
}  // namespace glint

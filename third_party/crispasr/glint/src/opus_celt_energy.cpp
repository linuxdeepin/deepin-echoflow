// CELT energy-envelope decoding — RFC 6716 section 4.3.2
// MIT License - Clean-room implementation

#include "opus_celt_energy.hpp"

#include <cmath>

#include "opus_celt_tables.hpp"
#include "opus_laplace.hpp"

namespace glint {
namespace opus {

void unquant_coarse_energy(int start, int end, double* old_ebands,
                           bool intra, RangeDecoder& dec, int channels,
                           int lm) {
    const uint8_t* prob = celt::kEProbModel[lm][intra ? 1 : 0];
    const double coef = intra ? 0.0 : celt::kPredCoef[lm];
    const double beta = intra ? celt::kBetaIntra : celt::kBetaCoef[lm];
    double prev[2] = { 0.0, 0.0 };
    const int32_t budget =
        static_cast<int32_t>(dec.storage_bytes()) * 8;

    for (int i = start; i < end; i++) {
        for (int c = 0; c < channels; c++) {
            int qi;
            int32_t tell = static_cast<int32_t>(dec.tell());
            if (budget - tell >= 15) {
                int pi = 2 * (i < 20 ? i : 20);
                qi = laplace_decode(dec,
                                    static_cast<unsigned>(prob[pi]) << 7,
                                    static_cast<int>(prob[pi + 1]) << 6);
            } else if (budget - tell >= 2) {
                // Zigzag over {0, -1, +1}.
                qi = dec.dec_icdf(celt::kSmallEnergyIcdf, 2);
                qi = (qi >> 1) ^ -(qi & 1);
            } else if (budget - tell >= 1) {
                qi = -dec.dec_bit_logp(1);
            } else {
                qi = -1;
            }
            double& e = old_ebands[c * celt::kNbEBands + i];
            if (e < -9.0) e = -9.0;  // limit prediction from silent bands
            e = coef * e + prev[c] + qi;
            prev[c] += qi - beta * qi;
        }
    }
}

void unquant_fine_energy(int start, int end, double* old_ebands,
                         const int* fine_quant, RangeDecoder& dec,
                         int channels) {
    for (int i = start; i < end; i++) {
        if (fine_quant[i] <= 0) continue;
        for (int c = 0; c < channels; c++) {
            int q2 = static_cast<int>(
                dec.dec_bits(static_cast<unsigned>(fine_quant[i])));
            double offset =
                (q2 + 0.5) * std::ldexp(1.0, -fine_quant[i]) - 0.5;
            old_ebands[c * celt::kNbEBands + i] += offset;
        }
    }
}

void unquant_energy_finalise(int start, int end, double* old_ebands,
                             const int* fine_quant,
                             const int* fine_priority, int bits_left,
                             RangeDecoder& dec, int channels) {
    for (int prio = 0; prio < 2; prio++) {
        for (int i = start; i < end && bits_left >= channels; i++) {
            if (fine_quant[i] >= kMaxFineBits || fine_priority[i] != prio)
                continue;
            for (int c = 0; c < channels; c++) {
                int q2 = static_cast<int>(dec.dec_bits(1));
                double offset =
                    (q2 - 0.5) * std::ldexp(1.0, -(fine_quant[i] + 1));
                old_ebands[c * celt::kNbEBands + i] += offset;
                bits_left--;
            }
        }
    }
}

}  // namespace opus
}  // namespace glint

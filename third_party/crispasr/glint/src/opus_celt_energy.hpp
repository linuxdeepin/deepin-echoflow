// CELT energy-envelope decoding — RFC 6716 sections 4.3.2 (coarse/fine/final)
// MIT License - Clean-room implementation
//
// CELT transmits per-band log2 energies in three layers:
//  - coarse (6 dB step): Laplace-coded prediction residuals with 2-D
//    prediction (coef * previous frame's band energy + a time-integrated
//    "prev" term per channel); intra frames drop the inter-frame term.
//    Near the end of the budget the model degrades gracefully: <15 bits →
//    a tiny zigzag icdf, <2 bits → a single sign bit, <1 bit → forced -1.
//  - fine: fine_quant[i] raw bits per band/channel, uniform refinement.
//  - final: leftover whole bits spent 1 bit/band/channel by priority class.
//
// Energies are stored as old_ebands[channel * kNbEBands + band] in log2
// units (base-2 dB/6.02); this array doubles as the inter-frame predictor
// state, exactly like the reference decoder.

#pragma once

#include "opus_ec.hpp"

namespace glint {
namespace opus {

constexpr int kMaxFineBits = 8;

void unquant_coarse_energy(int start, int end, double* old_ebands,
                           bool intra, RangeDecoder& dec, int channels,
                           int lm);

void unquant_fine_energy(int start, int end, double* old_ebands,
                         const int* fine_quant, RangeDecoder& dec,
                         int channels);

void unquant_energy_finalise(int start, int end, double* old_ebands,
                             const int* fine_quant,
                             const int* fine_priority, int bits_left,
                             RangeDecoder& dec, int channels);

}  // namespace opus
}  // namespace glint

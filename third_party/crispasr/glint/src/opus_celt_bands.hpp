// CELT band (PVQ shape) decoding — RFC 6716 section 4.3.4
// MIT License - Clean-room implementation
//
// Decode side of CELT's normalized-band machinery. Everything that touches
// the range decoder here is INTEGER (theta resolution qn, the three theta
// PDFs, bit splits via the bitexact cos/log2tan helpers, the PVQ pulse
// budget walk-down) — signal values never influence the wire, which is what
// makes the fuzz-oracle cross-check complete for this layer too.
//
// Band recursion (quant_band/quant_partition):
//  - splits halve the band and code the mid/side angle theta with one of
//    three PDFs (triangular for mono-long, uniform for transient/stereo,
//    step for stereo N>2); bit split derives from theta via the bit-exact
//    integer log2tan, with rebalancing between halves.
//  - transient frames interleave short blocks; tf_change recombines or
//    further splits time-frequency resolution via Haar butterflies, with
//    Hadamard-order deinterleaving of the folding source.
//  - bands with no pulses are filled by FOLDING a lower band (scaled copy +
//    LCG sign noise) or plain LCG noise, tracked by per-block collapse
//    masks; anti_collapse() later refills blocks that collapsed to zero in
//    transient frames.
//
// All arrays are the decoder's normalized spectra (unit-norm bands); the
// energy envelope is applied afterwards by denormalisation (top level).

#pragma once

#include <cstdint>

#include "opus_ec.hpp"

namespace glint {
namespace opus {

// Reference-compatible LCG; the seed is part of the decoder state and its
// evolution is bit-exact (it feeds noise fill and anti-collapse).
inline uint32_t celt_lcg_rand(uint32_t seed) {
    return 1664525u * seed + 1013904223u;
}

// Bit-exact integer trig used for the mid/side bit-split (these determine
// the allocation and therefore the wire; never replace with float math).
int16_t bitexact_cos(int16_t x);
int bitexact_log2tan(int isin, int icos);

// Haar butterfly with 1/sqrt(2) scaling on pairs at the given stride.
void haar1(double* x, int n0, int stride);

// Decode all bands' shapes into X_ (and Y_ for stereo). Mirrors the
// reference quant_all_bands with encode=0/resynth=1. total_bits is in
// 1/8-bit units (frame bits minus the anti-collapse reservation); balance
// comes from the allocator. seed is the decoder's rolling LCG state.
void quant_all_bands_dec(int start, int end, double* X_, double* Y_,
                         uint8_t* collapse_masks, const int* pulses,
                         int short_blocks, int spread, int dual_stereo,
                         int intensity, const int* tf_res,
                         int32_t total_bits, int32_t balance,
                         RangeDecoder& dec, int lm, int coded_bands,
                         uint32_t* seed, int disable_inv);

// Refill collapsed blocks of transient frames with low-level noise, using
// the current + two previous frames' band energies (log2 domain).
void anti_collapse(double* X_, const uint8_t* collapse_masks, int lm,
                   int channels, int size, int start, int end,
                   const double* log_e, const double* prev1_log_e,
                   const double* prev2_log_e, const int* pulses,
                   uint32_t seed);

}  // namespace opus
}  // namespace glint

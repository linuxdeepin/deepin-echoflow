// CELT implicit bit allocation — RFC 6716 section 4.3.3
// MIT License - Clean-room implementation
//
// This is the piece that makes CELT's design radically different from
// MP3/AAC: there are no per-band side-info fields. The decoder re-derives
// the entire per-band bit budget from (frame size, coded flags) by running
// the SAME integer computation as the encoder — every quantity here is in
// 1/8-bit units (kBitRes = 3) and every operation must match the reference
// bit-for-bit, or the PVQ decode desyncs.
//
// Structure: a quality row is picked from the static allocation table by
// bisection, interpolated 1/64ths toward the next row (interp_bits2pulses),
// trimmed by alloc_trim, boosted by dynalloc offsets, capped per band, with
// reserved bits for the skip flag / intensity position / dual-stereo flag.
// Skipped top bands are signaled with one bit each (decoded here); their
// bits are redistributed. Finally each band's budget is split into fine
// energy bits (ebits) and PVQ bits (pulses), with a rolling `balance`
// carrying rounding remainders to later bands.

#pragma once

#include <cstdint>

#include "opus_ec.hpp"

namespace glint {
namespace opus {

constexpr int kBitRes = 3;         // allocation works in 1/8-bit units
constexpr int kMaxPulseIndex = 40; // pseudo-pulse indices (get_pulses domain)
constexpr int kLogMaxPseudo = 6;
constexpr int kMaxPulses = 128;

// Pseudo-pulse index -> actual pulse count (indices >= 8 grow geometrically).
inline int get_pulses(int i) {
    return i < 8 ? i : (8 + (i & 7)) << ((i >> 3) - 1);
}

// Cache-based conversions between band bit budgets and pulse counts
// (band cache selected by (lm+1, band); bits in 1/8-bit units).
int bits2pulses(int band, int lm, int bits);
int pulses2bits(int band, int lm, int pulses);

// Per-band hard caps in 1/8 bits (from the generated caps table).
void init_caps(int* cap, int lm, int channels);

// Decoder-side allocation. Returns codedBands; fills pulses[], ebits[],
// fine_priority[] for all bands in [start, end), *intensity, *dual_stereo,
// and *balance (leftover 1/8 bits for quant_all_bands' rebalancing).
// offsets[] are the dynalloc boosts (1/8 bits), total is the remaining
// frame budget in 1/8 bits. Reads skip / intensity / dual-stereo symbols
// from dec exactly where the reference does.
int compute_allocation_dec(int start, int end, const int* offsets,
                           const int* cap, int alloc_trim, int* intensity,
                           int* dual_stereo, int32_t total,
                           int32_t* balance, int* pulses, int* ebits,
                           int* fine_priority, int channels, int lm,
                           RangeDecoder& dec);

// Encoder-side twin: identical integer logic, but DECIDES band skips (the
// depth-threshold policy driven by prev = last frame's codedBands and
// signal_bandwidth) and WRITES the skip/intensity/dual-stereo symbols.
// *intensity is clamped to codedBands and updated in place.
int compute_allocation_enc(int start, int end, const int* offsets,
                           const int* cap, int alloc_trim, int* intensity,
                           int* dual_stereo, int32_t total,
                           int32_t* balance, int* pulses, int* ebits,
                           int* fine_priority, int channels, int lm,
                           RangeEncoder& enc, int prev,
                           int signal_bandwidth);

}  // namespace opus
}  // namespace glint

// CELT energy-envelope ENCODING + float band analysis — RFC 6716 § 4.3.2
// MIT License - Clean-room implementation
//
// Encoder side of the energy pipeline (the decode side lives in
// opus_celt_energy.{hpp,cpp}):
//  - band analysis: per-band amplitude (sqrt energy with a 1e-27 epsilon),
//    log2 conversion minus the per-band means, and unit-energy
//    normalisation of the MDCT spectrum;
//  - coarse quantisation: 6 dB steps, 2-D prediction, Laplace coding, with
//    a full two-pass intra/inter decision (badness = total |qi| clamping
//    the budget forced, ties broken by a loss_rate-driven intra bias on
//    the 1/8-bit tell), max_decay clamping of downward steps, and the same
//    graceful <15/<2/<1-bit budget fallbacks the decoder applies;
//  - fine quantisation and finalisation: raw-bit refinements with error
//    feedback.
//
// PRECISION CONTRACT: everything in the encoder's decision chain is
// float32 — qi rounding (floor(.5f + f)), the error feedback, prev[] and
// oldEBands prediction state, loss_distortion, delayedIntra. These floats
// decide coded symbols, so a double-rounded tie flip would change the
// wire; the expressions mirror the float reference build term for term.
// (The decode side uses double because decoding takes no decisions.)
// The cross-check gate (tools/crosscheck_opus_enc_energy.py) compiles this
// file and the reference sources with -ffp-contract=off so the comparison
// is IEEE-exact; the library build may fuse — any fusion still yields a
// valid encoder, just not a byte-identical one to the pinned oracle.

#pragma once

#include <cstdint>

#include "opus_ec.hpp"

namespace glint {
namespace opus {

// Per-band amplitude: band_e[i + c*kNbEBands] = sqrt(1e-27 + sum x^2) over
// band i of channel c. x holds C channels of N = kShortMdctSize<<lm MDCT
// coefficients, channel-major.
void compute_band_energies(const float* x, float* band_e, int end,
                           int channels, int lm);

// band_log_e = log2(band_e) - eMeans per band; bands in [eff_end, end)
// are forced to -14 (silence floor).
void amp2Log2(int eff_end, int end, const float* band_e, float* band_log_e,
              int channels);

// x_norm = freq scaled so each band has unit energy (1/(1e-27 + band_e)).
// m = 1<<lm (number of short MDCTs per frame).
void normalise_bands(const float* freq, float* x_norm, const float* band_e,
                     int end, int channels, int m);

// Coarse energy encoder. e_bands is the target band_log_e; old_ebands is
// the inter-frame predictor state (updated in place to the quantised
// energies); error receives the residual for the fine stages; budget is
// the whole-frame bit budget (== buffer bytes * 8 — the decoder derives
// its budget from the buffer size, so they must agree); delayed_intra is
// persistent encoder state. two_pass encodes both intra and inter and
// keeps the better (badness, then biased tell) stream.
void quant_coarse_energy(int start, int end, int eff_end,
                         const float* e_bands, float* old_ebands,
                         uint32_t budget, float* error, RangeEncoder& enc,
                         int channels, int lm, int nb_available_bytes,
                         int force_intra, float* delayed_intra, int two_pass,
                         int loss_rate, int lfe);

// Fine energy: fine_quant[i] raw bits per band/channel; updates old_ebands
// and error in place.
void quant_fine_energy(int start, int end, float* old_ebands, float* error,
                       const int* fine_quant, RangeEncoder& enc,
                       int channels);

// Final single-bit refinements by priority class, spending bits_left.
void quant_energy_finalise(int start, int end, float* old_ebands,
                           float* error, const int* fine_quant,
                           const int* fine_priority, int bits_left,
                           RangeEncoder& enc, int channels);

}  // namespace opus
}  // namespace glint

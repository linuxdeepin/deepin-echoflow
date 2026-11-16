// SILK NLSF decoding and NLSF-to-LPC conversion — RFC 6716 section 4.2.7.5
// MIT License - Clean-room implementation
//
// Turns the two-stage NLSF codebook indices of one SILK frame into the Q12
// short-term whitening filter: residual dequant (backwards prediction, with
// the ±0.1 dead-zone level adjust), weighted stage-1 vector combination,
// spacing stabilization, then the cosine-domain P/Q polynomial construction
// with int16 fitting and stability-driven bandwidth expansion. Every step is
// exact int16/int32 fixed point — byte-identical with the reference decoder
// (gate: tools/crosscheck_opus_silk_nlsf.py).

#pragma once

#include <cstdint>

#include "opus_silk_tables.hpp"

namespace glint {
namespace opus {
namespace silk {

// Codebook indices -> stabilized NLSF vector in Q15 (ascending, spaced by
// the codebook's deltaMin). indices[0] = stage-1 vector, indices[1..order] =
// stage-2 residuals (range ±10 with the extension escapes decoded).
void nlsf_decode(int16_t* nlsf_q15, const int8_t* indices,
                 const NlsfCodebook& cb);

// NLSF vector (Q15) -> monic LPC coefficients in Q12 (order 10 or 16).
// Guarantees int16 range and filter stability via bandwidth expansion.
void nlsf2a(int16_t* a_q12, const int16_t* nlsf_q15, int order);

// Scale an AR filter (without the leading 1) by chirp, chirp^2, ... in Q16.
void bwexpander_32(int32_t* ar, int order, int32_t chirp_q16);

// Inverse prediction gain in the energy domain, Q30; 0 = unstable filter
// (any pole on/outside the unit circle, or gain beyond the decoder limit).
int32_t lpc_inverse_pred_gain(const int16_t* a_q12, int order);

}  // namespace silk
}  // namespace opus
}  // namespace glint

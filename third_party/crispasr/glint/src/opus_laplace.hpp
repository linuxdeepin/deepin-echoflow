// Opus/CELT Laplace-distributed symbol coder — RFC 6716 section 4.3.2.1
// MIT License - Clean-room implementation
//
// CELT codes coarse energy deltas with a two-sided geometric ("Laplace")
// model over a 15-bit total frequency: P(0) = fs0/32768, and each further
// magnitude step multiplies the probability by decay/16384. The far tail is
// flattened to a minimum probability of 1/32768 per value so every delta in
// a wide range stays codable; encoding may therefore CLAMP an extreme value
// (the encoder must use the returned, possibly-clamped value).

#pragma once

#include "opus_ec.hpp"

namespace glint {
namespace opus {

// Encode `value`; fs0 = P(value==0) in 1/32768 units, decay in 1/16384
// units (at most ~11456 in CELT's models). Returns the value actually
// coded (clamped if it fell off the representable tail).
int laplace_encode(RangeEncoder& enc, int value, unsigned fs0, int decay);

int laplace_decode(RangeDecoder& dec, unsigned fs0, int decay);

}  // namespace opus
}  // namespace glint

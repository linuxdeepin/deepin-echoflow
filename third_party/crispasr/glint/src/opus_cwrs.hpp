// Opus/CELT PVQ codeword enumeration (CWRS) — RFC 6716 section 4.3.4.4
// MIT License - Clean-room implementation
//
// A pulse vector y of dimension n with sum(|y|) == k is mapped bijectively
// to an index in [0, V(n,k)) and coded as a uniform integer (ec uint). The
// enumeration is governed by
//   V(n,k) = number of n-dim vectors with k unit pulses (signs on nonzero),
//   U(n,k) = number of those where the first n-1 dims carry at most k-1
//            pulses; symmetric (U(n,k) = U(k,n)), with
//   V(n,k) = U(n,k) + U(n,k+1)
// and both obey  f(n,k) = f(n-1,k) + f(n,k-1) + f(n-1,k-1).
//
// This implementation computes U rows on the fly with O(k) workspace
// (no static enumeration table — the values are exact integer math, and
// tools/crosscheck_opus_celt_prims.py proves index-for-index equivalence
// with libopus's table-driven version).
//
// Preconditions everywhere: n >= 2, 1 <= k <= kCwrsMaxK, and V(n,k) must
// fit in 32 bits — CELT's bit allocation caps guarantee this for every
// (band size, pulse count) it ever produces.

#pragma once

#include <cstdint>

#include "opus_ec.hpp"

namespace glint {
namespace opus {

// Largest pulse count CELT's allocator can produce for one band.
constexpr int kCwrsMaxK = 128;

// Encode pulse vector y[0..n) with sum(|y|) == k.
void encode_pulses(const int* y, int n, int k, RangeEncoder& enc);

// Decode into y[0..n); returns sum of y[i]^2 (the caller normalizes by it).
int32_t decode_pulses(int* y, int n, int k, RangeDecoder& dec);

}  // namespace opus
}  // namespace glint

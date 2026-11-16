// SILK fixed-point arithmetic kit — RFC 6716 section 4.2 (decode side)
// MIT License - Clean-room implementation
//
// SILK is specified in exact int16/int32 fixed point; every operation here
// must match the reference macro semantics BIT-FOR-BIT (verified by
// tools/crosscheck_opus_silk_math.py against the reference inline kit).
// Additive wraparound is intentional in several spots (the reference relies
// on two's-complement wrap), so sums are computed through uint32.

#pragma once

#include <cstdint>

namespace glint {
namespace opus {
namespace silk {

// ---- shifts / saturation --------------------------------------------------

inline int32_t sat16(int32_t a) {
    return a > 32767 ? 32767 : (a < -32768 ? -32768 : a);
}

// int16-saturated sum (inputs are int16-valued; the sum is exact in int32).
inline int16_t add_sat16(int32_t a, int32_t b) {
    return static_cast<int16_t>(sat16(a + b));
}

inline int32_t add_sat32(int32_t a, int32_t b) {
    int64_t s = static_cast<int64_t>(a) + b;
    if (s > INT32_MAX) return INT32_MAX;
    if (s < INT32_MIN) return INT32_MIN;
    return static_cast<int32_t>(s);
}

inline int32_t sub_sat32(int32_t a, int32_t b) {
    int64_t s = static_cast<int64_t>(a) - b;
    if (s > INT32_MAX) return INT32_MAX;
    if (s < INT32_MIN) return INT32_MIN;
    return static_cast<int32_t>(s);
}

// Clamp-then-shift (NOT saturate-to-MAX: the result's low `shift` bits are
// zero, e.g. lshift_sat32(BIG, 30) == 1<<30, matching the reference).
inline int32_t lshift_sat32(int32_t a, int shift) {
    int32_t lim_hi = INT32_MAX >> shift;
    int32_t lim_lo = INT32_MIN >> shift;
    int32_t clamped = a > lim_hi ? lim_hi : (a < lim_lo ? lim_lo : a);
    return static_cast<int32_t>(static_cast<uint32_t>(clamped) << shift);
}

// Right shift with round-to-nearest (ties away from zero for positives —
// exactly the reference's ((a >> (n-1)) + 1) >> 1 formulation).
inline int32_t rshift_round(int32_t a, int shift) {
    return shift == 1 ? (a >> 1) + (a & 1)
                      : ((a >> (shift - 1)) + 1) >> 1;
}

inline int64_t rshift_round64(int64_t a, int shift) {
    return shift == 1 ? (a >> 1) + (a & 1)
                      : ((a >> (shift - 1)) + 1) >> 1;
}

// Wrapping add/shift (two's-complement semantics via uint32).
inline int32_t add32_ovflw(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) +
                                static_cast<uint32_t>(b));
}

inline int32_t mla_ovflw(int32_t a, int32_t b, int32_t c) {
    return static_cast<int32_t>(
        static_cast<uint32_t>(a) +
        static_cast<uint32_t>(b) * static_cast<uint32_t>(c));
}

// ---- multiplies (Q-domain) ------------------------------------------------

// (a32 * bottom 16 bits of b32, signed) >> 16
inline int32_t smulwb(int32_t a, int32_t b) {
    return static_cast<int32_t>(
        (static_cast<int64_t>(a) * static_cast<int16_t>(b)) >> 16);
}
inline int32_t smlawb(int32_t a, int32_t b, int32_t c) {
    return add32_ovflw(a, smulwb(b, c));
}

// (a32 * top 16 bits of b32) >> 16
inline int32_t smulwt(int32_t a, int32_t b) {
    return static_cast<int32_t>(
        (static_cast<int64_t>(a) * (b >> 16)) >> 16);
}

// full 32x32 -> top 48, truncated: (a * b) >> 16
inline int32_t smulww(int32_t a, int32_t b) {
    return static_cast<int32_t>((static_cast<int64_t>(a) * b) >> 16);
}
inline int32_t smlaww(int32_t a, int32_t b, int32_t c) {
    return add32_ovflw(a, smulww(b, c));
}

// 16x16 products.
inline int32_t smulbb(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<int16_t>(a)) *
           static_cast<int16_t>(b);
}
inline int32_t smlabb(int32_t a, int32_t b, int32_t c) {
    return add32_ovflw(a, smulbb(b, c));
}
inline int32_t smulbt(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<int16_t>(a)) * (b >> 16);
}
inline int32_t smultt(int32_t a, int32_t b) { return (a >> 16) * (b >> 16); }

// Top word of the 64-bit product: (a * b) >> 32.
inline int32_t smmul(int32_t a, int32_t b) {
    return static_cast<int32_t>((static_cast<int64_t>(a) * b) >> 32);
}

// (a32 * b32) >> 15, rounded? — reference SMULWW-family rounding variant.
inline int32_t smulww_round(int32_t a, int32_t b) {
    return static_cast<int32_t>(
        rshift_round64(static_cast<int64_t>(a) * b, 16));
}

// ---- bit scans ------------------------------------------------------------

inline int clz32(uint32_t v) {
    if (!v) return 32;
    int n = 0;
    if (!(v & 0xFFFF0000u)) { n += 16; v <<= 16; }
    if (!(v & 0xFF000000u)) { n += 8; v <<= 8; }
    if (!(v & 0xF0000000u)) { n += 4; v <<= 4; }
    if (!(v & 0xC0000000u)) { n += 2; v <<= 2; }
    if (!(v & 0x80000000u)) { n += 1; }
    return n;
}

// CLZ on |a| with 3 fractional bits from the next bits (reference
// silk_CLZ_FRAC): lz = integer leading zeros, frac_q7 = Q7 fraction.
inline void clz_frac(int32_t in, int32_t* lz, int32_t* frac_q7) {
    int32_t lzeros = clz32(static_cast<uint32_t>(in));
    *lz = lzeros;
    *frac_q7 = static_cast<int32_t>(
        (static_cast<uint32_t>(in) << (lzeros & 31)) >> 24) & 0x7f;
}

// ---- division / inverse at variable Q (reference Inlines.h) ---------------

// "(a32 << qres) / b32" approximation: normalized 16-bit reciprocal plus
// one Newton correction step, then rescaled.
inline int32_t div32_varq(int32_t a32, int32_t b32, int qres) {
    int a_headrm = clz32(static_cast<uint32_t>(a32 < 0 ? -a32 : a32)) - 1;
    int32_t a32_nrm = a32 << a_headrm;
    int b_headrm = clz32(static_cast<uint32_t>(b32 < 0 ? -b32 : b32)) - 1;
    int32_t b32_nrm = b32 << b_headrm;
    int32_t b32_inv = (INT32_MAX >> 2) / (b32_nrm >> 16);
    int32_t result = smulwb(a32_nrm, b32_inv);
    // Residual (wrapping arithmetic on purpose), then correct.
    a32_nrm = static_cast<int32_t>(
        static_cast<uint32_t>(a32_nrm) -
        (static_cast<uint32_t>(smmul(b32_nrm, result)) << 3));
    result = smlawb(result, a32_nrm, b32_inv);
    int lshift = 29 + a_headrm - b_headrm - qres;
    if (lshift < 0) return lshift_sat32(result, -lshift);
    if (lshift < 32) return result >> lshift;
    return 0;
}

// "(1 << qres) / b32" approximation, same construction.
inline int32_t inverse32_varq(int32_t b32, int qres) {
    int b_headrm = clz32(static_cast<uint32_t>(b32 < 0 ? -b32 : b32)) - 1;
    int32_t b32_nrm = b32 << b_headrm;
    int32_t b32_inv = (INT32_MAX >> 2) / (b32_nrm >> 16);
    int32_t result = static_cast<int32_t>(static_cast<uint32_t>(b32_inv)
                                          << 16);
    int32_t err_q32 = static_cast<int32_t>(
        static_cast<uint32_t>((1 << 29) - smulwb(b32_nrm, b32_inv)) << 3);
    result = smlaww(result, err_q32, b32_inv);
    int lshift = 61 - b_headrm - qres;
    if (lshift <= 0) return lshift_sat32(result, -lshift);
    if (lshift < 32) return result >> lshift;
    return 0;
}

// ---- log2 <-> linear (reference lin2log.c / log2lin.c) ---------------------

// Approximate 128*log2(x).
inline int32_t lin2log(int32_t in_lin) {
    int32_t lz, frac_q7;
    clz_frac(in_lin, &lz, &frac_q7);
    // Piecewise-parabolic interpolation.
    return ((31 - lz) << 7) +
           smlawb(frac_q7, smulbb(frac_q7, 128 - frac_q7), 179);
}

// Approximate 2^(x/128).
inline int32_t log2lin(int32_t in_log_q7) {
    if (in_log_q7 < 0) return 0;
    if (in_log_q7 >= 3967) return INT32_MAX;
    int32_t out = 1 << (in_log_q7 >> 7);
    int32_t frac_q7 = in_log_q7 & 0x7F;
    if (in_log_q7 < 2048) {
        // Piecewise-linear with parabolic correction, low range.
        out = add32_ovflw(
            out, (out * smlawb(frac_q7,
                               smulbb(frac_q7, 128 - frac_q7), -174)) >> 7);
    } else {
        out = mla_ovflw(out, out >> 7,
                        smlawb(frac_q7,
                               smulbb(frac_q7, 128 - frac_q7), -174));
    }
    return out;
}

// SILK's LCG (RAND): wraps intentionally.
inline int32_t silk_rand(int32_t seed) {
    return mla_ovflw(907633515, seed, 196314165);
}

}  // namespace silk
}  // namespace opus
}  // namespace glint

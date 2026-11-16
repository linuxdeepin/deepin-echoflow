// glint - Fixed-point arithmetic (Q31 format)
// MIT License - Clean-room implementation

#ifndef GLINT_FIXEDPOINT_HPP
#define GLINT_FIXEDPOINT_HPP

#include <cstdint>

namespace glint {

// Q31: signed 32-bit, 1 sign bit + 31 fractional bits
// Range: [-1.0, +1.0), resolution ~4.66e-10

// Multiply two Q31 values, return Q31 (truncated)
inline int32_t fpmul(int32_t a, int32_t b) {
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 6)
    int32_t result;
#if defined(__aarch64__)
    asm("smmul %w0, %w1, %w2" : "=r"(result) : "r"(a), "r"(b));
#else
    asm("smmul %0, %1, %2" : "=r"(result) : "r"(a), "r"(b));
#endif
    return result;
#elif defined(__mips__) && !defined(__mips16)
    int32_t hi;
    asm("mult %1, %2\n\t"
        "mfhi %0"
        : "=r"(hi) : "r"(a), "r"(b) : "hi", "lo");
    return hi;
#else
    return static_cast<int32_t>((static_cast<int64_t>(a) * b) >> 32);
#endif
}

// Multiply two Q31 values with rounding
inline int32_t fpmul_round(int32_t a, int32_t b) {
    return static_cast<int32_t>((static_cast<int64_t>(a) * b + 0x80000000LL) >> 32);
}

// Multiply-accumulate: acc + (a * b >> 32)
inline int32_t fpmul_acc(int32_t acc, int32_t a, int32_t b) {
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 6)
    int32_t result;
#if defined(__aarch64__)
    asm("smmla %w0, %w1, %w2, %w3" : "=r"(result) : "r"(a), "r"(b), "r"(acc));
#else
    asm("smmla %0, %1, %2, %3" : "=r"(result) : "r"(a), "r"(b), "r"(acc));
#endif
    return result;
#elif defined(__mips__) && !defined(__mips16)
    int32_t hi;
    asm("mult %1, %2\n\t"
        "mfhi %0"
        : "=r"(hi) : "r"(a), "r"(b) : "hi", "lo");
    return acc + hi;
#else
    return acc + static_cast<int32_t>((static_cast<int64_t>(a) * b) >> 32);
#endif
}

// Multiply Q31 by Q30 coefficient (1-bit left shift): result = (a*b) >> 31
inline int32_t fpmul_shift(int32_t a, int32_t b) {
    return static_cast<int32_t>((static_cast<int64_t>(a) * b) >> 31);
}

// Convert double to Q31
constexpr int32_t to_q31(double v) {
    return static_cast<int32_t>(v * 2147483648.0 + (v >= 0 ? 0.5 : -0.5));
}

// Convert int16_t PCM to Q31 (left-shift by 16)
inline int32_t pcm_to_q31(int16_t s) {
    return static_cast<int32_t>(s) << 16;
}

// ---------------------------------------------------------------------------
// Self-test: compile-time verification of fixed-point primitives
// ---------------------------------------------------------------------------
namespace selftest {

inline bool run() {
    // fpmul computes (a * b) >> 32 (matches ARM smmul).
    // For Q31 inputs this gives Q30 result (inherent 1-bit right shift vs
    // a true Q31*Q31->Q31 multiply). This is by design to avoid overflow
    // when both operands are near -1.0.
    //
    // 0.5 in Q31 = 0x40000000
    // fpmul(0.5, 0.5) = (0x40000000 * 0x40000000) >> 32 = 0x10000000
    int32_t half = 0x40000000;
    int32_t result = fpmul(half, half);
    if (result != 0x10000000) return false;

    // fpmul_round: with rounding bit
    int32_t result_r = fpmul_round(half, half);
    if (result_r != 0x10000000) return false;

    // fpmul_acc: 0x10000000 + fpmul(0.5, 0.5) = 0x20000000
    int32_t acc_result = fpmul_acc(result, half, half);
    if (acc_result != 0x20000000) return false;

    // pcm_to_q31: 1 -> 0x10000, 32767 -> 0x7FFF0000, -1 -> 0xFFFF0000
    if (pcm_to_q31(1) != 0x00010000) return false;
    if (pcm_to_q31(32767) != 0x7FFF0000) return false;
    if (pcm_to_q31(-1) != static_cast<int32_t>(0xFFFF0000)) return false;

    return true;
}

} // namespace selftest
} // namespace glint

#endif // GLINT_FIXEDPOINT_HPP

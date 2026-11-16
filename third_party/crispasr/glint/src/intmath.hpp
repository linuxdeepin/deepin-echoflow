// glint - shared integer log2/exp2 helpers for the no-FPU hot paths
// MIT License - Clean-room implementation
//
// Q16 log2 and exp2 via 128-segment LUTs with linear interpolation
// (relative error ~1e-6). Used by the AAC integer quantizer
// (GLINT_AAC_INT) and the MP3 integer -q speed path (GLINT_MP3_INT):
// both quantizers are x -> floor(2^(0.75*log2|x| + step) + 0.4054) with an
// exactly-representable Q16 step (0.1875*65536 = 12288 for AAC's sf unit,
// 3/16*65536 = 12288 for MP3's quarter-dB gain — the same constant).

#ifndef GLINT_INTMATH_HPP
#define GLINT_INTMATH_HPP

#include <cmath>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace glint {
namespace intmath {

// Portable count-leading-zeros for v > 0 (MSVC has no __builtin_clz, and it
// compiles non-template inline bodies even when unused).
inline int clz32(uint32_t v) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse(&idx, v);
    return 31 - static_cast<int>(idx);
#else
    return __builtin_clz(v);
#endif
}

struct PowLuts {
    int32_t log2q16[130];   // log2(1 + i/128) * 65536
    int32_t exp2q15[130];   // 2^(i/128) * 32768
    PowLuts() {
        for (int i = 0; i <= 129; i++) {
            double f = i / 128.0;
            log2q16[i] = static_cast<int32_t>(std::lround(std::log2(1.0 + f) * 65536.0));
            exp2q15[i] = static_cast<int32_t>(std::lround(std::pow(2.0, f) * 32768.0));
        }
    }
};

// Global (startup-time) instance: keeps the trig/log constructor code in a
// global ctor instead of inlined into the hot functions' guard branches —
// tools/check_nofpu.sh proves the per-coefficient paths float-free that way.
inline const PowLuts g_pow_luts{};
inline const PowLuts& pow_luts() { return g_pow_luts; }

// Q16 log2 of v (v >= 1)
inline int32_t ilog2_q16(uint32_t v) {
    const PowLuts& t = pow_luts();
    int b = 31 - clz32(v);
    uint32_t u = v << (31 - b);           // MSB at bit 31
    int idx = (u >> 24) & 0x7F;           // 7 bits below the MSB
    int r = (u >> 16) & 0xFF;             // next 8 bits for interpolation
    int32_t lo = t.log2q16[idx];
    int32_t hi = t.log2q16[idx + 1];
    return (b << 16) + lo + static_cast<int32_t>((static_cast<int64_t>(hi - lo) * r) >> 8);
}

// floor(2^(l/65536) + magic/65536), clamped to cap. l may be negative;
// cap must be < 2^15 + 1 (the I >= 15 early-out returns cap directly).
inline int exp2_quant(int64_t l, int32_t magic_q16, int cap) {
    const PowLuts& t = pow_luts();
    int64_t I = l >> 16;                  // floor
    int f = static_cast<int>(l & 0xFFFF);
    if (I >= 15) return cap;
    int idx = (f >> 9) & 0x7F;
    int r = (f >> 1) & 0xFF;
    int32_t lo = t.exp2q15[idx];
    int32_t hi = t.exp2q15[idx + 1];
    int64_t m = lo + ((static_cast<int64_t>(hi - lo) * r) >> 8);  // Q15, [1,2)
    int sh = static_cast<int>(I) + 1;
    int64_t q16;
    if (sh >= 0) {
        q16 = m << sh;
    } else {
        q16 = m >> (-sh);
    }
    int v = static_cast<int>((q16 + magic_q16) >> 16);
    return v > cap ? cap : v;
}

}  // namespace intmath
}  // namespace glint

#endif  // GLINT_INTMATH_HPP

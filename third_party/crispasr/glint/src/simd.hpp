// glint - SIMD runtime dispatch
// MIT License - Clean-room implementation

#ifndef GLINT_SIMD_HPP
#define GLINT_SIMD_HPP

#include "glint/glint.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>  // __cpuid, __cpuidex, _xgetbv
#endif

namespace glint {

// Runtime SIMD level, set during glint_create() based on glint_config.simd.
// Checked in hot loops to select the code path.
// Values match glint_simd enum: 0=auto, 1=avx, 2=sse2, 3=none.
inline int g_simd_level = 0;  // 0 = not yet initialized

// Detect best available SIMD at runtime
inline int detect_simd() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // On x86, SSE2 is always available on x86-64
    int level = GLINT_SIMD_SSE2;
#if defined(__GNUC__) || defined(__clang__)
    // Use __builtin_cpu_supports for runtime AVX detection
    if (__builtin_cpu_supports("avx2") || __builtin_cpu_supports("avx"))
        level = GLINT_SIMD_AVX;
#elif defined(_MSC_VER)
    // MSVC has no __builtin_cpu_supports, so do the CPUID dance by hand.
    //
    // The AVX code paths here use only _mm256_*_pd — AVX1, no AVX2 integer
    // ops — so the bit to test is CPUID.1:ECX[28], not the AVX2 bit in leaf 7.
    // Testing AVX2 made MSVC fall back to SSE2 on every Sandy/Ivy Bridge part
    // that GCC happily ran with AVX.
    //
    // AVX also needs the OS to be saving YMM state, which is the OSXSAVE bit
    // plus XGETBV — without that check, enabling AVX on an OS that does not
    // preserve YMM corrupts registers across context switches.
    int cpuinfo[4];
    __cpuid(cpuinfo, 0);
    if (cpuinfo[0] >= 1) {  // leaf 1 exists (it always does; be explicit)
        __cpuidex(cpuinfo, 1, 0);
        const bool osxsave = (cpuinfo[2] & (1 << 27)) != 0;
        const bool avx = (cpuinfo[2] & (1 << 28)) != 0;
        // _XCR_XFEATURE_ENABLED_MASK == 0; bits 1 (XMM) and 2 (YMM) must be set.
        if (osxsave && avx && (_xgetbv(0) & 0x6) == 0x6)
            level = GLINT_SIMD_AVX;
    }
#endif
    return level;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    // AArch64 NEON
    return GLINT_SIMD_NEON;
#else
    // Non-x86, non-ARM: no SIMD
    return GLINT_SIMD_NONE;
#endif
}

// Initialize SIMD level from config
inline void init_simd(glint_simd requested) {
    if (requested == GLINT_SIMD_AUTO)
        g_simd_level = detect_simd();
    else
        g_simd_level = requested;
}

} // namespace glint

#endif // GLINT_SIMD_HPP

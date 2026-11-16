// Regression guard for #327 — glint's SIMD dispatch header.
//
// The reported symptom was a Windows/MSVC build break: simd.hpp calls the
// intrinsic __cpuidex without including <intrin.h>. What makes it worth a test
// rather than just a fix is how it hid. GCC/Clang take the
// __builtin_cpu_supports branch, so Linux and macOS never compiled the MSVC
// path at all; and our windows-latest CI job *did* build glint and *did* pass,
// because some MSVC toolsets pull <intrin.h> in transitively. Whether this
// compiled was luck, and a green CI proved nothing either way.
//
// It can also silently come back. tools/sync-glint.sh overwrites glint/ wholesale
// from upstream, so the include only survives as long as upstream keeps it.
//
// Hence the first line of this file: simd.hpp is included BEFORE anything else,
// so it has to be self-sufficient. On MSVC that alone reproduces #327 as a
// compile error. Everything below then checks the behaviour of the fix.
#include "simd.hpp"

// Deliberately after the header under test.
#include <catch2/catch_test_macros.hpp>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define SIMD_TEST_X86 1
#else
#define SIMD_TEST_X86 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SIMD_TEST_NEON 1
#else
#define SIMD_TEST_NEON 0
#endif

TEST_CASE("glint simd: detect_simd returns a level valid for this architecture", "[glint][simd]") {
    const int level = glint::detect_simd();

    // Never AUTO: AUTO is a request, not a result. init_simd() resolves it, so
    // detect_simd() returning 0 would leave g_simd_level permanently "not yet
    // initialized" and every hot-loop check would misread it.
    REQUIRE(level != GLINT_SIMD_AUTO);

#if SIMD_TEST_X86
    // SSE2 is architectural on x86-64, so the floor is SSE2 and the only
    // question is whether AVX was detected on top of it.
    REQUIRE((level == GLINT_SIMD_AVX || level == GLINT_SIMD_SSE2));
#elif SIMD_TEST_NEON
    REQUIRE(level == GLINT_SIMD_NEON);
#else
    REQUIRE(level == GLINT_SIMD_NONE);
#endif
}

#if SIMD_TEST_X86 && (defined(__GNUC__) || defined(__clang__))
// On GCC/Clang x86 the answer is independently checkable: ask the compiler
// builtin directly and compare. This is the arm that would have caught the
// second bug found while fixing #327 — the MSVC branch gated an AVX1-only code
// path on the AVX2 bit, so every Sandy/Ivy Bridge part ran SSE2 under MSVC
// while GCC ran the same machine with AVX. The two toolchains must agree.
TEST_CASE("glint simd: x86 detection agrees with the compiler builtin", "[glint][simd]") {
    const bool has_avx = __builtin_cpu_supports("avx") || __builtin_cpu_supports("avx2");
    const int expected = has_avx ? GLINT_SIMD_AVX : GLINT_SIMD_SSE2;
    REQUIRE(glint::detect_simd() == expected);
}
#endif

TEST_CASE("glint simd: init_simd honours an explicit request verbatim", "[glint][simd]") {
    // An explicit level must be taken as given, including one this CPU cannot
    // run — GLINT_SIMD_AVX is documented as "crashes if unsupported", i.e. the
    // caller is overriding detection on purpose and must not be second-guessed.
    for (const int requested : {GLINT_SIMD_AVX, GLINT_SIMD_SSE2, GLINT_SIMD_NONE, GLINT_SIMD_NEON}) {
        glint::init_simd(static_cast<glint_simd>(requested));
        REQUIRE(glint::g_simd_level == requested);
    }
}

TEST_CASE("glint simd: init_simd(AUTO) resolves to detect_simd", "[glint][simd]") {
    // Poison it first, so a no-op init_simd cannot pass by leaving behind a
    // value an earlier case happened to set.
    glint::g_simd_level = GLINT_SIMD_AUTO;
    glint::init_simd(GLINT_SIMD_AUTO);
    REQUIRE(glint::g_simd_level == glint::detect_simd());
    REQUIRE(glint::g_simd_level != GLINT_SIMD_AUTO);
}

TEST_CASE("glint simd: detection is pure", "[glint][simd]") {
    // Called once per glint_create(), and callers may create several engines.
    // CPUID does not change under us, so repeated calls must agree — and must
    // not depend on g_simd_level, which init_simd writes between them.
    const int first = glint::detect_simd();
    glint::init_simd(GLINT_SIMD_NONE);
    REQUIRE(glint::detect_simd() == first);
    glint::init_simd(GLINT_SIMD_AUTO);
    REQUIRE(glint::detect_simd() == first);
}

// crispasr_cpu_isa.cpp — see crispasr_cpu_isa.h for rationale (#261).

#include "crispasr_cpu_isa.h"

// core_cpu_backend::has_feature(): the ISA libggml-cpu was BUILT with, i.e. what
// ggml's own kernels actually emit — which is what SIGILLs in #261, and which
// our own TU macros cannot see. Safe to call before ggml_cpu_init().
//
// Routed through the shim rather than calling ggml_cpu_has_*() directly: those
// are symbols in libggml-cpu, and under GGML_BACKEND_DL the CPU backend is a
// dlopen'd module, so a direct call does not LINK. That broke every CUDA
// package in the v0.8.30 release run.
#include "core/ggml_cpu_backend.h"

#include <cstdint>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CRISPASR_ISA_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace crispasr_cpu_isa {

#if CRISPASR_ISA_X86

// Thin cpuid wrappers. leaf/subleaf → eax..edx.
static void cpuid_count(uint32_t leaf, uint32_t subleaf, uint32_t regs[4]) {
#if defined(_MSC_VER)
    int r[4];
    __cpuidex(r, (int)leaf, (int)subleaf);
    regs[0] = (uint32_t)r[0];
    regs[1] = (uint32_t)r[1];
    regs[2] = (uint32_t)r[2];
    regs[3] = (uint32_t)r[3];
#else
    unsigned int a = 0, b = 0, c = 0, d = 0;
    __get_cpuid_count(leaf, subleaf, &a, &b, &c, &d);
    regs[0] = a;
    regs[1] = b;
    regs[2] = c;
    regs[3] = d;
#endif
}

static uint64_t xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#endif
}

// Host CPU capabilities as actually usable (cpuid feature bit AND, where
// relevant, OS enablement of the register state via XCR0). Without OS XSAVE
// support an AVX/AVX-512 instruction #UDs even on a capable CPU, so the OS
// state must be part of "available".
struct HostCaps {
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;
    bool bmi2 = false;
    bool avx512f = false;
};

static HostCaps detect_host() {
    HostCaps h;
    uint32_t r1[4] = {0, 0, 0, 0};
    cpuid_count(1, 0, r1);
    const bool osxsave = (r1[2] & (1u << 27)) != 0;  // CPUID.1:ECX.OSXSAVE
    const bool cpu_avx = (r1[2] & (1u << 28)) != 0;  // CPUID.1:ECX.AVX
    const bool cpu_fma = (r1[2] & (1u << 12)) != 0;  // CPUID.1:ECX.FMA
    const bool cpu_f16c = (r1[2] & (1u << 29)) != 0; // CPUID.1:ECX.F16C

    // XCR0: bit1 = SSE(XMM), bit2 = AVX(YMM), bits5-7 = AVX-512 (opmask/ZMM_Hi256/Hi16_ZMM).
    bool ymm_enabled = false;
    bool zmm_enabled = false;
    if (osxsave) {
        const uint64_t xcr0 = xgetbv0();
        ymm_enabled = (xcr0 & 0x6) == 0x6;                  // XMM + YMM
        zmm_enabled = ymm_enabled && (xcr0 & 0xE0) == 0xE0; // + opmask/ZMM
    }

    // AVX-family requires OS YMM state; FMA/F16C are AVX-encoded too.
    h.avx = cpu_avx && ymm_enabled;
    h.fma = cpu_fma && ymm_enabled;
    h.f16c = cpu_f16c && ymm_enabled;

    uint32_t r7[4] = {0, 0, 0, 0};
    cpuid_count(7, 0, r7);
    h.avx2 = ((r7[1] & (1u << 5)) != 0) && ymm_enabled;     // CPUID.7.0:EBX.AVX2
    h.bmi2 = (r7[1] & (1u << 8)) != 0;                      // CPUID.7.0:EBX.BMI2 (no XSAVE dep)
    h.avx512f = ((r7[1] & (1u << 16)) != 0) && zmm_enabled; // CPUID.7.0:EBX.AVX512F

    return h;
}

#endif // CRISPASR_ISA_X86

IsaCheck check() {
    IsaCheck out;

#if CRISPASR_ISA_X86
    out.checked = true;
    const HostCaps h = detect_host();

    struct Feature {
        const char* name;
        bool required;  // does ANY binary in this process emit instructions for it?
        bool available; // does the host CPU (+OS) support it?
    };

    // "required" has TWO sources and must be the OR of both:
    //
    //   1. Our own TUs' compile-time macros (__AVX2__, __BMI2__, ...).
    //   2. What libggml-cpu.so was built with. `ggml_cpu_has_*()` return
    //      build-time constants compiled INTO libggml-cpu, so they report ggml's
    //      own ARCH_FLAGS (GGML_NATIVE / -march=native, GGML_BMI2, ...), which
    //      CMake picks independently of the flags used for CrispASR's sources.
    //
    // (2) is the one #261 turns on. Source 1 alone is a FALSE NEGATIVE: the
    // reporter's host printed "cpu-isa: OK — built for baseline x86-64 (no AVX)"
    // (true for our TUs) and then still died with SIGILL on a BMI2 instruction
    // inside libggml-cpu's ggml_cpu_init(). A baseline-built crispasr_cpu_isa.cpp
    // simply cannot see BMI2 that ggml baked into a different shared object.
    //
    // Calling these first is safe: they are constant returns and do NOT invoke
    // ggml_cpu_init() (the actual SIGILL site, reached only via
    // ggml_backend_cpu_reg → get_reg → ggml_backend_dev_count).
#if defined(__AVX512F__)
    constexpr bool own_avx512f = true;
#else
    constexpr bool own_avx512f = false;
#endif
#if defined(__AVX2__)
    constexpr bool own_avx2 = true;
#else
    constexpr bool own_avx2 = false;
#endif
#if defined(__FMA__)
    constexpr bool own_fma = true;
#else
    constexpr bool own_fma = false;
#endif
#if defined(__F16C__)
    constexpr bool own_f16c = true;
#else
    constexpr bool own_f16c = false;
#endif
#if defined(__BMI2__)
    constexpr bool own_bmi2 = true;
#else
    constexpr bool own_bmi2 = false;
#endif
#if defined(__AVX__)
    constexpr bool own_avx = true;
#else
    constexpr bool own_avx = false;
#endif

    const std::vector<Feature> feats = {
        {"AVX512F", own_avx512f || core_cpu_backend::has_feature("AVX512"), h.avx512f},
        {"AVX2", own_avx2 || core_cpu_backend::has_feature("AVX2"), h.avx2},
        {"FMA", own_fma || core_cpu_backend::has_feature("FMA"), h.fma},
        {"F16C", own_f16c || core_cpu_backend::has_feature("F16C"), h.f16c},
        {"BMI2", own_bmi2 || core_cpu_backend::has_feature("BMI2"), h.bmi2},
        {"AVX", own_avx || core_cpu_backend::has_feature("AVX"), h.avx},
    };

    for (const auto& f : feats) {
        if (!f.required)
            continue;
        if (!out.required.empty())
            out.required += "/";
        out.required += f.name;
        if (!f.available) {
            out.ok = false;
            if (!out.missing.empty())
                out.missing += ", ";
            out.missing += f.name;
        }
    }
    if (out.required.empty())
        out.required = "baseline x86-64 (no AVX)";
    out.host_note =
        out.ok ? "host CPU supports the build's ISA baseline" : "host CPU is missing instructions this build emits";
#else
    // Non-x86: we don't probe. The native-build/SIGILL mismatch is x86-specific
    // in the distributions we ship; ARM/Apple negotiate ISA state with the OS.
    out.checked = false;
    out.ok = true;
    out.required = "(non-x86 host — ISA self-check skipped)";
    out.host_note = "non-x86 platform";
#endif

    return out;
}

std::string banner(const IsaCheck& c) {
    if (!c.checked)
        return "cpu-isa: check skipped (" + c.host_note + ")";
    if (c.ok)
        return "cpu-isa: OK — built for " + c.required + "; " + c.host_note;

    // Mismatch: this is the #261 failure. Make it loud and actionable.
    return "cpu-isa: *** WARNING — instruction-set MISMATCH ***\n"
           "cpu-isa: this build emits " +
           c.required + " but the host CPU lacks: " + c.missing +
           "\n"
           "cpu-isa: CPU-only paths (Silero VAD, pyannote diarization) will raise SIGILL and\n"
           "cpu-isa: crash the process on this host. GPU ASR is unaffected. Use a portable\n"
           "cpu-isa: image built with -DGGML_NATIVE=OFF (AVX2 baseline), or omit vad/diarize.\n"
           "cpu-isa: (see issue #261) — such requests will be refused with HTTP 400 instead.";
}

} // namespace crispasr_cpu_isa

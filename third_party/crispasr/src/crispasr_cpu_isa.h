// crispasr_cpu_isa.h — CPU instruction-set self-check (#261).
//
// The CPU ggml backend is compiled for a FIXED instruction-set baseline
// (distributed images use -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON
// -DGGML_F16C=ON; see .devops/*.Dockerfile). If a binary is run on a CPU that
// lacks an instruction its kernels emit — e.g. an AVX-512 build (-march=native
// on a Xeon CI runner) run on a consumer / WSL2 CPU — those kernels raise
// SIGILL, which silently kills the process (issue #261). GPU ASR is unaffected;
// only the CPU-only paths (Silero VAD, pyannote diarization) hit it.
//
// This module compares what the binary was BUILT to emit (compile-time ISA
// macros) against what the host CPU actually supports (runtime cpuid), so
// callers can fail fast / degrade gracefully instead of crashing mid-inference.
// It cannot make a missing instruction runnable — the opcodes are baked into
// the compiled kernels; the only way to route around them is a multi-variant
// runtime-dispatch build (GGML_CPU_ALL_VARIANTS). Detection is the safety net.

#pragma once

#include <string>

namespace crispasr_cpu_isa {

struct IsaCheck {
    bool ok = true;        // true if the CPU supports every ISA the binary emits
    std::string missing;   // comma-separated missing features (empty when ok)
    std::string required;  // ISA the binary emits, for diagnostics
    std::string host_note; // short human note about the host CPU / OS state
    bool checked = false;  // false on platforms where we don't probe (non-x86)
};

// Compare the compile-time ISA baseline against the runtime CPU features.
// Cheap (a couple of cpuid calls); safe to call anytime, from any thread.
// On non-x86 hosts this returns ok=true, checked=false — the native-build
// SIGILL failure mode is x86-specific in practice (ARM / Apple ship fixed,
// OS-negotiated baselines and the release/docker builds don't use -march=native
// there in a way that mismatches).
IsaCheck check();

// A single startup line (may be multi-line on mismatch) describing the result.
// Does not include a trailing newline.
std::string banner(const IsaCheck& c);

} // namespace crispasr_cpu_isa

#include "core/chatterbox_f0_conv.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Exactly-representable grid: every value is n/4096 with |n| < 64, so every
// product is n1*n2/2^24 and every partial sum stays a multiple of 2^-24 below
// 1.0 — i.e. the whole reduction is EXACT. That makes this data useful for
// checking indexing and lane mapping, but blind to anything about rounding:
// float addition is associative when nothing rounds, so a kernel that summed
// in a different order would still match here. Measured: reordering the
// reduction from k-major to ci-major changes 0/8704 values on this grid and
// 5512/8704 on the rounding data below. Keep BOTH generators.
void fill_exact(std::vector<float>& values, std::uint32_t seed) {
    std::size_t index = 0;
    for (auto& value : values) {
        const auto integer = static_cast<int>((index++ * 37u + seed * 17u) % 127u) - 63;
        value = static_cast<float>(integer) / 4096.0f;
    }
}

// Values that actually round, so the per-lane accumulation ORDER is observable
// — that order is the whole reason the SIMD kernels can claim bit-identity.
// Deterministic LCG rather than <random>, so every host and libstdc++ version
// sees the same bytes.
void fill_rounding(std::vector<float>& values, std::uint32_t seed) {
    std::uint32_t state = seed * 2654435761u + 12345u;
    for (auto& value : values) {
        state = state * 1664525u + 1013904223u;
        // 24 random mantissa bits mapped to [-1, 1): not representable on any
        // coarse grid, so sums round.
        value = static_cast<float>((state >> 8) & 0xFFFFFFu) / 8388608.0f - 1.0f;
    }
}

using FillFn = void (*)(std::vector<float>&, std::uint32_t);

void require_matches_scalar(int T, int C_in, int C_out, int n_threads, core_chatterbox_f0::Isa isa, FillFn fill,
                            const char* fill_name) {
    std::vector<float> x(static_cast<std::size_t>(T) * C_in);
    std::vector<float> w(static_cast<std::size_t>(C_out) * C_in * 3);
    std::vector<float> bias(C_out);
    fill(x, 11);
    fill(w, 23);
    fill(bias, 47);

    std::vector<float> reference(static_cast<std::size_t>(T) * C_out);
    std::vector<float> actual(reference.size());
    core_chatterbox_f0::Conv1dEluK3{x.data(), w.data(), bias.data(), reference.data(), T, C_in, C_out}.run_scalar();
    core_chatterbox_f0::Conv1dEluK3{x.data(), w.data(), bias.data(), actual.data(), T, C_in, C_out}.run(n_threads, isa);

    INFO("isa=" << static_cast<int>(isa) << " data=" << fill_name << " T=" << T << " C_in=" << C_in
                << " C_out=" << C_out << " threads=" << n_threads);
    REQUIRE(std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) == 0);
}

} // namespace

// The F0 layers are all C_out = 512 (chatterbox_s3gen.cpp: `const int C = 512`),
// with C_in 80 on the first layer and 512 after it. Both shapes are covered.
TEST_CASE("Chatterbox F0 SIMD Conv1d is bit-identical to scalar", "[unit][tts][chatterbox][simd]") {
    const std::array<core_chatterbox_f0::Isa, 4> candidates{{
        core_chatterbox_f0::Isa::scalar,
        core_chatterbox_f0::Isa::neon,
        core_chatterbox_f0::Isa::avx2,
        core_chatterbox_f0::Isa::avx512f,
    }};

    for (const auto isa : candidates) {
        if (!core_chatterbox_f0::isa_available(isa))
            continue;
        for (const auto& data : {std::make_pair(&fill_exact, "exact"), std::make_pair(&fill_rounding, "rounding")}) {
            require_matches_scalar(7, 80, 512, 1, isa, data.first, data.second);
            require_matches_scalar(7, 512, 512, 1, isa, data.first, data.second);
            require_matches_scalar(17, 80, 512, 4, isa, data.first, data.second);
            require_matches_scalar(17, 512, 512, 4, isa, data.first, data.second);
        }
    }
}

// A green run on a host with no SIMD proves nothing about the SIMD kernels —
// the loop above just skips them. Say out loud what was exercised, so an ARM
// or pre-AVX2 CI leg cannot be mistaken for AVX2/AVX-512 coverage, and fail if
// the host advertises an ISA that dispatch then refused to use.
TEST_CASE("Chatterbox F0 SIMD reports which ISAs it actually exercised", "[unit][tts][chatterbox][simd]") {
    std::string exercised;
    const std::array<core_chatterbox_f0::Isa, 3> simd_isas{{
        core_chatterbox_f0::Isa::neon,
        core_chatterbox_f0::Isa::avx2,
        core_chatterbox_f0::Isa::avx512f,
    }};
    for (const auto isa : simd_isas) {
        if (!core_chatterbox_f0::isa_available(isa))
            continue;
        exercised += exercised.empty() ? "" : ",";
        exercised += isa_name(isa);
    }
    std::printf("[chatterbox-f0-simd] host: neon=%d avx2=%d avx512f=%d ; "
                "best_isa=%s ; SIMD exercised: %s\n",
                static_cast<int>(core_chatterbox_f0::isa_available(core_chatterbox_f0::Isa::neon)),
                static_cast<int>(core_chatterbox_f0::isa_available(core_chatterbox_f0::Isa::avx2)),
                static_cast<int>(core_chatterbox_f0::isa_available(core_chatterbox_f0::Isa::avx512f)),
                isa_name(core_chatterbox_f0::best_isa()),
                exercised.empty() ? "none" : exercised.c_str());

    // best_isa() must pick the widest ISA the host actually has, or dispatch
    // has silently regressed to a slower path while every output still matches.
    if (core_chatterbox_f0::isa_available(core_chatterbox_f0::Isa::avx512f))
        REQUIRE(core_chatterbox_f0::best_isa() == core_chatterbox_f0::Isa::avx512f);
    else if (core_chatterbox_f0::isa_available(core_chatterbox_f0::Isa::avx2))
        REQUIRE(core_chatterbox_f0::best_isa() == core_chatterbox_f0::Isa::avx2);
    else if (core_chatterbox_f0::isa_available(core_chatterbox_f0::Isa::neon))
        REQUIRE(core_chatterbox_f0::best_isa() == core_chatterbox_f0::Isa::neon);
    else
        REQUIRE(core_chatterbox_f0::best_isa() == core_chatterbox_f0::Isa::scalar);
}

// C_out that is not a multiple of the vector width must fall back to scalar
// rather than run a partial block: the packed-weight layout is sized in whole
// blocks, so a partial one would read and write past both buffers. The F0
// predictor only ever passes 512, so this pins the guard rather than a shape
// in use today.
TEST_CASE("Chatterbox F0 falls back cleanly when C_out is not a multiple of the width",
          "[unit][tts][chatterbox][simd]") {
    for (const int C_out : {1, 8, 100, 520}) {
        require_matches_scalar(9, 80, C_out, 2, core_chatterbox_f0::best_isa(), &fill_rounding, "rounding");
    }
}

TEST_CASE("Chatterbox F0 auto dispatch preserves scalar output", "[unit][tts][chatterbox][simd]") {
    require_matches_scalar(17, 512, 512, 4, core_chatterbox_f0::best_isa(), &fill_exact, "exact");
    require_matches_scalar(17, 512, 512, 4, core_chatterbox_f0::best_isa(), &fill_rounding, "rounding");
}

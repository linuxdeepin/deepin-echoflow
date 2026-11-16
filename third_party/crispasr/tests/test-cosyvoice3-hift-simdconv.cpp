#include "core/cosyvoice3_hift_simdconv.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

void fill_rounding(std::vector<float>& values, std::uint32_t seed) {
    std::uint32_t state = seed * 2654435761u + 12345u;
    for (auto& value : values) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<float>((state >> 8) & 0xFFFFFFu) / 8388608.0f - 1.0f;
    }
}

// Reference against the UNPACKED ggml weight byte order:
//   w ne=(K, IC, OC) => source index oc*K*IC + ic*K + k.
// Input/output here are the SIMDCONV time-major island: [T][C].
void reference_causal_conv(const float* input, const float* weights, const float* bias, float* output, int T, int C,
                           int K, int dilation) {
    const int pad = (K - 1) * dilation;
    for (int t = 0; t < T; ++t) {
        for (int oc = 0; oc < C; ++oc) {
            float sum = bias[oc];
            for (int k = 0; k < K; ++k) {
                const int src_t = t + k * dilation - pad;
                if (src_t < 0)
                    continue;
                const float* in = input + static_cast<std::size_t>(src_t) * C;
                for (int ic = 0; ic < C; ++ic) {
                    const float w = weights[static_cast<std::size_t>(oc) * C * K +
                                            static_cast<std::size_t>(ic) * K + k];
                    sum += w * in[ic];
                }
            }
            output[static_cast<std::size_t>(t) * C + oc] = sum;
        }
    }
}

void require_kernel(int K, int dilation, core_cv3_hift_simdconv::Isa isa, int n_threads) {
    constexpr int T = 17;
    constexpr int C = 32; // divisible by NEON4, AVX2x8 and AVX-512x16
    std::vector<float> input(static_cast<std::size_t>(T) * C);
    std::vector<float> weights(static_cast<std::size_t>(K) * C * C);
    std::vector<float> bias(C);
    fill_rounding(input, 11u + static_cast<std::uint32_t>(K * 7 + dilation));
    fill_rounding(weights, 23u + static_cast<std::uint32_t>(K * 11 + dilation));
    fill_rounding(bias, 47u + static_cast<std::uint32_t>(K * 13 + dilation));

    std::vector<float> reference(static_cast<std::size_t>(T) * C);
    std::vector<float> scalar(reference.size());
    std::vector<float> actual(reference.size());
    reference_causal_conv(input.data(), weights.data(), bias.data(), reference.data(), T, C, K, dilation);

    core_cv3_hift_simdconv::PackedConv scalar_conv;
    REQUIRE(scalar_conv.reset(weights.data(), bias.data(), K, C, dilation,
                              core_cv3_hift_simdconv::Isa::scalar));
    scalar_conv.run(input.data(), scalar.data(), T, n_threads);
    INFO("K=" << K << " dilation=" << dilation << " scalar");
    REQUIRE(std::memcmp(reference.data(), scalar.data(), reference.size() * sizeof(float)) == 0);

    core_cv3_hift_simdconv::PackedConv conv;
    REQUIRE(conv.reset(weights.data(), bias.data(), K, C, dilation, isa));
    conv.run(input.data(), actual.data(), T, n_threads);
    INFO("K=" << K << " dilation=" << dilation << " isa=" << core_cv3_hift_simdconv::isa_name(isa)
              << " resolved=" << core_cv3_hift_simdconv::isa_name(conv.isa) << " threads=" << n_threads);
    REQUIRE(std::memcmp(scalar.data(), actual.data(), scalar.size() * sizeof(float)) == 0);
}

} // namespace

TEST_CASE("CosyVoice3 HiFT SIMDCONV scalar matches unpacked causal Conv1d",
          "[unit][tts][cosyvoice3][simdconv]") {
    for (const int K : {3, 7, 11})
        for (const int dilation : {1, 3, 5})
            require_kernel(K, dilation, core_cv3_hift_simdconv::Isa::scalar, 4);
}

TEST_CASE("CosyVoice3 HiFT SIMDCONV SIMD is bit-identical to its scalar reference",
          "[unit][tts][cosyvoice3][simdconv]") {
    const std::array<core_cv3_hift_simdconv::Isa, 3> candidates{{
        core_cv3_hift_simdconv::Isa::neon,
        core_cv3_hift_simdconv::Isa::avx2,
        core_cv3_hift_simdconv::Isa::avx512f,
    }};
    std::string exercised;
    for (const auto isa : candidates) {
        if (!core_cv3_hift_simdconv::isa_available(isa))
            continue;
        exercised += exercised.empty() ? "" : ",";
        exercised += core_cv3_hift_simdconv::isa_name(isa);
        for (const int K : {3, 7, 11})
            for (const int dilation : {1, 3, 5})
                require_kernel(K, dilation, isa, 4);
    }
    std::printf("[cosyvoice3-hift-simdconv] best=%s ; SIMD exercised: %s\n",
                core_cv3_hift_simdconv::isa_name(core_cv3_hift_simdconv::best_isa()),
                exercised.empty() ? "none" : exercised.c_str());
}

TEST_CASE("CosyVoice3 HiFT SIMDCONV falls back when channel width cannot be vectorized",
          "[unit][tts][cosyvoice3][simdconv]") {
    constexpr int K = 7;
    constexpr int C = 30;
    std::vector<float> weights(static_cast<std::size_t>(K) * C * C);
    std::vector<float> bias(C);
    fill_rounding(weights, 101);
    fill_rounding(bias, 103);
    core_cv3_hift_simdconv::PackedConv conv;
    REQUIRE(conv.reset(weights.data(), bias.data(), K, C, 3, core_cv3_hift_simdconv::best_isa()));
    REQUIRE(conv.isa == core_cv3_hift_simdconv::Isa::scalar);
    REQUIRE(conv.width == 1);
}

#include "core/chatterbox_hift_simdconv.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

void fill_rounding(std::vector<float>& values, std::uint32_t seed) {
    std::uint32_t state = seed * 2654435761u + 12345u;
    for (auto& value : values) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<float>((state >> 8) & 0xFFFFFFu) / 8388608.0f - 1.0f;
    }
}

// Reference against unpacked ggml weights: ne=(K, IC, OC), bytes [OC][IC][K].
// Chatterbox HiFT uses symmetric zero padding and keeps the output length T.
void reference_symmetric_conv(const float* input, const float* weights, const float* bias, float* output, int T, int C,
                              int K, int dilation) {
    const int pad = (K * dilation - dilation) / 2;
    for (int t = 0; t < T; ++t) {
        for (int oc = 0; oc < C; ++oc) {
            float sum = bias[oc];
            for (int k = 0; k < K; ++k) {
                const int src_t = t + k * dilation - pad;
                if (src_t < 0 || src_t >= T)
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

void require_kernel(int K, int dilation, core_chatterbox_hift_simdconv::Isa isa, int n_threads) {
    constexpr int T = 17;
    constexpr int C = 32;
    std::vector<float> input(static_cast<std::size_t>(T) * C);
    std::vector<float> weights(static_cast<std::size_t>(K) * C * C);
    std::vector<float> bias(C);
    fill_rounding(input, 11u + static_cast<std::uint32_t>(K * 7 + dilation));
    fill_rounding(weights, 23u + static_cast<std::uint32_t>(K * 11 + dilation));
    fill_rounding(bias, 47u + static_cast<std::uint32_t>(K * 13 + dilation));

    std::vector<float> reference(static_cast<std::size_t>(T) * C);
    std::vector<float> scalar(reference.size());
    std::vector<float> actual(reference.size());
    reference_symmetric_conv(input.data(), weights.data(), bias.data(), reference.data(), T, C, K, dilation);

    core_chatterbox_hift_simdconv::PackedConv scalar_conv;
    REQUIRE(scalar_conv.reset(weights.data(), bias.data(), K, C, dilation,
                              core_chatterbox_hift_simdconv::Isa::scalar));
    scalar_conv.run(input.data(), scalar.data(), T, n_threads);
    INFO("K=" << K << " dilation=" << dilation << " scalar");
    REQUIRE(std::memcmp(reference.data(), scalar.data(), reference.size() * sizeof(float)) == 0);

    core_chatterbox_hift_simdconv::PackedConv conv;
    REQUIRE(conv.reset(weights.data(), bias.data(), K, C, dilation, isa));
    conv.run(input.data(), actual.data(), T, n_threads);
    INFO("K=" << K << " dilation=" << dilation << " isa=" << core_chatterbox_hift_simdconv::isa_name(isa)
              << " resolved=" << core_chatterbox_hift_simdconv::isa_name(conv.isa) << " threads=" << n_threads);
    REQUIRE(std::memcmp(scalar.data(), actual.data(), scalar.size() * sizeof(float)) == 0);
}

} // namespace

TEST_CASE("Chatterbox HiFT SIMDCONV scalar matches symmetric Conv1d", "[unit][tts][chatterbox][hift][simdconv]") {
    for (const int K : {3, 7, 11})
        for (const int dilation : {1, 3, 5})
            require_kernel(K, dilation, core_chatterbox_hift_simdconv::Isa::scalar, 4);
}

TEST_CASE("Chatterbox HiFT SIMDCONV SIMD is bit-identical to scalar", "[unit][tts][chatterbox][hift][simdconv]") {
    const std::array<core_chatterbox_hift_simdconv::Isa, 3> candidates{{
        core_chatterbox_hift_simdconv::Isa::neon,
        core_chatterbox_hift_simdconv::Isa::avx2,
        core_chatterbox_hift_simdconv::Isa::avx512f,
    }};
    std::string exercised;
    for (const auto isa : candidates) {
        if (!core_chatterbox_hift_simdconv::isa_available(isa))
            continue;
        exercised += exercised.empty() ? "" : ",";
        exercised += core_chatterbox_hift_simdconv::isa_name(isa);
        for (const int K : {3, 7, 11})
            for (const int dilation : {1, 3, 5})
                require_kernel(K, dilation, isa, 4);
    }
    std::printf("[chatterbox-hift-simdconv] best=%s ; SIMD exercised: %s\n",
                core_chatterbox_hift_simdconv::isa_name(core_chatterbox_hift_simdconv::best_isa()),
                exercised.empty() ? "none" : exercised.c_str());
}

TEST_CASE("Chatterbox HiFT SIMDCONV falls back for an unaligned channel count",
          "[unit][tts][chatterbox][hift][simdconv]") {
    constexpr int K = 7;
    constexpr int C = 30;
    std::vector<float> weights(static_cast<std::size_t>(K) * C * C);
    std::vector<float> bias(C);
    fill_rounding(weights, 101);
    fill_rounding(bias, 103);
    core_chatterbox_hift_simdconv::PackedConv conv;
    REQUIRE(conv.reset(weights.data(), bias.data(), K, C, 3, core_chatterbox_hift_simdconv::best_isa()));
    REQUIRE(conv.isa == core_chatterbox_hift_simdconv::Isa::scalar);
    REQUIRE(conv.width == 1);
}

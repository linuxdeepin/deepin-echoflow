#pragma once

#include "core/parallel_for.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CRISPASR_CPU_PACKED_CONV1D_X86 1
#else
#define CRISPASR_CPU_PACKED_CONV1D_X86 0
#endif

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define CRISPASR_CPU_PACKED_CONV1D_NEON 1
#else
#define CRISPASR_CPU_PACKED_CONV1D_NEON 0
#endif

namespace core_cpu_conv1d {

enum class Isa {
    scalar,
    neon,
    avx2,
    avx512f,
};

inline const char* isa_name(Isa isa) {
    switch (isa) {
    case Isa::neon:
        return "neon";
    case Isa::avx2:
        return "avx2";
    case Isa::avx512f:
        return "avx512f";
    default:
        return "scalar";
    }
}

inline int isa_width(Isa isa) {
    switch (isa) {
    case Isa::neon:
        return 4;
    case Isa::avx2:
        return 8;
    case Isa::avx512f:
        return 16;
    default:
        return 1;
    }
}

inline bool isa_available(Isa isa) {
    if (isa == Isa::scalar)
        return true;
#if CRISPASR_CPU_PACKED_CONV1D_NEON
    if (isa == Isa::neon)
        return true;
#endif
#if CRISPASR_CPU_PACKED_CONV1D_X86
    if (isa == Isa::avx512f)
        return __builtin_cpu_supports("avx512f");
    if (isa == Isa::avx2)
        return __builtin_cpu_supports("avx2");
#endif
    return false;
}

inline Isa best_isa() {
    static const Isa isa = [] {
#if CRISPASR_CPU_PACKED_CONV1D_X86
        if (isa_available(Isa::avx512f))
            return Isa::avx512f;
        if (isa_available(Isa::avx2))
            return Isa::avx2;
#endif
#if CRISPASR_CPU_PACKED_CONV1D_NEON
        if (isa_available(Isa::neon))
            return Isa::neon;
#endif
        return Isa::scalar;
    }();
    return isa;
}

// requested is a ceiling: fall back to a narrower SIMD ISA when C_out
// is not divisible by the requested vector width.
inline Isa resolve_isa(Isa requested, int output_channels) {
#if CRISPASR_CPU_PACKED_CONV1D_X86
    if (requested == Isa::avx512f && isa_available(Isa::avx512f) && output_channels % 16 == 0)
        return Isa::avx512f;
    if ((requested == Isa::avx512f || requested == Isa::avx2) && isa_available(Isa::avx2) && output_channels % 8 == 0)
        return Isa::avx2;
#endif
#if CRISPASR_CPU_PACKED_CONV1D_NEON
    if (requested == Isa::neon && isa_available(Isa::neon) && output_channels % 4 == 0)
        return Isa::neon;
#endif
    return Isa::scalar;
}

struct Identity {
    constexpr float operator()(float value) const noexcept { return value; }
};

// F32 Conv1d with packed weights.
// Input/output layout is time-major [T][C]. Original weights are [OC][IC][K],
// matching ggml's contiguous K dimension for ne=(K, IC, OC).
struct PackedConv1d {
    int kernel = 0;
    int input_channels = 0;
    int output_channels = 0;
    int dilation = 1;
    int left_padding = 0;
    int width = 1;
    Isa isa = Isa::scalar;
    std::vector<float> weights; // [OC_block][K][IC][lane]
    std::vector<float> bias;    // [OC]

    // Pack [OC][IC][K] into [OC_block][K][IC][lane] once at model load.
    bool reset(const float* weights_ocick, const float* bias_oc, int kernel_size, int n_input_channels,
               int n_output_channels, int dilation_value, int left_padding_value, Isa requested = best_isa()) {
        if (!weights_ocick || kernel_size <= 0 || n_input_channels <= 0 || n_output_channels <= 0 ||
            dilation_value <= 0 || left_padding_value < 0)
            return false;

        kernel = kernel_size;
        input_channels = n_input_channels;
        output_channels = n_output_channels;
        dilation = dilation_value;
        left_padding = left_padding_value;
        isa = resolve_isa(requested, output_channels);
        width = isa_width(isa);
        assert(output_channels % width == 0);

        weights.resize(static_cast<std::size_t>(kernel) * input_channels * output_channels);
        bias.assign(static_cast<std::size_t>(output_channels), 0.0f);
        if (bias_oc)
            std::memcpy(bias.data(), bias_oc, static_cast<std::size_t>(output_channels) * sizeof(float));

        for (int oc = 0; oc < output_channels; oc += width) {
            const int block = oc / width;
            for (int k = 0; k < kernel; ++k) {
                for (int ic = 0; ic < input_channels; ++ic) {
                    float* dst =
                        weights.data() + ((static_cast<std::size_t>(block) * kernel + k) * input_channels + ic) * width;
                    for (int lane = 0; lane < width; ++lane) {
                        const int out = oc + lane;
                        dst[lane] = weights_ocick[static_cast<std::size_t>(out) * input_channels * kernel +
                                                  static_cast<std::size_t>(ic) * kernel + k];
                    }
                }
            }
        }
        return true;
    }

    bool valid() const {
        return kernel > 0 && input_channels > 0 && output_channels > 0 && dilation > 0 && left_padding >= 0 &&
               width > 0 && output_channels % width == 0 &&
               weights.size() == static_cast<std::size_t>(kernel) * input_channels * output_channels &&
               bias.size() == static_cast<std::size_t>(output_channels);
    }

    template <class InputTransform, class OutputTransform>
    void run_rows(const float* input, float* output, int frames, int begin, int end, InputTransform input_transform,
                  OutputTransform output_transform) const;

    void run_rows(const float* input, float* output, int frames, int begin, int end) const {
        run_rows(input, output, frames, begin, end, Identity{}, Identity{});
    }

    template <class OutputTransform>
    void run_rows(const float* input, float* output, int frames, int begin, int end,
                  OutputTransform output_transform) const {
        run_rows(input, output, frames, begin, end, Identity{}, output_transform);
    }

    void run(const float* input, float* output, int frames, int n_threads = 1) const {
        run(input, output, frames, n_threads, Identity{}, Identity{});
    }

    // Common case: a post-conv transform, e.g. F0 Conv1d + ELU.
    template <class OutputTransform>
    void run(const float* input, float* output, int frames, int n_threads, OutputTransform output_transform) const {
        run(input, output, frames, n_threads, Identity{}, output_transform);
    }

    template <class InputTransform, class OutputTransform>
    void run(const float* input, float* output, int frames, int n_threads, InputTransform input_transform,
             OutputTransform output_transform) const {
        core_parallel::for_each_chunk(frames, n_threads, [&](int begin, int end) {
            run_rows(input, output, frames, begin, end, input_transform, output_transform);
        });
    }
};

namespace detail {

// Keep the K -> IC accumulation order identical across scalar/SIMD.
// Input/output transforms are scalar policies intentionally applied per value/lane.

#if defined(__clang__)
#define CRISPASR_CPU_PACKED_CONV1D_NO_CONTRACT
#define CRISPASR_CPU_PACKED_CONV1D_CLANG_NO_CONTRACT _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#define CRISPASR_CPU_PACKED_CONV1D_NO_CONTRACT __attribute__((optimize("fp-contract=off")))
#define CRISPASR_CPU_PACKED_CONV1D_CLANG_NO_CONTRACT
#else
#define CRISPASR_CPU_PACKED_CONV1D_NO_CONTRACT
#define CRISPASR_CPU_PACKED_CONV1D_CLANG_NO_CONTRACT
#endif

template <class InputTransform, class OutputTransform>
CRISPASR_CPU_PACKED_CONV1D_NO_CONTRACT inline void run_scalar_rows(const PackedConv1d& conv, const float* input,
                                                                   float* output, int frames, int begin, int end,
                                                                   InputTransform input_transform,
                                                                   OutputTransform output_transform) {
    CRISPASR_CPU_PACKED_CONV1D_CLANG_NO_CONTRACT
    assert(conv.width == 1);
    for (int t = begin; t < end; ++t) {
        float* out = output + static_cast<std::size_t>(t) * conv.output_channels;
        for (int oc = 0; oc < conv.output_channels; ++oc) {
            float sum = conv.bias[oc];
            const float* block_weights =
                conv.weights.data() + static_cast<std::size_t>(oc) * conv.kernel * conv.input_channels;
            for (int k = 0; k < conv.kernel; ++k) {
                const int src_t = t + k * conv.dilation - conv.left_padding;
                if (src_t < 0 || src_t >= frames)
                    continue;
                const float* in = input + static_cast<std::size_t>(src_t) * conv.input_channels;
                const float* wk = block_weights + static_cast<std::size_t>(k) * conv.input_channels;
                for (int ic = 0; ic < conv.input_channels; ++ic)
                    sum += wk[ic] * input_transform(in[ic]);
            }
            out[oc] = output_transform(sum);
        }
    }
}

#if CRISPASR_CPU_PACKED_CONV1D_NEON
template <class InputTransform, class OutputTransform>
CRISPASR_CPU_PACKED_CONV1D_NO_CONTRACT inline void run_neon_rows(const PackedConv1d& conv, const float* input,
                                                                 float* output, int frames, int begin, int end,
                                                                 InputTransform input_transform,
                                                                 OutputTransform output_transform) {
    CRISPASR_CPU_PACKED_CONV1D_CLANG_NO_CONTRACT
    constexpr int W = 4;
    assert(conv.width == W);
    for (int t = begin; t < end; ++t) {
        float* out = output + static_cast<std::size_t>(t) * conv.output_channels;
        for (int block = 0; block < conv.output_channels / W; ++block) {
            float32x4_t sum = vld1q_f32(conv.bias.data() + block * W);
            const float* bw =
                conv.weights.data() + static_cast<std::size_t>(block) * conv.kernel * conv.input_channels * W;
            for (int k = 0; k < conv.kernel; ++k) {
                const int src_t = t + k * conv.dilation - conv.left_padding;
                if (src_t < 0 || src_t >= frames)
                    continue;
                const float* in = input + static_cast<std::size_t>(src_t) * conv.input_channels;
                const float* wk = bw + static_cast<std::size_t>(k) * conv.input_channels * W;
                for (int ic = 0; ic < conv.input_channels; ++ic) {
                    const float32x4_t w = vld1q_f32(wk + static_cast<std::size_t>(ic) * W);
                    const float32x4_t x = vdupq_n_f32(input_transform(in[ic]));
                    sum = vaddq_f32(sum, vmulq_f32(w, x));
                }
            }
            std::array<float, W> lanes;
            vst1q_f32(lanes.data(), sum);
            for (int lane = 0; lane < W; ++lane)
                out[block * W + lane] = output_transform(lanes[lane]);
        }
    }
}
#endif

#if CRISPASR_CPU_PACKED_CONV1D_X86
#if defined(__clang__)
#define CRISPASR_CPU_PACKED_CONV1D_TARGET_AVX2 __attribute__((target("avx2")))
#define CRISPASR_CPU_PACKED_CONV1D_TARGET_AVX512 __attribute__((target("avx512f")))
#else
#define CRISPASR_CPU_PACKED_CONV1D_TARGET_AVX2 __attribute__((target("avx2")))
#define CRISPASR_CPU_PACKED_CONV1D_TARGET_AVX512 __attribute__((target("avx512f")))
#endif

template <class InputTransform, class OutputTransform>
CRISPASR_CPU_PACKED_CONV1D_TARGET_AVX2 CRISPASR_CPU_PACKED_CONV1D_NO_CONTRACT inline void run_avx2_rows(
    const PackedConv1d& conv, const float* input, float* output, int frames, int begin, int end,
    InputTransform input_transform, OutputTransform output_transform) {
    CRISPASR_CPU_PACKED_CONV1D_CLANG_NO_CONTRACT
    constexpr int W = 8;
    assert(conv.width == W);
    for (int t = begin; t < end; ++t) {
        float* out = output + static_cast<std::size_t>(t) * conv.output_channels;
        for (int block = 0; block < conv.output_channels / W; ++block) {
            __m256 sum = _mm256_loadu_ps(conv.bias.data() + block * W);
            const float* bw =
                conv.weights.data() + static_cast<std::size_t>(block) * conv.kernel * conv.input_channels * W;
            for (int k = 0; k < conv.kernel; ++k) {
                const int src_t = t + k * conv.dilation - conv.left_padding;
                if (src_t < 0 || src_t >= frames)
                    continue;
                const float* in = input + static_cast<std::size_t>(src_t) * conv.input_channels;
                const float* wk = bw + static_cast<std::size_t>(k) * conv.input_channels * W;
                for (int ic = 0; ic < conv.input_channels; ++ic) {
                    const __m256 w = _mm256_loadu_ps(wk + static_cast<std::size_t>(ic) * W);
                    const __m256 x = _mm256_set1_ps(input_transform(in[ic]));
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(w, x));
                }
            }
            std::array<float, W> lanes;
            _mm256_storeu_ps(lanes.data(), sum);
            for (int lane = 0; lane < W; ++lane)
                out[block * W + lane] = output_transform(lanes[lane]);
        }
    }
}

template <class InputTransform, class OutputTransform>
CRISPASR_CPU_PACKED_CONV1D_TARGET_AVX512 CRISPASR_CPU_PACKED_CONV1D_NO_CONTRACT inline void run_avx512_rows(
    const PackedConv1d& conv, const float* input, float* output, int frames, int begin, int end,
    InputTransform input_transform, OutputTransform output_transform) {
    CRISPASR_CPU_PACKED_CONV1D_CLANG_NO_CONTRACT
    constexpr int W = 16;
    assert(conv.width == W);
    for (int t = begin; t < end; ++t) {
        float* out = output + static_cast<std::size_t>(t) * conv.output_channels;
        for (int block = 0; block < conv.output_channels / W; ++block) {
            __m512 sum = _mm512_loadu_ps(conv.bias.data() + block * W);
            const float* bw =
                conv.weights.data() + static_cast<std::size_t>(block) * conv.kernel * conv.input_channels * W;
            for (int k = 0; k < conv.kernel; ++k) {
                const int src_t = t + k * conv.dilation - conv.left_padding;
                if (src_t < 0 || src_t >= frames)
                    continue;
                const float* in = input + static_cast<std::size_t>(src_t) * conv.input_channels;
                const float* wk = bw + static_cast<std::size_t>(k) * conv.input_channels * W;
                for (int ic = 0; ic < conv.input_channels; ++ic) {
                    const __m512 w = _mm512_loadu_ps(wk + static_cast<std::size_t>(ic) * W);
                    const __m512 x = _mm512_set1_ps(input_transform(in[ic]));
                    sum = _mm512_add_ps(sum, _mm512_mul_ps(w, x));
                }
            }
            std::array<float, W> lanes;
            _mm512_storeu_ps(lanes.data(), sum);
            for (int lane = 0; lane < W; ++lane)
                out[block * W + lane] = output_transform(lanes[lane]);
        }
    }
}
#endif

#undef CRISPASR_CPU_PACKED_CONV1D_TARGET_AVX2
#undef CRISPASR_CPU_PACKED_CONV1D_TARGET_AVX512
#undef CRISPASR_CPU_PACKED_CONV1D_CLANG_NO_CONTRACT
#undef CRISPASR_CPU_PACKED_CONV1D_NO_CONTRACT

} // namespace detail

template <class InputTransform, class OutputTransform>
inline void PackedConv1d::run_rows(const float* input, float* output, int frames, int begin, int end,
                                   InputTransform input_transform, OutputTransform output_transform) const {
    assert(valid());
    assert(input && output && frames >= 0 && begin >= 0 && begin <= end && end <= frames);
    switch (isa) {
#if CRISPASR_CPU_PACKED_CONV1D_X86
    case Isa::avx512f:
        detail::run_avx512_rows(*this, input, output, frames, begin, end, input_transform, output_transform);
        return;
    case Isa::avx2:
        detail::run_avx2_rows(*this, input, output, frames, begin, end, input_transform, output_transform);
        return;
#endif
#if CRISPASR_CPU_PACKED_CONV1D_NEON
    case Isa::neon:
        detail::run_neon_rows(*this, input, output, frames, begin, end, input_transform, output_transform);
        return;
#endif
    default:
        detail::run_scalar_rows(*this, input, output, frames, begin, end, input_transform, output_transform);
        return;
    }
}

} // namespace core_cpu_conv1d

#undef CRISPASR_CPU_PACKED_CONV1D_X86
#undef CRISPASR_CPU_PACKED_CONV1D_NEON

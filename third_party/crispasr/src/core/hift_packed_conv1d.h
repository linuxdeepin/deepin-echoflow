#pragma once

#include "core/cpu_packed_conv1d.h"

namespace core_hift_conv1d {

using Isa = core_cpu_conv1d::Isa;
using core_cpu_conv1d::best_isa;
using core_cpu_conv1d::isa_available;
using core_cpu_conv1d::isa_name;
using core_cpu_conv1d::isa_width;
using core_cpu_conv1d::resolve_isa;

enum class Padding {
    causal,
    symmetric,
};

inline int padding_left(Padding padding, int kernel, int dilation) {
    const int full = (kernel - 1) * dilation;
    return padding == Padding::causal ? full : full / 2;
}

// Square HiFT Conv1d (IC == OC) on time-major [T][C] buffers. Model adapters
// choose the padding policy while packed SIMD storage/execution stays shared.
struct PackedConv : core_cpu_conv1d::PackedConv1d {
    int channels = 0;

    bool reset(const float* weights_ocick, const float* bias_oc, int kernel_size, int n_channels, int dilation_value,
               Padding padding, Isa requested = best_isa()) {
        channels = n_channels;
        if (!core_cpu_conv1d::PackedConv1d::reset(weights_ocick, bias_oc, kernel_size, n_channels, n_channels,
                                                  dilation_value, padding_left(padding, kernel_size, dilation_value),
                                                  requested)) {
            channels = 0;
            return false;
        }
        return true;
    }
};

} // namespace core_hift_conv1d

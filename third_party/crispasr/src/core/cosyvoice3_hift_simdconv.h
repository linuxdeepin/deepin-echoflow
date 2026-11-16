#pragma once

#include "core/hift_packed_conv1d.h"

namespace core_cv3_hift_simdconv {

using Isa = core_hift_conv1d::Isa;
using core_hift_conv1d::best_isa;
using core_hift_conv1d::isa_available;
using core_hift_conv1d::isa_name;
using core_hift_conv1d::isa_width;
using core_hift_conv1d::resolve_isa;

// CosyVoice3 policy adapter: square HiFT Conv1d with causal left padding.
struct PackedConv : core_hift_conv1d::PackedConv {
    bool reset(const float* weights_ocick, const float* bias_oc, int kernel_size, int n_channels, int dilation_value,
               Isa requested = best_isa()) {
        return core_hift_conv1d::PackedConv::reset(weights_ocick, bias_oc, kernel_size, n_channels, dilation_value,
                                                   core_hift_conv1d::Padding::causal, requested);
    }
};

} // namespace core_cv3_hift_simdconv

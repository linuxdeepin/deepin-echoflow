#pragma once

#include "core/cpu_packed_conv1d.h"

#include <cassert>
#include <cmath>

namespace core_chatterbox_f0 {

using Isa = core_cpu_conv1d::Isa;
using core_cpu_conv1d::best_isa;
using core_cpu_conv1d::isa_available;

struct Elu {
    float operator()(float value) const { return value < 0.0f ? std::exp(value) - 1.0f : value; }
};

// Non-owning view of one k=3, p=1 Conv1d + ELU operation.
struct Conv1dEluK3 {
    const float* input;
    const float* weights;
    const float* bias;
    float* output;
    int frames;
    int input_channels;
    int output_channels;

    void run(int n_threads, Isa isa = best_isa()) const;
    void run_scalar(int n_threads = 1) const;
};

inline void Conv1dEluK3::run_scalar(int n_threads) const {
    core_cpu_conv1d::PackedConv1d conv;
    const bool packed = conv.reset(weights, bias, 3, input_channels, output_channels, 1, 1, Isa::scalar);
    assert(packed);

    conv.run(input, output, frames, n_threads, Elu{});
}

inline void Conv1dEluK3::run(int n_threads, Isa isa) const {
    core_cpu_conv1d::PackedConv1d conv;
    const bool packed = conv.reset(weights, bias, 3, input_channels, output_channels, 1, 1, isa);
    assert(packed);

    conv.run(input, output, frames, n_threads, Elu{});
}

} // namespace core_chatterbox_f0
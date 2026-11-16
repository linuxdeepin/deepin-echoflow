#pragma once

#include "core/cpu_ops.h"
#include "core/hift_packed_conv1d.h"

#include "ggml.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace core_hift_simdconv {

using core_hift_conv1d::PackedConv;

// Shared load-time pack/validation/rollback helper. PackedConvT may be a thin
// model adapter around PackedConv (for example causal vs symmetric padding).
class Packer {
public:
    explicit Packer(bool enabled) : ok_(enabled) {}

    template <class PackedConvT> void add(PackedConvT& dst, ggml_tensor* weights, ggml_tensor* bias, int dilation) {
        if (!ok_)
            return;
        if (!weights || ggml_n_dims(weights) != 3 || weights->ne[0] <= 0 || weights->ne[1] <= 0 ||
            weights->ne[1] != weights->ne[2]) {
            ok_ = false;
            return;
        }

        const int kernel = static_cast<int>(weights->ne[0]);
        const int channels = static_cast<int>(weights->ne[1]);
        std::vector<float> wf = core_cpu::to_f32(weights);
        std::vector<float> bf =
            bias ? core_cpu::to_f32(bias) : std::vector<float>(static_cast<std::size_t>(channels), 0.0f);

        if (wf.size() != static_cast<std::size_t>(kernel) * channels * channels ||
            bf.size() != static_cast<std::size_t>(channels) ||
            !dst.reset(wf.data(), bf.data(), kernel, channels, dilation)) {
            ok_ = false;
            return;
        }

        packed_.push_back(&dst);
        ++count_;
    }

    bool finish(int expected) {
        const bool complete = ok_ && count_ == expected;
        if (!complete) {
            for (std::size_t i = 0; i < packed_.size(); ++i)
                *packed_[i] = PackedConv();
            packed_.clear();
        }
        return complete;
    }

    int count() const { return count_; }

private:
    bool ok_ = false;
    int count_ = 0;
    std::vector<PackedConv*> packed_;
};

// Normal HiFT ggml tensors are ne=(T,C), so bytes are channel-major [C][T].
// PackedConv wants channels contiguous; transpose+cont materialises ne=(C,T)
// with time-major [T][C] bytes for the whole ResBlock island.
inline ggml_tensor* to_time_major(ggml_context* ctx, ggml_tensor* x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

inline ggml_tensor* from_time_major(ggml_context* ctx, ggml_tensor* x_tm) {
    return ggml_cont(ctx, ggml_transpose(ctx, x_tm));
}

inline void conv_op(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    const PackedConv* conv = static_cast<const PackedConv*>(userdata);
    GGML_ASSERT(conv && conv->valid());
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->ne[0] == conv->channels && dst->ne[0] == conv->channels);
    GGML_ASSERT(src->ne[1] == dst->ne[1]);

    const int frames = static_cast<int>(src->ne[1]);
    const int begin = (frames * ith) / nth;
    const int end = (frames * (ith + 1)) / nth;
    conv->run_rows(static_cast<const float*>(src->data), static_cast<float*>(dst->data), frames, begin, end);
}

inline ggml_tensor* conv_tm(ggml_context* ctx, ggml_tensor* x_tm, const PackedConv& conv, int n_threads) {
    GGML_ASSERT(conv.valid());
    GGML_ASSERT(x_tm->ne[0] == conv.channels);
    return ggml_map_custom1(ctx, x_tm, conv_op, std::max(1, n_threads), const_cast<PackedConv*>(&conv));
}

} // namespace core_hift_simdconv

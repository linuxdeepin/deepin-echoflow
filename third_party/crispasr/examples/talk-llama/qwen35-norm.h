#pragma once
#include "ggml.h"
#include <cmath>

namespace llama_qwen35 {
// Transformers 5.6.0 l2norm: x * rsqrt(sum(x*x) + 1e-6).
// ggml_l2_norm instead clamps sqrt(sum) to epsilon. RMS with epsilon / D,
// followed by 1/sqrt(D), gives the released formula using portable kernels.
inline ggml_tensor* l2_norm(ggml_context* ctx, ggml_tensor* input) {
    const float width = (float)input->ne[0];
    return ggml_scale(ctx, ggml_rms_norm(ctx, input, 1e-6f / width), 1.0f / std::sqrt(width));
}
} // namespace llama_qwen35

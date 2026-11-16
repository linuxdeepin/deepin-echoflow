// src/core/beatrice_ops.h — graph helpers shared by the Beatrice v2 components.
//
// PitchEstimator and PhoneExtractor are different networks but are built from
// the same ConvNeXtStack primitives, so these live in one place rather than
// being copied per backend (see the multi-surface-dispatch lesson in
// docs/contributing.md: a fix applied to one copy and not the other is the
// failure mode).
//
// LAYOUT CONVENTION, and it is load-bearing: tensors are ggml ne = [time,
// channels] (time fastest), byte-identical to torch's [batch, channels, time]
// dump. Blocks transpose to [channels, time] internally for LayerNorm and the
// pointwise projections -- ggml_norm normalises over ne0 and mul_mat contracts
// over ne0 -- then back. That mirrors the transpose(1, 2) in the reference.
// Transposing at the capture boundary instead has caused four separate bugs
// across the RVC and Beatrice ports, each showing ~0 cosine on a correct graph.

#pragma once

#include "ggml.h"

namespace beatrice_ops {

// torch nn.LayerNorm default
constexpr float kLayerNormEps = 1e-5f;

// Pointwise (1x1) projection over the channel axis.
// x is [T, Cin]; w is [Cin, Cout]; returns [T, Cout].
inline ggml_tensor* pointwise(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [Cin, T]
    ggml_tensor* y = ggml_mul_mat(ctx, w, xt);                // [Cout, T]
    if (b)
        y = ggml_add(ctx, y, b);
    return ggml_cont(ctx, ggml_transpose(ctx, y)); // [T, Cout]
}

// LayerNorm over the channel axis of an [T, C] tensor.
//
// `g`/`b` may be null. That is NOT a convenience: inside a ConvNeXt block the
// affine has been folded into pwconv1 by merge_weights and is not stored at
// all, while the stack-level norm and final_layer_norm keep theirs. Using one
// form in both places is wrong in exactly one of them.
inline ggml_tensor* layernorm_tc(ggml_context* ctx, ggml_tensor* x, ggml_tensor* g, ggml_tensor* b) {
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C, T]
    xt = ggml_norm(ctx, xt, kLayerNormEps);                   // normalises over ne0 = C
    if (g)
        xt = ggml_mul(ctx, xt, g);
    if (b)
        xt = ggml_add(ctx, xt, b);
    return ggml_cont(ctx, ggml_transpose(ctx, xt)); // [T, C]
}

// Trim `n` frames off the END of a [T, C] tensor -- the CausalConv1d trim.
// Note trim != padding whenever the layer has delay > 0.
inline ggml_tensor* trim_tail(ggml_context* ctx, ggml_tensor* x, int n) {
    if (n <= 0)
        return x;
    const int64_t T = x->ne[0] - n;
    return ggml_cont(ctx, ggml_view_2d(ctx, x, T, x->ne[1], x->nb[1], 0));
}

// One ConvNeXt block's convolutional branch, shared by both components:
//   depthwise causal conv -> LayerNorm (no affine) -> pw1 -> gelu -> pw2 -> +residual
// `gamma`/`pre_scale`/`post_scale` are identically 1 after fusion and are
// deliberately absent from the GGUF, so nothing applies them here.
//
// ggml_gelu is the TANH approximation (verified in ggml-cpu/vec.h), which is
// what the reference uses via approximate="tanh". ggml_gelu_erf would be a
// different function -- and one that still scores cos 0.9999996, so it cannot
// be caught by a cosine check.
inline ggml_tensor* convnext_conv_branch(ggml_context* ctx, ggml_tensor* x, ggml_tensor* dw_w, ggml_tensor* dw_b,
                                         ggml_tensor* pw1_w, ggml_tensor* pw1_b, ggml_tensor* pw2_w, ggml_tensor* pw2_b,
                                         int dw_padding, int dw_trim, int channels, ggml_tensor** dwconv_out) {
    ggml_tensor* identity = x;
    ggml_tensor* h = ggml_conv_1d_dw(ctx, dw_w, x, 1, dw_padding, 1);
    h = ggml_add(ctx, h, ggml_reshape_2d(ctx, dw_b, 1, channels));
    h = trim_tail(ctx, h, dw_trim);
    if (dwconv_out)
        *dwconv_out = h;
    h = layernorm_tc(ctx, h, nullptr, nullptr);
    h = pointwise(ctx, h, pw1_w, pw1_b);
    h = ggml_gelu(ctx, h);
    h = pointwise(ctx, h, pw2_w, pw2_b);
    return ggml_add(ctx, h, identity);
}

} // namespace beatrice_ops

// src/core/gru.h — bidirectional GRU helpers for ggml graphs.
//
// Sibling of core/lstm.h; same storage convention, same per-step graph shape,
// same output-container write pattern. Read that file first if this one is new
// to you — only the cell math differs.
//
// PyTorch nn.GRU forward (single layer, single direction):
//
//   r_t = sigmoid(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
//   z_t = sigmoid(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
//   n_t = tanh   (W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))
//   h_t = (1 - z_t) * n_t + z_t * h_{t-1}
//
// ⚠ TWO TRAPS, both of which silently produce plausible-but-wrong output:
//
//  1. **r_t multiplies the RECURRENT TERM, not h_{t-1}.** PyTorch applies the
//     reset gate AFTER the W_hn matmul and AFTER adding b_hn — i.e.
//     `r * (W_hn h + b_hn)`, not `W_hn (r * h)`. The textbook/original GRU
//     formulation does the latter. They are NOT equivalent. This is the single
//     most common GRU porting bug.
//
//  2. **b_ih and b_hh cannot be pre-summed.** For LSTM they can (both are added
//     into the same pre-activation), which is why core/lstm.h is free to fold
//     them. Here b_hn sits INSIDE the r_t product while b_in sits outside, so
//     folding them changes the result. Keep them separate.
//
// Gate order along the 3H axis is PyTorch's (r, z, n).
//
// Storage convention (mirrors core/lstm.h, with 4H -> 3H):
//
//   weight_ih_l0          ne = (input_size,  3 * hidden_size)   F16/F32
//   weight_hh_l0          ne = (hidden_size, 3 * hidden_size)   F16/F32
//   bias_ih_l0            ne = (3 * hidden_size,)               F32
//   bias_hh_l0            ne = (3 * hidden_size,)               F32
//   *_reverse             same shapes, for the backward direction
//
// As in core/lstm.h the input projection (W_ih @ X + b_ih) is hoisted out of
// the loop and computed for all T at once; only the recurrent matmul is
// genuinely sequential. At t=0, h_{-1} = 0, so the W_hh term vanishes — but
// NOTE the b_hh term does NOT vanish (it is still inside the r * (...) product
// for the n gate), which is why the t=0 branch below still adds b_hh rather
// than skipping it.

#pragma once

#include "ggml.h"

#include <cstddef>

namespace core_gru {

// Single-direction GRU forward over T timesteps.
//
//   ctx          per-graph ggml context (no_alloc=true)
//   gf           graph being built — cpy ops appended via
//                ggml_build_forward_expand so writes precede downstream reads
//   X            input, ne = (input_size, T)  F32
//   W_ih         ne = (input_size,  3H)        F16/F32
//   W_hh         ne = (hidden_size, 3H)        F16/F32
//   b_ih, b_hh   ne = (3H,)                    F32
//   H            hidden_size
//   reverse      iterate t = T-1 .. 0 (use the *_reverse weights)
//
// Returns the GRU output as a contiguous (H, T) F32 tensor.
static inline ggml_tensor* gru_unidir(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* X, ggml_tensor* W_ih,
                                      ggml_tensor* W_hh, ggml_tensor* b_ih, ggml_tensor* b_hh, int H, bool reverse) {
    const int T = (int)X->ne[1];
    const int H3 = 3 * H;

    // Input projection over all T timesteps at once. proj_x ne = (3H, T)
    ggml_tensor* proj_x = ggml_mul_mat(ctx, W_ih, X);
    proj_x = ggml_add(ctx, proj_x, b_ih);

    ggml_tensor* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);

    ggml_tensor* h = nullptr;

    const size_t row_stride_3h = proj_x->nb[1];
    const size_t row_stride_h = output->nb[1];
    const size_t f32_size = ggml_type_size(GGML_TYPE_F32);

    const int t0 = reverse ? T - 1 : 0;
    const int dt = reverse ? -1 : 1;
    for (int step = 0; step < T; step++) {
        const int t = t0 + step * dt;

        // Input-side pre-activations for this timestep: (3H, 1) F32.
        ggml_tensor* px = ggml_view_2d(ctx, proj_x, H3, 1, row_stride_3h, (size_t)t * row_stride_3h);

        // Recurrent-side: W_hh @ h_{t-1} + b_hh. At t=0 h is zero so the matmul
        // vanishes, but b_hh does NOT (see the header note) — it still feeds the
        // n gate inside the reset product.
        ggml_tensor* rh = h ? ggml_add(ctx, ggml_mul_mat(ctx, W_hh, h), b_hh) : b_hh;

        // Split both sides into (r, z, n) stripes of width H.
        const size_t px_stride = px->nb[1];
        ggml_tensor* px_r = ggml_view_2d(ctx, px, H, 1, px_stride, (size_t)0 * H * f32_size);
        ggml_tensor* px_z = ggml_view_2d(ctx, px, H, 1, px_stride, (size_t)1 * H * f32_size);
        ggml_tensor* px_n = ggml_view_2d(ctx, px, H, 1, px_stride, (size_t)2 * H * f32_size);

        // b_hh alone is 1-D (3H,) at t=0; the summed form is (3H,1). Views must
        // use the right stride for each case, so normalise to 2-D first.
        ggml_tensor* rh2 = h ? rh : ggml_reshape_2d(ctx, rh, H3, 1);
        const size_t rh_stride = rh2->nb[1];
        ggml_tensor* rh_r = ggml_view_2d(ctx, rh2, H, 1, rh_stride, (size_t)0 * H * f32_size);
        ggml_tensor* rh_z = ggml_view_2d(ctx, rh2, H, 1, rh_stride, (size_t)1 * H * f32_size);
        ggml_tensor* rh_n = ggml_view_2d(ctx, rh2, H, 1, rh_stride, (size_t)2 * H * f32_size);

        ggml_tensor* r = ggml_sigmoid(ctx, ggml_add(ctx, px_r, rh_r));
        ggml_tensor* z = ggml_sigmoid(ctx, ggml_add(ctx, px_z, rh_z));

        // n = tanh(px_n + r * rh_n)  <-- reset gate multiplies the RECURRENT
        // term (already including b_hn), NOT h_{t-1}. See trap 1.
        ggml_tensor* n = ggml_tanh(ctx, ggml_add(ctx, px_n, ggml_mul(ctx, r, rh_n)));

        // h_t = (1 - z) * n + z * h_{t-1}; at t=0 the second term vanishes.
        ggml_tensor* one_minus_z = ggml_scale_bias(ctx, z, -1.0f, 1.0f);
        ggml_tensor* h_new = ggml_mul(ctx, one_minus_z, n);
        if (h)
            h_new = ggml_add(ctx, h_new, ggml_mul(ctx, z, h));
        h = h_new;

        ggml_tensor* slot = ggml_view_2d(ctx, output, H, 1, row_stride_h, (size_t)t * row_stride_h);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h, slot));
    }
    return output;
}

// Bidirectional GRU. Two unidirectional passes concatenated along the feature
// dim: output ne = (2H, T), forward H first then backward H — matching
// PyTorch's nn.GRU(bidirectional=True) output layout.
static inline ggml_tensor* gru_bidir(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* X, ggml_tensor* W_ih_f,
                                     ggml_tensor* W_hh_f, ggml_tensor* b_ih_f, ggml_tensor* b_hh_f, ggml_tensor* W_ih_r,
                                     ggml_tensor* W_hh_r, ggml_tensor* b_ih_r, ggml_tensor* b_hh_r, int H) {
    ggml_tensor* fwd = gru_unidir(ctx, gf, X, W_ih_f, W_hh_f, b_ih_f, b_hh_f, H, /*reverse=*/false);
    ggml_tensor* bwd = gru_unidir(ctx, gf, X, W_ih_r, W_hh_r, b_ih_r, b_hh_r, H, /*reverse=*/true);
    return ggml_concat(ctx, fwd, bwd, /*dim=*/0);
}

} // namespace core_gru

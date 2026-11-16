// core/ecapa_tdnn.h — shared ECAPA-TDNN speaker encoder graph.
//
// The Qwen3-TTS speaker encoder (Qwen3TTSSpeakerEncoder: initial TDNN, three
// SE-Res2Net blocks at dilation 2/3/4, multi-layer feature aggregation,
// attentive-statistics pooling, FC) is used verbatim by more than one backend
// — Confucius4-TTS imports the very same class, differing only in the input
// dimension (w2v-BERT features, 1024) and the output dimension (1280).
//
// Extracted from the validated qwen3_tts.cpp implementation so both can share
// one graph rather than keeping two copies in step.  The only generalisation is
// the Res2Net chunk width, which qwen3 hardcoded to 64 (its C/8): it is now
// derived from the tensor, so any channel count works.
//
// Layout convention throughout is channels-first [C, T].
//
// Weights are bound by the caller, which owns the GGUF naming:
//   qwen3-tts        speaker.blocks.N.*        (mel input, 128)
//   confucius4-tts   speaker_encoder.blocks.N.*(w2v-BERT input, 1024)

#pragma once

#include "ggml.h"

#include <vector>

namespace core_ecapa {

struct tdnn_w {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};

struct res2net_w {
    tdnn_w blocks[7]; // scale=8 → 7 inner TDNNs
};

struct se_w {
    ggml_tensor* c1w = nullptr;
    ggml_tensor* c1b = nullptr;
    ggml_tensor* c2w = nullptr;
    ggml_tensor* c2b = nullptr;
};

struct se_res2net_w {
    tdnn_w tdnn1, tdnn2;
    res2net_w res2net;
    se_w se;
};

struct asp_w {
    tdnn_w tdnn;                   // (3C → attention_channels, k=1)
    ggml_tensor* conv_w = nullptr; // (attention_channels → C, k=1)
    ggml_tensor* conv_b = nullptr;
};

struct model {
    tdnn_w blk0;         // initial TDNN, k=5, d=1
    se_res2net_w blk[3]; // SE-Res2Net, d = 2/3/4
    tdnn_w mfa;          // multi-layer feature aggregation, k=1
    asp_w asp;           // attentive-statistics pooling
    ggml_tensor* fc_w = nullptr;
    ggml_tensor* fc_b = nullptr; // 2C → enc_dim
    bool loaded = false;
};

// Conv1d with symmetric REFLECT padding ("same", PyTorch padding_mode='reflect').
inline ggml_tensor* same_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int dilation) {
    const int K = (int)w->ne[0];
    const int pad = (K - 1) * dilation / 2;
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C, T] → [T, C]
    if (pad > 0)
        x = ggml_pad_reflect_1d(ctx, x, pad, pad);
    x = ggml_conv_1d(ctx, w, x, 1, 0, dilation); // already padded, so p0 = 0
    x = ggml_cont(ctx, ggml_transpose(ctx, x));  // [C_out, T_out]
    if (b)
        x = ggml_add(ctx, x, b);
    return x;
}

inline ggml_tensor* tdnn_block(ggml_context* ctx, ggml_tensor* x, const tdnn_w& t, int dilation) {
    return ggml_relu(ctx, same_conv1d(ctx, x, t.w, t.b, dilation));
}

inline ggml_tensor* se_block(ggml_context* ctx, ggml_tensor* x, const se_w& se) {
    const int T = (int)x->ne[1];
    ggml_tensor* m =
        ggml_cont(ctx, ggml_transpose(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, x))),
                                                      1.0f / T))); // [C, 1]
    auto w1 = ggml_reshape_2d(ctx, se.c1w, se.c1w->ne[1], se.c1w->ne[2]);
    ggml_tensor* h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w1, m), se.c1b));
    auto w2 = ggml_reshape_2d(ctx, se.c2w, se.c2w->ne[1], se.c2w->ne[2]);
    ggml_tensor* sc = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w2, h), se.c2b));
    return ggml_mul(ctx, x, ggml_repeat(ctx, sc, x));
}

inline ggml_tensor* res2net_block(ggml_context* ctx, ggml_tensor* x, const res2net_w& r, int dilation) {
    const int T = (int)x->ne[1];
    const int C = (int)x->ne[0];
    const int chunk = C / 8; // scale = 8; derived, not hardcoded
    ggml_tensor* outs[8];
    for (int i = 0; i < 8; i++) {
        ggml_tensor* ci = ggml_cont(ctx, ggml_view_2d(ctx, x, chunk, T, x->nb[1], (size_t)i * chunk * sizeof(float)));
        if (i == 0) {
            outs[i] = ci;
            continue;
        }
        ggml_tensor* in = (i == 1) ? ci : ggml_add(ctx, ci, outs[i - 1]);
        outs[i] = tdnn_block(ctx, in, r.blocks[i - 1], dilation);
    }
    ggml_tensor* out = outs[0];
    for (int i = 1; i < 8; i++)
        out = ggml_concat(ctx, out, outs[i], 0);
    return out;
}

inline ggml_tensor* se_res2net(ggml_context* ctx, ggml_tensor* x, const se_res2net_w& blk, int d) {
    ggml_tensor* res = x;
    x = tdnn_block(ctx, x, blk.tdnn1, 1);
    x = res2net_block(ctx, x, blk.res2net, d);
    x = tdnn_block(ctx, x, blk.tdnn2, 1);
    x = se_block(ctx, x, blk.se);
    return ggml_add(ctx, x, res);
}

inline ggml_tensor* asp_block(ggml_context* ctx, ggml_tensor* x, const asp_w& asp) {
    const int T = (int)x->ne[1];
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* m1C = ggml_scale(ctx, ggml_sum_rows(ctx, xT), 1.0f / T);
    ggml_tensor* mC1 = ggml_cont(ctx, ggml_transpose(ctx, m1C));
    ggml_tensor* mCT = ggml_repeat(ctx, mC1, x);
    ggml_tensor* d2 = ggml_mul(ctx, ggml_sub(ctx, x, mCT), ggml_sub(ctx, x, mCT));
    ggml_tensor* s1C =
        ggml_sqrt(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, d2))), 1.0f / T));
    ggml_tensor* sCT = ggml_repeat(ctx, ggml_cont(ctx, ggml_transpose(ctx, s1C)), x);

    ggml_tensor* att = ggml_concat(ctx, ggml_concat(ctx, x, mCT, 0), sCT, 0);
    att = tdnn_block(ctx, att, asp.tdnn, 1);
    att = ggml_tanh(ctx, att);
    auto cw = ggml_reshape_2d(ctx, asp.conv_w, asp.conv_w->ne[1], asp.conv_w->ne[2]);
    att = ggml_add(ctx, ggml_mul_mat(ctx, cw, att), asp.conv_b);
    att = ggml_cont(ctx, ggml_transpose(ctx, att));
    att = ggml_soft_max(ctx, att); // over T (ne[0] after the transpose)
    att = ggml_cont(ctx, ggml_transpose(ctx, att));

    ggml_tensor* wx = ggml_mul(ctx, att, x);
    ggml_tensor* wm = ggml_cont(ctx, ggml_transpose(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, wx)))));
    ggml_tensor* wmCT = ggml_repeat(ctx, wm, x);
    ggml_tensor* dd = ggml_sub(ctx, x, wmCT);
    ggml_tensor* ws = ggml_sqrt(
        ctx,
        ggml_cont(ctx,
                  ggml_transpose(
                      ctx, ggml_sum_rows(
                               ctx, ggml_cont(ctx, ggml_transpose(ctx, ggml_mul(ctx, att, ggml_mul(ctx, dd, dd))))))));
    return ggml_concat(ctx, wm, ws, 0); // [2C, 1]
}

// Full forward: [in_dim, T] → [enc_dim].  `gf` must already exist; the caller
// owns allocation and supplies `x` as a graph input.
inline ggml_tensor* forward(ggml_context* ctx, const model& m, ggml_tensor* x) {
    ggml_tensor* h = tdnn_block(ctx, x, m.blk0, 1);

    static const int dilations[3] = {2, 3, 4};
    ggml_tensor* blk_outs[3];
    for (int i = 0; i < 3; i++) {
        h = se_res2net(ctx, h, m.blk[i], dilations[i]);
        blk_outs[i] = h;
    }

    ggml_tensor* mfa_in = ggml_concat(ctx, ggml_concat(ctx, blk_outs[0], blk_outs[1], 0), blk_outs[2], 0);
    h = tdnn_block(ctx, mfa_in, m.mfa, 1);
    h = asp_block(ctx, h, m.asp);

    auto fcw = ggml_reshape_2d(ctx, m.fc_w, m.fc_w->ne[1], m.fc_w->ne[2]);
    h = ggml_add(ctx, ggml_mul_mat(ctx, fcw, h), m.fc_b);
    return ggml_reshape_1d(ctx, h, (int)m.fc_w->ne[2]);
}

} // namespace core_ecapa

// src/tada_encoder.cpp — TADA encoder for voice reference creation.
//
// Converts 24 kHz audio + transcript → aligned acoustic features
// (token_values + token_positions) for saving as voice reference GGUF.
//
// Architecture:
//   1. WavEncoder: DAC-style strided conv encoder
//      Conv1d(1→64, k=7, p=3) → 4× EncoderBlock(strides [6,5,4,4])
//      EncoderBlock: 3× ResUnit(dil=1,3,9) → Snake1d → stride Conv1d
//      Final: Snake1d(1024) → Conv1d(1024→latent_dim, k=3, p=1)
//      Total: 480× downsample → 24kHz becomes 50Hz
//
//   2. LocalAttentionEncoder: 6 layers (pre-norm)
//      LN → QKV → RoPE → attention → add → LN → FFN(GELU) → add
//      + pos_emb(token_masks) added to input
//      + segment attention mask (v2)
//      + final LayerNorm
//
//   3. hidden_linear: Linear(1024→512)
//
//   4. Post-processing: zero non-token, add noise, gather, normalize

#include "tada_encoder.h"
#include "wav2vec2-ggml.h"
#include "core/bpe.h"
#include "core/cpu_ops.h" // core_cpu::to_f32 (quantized-safe weight read)
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─────────────────────────── debug gate ────────────────────────────
static bool tada_enc_debug() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_TADA_ENCODER_DEBUG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// ──────────────────────── internal types ───────────────────────────

namespace {

struct wn_conv {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};

struct enc_res_unit {
    ggml_tensor* alpha0 = nullptr;
    ggml_tensor* inv_alpha0 = nullptr;
    wn_conv conv0; // k=7, dilated
    ggml_tensor* alpha1 = nullptr;
    ggml_tensor* inv_alpha1 = nullptr;
    wn_conv conv1; // k=1
};

struct enc_block {
    enc_res_unit res[3]; // dilation 1, 3, 9
    ggml_tensor* snake_alpha = nullptr;
    ggml_tensor* inv_snake_alpha = nullptr;
    wn_conv stride_conv; // strided downsample conv
};

struct enc_attn_layer {
    ggml_tensor* qkv_w = nullptr;
    ggml_tensor* qkv_b = nullptr;
    ggml_tensor* out_w = nullptr;
    ggml_tensor* out_b = nullptr;
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
};

} // namespace

struct tada_encoder_context {
    tada_encoder_params params;

    // WavEncoder
    wn_conv first_conv;  // Conv1d(1→64, k=7, p=3)
    enc_block blocks[4]; // strides [6,5,4,4]
    ggml_tensor* final_snake_alpha = nullptr;
    ggml_tensor* inv_final_snake_alpha = nullptr;
    wn_conv final_conv; // Conv1d(1024→latent_dim, k=3, p=1)

    // LocalAttentionEncoder
    enc_attn_layer attn_layers[6];
    ggml_tensor* final_norm_w = nullptr;
    ggml_tensor* final_norm_b = nullptr;

    // hidden_linear
    ggml_tensor* hidden_w = nullptr;
    ggml_tensor* hidden_b = nullptr;

    // pos_emb: Embedding(2, 1024)
    ggml_tensor* pos_emb_w = nullptr;

    // Config
    int hidden_dim = 1024;
    int embed_dim = 512;
    int n_attn_layers = 6;
    int n_attn_heads = 8;
    int head_dim = 128;
    int strides[4] = {6, 5, 4, 4};
    float noise_std = 0.5f;
    float acoustic_mean = 0.0f;
    float acoustic_std = 1.5f;

    // ggml state
    ggml_backend_t backend = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context* ctx_inv = nullptr;
    ggml_backend_buffer_t buf_inv = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<uint8_t> compute_meta;
};

// ────────────────────── weight loading ─────────────────────────────

static void bind_wn(tada_encoder_context* c, wn_conv& wn, const char* prefix) {
    char key[256];
    snprintf(key, sizeof(key), "%s.weight", prefix);
    wn.w = core_gguf::try_get(c->tensors, key);
    snprintf(key, sizeof(key), "%s.bias", prefix);
    wn.b = core_gguf::try_get(c->tensors, key);
}

static bool bind_weights(tada_encoder_context* c) {
    auto& m = c->tensors;
    char key[256];

    // WavEncoder: enc.wav.{0..6}
    // block 0: first Conv1d(1→64, k=7, p=3)
    bind_wn(c, c->first_conv, "enc.wav.0");

    // blocks 1-4: EncoderBlocks
    for (int b = 0; b < 4; b++) {
        auto& blk = c->blocks[b];
        int wav_idx = b + 1; // enc.wav.{1,2,3,4}

        // 3 ResidualUnits: enc.wav.{b+1}.block.{0,1,2}
        for (int r = 0; r < 3; r++) {
            auto& ru = blk.res[r];
            // Snake1d alpha: block.{r}.block.0.alpha
            snprintf(key, sizeof(key), "enc.wav.%d.block.%d.block.0.alpha", wav_idx, r);
            ru.alpha0 = core_gguf::try_get(m, key);

            // Conv1d k=7: block.{r}.block.1
            snprintf(key, sizeof(key), "enc.wav.%d.block.%d.block.1", wav_idx, r);
            bind_wn(c, ru.conv0, key);

            // Snake1d alpha: block.{r}.block.2.alpha
            snprintf(key, sizeof(key), "enc.wav.%d.block.%d.block.2.alpha", wav_idx, r);
            ru.alpha1 = core_gguf::try_get(m, key);

            // Conv1d k=1: block.{r}.block.3
            snprintf(key, sizeof(key), "enc.wav.%d.block.%d.block.3", wav_idx, r);
            bind_wn(c, ru.conv1, key);
        }

        // Snake1d before stride conv: block.3.alpha
        snprintf(key, sizeof(key), "enc.wav.%d.block.3.alpha", wav_idx);
        blk.snake_alpha = core_gguf::try_get(m, key);

        // Stride conv: block.4
        snprintf(key, sizeof(key), "enc.wav.%d.block.4", wav_idx);
        bind_wn(c, blk.stride_conv, key);
    }

    // Final Snake1d + Conv1d: enc.wav.5 and enc.wav.6
    c->final_snake_alpha = core_gguf::try_get(m, "enc.wav.5.alpha");
    bind_wn(c, c->final_conv, "enc.wav.6");

    // LocalAttentionEncoder
    for (int i = 0; i < c->n_attn_layers; i++) {
        auto& l = c->attn_layers[i];
        snprintf(key, sizeof(key), "enc.attn.blk.%d.attn_qkv.weight", i);
        l.qkv_w = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.attn_qkv.bias", i);
        l.qkv_b = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.attn_output.weight", i);
        l.out_w = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.attn_output.bias", i);
        l.out_b = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.attn_norm.weight", i);
        l.attn_norm_w = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.attn_norm.bias", i);
        l.attn_norm_b = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.ffn_up.weight", i);
        l.ffn_up_w = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.ffn_up.bias", i);
        l.ffn_up_b = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.ffn_down.weight", i);
        l.ffn_down_w = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.ffn_down.bias", i);
        l.ffn_down_b = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.ffn_norm.weight", i);
        l.ffn_norm_w = core_gguf::try_get(m, key);
        snprintf(key, sizeof(key), "enc.attn.blk.%d.ffn_norm.bias", i);
        l.ffn_norm_b = core_gguf::try_get(m, key);
    }

    c->final_norm_w = core_gguf::try_get(m, "enc.attn.final_norm.weight");
    c->final_norm_b = core_gguf::try_get(m, "enc.attn.final_norm.bias");

    // hidden_linear
    c->hidden_w = core_gguf::try_get(m, "enc.hidden_linear.weight");
    c->hidden_b = core_gguf::try_get(m, "enc.hidden_linear.bias");

    // pos_emb
    c->pos_emb_w = core_gguf::try_get(m, "enc.pos_emb.weight");

    return true;
}

// ─────────────── precompute inverse alphas ─────────────────────────

static void precompute_inv_alphas(tada_encoder_context* c) {
    struct alpha_pair {
        ggml_tensor* src;
        ggml_tensor** dst;
    };
    std::vector<alpha_pair> pairs;

    for (int b = 0; b < 4; b++) {
        if (c->blocks[b].snake_alpha)
            pairs.push_back({c->blocks[b].snake_alpha, &c->blocks[b].inv_snake_alpha});
        for (int r = 0; r < 3; r++) {
            if (c->blocks[b].res[r].alpha0)
                pairs.push_back({c->blocks[b].res[r].alpha0, &c->blocks[b].res[r].inv_alpha0});
            if (c->blocks[b].res[r].alpha1)
                pairs.push_back({c->blocks[b].res[r].alpha1, &c->blocks[b].res[r].inv_alpha1});
        }
    }
    if (c->final_snake_alpha)
        pairs.push_back({c->final_snake_alpha, &c->inv_final_snake_alpha});

    if (pairs.empty())
        return;

    size_t ctx_size = ggml_tensor_overhead() * (pairs.size() + 4) + 4096;
    ggml_init_params ip = {ctx_size, nullptr, true};
    c->ctx_inv = ggml_init(ip);

    for (auto& p : pairs) {
        *p.dst = ggml_new_tensor(c->ctx_inv, GGML_TYPE_F32, ggml_n_dims(p.src), p.src->ne);
    }

    c->buf_inv = ggml_backend_alloc_ctx_tensors(c->ctx_inv, c->backend);

    for (auto& p : pairs) {
        int64_t n = ggml_nelements(p.src);
        std::vector<float> a = core_cpu::to_f32(p.src); // F32/F16/quantized-safe
        std::vector<float> inv(n);
        for (int64_t i = 0; i < n; i++)
            inv[i] = 1.0f / (a[i] + 1e-12f);
        ggml_backend_tensor_set(*p.dst, inv.data(), 0, n * sizeof(float));
    }
}

// ──────────────── graph building helpers ───────────────────────────

// Snake1d: y = x + sin²(alpha * x) * inv_alpha
static ggml_tensor* snake1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* inv_alpha) {
    if (!alpha || !inv_alpha)
        return x;
    const int C = (int)x->ne[0];
    ggml_tensor* a = ggml_cast(ctx, ggml_cont(ctx, ggml_reshape_2d(ctx, alpha, C, 1)), GGML_TYPE_F32);
    ggml_tensor* ia = ggml_cast(ctx, ggml_cont(ctx, ggml_reshape_2d(ctx, inv_alpha, C, 1)), GGML_TYPE_F32);
    ggml_tensor* ax = ggml_mul(ctx, x, a);
    ggml_tensor* sin_ax = ggml_sin(ctx, ax);
    ggml_tensor* sin2 = ggml_mul(ctx, sin_ax, sin_ax);
    ggml_tensor* term = ggml_mul(ctx, sin2, ia);
    return ggml_add(ctx, x, term);
}

// Conv1d with symmetric padding. x: (C_in, T) → (C_out, T_out)
// For dilation>1 convs (ResidualUnits): pad = dilation*(K-1)/2, stride=1
// For stride convs (EncoderBlocks): pad = ceil(stride/2), dilation=1
static ggml_tensor* wn_conv1d(ggml_context* ctx, ggml_tensor* x, const wn_conv& wn, int dilation, int stride = 1,
                              int pad_override = -1) {
    if (!wn.w)
        return x;
    const int K = (int)wn.w->ne[0];
    const int p = (pad_override >= 0) ? pad_override : dilation * (K - 1) / 2;

    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C_in)
    ggml_tensor* y = ggml_conv_1d(ctx, wn.w, xt, stride, p, dilation);
    const int Cout = (int)wn.w->ne[2];
    const int T_out = (int)((x->ne[1] + 2 * p - dilation * (K - 1) - 1) / stride + 1);
    y = ggml_reshape_2d(ctx, y, T_out, Cout);
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (C_out, T_out)
    if (wn.b)
        y = ggml_add(ctx, y, ggml_cast(ctx, wn.b, GGML_TYPE_F32));
    return y;
}

// ResidualUnit: Snake → Conv(k=7,dil) → Snake → Conv(k=1), + residual
static ggml_tensor* res_unit(ggml_context* ctx, ggml_tensor* x, const enc_res_unit& ru, int dilation) {
    ggml_tensor* y = snake1d(ctx, x, ru.alpha0, ru.inv_alpha0);
    y = wn_conv1d(ctx, y, ru.conv0, dilation);
    y = snake1d(ctx, y, ru.alpha1, ru.inv_alpha1);
    y = wn_conv1d(ctx, y, ru.conv1, 1);
    // The encoder ResidualUnit has a pad-and-add residual if lengths differ
    // For k=7 with symmetric padding, output length equals input, so just add.
    return ggml_add(ctx, x, y);
}

// EncoderBlock: 3× ResUnit(d=1,3,9) → Snake → stride Conv
// Python: WNConv1d(dim//2, dim, kernel_size=2*stride, stride=stride, padding=ceil(stride/2))
static ggml_tensor* enc_block_forward(ggml_context* ctx, ggml_tensor* x, const enc_block& blk, int stride) {
    static const int dilations[3] = {1, 3, 9};
    for (int r = 0; r < 3; r++)
        x = res_unit(ctx, x, blk.res[r], dilations[r]);
    x = snake1d(ctx, x, blk.snake_alpha, blk.inv_snake_alpha);
    int pad = (stride + 1) / 2; // ceil(stride/2)
    x = wn_conv1d(ctx, x, blk.stride_conv, /*dilation=*/1, stride, pad);
    return x;
}

// ────────────── v2 segment attention mask ──────────────────────────

// Build the v2 segment attention mask from token_masks.
// token_masks: (T,) int32 with 1 at token boundaries, 0 elsewhere.
// Returns: (T, T) float16 mask where -inf = masked, 0 = attend.
// This matches tada/modules/encoder.py _create_segment_attention_mask(v2).
static std::vector<ggml_fp16_t> build_v2_segment_mask(const int32_t* token_masks, int T) {
    // Compute block IDs: cumsum of token_masks
    std::vector<int> block_ids(T);
    int cumsum = 0;
    for (int i = 0; i < T; i++) {
        cumsum += token_masks[i];
        block_ids[i] = cumsum;
    }

    std::vector<ggml_fp16_t> mask(T * T);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);

    for (int i = 0; i < T; i++) {
        bool is_marked_i = token_masks[i] != 0;
        for (int j = 0; j < T; j++) {
            bool is_marked_j = token_masks[j] != 0;
            bool same_block = block_ids[i] == block_ids[j];
            bool prev_block = block_ids[j] == block_ids[i] - 1;

            // v2 rules:
            // same_block_valid = same_block && (!is_marked_j || (is_marked_i && same_block))
            bool same_block_valid = same_block && (!is_marked_j || is_marked_i);
            // prev_block_valid = prev_block && !is_marked_j
            bool prev_block_valid = prev_block && !is_marked_j;
            // can_attend = same_block_valid || (is_marked_i && prev_block_valid)
            bool can_attend = same_block_valid || (is_marked_i && prev_block_valid);

            mask[i * T + j] = can_attend ? zero : neg_inf;
        }
    }
    return mask;
}

// ──────────── full WavEncoder + attention graph ────────────────────

static ggml_cgraph* build_encoder_graph(tada_encoder_context* c, int n_audio_samples) {
    const int d = c->hidden_dim;
    const int ed = c->embed_dim;
    const int nh = c->n_attn_heads;
    const int hd = c->head_dim;
    const float eps = 1e-5f;

    // Compute output frame count after WavEncoder
    // Pad input by 960 samples (matches Python: F.pad(audio, (0, 960)))
    int padded_len = n_audio_samples + 960;
    // First conv: k=7, p=3, stride=1 → same length
    int T = padded_len;
    // 4 encoder blocks with strides
    for (int b = 0; b < 4; b++) {
        int s = c->strides[b];
        int K = 2 * s;       // kernel = 2*stride
        int p = (s + 1) / 2; // ceil(stride/2) — matches Python math.ceil(stride/2)
        T = (T + 2 * p - K) / s + 1;
    }
    // Final conv: k=3, p=1, stride=1 → same length
    int n_frames = T;

    if (tada_enc_debug())
        fprintf(stderr, "[tada-enc] audio=%d padded=%d → %d frames\n", n_audio_samples, padded_len, n_frames);

    size_t meta_size = 64 * 1024 * 1024; // 64 MB for graph metadata
    c->compute_meta.resize(meta_size);
    ggml_init_params ip = {meta_size, c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    // Input: mono audio — channel-first (C_in=1, T=padded_len)
    // In ggml ne[0]=C_in=1, ne[1]=T=padded_len
    ggml_tensor* audio = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, padded_len);
    ggml_set_name(audio, "audio_input");
    ggml_set_input(audio);

    // ── WavEncoder ──
    // First conv: Conv1d(1→64, k=7, p=3)
    if (tada_enc_debug())
        fprintf(stderr, "[tada-enc] audio ne=[%lld,%lld] first_conv.w ne=[%lld,%lld,%lld]\n", (long long)audio->ne[0],
                (long long)audio->ne[1], (long long)c->first_conv.w->ne[0], (long long)c->first_conv.w->ne[1],
                (long long)c->first_conv.w->ne[2]);
    ggml_tensor* cur = wn_conv1d(ctx0, audio, c->first_conv, 1);

    // 4 encoder blocks
    for (int b = 0; b < 4; b++)
        cur = enc_block_forward(ctx0, cur, c->blocks[b], c->strides[b]);

    // Final: Snake1d → Conv1d(1024→latent_dim, k=3, p=1)
    cur = snake1d(ctx0, cur, c->final_snake_alpha, c->inv_final_snake_alpha);
    cur = wn_conv1d(ctx0, cur, c->final_conv, 1);

    // cur is channel-first (hidden_dim, n_frames) = ne[0]=d, ne[1]=T

    // Dump: wav_encoder output — cur is (d, T) = ne[0]=d, ne[1]=T.
    // ggml_backend_tensor_get reads ne[0]-contiguous, giving (T, d) row-major
    // which matches the Python reference layout.
    ggml_tensor* dump_wav = ggml_cont(ctx0, cur);
    ggml_set_name(dump_wav, "encoder_wav_out");
    ggml_set_output(dump_wav);
    ggml_build_forward_expand(gf, dump_wav);

    // ── Add pos_emb ──
    // token_masks input: (n_frames,) int32
    ggml_tensor* masks = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(masks, "token_masks");
    ggml_set_input(masks);

    // pos_emb: Embedding(2, hidden_dim) — index by token_masks
    // ggml_get_rows returns (hidden_dim, n_frames) = ne[0]=d, ne[1]=T
    // which matches cur's layout
    if (c->pos_emb_w) {
        ggml_tensor* pos = ggml_get_rows(ctx0, c->pos_emb_w, masks);
        cur = ggml_add(ctx0, cur, pos);
    }

    // cur stays as (hidden_dim, n_frames) = (d, T) for attention

    // ── LocalAttentionEncoder (pre-norm) ──
    ggml_tensor* attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_frames, n_frames);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    for (int il = 0; il < c->n_attn_layers; il++) {
        const auto& l = c->attn_layers[il];
        ggml_tensor* residual = cur;

        // Pre-norm: LayerNorm before attention
        ggml_tensor* attn_in = ggml_norm(ctx0, cur, eps);
        // Note: the encoder uses the LocalSelfAttention's internal layer_norm
        // which is a POST-attention norm: output = layer_norm(x + attn(x))
        // This matches the codec decoder's post-norm pattern.

        // QKV projection
        ggml_tensor* qkv = ggml_mul_mat(ctx0, l.qkv_w, cur);
        if (l.qkv_b)
            qkv = ggml_add(ctx0, qkv, l.qkv_b);

        ggml_tensor* q = ggml_view_2d(ctx0, qkv, d, n_frames, qkv->nb[1], 0);
        ggml_tensor* k = ggml_view_2d(ctx0, qkv, d, n_frames, qkv->nb[1], (size_t)d * sizeof(float));
        ggml_tensor* v = ggml_view_2d(ctx0, qkv, d, n_frames, qkv->nb[1], (size_t)2 * d * sizeof(float));
        q = ggml_cont(ctx0, q);
        k = ggml_cont(ctx0, k);
        v = ggml_cont(ctx0, v);

        q = ggml_reshape_3d(ctx0, q, hd, nh, n_frames);
        k = ggml_reshape_3d(ctx0, k, hd, nh, n_frames);
        v = ggml_reshape_3d(ctx0, v, hd, nh, n_frames);

        // RoPE — interleaved-pair style (NORMAL), θ=10000
        q = ggml_rope_ext(ctx0, q, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);
        k = ggml_rope_ext(ctx0, k, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);

        // Permute to (hd, T, nh) for flash attention
        q = ggml_permute(ctx0, q, 0, 2, 1, 3);
        k = ggml_permute(ctx0, k, 0, 2, 1, 3);
        v = ggml_permute(ctx0, v, 0, 2, 1, 3);

        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, q, k, v, attn_mask, scale, 0, 0);
        attn = ggml_cont(ctx0, attn);
        attn = ggml_reshape_2d(ctx0, attn, d, n_frames);

        // Output projection + dropout (dropout is no-op at inference)
        attn = ggml_mul_mat(ctx0, l.out_w, attn);
        if (l.out_b)
            attn = ggml_add(ctx0, attn, l.out_b);

        // Post-norm: cur = LayerNorm(residual + attn)
        // This matches LocalSelfAttention.forward():
        //   output = self.layer_norm(x + self.out_proj(attn_output))
        cur = ggml_add(ctx0, residual, attn);
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, l.attn_norm_w);
        if (l.attn_norm_b)
            cur = ggml_add(ctx0, cur, l.attn_norm_b);

        // FFN: Linear(d, 4d) → GELU → Linear(4d, d)
        residual = cur;
        ggml_tensor* ffn = ggml_mul_mat(ctx0, l.ffn_up_w, cur);
        if (l.ffn_up_b)
            ffn = ggml_add(ctx0, ffn, l.ffn_up_b);
        ffn = ggml_gelu(ctx0, ffn);
        ffn = ggml_mul_mat(ctx0, l.ffn_down_w, ffn);
        if (l.ffn_down_b)
            ffn = ggml_add(ctx0, ffn, l.ffn_down_b);

        // Post-norm: cur = LayerNorm(cur + ffn)
        // Matches LocalAttentionEncoderLayer.forward():
        //   x = self.norm(x + self.ffn(x))
        cur = ggml_add(ctx0, residual, ffn);
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, l.ffn_norm_w);
        if (l.ffn_norm_b)
            cur = ggml_add(ctx0, cur, l.ffn_norm_b);
    }

    // Final norm
    cur = ggml_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, c->final_norm_w);
    if (c->final_norm_b)
        cur = ggml_add(ctx0, cur, c->final_norm_b);

    // Dump: attention output — cur is (d, T), read back as (T, d) row-major
    ggml_tensor* dump_attn = ggml_cont(ctx0, cur);
    ggml_set_name(dump_attn, "encoder_attn_out");
    ggml_set_output(dump_attn);
    ggml_build_forward_expand(gf, dump_attn);

    // ── hidden_linear: (d, T) → (embed_dim, T) ──
    if (c->hidden_w) {
        cur = ggml_mul_mat(ctx0, c->hidden_w, cur);
        if (c->hidden_b)
            cur = ggml_add(ctx0, cur, c->hidden_b);
    }

    // Dump: hidden output — cur is (embed_dim, T), read back as (T, embed_dim) row-major
    ggml_tensor* dump_hidden = ggml_cont(ctx0, cur);
    ggml_set_name(dump_hidden, "encoder_hidden");
    ggml_set_output(dump_hidden);
    ggml_build_forward_expand(gf, dump_hidden);

    ggml_free(ctx0);
    return gf;
}

// ──────────────────── DP alignment ────────────────────────────────

int tada_dp_align(const float* probs, int T, int V, const int32_t* tokens, int N, int32_t* positions) {
    if (N <= 0 || T <= 0)
        return -1;

    // F[i][j] = best score aligning tokens[0..j] to probs[0..i]
    std::vector<float> F((size_t)T * N, -INFINITY);
    std::vector<int32_t> backptr((size_t)T * N, 0);

    // Extract token probabilities: probs[t, tokens[j]] for all t, j
    // to avoid repeated V-stride access
    std::vector<float> tp((size_t)T * N);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < N; j++)
            tp[t * N + j] = probs[t * V + tokens[j]];

    // Initialize first column: best position for first token
    float best = -INFINITY;
    int best_i = 0;
    for (int i = 0; i < T; i++) {
        float v = tp[i * N + 0];
        if (v >= best) {
            best = v;
            best_i = i;
        }
        F[i * N + 0] = best;
        backptr[i * N + 0] = best_i;
    }

    // Initialize diagonal: forced alignment
    if (N <= T) {
        float cumsum = 0.0f;
        for (int j = 0; j < N; j++) {
            cumsum += tp[j * N + j];
            F[j * N + j] = cumsum;
            backptr[j * N + j] = j;
        }
    }

    // Fill DP table
    for (int i = 1; i < T; i++) {
        int max_j = std::min(i, N - 1);
        for (int j = 1; j <= max_j; j++) {
            float skip = F[(i - 1) * N + j];                      // don't use position i
            float use = F[(i - 1) * N + (j - 1)] + tp[i * N + j]; // use position i for token j

            if (use >= skip) {
                F[i * N + j] = use;
                backptr[i * N + j] = i;
            } else {
                F[i * N + j] = skip;
                backptr[i * N + j] = -1; // signal: inherited from previous row
            }
        }
    }

    // Traceback
    int i = T - 1, j = N - 1;
    int pos_idx = N - 1;
    while (j >= 0) {
        if (j == 0) {
            positions[pos_idx] = backptr[i * N + j];
            break;
        } else if (backptr[i * N + j] == -1) {
            i--;
        } else {
            positions[pos_idx] = backptr[i * N + j];
            pos_idx--;
            i--;
            j--;
        }
    }

    return 0;
}

// ──────────────────── public API ──────────────────────────────────

tada_encoder_params tada_encoder_default_params() {
    return {};
}

tada_encoder_context* tada_encoder_init(const char* path, tada_encoder_params params) {
    auto* c = new tada_encoder_context();
    c->params = params;
    c->compute_meta.resize(64 * 1024 * 1024);

    // Load metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta) {
        delete c;
        return nullptr;
    }

    auto u32 = [&](const char* key, int def) -> int {
        int idx = gguf_find_key(meta, key);
        return idx >= 0 ? (int)gguf_get_val_u32(meta, idx) : def;
    };
    auto f32v = [&](const char* key, float def) -> float {
        int idx = gguf_find_key(meta, key);
        return idx >= 0 ? gguf_get_val_f32(meta, idx) : def;
    };

    c->hidden_dim = u32("tada_encoder.hidden_dim", 1024);
    c->embed_dim = u32("tada_encoder.embed_dim", 512);
    c->n_attn_layers = u32("tada_encoder.num_attn_layers", 6);
    c->n_attn_heads = u32("tada_encoder.num_attn_heads", 8);
    c->head_dim = c->hidden_dim / c->n_attn_heads;
    c->noise_std = f32v("tada_encoder.noise_std", 0.5f);
    c->acoustic_mean = f32v("tada_encoder.acoustic_mean", 0.0f);
    c->acoustic_std = f32v("tada_encoder.acoustic_std", 1.5f);

    // Read strides array
    {
        int idx = gguf_find_key(meta, "tada_encoder.strides");
        if (idx >= 0) {
            int n = (int)gguf_get_arr_n(meta, idx);
            for (int i = 0; i < std::min(n, 4); i++)
                c->strides[i] = (int)((const int32_t*)gguf_get_arr_data(meta, idx))[i];
        }
    }
    core_gguf::free_metadata(meta);

    // Load weights
    c->backend = core_cpu_backend::init();
    if (!c->backend) {
        delete c;
        return nullptr;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "tada-encoder", wl)) {
        fprintf(stderr, "[tada-encoder] failed to load weights from %s\n", path);
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!bind_weights(c)) {
        delete c;
        return nullptr;
    }

    precompute_inv_alphas(c);

    if (params.verbosity > 0) {
        fprintf(stderr,
                "[tada-encoder] hidden=%d embed=%d attn_layers=%d heads=%d "
                "strides=[%d,%d,%d,%d] noise_std=%.2f\n",
                c->hidden_dim, c->embed_dim, c->n_attn_layers, c->n_attn_heads, c->strides[0], c->strides[1],
                c->strides[2], c->strides[3], c->noise_std);
    }

    return c;
}

void tada_encoder_free(tada_encoder_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf_inv)
        ggml_backend_buffer_free(ctx->buf_inv);
    if (ctx->ctx_inv)
        ggml_free(ctx->ctx_inv);
    if (ctx->buf_w)
        core_gguf::release_weight_buffer(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

int tada_encoder_encode_with_positions(tada_encoder_context* ctx, const float* audio_24k, int n_samples_24k,
                                       const int32_t* token_masks, int n_masks, const int32_t* token_positions,
                                       int n_tokens, tada_encoder_result& result) {
    // Pad audio with 960 zeros (matches Python)
    int padded = n_samples_24k + 960;
    std::vector<float> audio_padded(padded, 0.0f);
    std::memcpy(audio_padded.data(), audio_24k, n_samples_24k * sizeof(float));

    // Build graph
    ggml_cgraph* gf = build_encoder_graph(ctx, n_samples_24k);
    if (!gf)
        return -1;

    // Allocate
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "[tada-encoder] graph allocation failed\n");
        ggml_gallocr_free(alloc);
        return -2;
    }

    // Set inputs
    ggml_tensor* audio_t = ggml_graph_get_tensor(gf, "audio_input");
    ggml_backend_tensor_set(audio_t, audio_padded.data(), 0, padded * sizeof(float));

    // Compute expected frame count
    int T = padded;
    for (int b = 0; b < 4; b++) {
        int s = ctx->strides[b];
        int K = 2 * s;
        int p = (s + 1) / 2;
        T = (T + 2 * p - K) / s + 1;
    }
    int n_frames = T;

    // Pad token_masks to n_frames
    std::vector<int32_t> masks_padded(n_frames, 0);
    int copy_len = std::min(n_masks, n_frames);
    std::memcpy(masks_padded.data(), token_masks, copy_len * sizeof(int32_t));

    ggml_tensor* masks_t = ggml_graph_get_tensor(gf, "token_masks");
    ggml_backend_tensor_set(masks_t, masks_padded.data(), 0, n_frames * sizeof(int32_t));

    // Build attention mask
    auto attn_mask_data = build_v2_segment_mask(masks_padded.data(), n_frames);
    ggml_tensor* attn_mask_t = ggml_graph_get_tensor(gf, "attn_mask");
    ggml_backend_tensor_set(attn_mask_t, attn_mask_data.data(), 0, (size_t)n_frames * n_frames * sizeof(ggml_fp16_t));

    // Position IDs: 0, 1, 2, ...
    std::vector<int32_t> pos_ids(n_frames);
    for (int i = 0; i < n_frames; i++)
        pos_ids[i] = i;
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "pos_ids");
    ggml_backend_tensor_set(pos_t, pos_ids.data(), 0, n_frames * sizeof(int32_t));

    // Compute
    core_cpu_backend::set_n_threads(ctx->backend, ctx->params.n_threads);
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[tada-encoder] graph compute failed\n");
        ggml_gallocr_free(alloc);
        return -3;
    }

    // Read hidden output: the dump tensor is already (T, embed_dim) after
    // the ggml_transpose + ggml_cont in the graph. ggml stores this as
    // ne[0]=embed_dim (fast axis), ne[1]=T, so backend_tensor_get gives
    // row-major (T, embed_dim) directly.
    ggml_tensor* hidden_out = ggml_graph_get_tensor(gf, "encoder_hidden");
    int ed = ctx->embed_dim;
    std::vector<float> features((size_t)n_frames * ed);
    ggml_backend_tensor_get(hidden_out, features.data(), 0, features.size() * sizeof(float));

    // Zero out non-token frames
    for (int t = 0; t < n_frames; t++) {
        if (masks_padded[t] == 0) {
            std::memset(&features[t * ed], 0, ed * sizeof(float));
        }
    }

    // Add noise to voiced frames
    std::mt19937 rng(ctx->params.seed);
    std::normal_distribution<float> noise(0.0f, ctx->noise_std);
    for (int t = 0; t < n_frames; t++) {
        if (masks_padded[t] != 0) {
            for (int d = 0; d < ed; d++)
                features[t * ed + d] += noise(rng);
        }
    }

    // Gather at token positions and normalize
    result.n_tokens = n_tokens;
    result.embed_dim = ed;
    result.token_values.resize((size_t)n_tokens * ed);
    result.token_positions.resize(n_tokens);

    for (int i = 0; i < n_tokens; i++) {
        int pos = std::max(0, token_positions[i] - 1); // positions are 1-indexed
        pos = std::min(pos, n_frames - 1);
        result.token_positions[i] = (float)token_positions[i];
        for (int d = 0; d < ed; d++) {
            float val = features[pos * ed + d];
            result.token_values[i * ed + d] = (val - ctx->acoustic_mean) / ctx->acoustic_std;
        }
    }

    ggml_gallocr_free(alloc);
    return 0;
}

// Text normalization matching tada/utils/text.py normalize_text().
static std::string normalize_text(std::string text) {
    auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
        if (from.empty())
            return;
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all(text, "; ", ". ");
    replace_all(text, "\"", "");
    replace_all(text, ":", ",");
    replace_all(text, "(", "");
    replace_all(text, ")", "");
    replace_all(text, "--", "-");
    replace_all(text, "-", ", ");
    replace_all(text, ",,", ",");
    replace_all(text, " '", " ");
    replace_all(text, "' ", " ");
    replace_all(text, "  ", " ");

    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); i++) {
        if (std::isspace((unsigned char)text[i]) && i + 1 < text.size() &&
            (text[i + 1] == '.' || text[i + 1] == ',' || text[i + 1] == '?' || text[i + 1] == '!'))
            continue;
        out.push_back(text[i]);
    }
    return out;
}

int tada_encoder_encode(tada_encoder_context* ctx, const char* aligner_gguf, const float* audio_24k, int n_samples_24k,
                        const char* transcript, tada_encoder_result& result) {
    // Step 1: Resample 24kHz → 16kHz for aligner
    int n_16k = (int)((int64_t)n_samples_24k * 16000 / 24000);
    std::vector<float> audio_16k(n_16k);
    for (int i = 0; i < n_16k; i++) {
        float src_pos = (float)i * 24000.0f / 16000.0f;
        int idx = (int)src_pos;
        float frac = src_pos - idx;
        if (idx + 1 < n_samples_24k)
            audio_16k[i] = audio_24k[idx] * (1.0f - frac) + audio_24k[idx + 1] * frac;
        else if (idx < n_samples_24k)
            audio_16k[i] = audio_24k[idx];
        else
            audio_16k[i] = 0.0f;
    }

    // Step 2: Load aligner (wav2vec2) and compute logits
    if (ctx->params.verbosity > 0)
        fprintf(stderr, "[tada-encoder] loading aligner from %s\n", aligner_gguf);
    wav2vec2_model aligner;
    if (!wav2vec2_load(aligner_gguf, aligner)) {
        fprintf(stderr, "[tada-encoder] failed to load aligner from %s\n", aligner_gguf);
        return -1;
    }

    std::vector<float> logits = wav2vec2_compute_logits(aligner, audio_16k.data(), n_16k, ctx->params.n_threads);
    if (logits.empty()) {
        fprintf(stderr, "[tada-encoder] aligner logits computation failed\n");
        return -2;
    }

    int V = (int)aligner.hparams.vocab_size;
    int T_ctc = (int)(logits.size() / V);

    // Diagnostic: free (text-less) greedy CTC decode of the aligner logits, to
    // probe whether this alignment model doubles as a CTC ASR / word-timestamp
    // model. Gated by CRISPASR_TADA_CTC_ASR=1. The aligner is trained for FORCED
    // alignment, so this is exploratory — but the logits are a full distribution
    // over the Llama-3.2 BPE vocab per frame, so argmax+collapse gives a
    // transcript hypothesis (and per-token frame indices = timecodes).
    if (const char* e = std::getenv("CRISPASR_TADA_CTC_ASR"); e && e[0] == '1') {
        // blank id: wav2vec2 CTC uses the pad token as blank.
        int blank = 0;
        {
            gguf_init_params mp = {true, nullptr};
            if (gguf_context* g = gguf_init_from_file(aligner_gguf, mp)) {
                int k = gguf_find_key(g, "wav2vec2.pad_token_id");
                if (k >= 0)
                    blank = (int)gguf_get_val_u32(g, k);
                gguf_free(g);
            }
        }
        std::vector<int32_t> ids;
        std::vector<int> frames;
        int prev = -1;
        for (int t = 0; t < T_ctc; t++) {
            const float* row = logits.data() + (size_t)t * V;
            int am = (int)(std::max_element(row, row + V) - row);
            if (am != blank && am != prev)
                ids.push_back(am), frames.push_back(t);
            prev = am;
        }
        std::string txt = core_bpe::detokenize(aligner.vocab, ids.data(), ids.size());
        fprintf(stderr, "[tada-ctc-asr] blank=%d frames=%d decoded %zu tokens\n[tada-ctc-asr] TEXT: %s\n", blank, T_ctc,
                ids.size(), txt.c_str());
        // Timecodes: first ~12 tokens with their frame index (wav2vec2 frame rate).
        fprintf(stderr, "[tada-ctc-asr] timecodes (token@frame):");
        for (size_t i = 0; i < ids.size() && i < 12; i++)
            fprintf(stderr, " '%s'@%d", core_bpe::token_bytes_to_utf8(aligner.vocab[ids[i]]).c_str(), frames[i]);
        fprintf(stderr, "\n");
    }

    // Step 3: Tokenize transcript using BPE from the aligner GGUF
    // The aligner GGUF embeds tokenizer.ggml.tokens + tokenizer.ggml.merges
    // (same Llama-3.2 vocab used by tada_tts.cpp).
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
    for (size_t i = 0; i < aligner.vocab.size(); i++)
        token_to_id[aligner.vocab[i]] = (int32_t)i;

    // Load merges from the aligner GGUF metadata
    {
        gguf_init_params mp = {true, nullptr};
        gguf_context* gctx = gguf_init_from_file(aligner_gguf, mp);
        if (gctx) {
            int midx = gguf_find_key(gctx, "tokenizer.ggml.merges");
            if (midx >= 0) {
                int n = (int)gguf_get_arr_n(gctx, midx);
                for (int i = 0; i < n; i++) {
                    const char* s = gguf_get_arr_str(gctx, midx, i);
                    if (s)
                        merge_rank[s] = i;
                }
                if (ctx->params.verbosity > 0)
                    fprintf(stderr, "[tada-encoder] loaded %d BPE merges from aligner\n", n);
            }
            gguf_free(gctx);
        }
    }

    if (token_to_id.empty()) {
        fprintf(stderr, "[tada-encoder] aligner has no tokenizer vocab\n");
        return -3;
    }

    std::string norm = normalize_text(std::string(transcript));
    std::vector<int32_t> text_tokens = core_bpe::tokenize_simple(token_to_id, merge_rank, norm);

    if (text_tokens.empty()) {
        fprintf(stderr, "[tada-encoder] tokenization produced 0 tokens for '%s'\n", transcript);
        return -4;
    }
    if (ctx->params.verbosity > 0)
        fprintf(stderr, "[tada-encoder] tokenized '%s' → %zu tokens\n", norm.c_str(), text_tokens.size());

    // Step 4: Softmax logits and run DP alignment
    // Compute softmax per-frame (in-place on logits)
    for (int t = 0; t < T_ctc; t++) {
        float* row = logits.data() + (size_t)t * V;
        float mx = *std::max_element(row, row + V);
        float sum = 0.0f;
        for (int v = 0; v < V; v++) {
            row[v] = std::exp(row[v] - mx);
            sum += row[v];
        }
        for (int v = 0; v < V; v++)
            row[v] /= sum;
    }

    int N = (int)text_tokens.size();
    // Prepend blank token (id=0) to match Python's
    // valid_tokens = F.pad(text_tokens, (1, 0))
    std::vector<int32_t> valid_tokens(N + 1);
    valid_tokens[0] = 0; // blank / pad
    std::memcpy(valid_tokens.data() + 1, text_tokens.data(), N * sizeof(int32_t));

    // Build reduced probability matrix: only keep probs for valid_tokens
    // to avoid the full T×V softmax matrix
    // Actually, the DP operates on probs[T, valid_tokens_subset], but the
    // Python code uses probs[:, valid_tokens] which is the full vocab filtered.
    // Our DP function takes the full probs matrix, so just pass it directly.

    // The Python aligns text_tokens (without the leading blank).
    // Filter out eos tokens from text_tokens
    int eos_id = (V > 128009) ? 128009 : (int)V - 1;
    std::vector<int32_t> align_tokens;
    for (int32_t tok : text_tokens) {
        if (tok != eos_id)
            align_tokens.push_back(tok);
    }
    N = (int)align_tokens.size();

    std::vector<int32_t> positions(N);
    int rc = tada_dp_align(logits.data(), T_ctc, V, align_tokens.data(), N, positions.data());
    if (rc != 0) {
        fprintf(stderr, "[tada-encoder] DP alignment failed (rc=%d)\n", rc);
        return -5;
    }

    if (ctx->params.verbosity > 0)
        fprintf(stderr, "[tada-encoder] aligned %d tokens to %d CTC frames\n", N, T_ctc);

    // Step 5: Build token_masks from positions
    // CTC frames → encoder frames: CTC operates at 50Hz (same as encoder)
    // The aligner resamples 24kHz→16kHz internally, runs wav2vec2 at 16kHz.
    // wav2vec2 output is at 50fps (16000 / stride_product = 16000/320 = 50).
    // The encoder also outputs at 50Hz (24000 / 480 = 50). So positions map 1:1.
    int padded_24k = n_samples_24k + 960;
    int T_enc = padded_24k;
    for (int b = 0; b < 4; b++) {
        int s = ctx->strides[b];
        int K = 2 * s;
        int p = (s + 1) / 2;
        T_enc = (T_enc + 2 * p - K) / s + 1;
    }

    // Convert CTC positions to 1-indexed (matching Python's selected_positions = 1 + positions)
    std::vector<int32_t> positions_1idx(N);
    for (int i = 0; i < N; i++)
        positions_1idx[i] = positions[i] + 1;

    // Build token_masks: 0 everywhere, 1 at position indices
    // input_lengths = ceil(audio_length / sample_rate * 50)
    int T_mask = std::max(T_ctc, T_enc);
    std::vector<int32_t> token_masks(T_mask, 0);
    for (int i = 0; i < N; i++) {
        int p = positions[i];
        if (p >= 0 && p < T_mask)
            token_masks[p] = 1;
    }

    if (ctx->params.verbosity > 0)
        fprintf(stderr, "[tada-encoder] masks: %d/%d voiced, encoder frames=%d\n",
                (int)std::count(token_masks.begin(), token_masks.end(), 1), T_mask, T_enc);

    // Step 6: Run encoder with positions
    int rc6 = tada_encoder_encode_with_positions(ctx, audio_24k, n_samples_24k, token_masks.data(), T_mask,
                                                 positions_1idx.data(), N, result);
    // Expose the aligned BPE token ids + decoded text (1:1 with token_positions)
    // for the --align / forced-alignment path. align_tokens[i] was aligned to
    // frame positions[i]; result.token_positions[i] mirrors that.
    if (rc6 == 0) {
        result.token_ids = align_tokens;
        result.token_texts.clear();
        result.token_texts.reserve(align_tokens.size());
        for (int32_t id : align_tokens)
            result.token_texts.push_back((id >= 0 && (size_t)id < aligner.vocab.size())
                                             ? core_bpe::token_bytes_to_utf8(aligner.vocab[id])
                                             : std::string());
    }
    return rc6;
}

float* tada_encoder_extract_stage(tada_encoder_context* ctx, const float* audio_24k, int n_samples_24k,
                                  const int32_t* token_masks, int n_masks, const char* stage, int* n_elem) {
    *n_elem = 0;

    // Build and run the graph
    int padded = n_samples_24k + 960;
    std::vector<float> audio_padded(padded, 0.0f);
    std::memcpy(audio_padded.data(), audio_24k, n_samples_24k * sizeof(float));

    ggml_cgraph* gf = build_encoder_graph(ctx, n_samples_24k);
    if (!gf)
        return nullptr;

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        ggml_gallocr_free(alloc);
        return nullptr;
    }

    // Set inputs
    ggml_tensor* audio_t = ggml_graph_get_tensor(gf, "audio_input");
    ggml_backend_tensor_set(audio_t, audio_padded.data(), 0, padded * sizeof(float));

    int T = padded;
    for (int b = 0; b < 4; b++) {
        int s = ctx->strides[b];
        int K = 2 * s;
        int p = (s + 1) / 2;
        T = (T + 2 * p - K) / s + 1;
    }
    int n_frames = T;

    std::vector<int32_t> masks_padded(n_frames, 0);
    int copy_len = std::min(n_masks, n_frames);
    std::memcpy(masks_padded.data(), token_masks, copy_len * sizeof(int32_t));

    ggml_tensor* masks_t = ggml_graph_get_tensor(gf, "token_masks");
    ggml_backend_tensor_set(masks_t, masks_padded.data(), 0, n_frames * sizeof(int32_t));

    auto attn_mask_data = build_v2_segment_mask(masks_padded.data(), n_frames);
    ggml_tensor* attn_mask_t = ggml_graph_get_tensor(gf, "attn_mask");
    ggml_backend_tensor_set(attn_mask_t, attn_mask_data.data(), 0, (size_t)n_frames * n_frames * sizeof(ggml_fp16_t));

    std::vector<int32_t> pos_ids(n_frames);
    for (int i = 0; i < n_frames; i++)
        pos_ids[i] = i;
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "pos_ids");
    ggml_backend_tensor_set(pos_t, pos_ids.data(), 0, n_frames * sizeof(int32_t));

    core_cpu_backend::set_n_threads(ctx->backend, ctx->params.n_threads);
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(alloc);
        return nullptr;
    }

    // Find the requested stage tensor
    ggml_tensor* out = ggml_graph_get_tensor(gf, stage);
    if (!out) {
        fprintf(stderr, "[tada-encoder] unknown stage '%s'\n", stage);
        ggml_gallocr_free(alloc);
        return nullptr;
    }

    size_t n = ggml_nelements(out);
    float* data = (float*)malloc(n * sizeof(float));
    ggml_backend_tensor_get(out, data, 0, n * sizeof(float));
    *n_elem = (int)n;

    ggml_gallocr_free(alloc);
    return data;
}

// ──────────── GGUF writer for voice reference ─────────────────────

int tada_encoder_write_ref_gguf(const char* path, const tada_encoder_result& result, const char* transcript,
                                const char* language, bool cloned_from_recording, const char* consent_attestation) {
    if (result.n_tokens <= 0 || result.token_values.empty())
        return -1;

    // Create ggml context for tensor descriptors (no_alloc — we just need metadata)
    size_t ctx_size = ggml_tensor_overhead() * 4 + 4096;
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    if (!ctx)
        return -2;

    gguf_context* gw = gguf_init_empty();
    gguf_set_val_str(gw, "general.architecture", "crispasr.reference");
    gguf_set_val_str(gw, "general.name", "tada-ref-custom");
    if (transcript)
        gguf_set_val_str(gw, "crispasr.ref.tada_tts_prompt_text", transcript);
    if (language && language[0])
        gguf_set_val_str(gw, "crispasr.ref.tada_tts_language", language);
    // Voice-clone provenance. A pack baked from a person's recording is exactly
    // as much a deepfake as the recording; stamping it is what lets the consent
    // and Art. 50(4) disclosure gates see that, since they can no longer tell
    // from the .gguf suffix alone. Key names are mirrored in
    // examples/cli/crispasr_voice_clone_policy.h — change both together.
    if (cloned_from_recording) {
        gguf_set_val_bool(gw, "crispasr.voice.cloned_from_recording", true);
        if (consent_attestation && consent_attestation[0])
            gguf_set_val_str(gw, "crispasr.voice.consent_attestation", consent_attestation);
    }

    // prompt_token_values: (n_tokens, embed_dim) F32
    {
        int64_t ne[2] = {result.embed_dim, result.n_tokens};
        ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
        ggml_set_name(t, "prompt_token_values");
        t->data = (void*)result.token_values.data();
        gguf_add_tensor(gw, t);
    }

    // prompt_token_positions: (n_tokens,) F32
    {
        int64_t ne[1] = {result.n_tokens};
        ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, ne);
        ggml_set_name(t, "prompt_token_positions");
        t->data = (void*)result.token_positions.data();
        gguf_add_tensor(gw, t);
    }

    gguf_write_to_file(gw, path, false);
    gguf_free(gw);
    ggml_free(ctx);
    return 0;
}

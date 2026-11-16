// moss_tts_local.cpp — MOSS-TTS-Local-Transformer-v1.5 (4B) ggml runtime.
//
// Backbone: Qwen3-4B KV-cached decoder (same in-house Qwen3 path as moss_tts.cpp,
// smaller dims + tied embeddings) → one hidden per frame. A 1-layer local/depth
// transformer (GPT2-style: LayerNorm+bias, fused-QKV, RoPE NEOX 1e6, SiLU MLP)
// then AR-generates the 12 RVQ codebooks WITHIN each frame (RQ-Transformer style),
// each conditioned on the previous — replaces the 8B's delay pattern. A binary
// local text head decides continue/stop per frame. Decoded to 48 kHz by the
// companion codec (MOSS-Audio-Tokenizer-v2; Phase 3).
//
// Decoded line-by-line from the HF modeling code — see docs/moss-tts/STUDY-4B.md.
// Graph math written by hand (validated by the Kaggle round-trip, HARD RULE #3).

#include "moss_tts_local.h"

#include "moss_tts_local_codec.h"

#include "core/attention.h"
#include "core/audio_resample.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct mtl_hparams {
    // Qwen3-4B backbone
    uint32_t llm_hidden = 2560;
    uint32_t llm_layers = 36;
    uint32_t llm_n_heads = 32;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 9728;
    uint32_t llm_vocab_size = 151936;
    uint32_t llm_max_pos = 32768;
    float llm_rope_theta = 1000000.0f;
    float llm_rms_eps = 1e-6f;

    // Local (depth) transformer
    uint32_t local_layers = 1;
    uint32_t local_hidden = 2560;
    uint32_t local_n_heads = 32;
    uint32_t local_ff_dim = 9728;
    float local_rope_base = 1000000.0f;
    float local_ln_eps = 1e-6f;

    // Audio
    uint32_t n_vq = 12;
    uint32_t audio_vocab_size = 1024; // codes 0..1023; pad (1024) is masked, NOT a row
    uint32_t audio_pad_code = 1024;
    uint32_t sampling_rate = 48000;
    uint32_t downsample_rate = 3840; // MOSS-Audio-Tokenizer-v2 hop (48000/3840 = 12.5 Hz); stereo

    // Special token ids
    uint32_t tok_audio_start = 151669;
    uint32_t tok_audio_end = 151670;
    uint32_t tok_audio_user_slot = 151654;
    uint32_t tok_audio_gen_slot = 151656;
    uint32_t tok_im_start = 151644;
    uint32_t tok_im_end = 151645;
    uint32_t tok_pad = 151643;
};

// ===========================================================================
// Model / vocab / context
// ===========================================================================

struct mtl_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr;
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct mtl_local_block {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr; // attn_norm (LayerNorm + bias)
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr; // fused QKV (3*d, d)
    ggml_tensor *o_w = nullptr, *o_b = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr; // ffn_norm
    ggml_tensor *fc_in_w = nullptr, *fc_in_b = nullptr;
    ggml_tensor *fc_out_w = nullptr, *fc_out_b = nullptr;
};

struct mtl_model {
    mtl_hparams hparams;

    ggml_tensor* token_embd_w = nullptr;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_w = nullptr; // llm.lm_head (re-emitted from tied embed)
    std::vector<mtl_llm_block> blocks;

    std::vector<mtl_local_block> local_blocks;
    ggml_tensor *local_final_norm_w = nullptr, *local_final_norm_b = nullptr;

    std::vector<ggml_tensor*> audio_embed;    // moss.audio_embed.{k} (hidden, 1024)
    std::vector<ggml_tensor*> audio_head;     // moss.audio_head.{k}  (hidden, 1024)
    ggml_tensor* local_text_head_w = nullptr; // (hidden, 2) binary continue/stop

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

struct mtl_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

struct moss_tts_local_context {
    moss_tts_local_context_params params;
    mtl_model model;
    mtl_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    int n_threads = 4;
    uint32_t seed = 0;

    std::string codec_path;
    bool codec_loaded = false;
    moss_tts_local_codec::Codec* codec = nullptr;
};

static ggml_tensor* mtl_req(mtl_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "moss_tts_local");
}
static ggml_tensor* mtl_try(mtl_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool mtl_load_model(mtl_model& model, mtl_vocab& vocab, const char* path, ggml_backend_t backend) {
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;
        auto& hp = model.hparams;
        hp.llm_hidden = core_gguf::kv_u32(gctx, "moss_tts_local.llm.hidden_size", hp.llm_hidden);
        hp.llm_layers = core_gguf::kv_u32(gctx, "moss_tts_local.llm.num_layers", hp.llm_layers);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "moss_tts_local.llm.num_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "moss_tts_local.llm.num_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "moss_tts_local.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "moss_tts_local.llm.intermediate_size", hp.llm_ff_dim);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "moss_tts_local.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "moss_tts_local.llm.max_position_embeddings", hp.llm_max_pos);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "moss_tts_local.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "moss_tts_local.llm.rms_norm_eps", hp.llm_rms_eps);

        hp.local_layers = core_gguf::kv_u32(gctx, "moss_tts_local.local.num_layers", hp.local_layers);
        hp.local_hidden = core_gguf::kv_u32(gctx, "moss_tts_local.local.hidden_size", hp.local_hidden);
        hp.local_n_heads = core_gguf::kv_u32(gctx, "moss_tts_local.local.num_heads", hp.local_n_heads);
        hp.local_ff_dim = core_gguf::kv_u32(gctx, "moss_tts_local.local.intermediate_size", hp.local_ff_dim);
        hp.local_rope_base = core_gguf::kv_f32(gctx, "moss_tts_local.local.rope_base", hp.local_rope_base);
        hp.local_ln_eps = core_gguf::kv_f32(gctx, "moss_tts_local.local.layer_norm_eps", hp.local_ln_eps);

        hp.n_vq = core_gguf::kv_u32(gctx, "moss.n_vq", hp.n_vq);
        hp.audio_vocab_size = core_gguf::kv_u32(gctx, "moss.audio_vocab_size", hp.audio_vocab_size);
        hp.audio_pad_code = core_gguf::kv_u32(gctx, "moss.audio_pad_code", hp.audio_pad_code);
        hp.sampling_rate = core_gguf::kv_u32(gctx, "moss.sampling_rate", hp.sampling_rate);
        hp.downsample_rate = core_gguf::kv_u32(gctx, "moss.downsample_rate", hp.downsample_rate);

        hp.tok_audio_start = core_gguf::kv_u32(gctx, "moss.token.audio_start", hp.tok_audio_start);
        hp.tok_audio_end = core_gguf::kv_u32(gctx, "moss.token.audio_end", hp.tok_audio_end);
        hp.tok_audio_user_slot = core_gguf::kv_u32(gctx, "moss.token.audio_user_slot", hp.tok_audio_user_slot);
        hp.tok_audio_gen_slot = core_gguf::kv_u32(gctx, "moss.token.audio_gen_slot", hp.tok_audio_gen_slot);
        hp.tok_im_start = core_gguf::kv_u32(gctx, "moss.token.im_start", hp.tok_im_start);
        hp.tok_im_end = core_gguf::kv_u32(gctx, "moss.token.im_end", hp.tok_im_end);
        hp.tok_pad = core_gguf::kv_u32(gctx, "moss.token.pad", hp.tok_pad);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++)
                vocab.token_to_id[vocab.id_to_token[i]] = i;
        }
        auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++)
            vocab.merge_rank[merges[i]] = i;
        core_gguf::free_metadata(gctx);
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "moss_tts_local", wl))
        return false;
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    model.token_embd_w = mtl_req(model, "llm.embed.weight");
    model.output_norm_w = mtl_req(model, "llm.final_norm.weight");
    model.output_w = mtl_req(model, "llm.lm_head.weight");

    model.blocks.resize(model.hparams.llm_layers);
    for (uint32_t i = 0; i < model.hparams.llm_layers; i++) {
        char buf[128];
        auto& b = model.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "llm.blk.%u.%s", i, suf);
            return mtl_req(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn.q.weight");
        b.attn_k_w = get("attn.k.weight");
        b.attn_v_w = get("attn.v.weight");
        b.attn_output_w = get("attn.o.weight");
        b.attn_q_norm_w = get("attn.q_norm.weight");
        b.attn_k_norm_w = get("attn.k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn.gate.weight");
        b.ffn_up_w = get("ffn.up.weight");
        b.ffn_down_w = get("ffn.down.weight");
    }

    model.local_blocks.resize(model.hparams.local_layers);
    for (uint32_t i = 0; i < model.hparams.local_layers; i++) {
        char buf[128];
        auto& b = model.local_blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "local.blk.%u.%s", i, suf);
            return mtl_req(model, buf);
        };
        b.ln1_w = get("attn_norm.weight");
        b.ln1_b = get("attn_norm.bias");
        b.qkv_w = get("attn.qkv.weight");
        b.qkv_b = get("attn.qkv.bias");
        b.o_w = get("attn.o.weight");
        b.o_b = get("attn.o.bias");
        b.ln2_w = get("ffn_norm.weight");
        b.ln2_b = get("ffn_norm.bias");
        b.fc_in_w = get("ffn.in.weight");
        b.fc_in_b = get("ffn.in.bias");
        b.fc_out_w = get("ffn.out.weight");
        b.fc_out_b = get("ffn.out.bias");
    }
    model.local_final_norm_w = mtl_req(model, "local.final_norm.weight");
    model.local_final_norm_b = mtl_req(model, "local.final_norm.bias");

    model.audio_embed.resize(model.hparams.n_vq);
    model.audio_head.resize(model.hparams.n_vq);
    for (uint32_t k = 0; k < model.hparams.n_vq; k++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "moss.audio_embed.%u.weight", k);
        model.audio_embed[k] = mtl_req(model, buf);
        snprintf(buf, sizeof(buf), "moss.audio_head.%u.weight", k);
        model.audio_head[k] = mtl_req(model, buf);
    }
    model.local_text_head_w = mtl_req(model, "moss.local_text_head.weight");
    return true;
}

// ===========================================================================
// Backbone Qwen3 graph (KV-cached) — same structure as moss_tts.cpp
// ===========================================================================

static ggml_cgraph* mtl_build_graph_llm_kv(moss_tts_local_context* ctx, int n_past, int n_tokens) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_hidden;
    const int n_q = (int)hp.llm_n_heads;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;
    const float eps = hp.llm_rms_eps;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = n_past + T;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    const core_attn::KvSelfAttnParams kvp = {
        n_q,   n_kv, hd,         n_q / n_kv, (int)hp.llm_max_pos,        hp.llm_rope_theta,
        32.0f, 1.0f, attn_scale, eps,        core_attn::GQA_MANUAL_CONT,
    };

    // #249 per-layer diff: on the prompt prefill, expose each block's output so
    // mtl_run_backbone can dump it and compare to the reference per layer.
    const bool dump_layers = (n_past == 0) && (getenv("CRISPASR_MOSS_TTS_LOCAL_DUMP_LAYERS") != nullptr);
    // #249 option 2 — pin the layer-10 op: for one target layer, expose the
    // attention output and the SwiGLU-MLP output (both pre-residual, last
    // position) so the per-sublayer diff can tell whether attention or the MLP is
    // where the block first diverges from the reference (block input matches).
    int dump_sub = -1;
    if (n_past == 0) {
        if (const char* s = getenv("CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER"))
            dump_sub = atoi(s);
    }
    // #249 gold-standard test: inject the REFERENCE's exact block-(L-1) output as
    // the input to block L, so block L computes from a byte-identical input. If
    // block L's output then matches the reference, layer L is correct and the
    // divergence is pure amplification of accumulated drift; if it still diverges,
    // there is a real per-layer op bug. inject_in is filled in mtl_run_backbone.
    int inject_layer = -1;
    if (n_past == 0) {
        if (const char* s = getenv("CRISPASR_MOSS_TTS_LOCAL_INJECT_LAYER"))
            inject_layer = atoi(s);
    }
    ggml_tensor* inject_in = nullptr;
    if (inject_layer >= 0) {
        inject_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
        ggml_set_name(inject_in, "inject_in");
        ggml_set_input(inject_in);
    }
    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.llm_layers; il++) {
        const auto& b = m.blocks[il];
        if ((int)il == inject_layer && inject_in)
            cur = inject_in; // block L reads the reference's exact input
        ggml_tensor* residual = cur;
        ggml_tensor* x = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), b.attn_norm_w);
        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w, b.attn_k_norm_w,
            positions, (T == 1) ? nullptr : causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, kvp, nullptr);
        cur = ggml_add(ctx0, residual, attn);
        residual = cur;
        x = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
        if (dump_sub == (int)il) {
            char nm[24];
            snprintf(nm, sizeof(nm), "sub_attn_%u", il);
            ggml_set_name(attn, nm);
            ggml_set_output(attn);
            ggml_build_forward_expand(gf, attn);
            snprintf(nm, sizeof(nm), "sub_mlp_%u", il);
            ggml_set_name(mlp, nm);
            ggml_set_output(mlp);
            ggml_build_forward_expand(gf, mlp);
        }
        if (dump_layers) {
            char nm[24];
            snprintf(nm, sizeof(nm), "blk_%u", il);
            ggml_set_name(cur, nm);
            ggml_set_output(cur);
            ggml_build_forward_expand(gf, cur);
        }
    }
    cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), m.output_norm_w);
    if (T > 1)
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    ggml_tensor* hidden_last = ggml_cont(ctx0, cur);
    ggml_set_name(hidden_last, "hidden_last");
    ggml_set_output(hidden_last);
    ggml_build_forward_expand(gf, hidden_last);
    ggml_free(ctx0);
    return gf;
}

static bool mtl_kv_init(moss_tts_local_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    if (ctx->kv_k && ctx->kv_max_ctx >= max_ctx)
        return true;
    if (ctx->kv_buf) {
        ggml_backend_buffer_free(ctx->kv_buf);
        ctx->kv_buf = nullptr;
    }
    if (ctx->kv_ctx) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
    }
    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int nl = (int)hp.llm_layers;
    // #249: the 4B's marginal binary stop head is sensitive to KV precision —
    // an f16 KV cache measurably delays/breaks the wind-down (an f32 KV moves
    // the "Hello world" stop earlier, toward the reference). Default to F32 KV;
    // CRISPASR_KV_QUANT still overrides for users who want the smaller cache.
    const ggml_type kv_type = core_attn::kv_dtype_parse(std::getenv("CRISPASR_KV_QUANT"), "moss_tts_local",
                                                        "CRISPASR_KV_QUANT", GGML_TYPE_F32);
    ggml_init_params ip = {ggml_tensor_overhead() * 4, nullptr, true};
    ctx->kv_ctx = ggml_init(ip);
    if (!ctx->kv_ctx)
        return false;
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, nl);
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf)
        return false;
    ctx->kv_max_ctx = max_ctx;
    return true;
}

// Run the backbone; returns malloc'd (hidden,) F32 last hidden (or null).
static float* mtl_run_backbone(moss_tts_local_context* ctx, const float* inputs_embeds, int n_tokens, int n_past) {
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_hidden;
    const int Lk = n_past + n_tokens;
    if (!ctx->kv_k || Lk > ctx->kv_max_ctx)
        return nullptr;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), neg = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, z);
        for (int q = 0; q < n_tokens; q++)
            for (int k = n_past + q + 1; k < Lk; k++)
                mask[(size_t)q * Lk + k] = neg;
    }
    ggml_cgraph* gf = mtl_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), inputs_embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1)
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    // #249 gold-standard test: load the reference block-(L-1) output (all positions,
    // layout [pos][dim] == our (d, T) ggml order) into the injection tensor.
    if (n_past == 0) {
        if (ggml_tensor* it = ggml_graph_get_tensor(gf, "inject_in")) {
            const char* ip = getenv("CRISPASR_MOSS_TTS_LOCAL_INJECT_PATH");
            if (ip) {
                std::vector<float> buf((size_t)d * n_tokens);
                if (FILE* f = fopen(ip, "rb")) {
                    const size_t got = fread(buf.data(), sizeof(float), buf.size(), f);
                    fclose(f);
                    if (got == buf.size())
                        ggml_backend_tensor_set(it, buf.data(), 0, buf.size() * sizeof(float));
                    else
                        fprintf(stderr, "moss_tts_local[inject] short read %zu/%zu\n", got, buf.size());
                }
            }
        }
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* h_t = ggml_graph_get_tensor(gf, "hidden_last");
    float* out = (float*)malloc((size_t)d * sizeof(float));
    ggml_backend_tensor_get(h_t, out, 0, (size_t)d * sizeof(float));
    // #249: dump each block's last-position hidden on the prompt prefill.
    if (n_past == 0) {
        if (const char* lp = getenv("CRISPASR_MOSS_TTS_LOCAL_DUMP_LAYERS")) {
            if (FILE* lf = fopen(lp, "w")) {
                std::vector<float> buf(d);
                for (uint32_t il = 0; il < ctx->model.hparams.llm_layers; il++) {
                    char nm[24];
                    snprintf(nm, sizeof(nm), "blk_%u", il);
                    ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
                    if (!t)
                        continue;
                    ggml_backend_tensor_get(t, buf.data(), (size_t)(n_tokens - 1) * d * sizeof(float),
                                            (size_t)d * sizeof(float));
                    fprintf(lf, "blk_%u", il);
                    for (int i = 0; i < d; i++)
                        fprintf(lf, " %.6f", buf[i]);
                    fprintf(lf, "\n");
                }
                fclose(lf);
            }
        }
        // #249 option 2: dump one target layer's attention + MLP outputs.
        if (const char* sl = getenv("CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER")) {
            if (const char* sp = getenv("CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER_PATH")) {
                const int il = atoi(sl);
                if (FILE* sf = fopen(sp, "w")) {
                    std::vector<float> buf(d);
                    for (const char* which : {"sub_attn", "sub_mlp"}) {
                        char nm[24];
                        snprintf(nm, sizeof(nm), "%s_%d", which, il);
                        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
                        if (!t)
                            continue;
                        ggml_backend_tensor_get(t, buf.data(), (size_t)(n_tokens - 1) * d * sizeof(float),
                                                (size_t)d * sizeof(float));
                        fprintf(sf, "%s", nm);
                        for (int i = 0; i < d; i++)
                            fprintf(sf, " %.6f", buf[i]);
                        fprintf(sf, "\n");
                    }
                    fclose(sf);
                }
            }
        }
        // #249 option 2: dump the shared-attention debug tensors (named by the
        // core_attn CRISPASR_CORE_ATTN_DUMP_FA_LAYER hook) so a numpy flash-vs-eager
        // self-check and the Q/K/V-vs-reference diff can localize the attention op.
        if (const char* fp = getenv("CRISPASR_MOSS_TTS_LOCAL_DUMP_FA_PATH")) {
            if (FILE* ff = fopen(fp, "w")) {
                for (const char* nm : {"DBG_Q_prerope", "DBG_K_prerope", "DBG_V_new", "DBG_Q_post_rope", "DBG_Kfull",
                                       "DBG_Vfull", "DBG_fa_out", "DBG_fa_reshaped"}) {
                    ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
                    if (!t)
                        continue;
                    const int64_t n = ggml_nelements(t);
                    std::vector<float> buf((size_t)n);
                    ggml_backend_tensor_get(t, buf.data(), 0, (size_t)n * sizeof(float));
                    fprintf(ff, "%s %lld %lld %lld %lld", nm, (long long)t->ne[0], (long long)t->ne[1],
                            (long long)t->ne[2], (long long)t->ne[3]);
                    for (int64_t i = 0; i < n; i++)
                        fprintf(ff, " %.6f", buf[(size_t)i]);
                    fprintf(ff, "\n");
                }
                fclose(ff);
            }
        }
    }
    return out;
}

// ===========================================================================
// Summed input embedding for a (n_pos, 1+n_vq) grid. PAD channels (== pad_code)
// are masked to zero (the audio_embed table has NO pad row — 4B specific).
// Returns malloc'd (n_pos, hidden) F32 row-major (ne0=hidden per position).
// ===========================================================================

static float* mtl_input_embeddings(moss_tts_local_context* ctx, const int32_t* grid, int n_pos) {
    const auto& m = ctx->model;
    const int n_vq = (int)m.hparams.n_vq;
    const int hidden = (int)m.hparams.llm_hidden;
    const int pad = (int)m.hparams.audio_pad_code;
    const int stride = 1 + n_vq;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8 * n_vq + 64, false);

    ggml_tensor* text_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(text_ids, "text_ids");
    ggml_set_input(text_ids);
    std::vector<ggml_tensor*> audio_ids(n_vq), audio_mask(n_vq);
    for (int i = 0; i < n_vq; i++) {
        audio_ids[i] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        audio_mask[i] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_pos); // broadcast over hidden
        char nm[32];
        snprintf(nm, sizeof(nm), "aid_%d", i);
        ggml_set_name(audio_ids[i], nm);
        ggml_set_input(audio_ids[i]);
        snprintf(nm, sizeof(nm), "amask_%d", i);
        ggml_set_name(audio_mask[i], nm);
        ggml_set_input(audio_mask[i]);
    }

    ggml_tensor* out = ggml_cast(ctx0, ggml_get_rows(ctx0, m.token_embd_w, text_ids), GGML_TYPE_F32);
    for (int i = 0; i < n_vq; i++) {
        ggml_tensor* e = ggml_cast(ctx0, ggml_get_rows(ctx0, m.audio_embed[i], audio_ids[i]), GGML_TYPE_F32);
        e = ggml_mul(ctx0, e, audio_mask[i]); // zero out pad positions
        out = ggml_add(ctx0, out, e);
    }
    ggml_set_name(out, "input_embeds");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;
    std::vector<int32_t> col(n_pos);
    std::vector<float> mcol(n_pos);
    for (int r = 0; r < n_pos; r++)
        col[r] = grid[r * stride + 0];
    ggml_backend_tensor_set(text_ids, col.data(), 0, (size_t)n_pos * sizeof(int32_t));
    for (int i = 0; i < n_vq; i++) {
        for (int r = 0; r < n_pos; r++) {
            const int32_t id = grid[r * stride + 1 + i];
            const bool valid = (id != pad) && (id >= 0) && (id < (int)m.hparams.audio_vocab_size);
            col[r] = valid ? id : 0;
            mcol[r] = valid ? 1.0f : 0.0f;
        }
        ggml_backend_tensor_set(audio_ids[i], col.data(), 0, (size_t)n_pos * sizeof(int32_t));
        ggml_backend_tensor_set(audio_mask[i], mcol.data(), 0, (size_t)n_pos * sizeof(float));
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "input_embeds");
    float* result = (float*)malloc((size_t)n_pos * hidden * sizeof(float));
    ggml_backend_tensor_get(out_t, result, 0, (size_t)n_pos * hidden * sizeof(float));
    return result;
}

// ===========================================================================
// Local (depth) transformer forward — 1-layer GPT2-style, rebuilt over the
// growing per-frame sequence (<= n_vq+1 tokens). Returns malloc'd (hidden,) F32
// last-position hidden. inputs_embeds is (hidden, S) row-major (ne0=hidden).
// ===========================================================================

static float* mtl_local_forward(moss_tts_local_context* ctx, const float* inputs_embeds, int S) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.local_hidden;
    const int nh = (int)hp.local_n_heads;
    const int hd = d / nh;
    const float eps = hp.local_ln_eps;
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 512, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, S);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, S);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);
    ggml_tensor* mask = nullptr;
    if (S > 1) {
        mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, S, S);
        ggml_set_name(mask, "mask");
        ggml_set_input(mask);
    }

    auto layernorm = [&](ggml_tensor* t, ggml_tensor* w, ggml_tensor* b) {
        return ggml_add(ctx0, ggml_mul(ctx0, ggml_norm(ctx0, t, eps), w), b);
    };

    ggml_tensor* cur = x;
    for (uint32_t il = 0; il < hp.local_layers; il++) {
        const auto& b = m.local_blocks[il];
        ggml_tensor* residual = cur;
        ggml_tensor* h = layernorm(cur, b.ln1_w, b.ln1_b);

        ggml_tensor* qkv = ggml_add(ctx0, ggml_mul_mat(ctx0, b.qkv_w, h), b.qkv_b); // (3d, S)
        ggml_tensor* Q = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, d, S, qkv->nb[1], 0));
        ggml_tensor* K = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, d, S, qkv->nb[1], (size_t)d * qkv->nb[0]));
        ggml_tensor* V = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, d, S, qkv->nb[1], (size_t)2 * d * qkv->nb[0]));
        Q = ggml_reshape_3d(ctx0, Q, hd, nh, S);
        K = ggml_reshape_3d(ctx0, K, hd, nh, S);
        V = ggml_reshape_3d(ctx0, V, hd, nh, S);
        Q = ggml_rope_ext(ctx0, Q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.local_rope_base, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);
        K = ggml_rope_ext(ctx0, K, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.local_rope_base, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // (hd, S, nh)
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q); // (S, S, nh)
        scores = ggml_soft_max_ext(ctx0, scores, mask, scale, 0.0f);
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // (S, hd, nh)
        ggml_tensor* attn = ggml_mul_mat(ctx0, V, scores);      // (hd, S, nh)
        attn = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3)), d, S);
        attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.o_w, attn), b.o_b);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        h = layernorm(cur, b.ln2_w, b.ln2_b);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, b.fc_in_w, h), b.fc_in_b);
        h = ggml_silu(ctx0, h);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, b.fc_out_w, h), b.fc_out_b);
        cur = ggml_add(ctx0, residual, h);
    }
    cur = layernorm(cur, m.local_final_norm_w, m.local_final_norm_b);
    // last position
    if (S > 1)
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(S - 1) * cur->nb[1]);
    cur = ggml_cont(ctx0, cur);
    ggml_set_name(cur, "local_hidden");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;
    std::vector<int32_t> positions(S);
    for (int i = 0; i < S; i++)
        positions[i] = i;
    ggml_backend_tensor_set(x, inputs_embeds, 0, (size_t)d * S * sizeof(float));
    ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (S > 1) {
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), neg = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mv((size_t)S * S, z);
        for (int q = 0; q < S; q++)
            for (int k = q + 1; k < S; k++)
                mv[(size_t)q * S + k] = neg;
        ggml_backend_tensor_set(mask, mv.data(), 0, mv.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "local_hidden");
    float* out = (float*)malloc((size_t)d * sizeof(float));
    ggml_backend_tensor_get(out_t, out, 0, (size_t)d * sizeof(float));
    return out;
}

// Apply a linear head (weight (in, out)) to a (in,) hidden -> malloc'd (out,).
static float* mtl_apply_head(moss_tts_local_context* ctx, ggml_tensor* w, const float* hidden, int in_dim,
                             int out_dim) {
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16, false);
    ggml_tensor* h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, in_dim, 1);
    ggml_set_name(h, "h");
    ggml_set_input(h);
    ggml_tensor* y = ggml_mul_mat(ctx0, w, h); // (out, 1)
    ggml_set_name(y, "y");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;
    ggml_backend_tensor_set(h, hidden, 0, (size_t)in_dim * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    float* out = (float*)malloc((size_t)out_dim * sizeof(float));
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "y"), out, 0, (size_t)out_dim * sizeof(float));
    return out;
}

// Embed one audio code from codebook k -> malloc'd (hidden,) F32.
static float* mtl_embed_audio_code(moss_tts_local_context* ctx, int k, int32_t code) {
    const int d = (int)ctx->model.hparams.local_hidden;
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16, false);
    ggml_tensor* id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(id, "id");
    ggml_set_input(id);
    ggml_tensor* e = ggml_cast(ctx0, ggml_get_rows(ctx0, ctx->model.audio_embed[k], id), GGML_TYPE_F32);
    ggml_set_name(e, "e");
    ggml_set_output(e);
    ggml_build_forward_expand(gf, e);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;
    ggml_backend_tensor_set(id, &code, 0, sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    float* out = (float*)malloc((size_t)d * sizeof(float));
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "e"), out, 0, (size_t)d * sizeof(float));
    return out;
}

// ===========================================================================
// Sampling helpers (CPU)
// ===========================================================================

namespace {
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) {
        if (seed == 0) {
            std::random_device rd;
            seed = ((uint64_t)rd() << 32) | rd();
        }
        g.seed(seed);
    }
    float uniform01() { return std::uniform_real_distribution<float>(0.f, 1.f)(g); }
};

void softmax_inplace(float* x, int n) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; i++)
        if (x[i] > mx)
            mx = x[i];
    if (!std::isfinite(mx)) {
        std::fill(x, x + n, 0.f);
        return;
    }
    float sum = 0.f;
    for (int i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    if (sum <= 0.f) {
        std::fill(x, x + n, 0.f);
        return;
    }
    const float inv = 1.f / sum;
    for (int i = 0; i < n; i++)
        x[i] *= inv;
}

void apply_repetition_penalty(float* logits, int vocab, const std::vector<int32_t>& history, float penalty) {
    if (penalty == 1.0f)
        return;
    std::vector<bool> seen((size_t)vocab, false);
    for (int32_t id : history) {
        if (id < 0 || id >= vocab || seen[(size_t)id])
            continue;
        seen[(size_t)id] = true;
        logits[id] = (logits[id] > 0) ? logits[id] / penalty : logits[id] * penalty;
    }
}

int32_t sample_one(float* logits, int vocab, float temperature, float top_p, int top_k, bool do_sample, Rng& rng) {
    if (!do_sample) {
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (logits[i] > logits[best])
                best = i;
        return best;
    }
    if (temperature > 0.f && temperature != 1.f) {
        const float inv = 1.f / temperature;
        for (int i = 0; i < vocab; i++)
            logits[i] *= inv;
    }
    if (top_k > 0 && top_k < vocab) {
        std::vector<std::pair<float, int32_t>> v;
        v.reserve(vocab);
        for (int i = 0; i < vocab; i++)
            if (std::isfinite(logits[i]))
                v.emplace_back(logits[i], i);
        if ((int)v.size() > top_k) {
            std::nth_element(v.begin(), v.begin() + top_k, v.end(), [](auto& a, auto& b) { return a.first > b.first; });
            v.resize(top_k);
        }
        std::vector<bool> keep(vocab, false);
        for (auto& p : v)
            keep[p.second] = true;
        for (int i = 0; i < vocab; i++)
            if (!keep[i])
                logits[i] = -std::numeric_limits<float>::infinity();
    }
    softmax_inplace(logits, vocab);
    if (top_p > 0.f && top_p < 1.f) {
        std::vector<int32_t> order(vocab);
        for (int i = 0; i < vocab; i++)
            order[i] = i;
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        float acc = 0.f;
        size_t cut = order.size();
        for (size_t i = 0; i < order.size(); i++) {
            acc += logits[order[i]];
            if (acc >= top_p) {
                cut = i + 1;
                break;
            }
        }
        for (size_t i = cut; i < order.size(); i++)
            logits[order[i]] = 0.f;
        float sum = 0.f;
        for (size_t i = 0; i < cut; i++)
            sum += logits[order[i]];
        if (sum > 0.f) {
            const float inv = 1.f / sum;
            for (size_t i = 0; i < cut; i++)
                logits[order[i]] *= inv;
        }
    }
    const float u = rng.uniform01();
    float acc = 0.f;
    for (int i = 0; i < vocab; i++) {
        acc += logits[i];
        if (acc >= u)
            return i;
    }
    for (int i = vocab - 1; i >= 0; i--)
        if (logits[i] > 0.f)
            return i;
    return 0;
}
} // namespace

// ===========================================================================
// Tokenizer (Qwen3 gpt2-style BPE, special-token aware) — as moss_tts.cpp
// ===========================================================================

static std::vector<std::string> mtl_pretokenize(const std::string& s) {
    std::vector<std::string> out;
    const size_t n = s.size();
    auto is_letter = [](unsigned char c) { return std::isalpha(c) != 0 || c >= 0x80; };
    auto is_digit = [](unsigned char c) { return std::isdigit(c) != 0; };
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    auto is_nl = [](unsigned char c) { return c == '\r' || c == '\n'; };
    size_t i = 0;
    while (i < n) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '\'' && i + 1 < n) {
            static const char* cons[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            bool matched = false;
            for (const char* cc : cons) {
                const size_t len = std::strlen(cc);
                if (i + len > n)
                    continue;
                bool eq = true;
                for (size_t k = 0; k < len; k++)
                    if (std::tolower((unsigned char)s[i + k]) != std::tolower((unsigned char)cc[k])) {
                        eq = false;
                        break;
                    }
                if (eq) {
                    out.push_back(s.substr(i, len));
                    i += len;
                    matched = true;
                    break;
                }
            }
            if (matched)
                continue;
        }
        {
            size_t j = i;
            if (j < n && !is_nl((unsigned char)s[j]) && !is_letter((unsigned char)s[j]) &&
                !is_digit((unsigned char)s[j]))
                j++;
            if (j < n && is_letter((unsigned char)s[j])) {
                while (j < n && is_letter((unsigned char)s[j]))
                    j++;
                out.push_back(s.substr(i, j - i));
                i = j;
                continue;
            }
        }
        if (is_digit(c)) {
            out.push_back(s.substr(i, 1));
            i++;
            continue;
        }
        {
            size_t j = i;
            if (s[j] == ' ')
                j++;
            const size_t p0 = j;
            while (j < n && !is_space((unsigned char)s[j]) && !is_letter((unsigned char)s[j]) &&
                   !is_digit((unsigned char)s[j]))
                j++;
            if (j > p0) {
                while (j < n && is_nl((unsigned char)s[j]))
                    j++;
                out.push_back(s.substr(i, j - i));
                i = j;
                continue;
            }
        }
        {
            size_t j = i;
            while (j < n && is_space((unsigned char)s[j]) && !is_nl((unsigned char)s[j]))
                j++;
            if (j < n && is_nl((unsigned char)s[j])) {
                while (j < n && is_nl((unsigned char)s[j]))
                    j++;
                out.push_back(s.substr(i, j - i));
                i = j;
                continue;
            }
        }
        if (is_space(c)) {
            size_t j = i;
            while (j < n && is_space((unsigned char)s[j]))
                j++;
            out.push_back(s.substr(i, j - i));
            i = j;
            continue;
        }
        out.push_back(s.substr(i, 1));
        i++;
    }
    return out;
}

extern "C" int32_t* moss_tts_local_tokenize(moss_tts_local_context* ctx, const char* text, int* out_n_tokens) {
    if (!ctx || !text) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;
    auto match_special = [&](const std::string& str, size_t pos, int32_t& id) -> size_t {
        if (str[pos] != '<')
            return 0;
        const bool pipe = pos + 1 < str.size() && str[pos + 1] == '|';
        const size_t close = pipe ? str.find("|>", pos + 2) : str.find('>', pos + 1);
        if (close == std::string::npos)
            return 0;
        const size_t len = close + (pipe ? 2 : 1) - pos;
        if (!pipe && len > 24)
            return 0;
        auto it = v.token_to_id.find(str.substr(pos, len));
        if (it == v.token_to_id.end())
            return 0;
        id = it->second;
        return len;
    };
    const std::string s = text;
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '<') {
            int32_t id = 0;
            const size_t l = match_special(s, i, id);
            if (l > 0) {
                result.push_back(id);
                i += l;
                continue;
            }
        }
        size_t j = i;
        if (s[j] == '<')
            j++;
        while (j < s.size()) {
            if (s[j] == '<') {
                int32_t id = 0;
                if (match_special(s, j, id) > 0)
                    break;
            }
            j++;
        }
        std::string chunk = s.substr(i, j - i);
        i = j;
        if (chunk.empty())
            continue;
        for (const std::string& pre : mtl_pretokenize(chunk)) {
            std::string enc = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(v.token_to_id, v.merge_rank, enc, result);
        }
    }
    if (out_n_tokens)
        *out_n_tokens = (int)result.size();
    int32_t* out = (int32_t*)malloc(result.size() * sizeof(int32_t));
    if (out)
        std::memcpy(out, result.data(), result.size() * sizeof(int32_t));
    return out;
}

extern "C" const char* moss_tts_local_token_text(moss_tts_local_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[id].c_str();
}

// ===========================================================================
// Generate: depth-first RVQ code grid
// ===========================================================================

static bool mtl_generate_grid(moss_tts_local_context* ctx, const char* text, const moss_tts_local_synth_params& sp,
                              std::vector<int32_t>& out_codes, int& out_t_audio) {
    const auto& hp = ctx->model.hparams;
    const int n_vq = (int)hp.n_vq;
    const int d = (int)hp.llm_hidden;
    const int aud_v = (int)hp.audio_vocab_size;
    int max_frames = sp.max_new_frames > 0 ? sp.max_new_frames : 4096;
    if (const char* mf = getenv("CRISPASR_MOSS_TTS_LOCAL_MAX_FRAMES")) // #249 diag: bound runaways for A/B tests
        max_frames = atoi(mf) > 0 ? atoi(mf) : max_frames;
    const int stride = 1 + n_vq;
    Rng rng(sp.seed ? sp.seed : ctx->seed);
    // CRISPASR_MOSS_TTS_LOCAL_GREEDY_TEXT=1 forces the binary stop head greedy
    // (argmax) — used by the #249 trajectory diff so the stop decision is
    // deterministic and directly comparable to the HF reference run greedy.
    const bool text_greedy = (sp.text_temperature <= 0.f) || (getenv("CRISPASR_MOSS_TTS_LOCAL_GREEDY_TEXT") != nullptr);
    // CRISPASR_MOSS_TTS_LOCAL_GREEDY_AUDIO=1 forces greedy audio codebooks (A/B the
    // stop-runaway hypothesis: sampled audio feeds back and may prevent the binary
    // stop head from firing). CRISPASR_MOSS_TTS_LOCAL_DEBUG=1 traces stop logits.
    const bool audio_greedy =
        (sp.audio_temperature <= 0.f) || (getenv("CRISPASR_MOSS_TTS_LOCAL_GREEDY_AUDIO") != nullptr);
    const bool dbg = getenv("CRISPASR_MOSS_TTS_LOCAL_DEBUG") != nullptr;
    // CRISPASR_MOSS_TTS_LOCAL_DUMP_STOP=1 emits the RAW (pre-softmax) stop-head
    // logits every frame, parseable, for the reference-vs-port trajectory diff.
    const bool dump_stop = getenv("CRISPASR_MOSS_TTS_LOCAL_DUMP_STOP") != nullptr;

    // CRISPASR_MOSS_TTS_LOCAL_FORCE_FRAMES=<path>: teacher-forcing for the #249
    // trajectory diff. The file is whitespace-separated audio codes, n_vq per
    // frame; we feed those exact frames back (skipping audio sampling) and dump
    // our stop logit per frame — directly comparable to the reference run on the
    // SAME frames, isolating forward-correctness from the sampled-code choices.
    std::vector<std::vector<int32_t>> forced;
    if (const char* fp = getenv("CRISPASR_MOSS_TTS_LOCAL_FORCE_FRAMES")) {
        if (FILE* ff = fopen(fp, "r")) {
            std::vector<int32_t> row;
            int v;
            while (fscanf(ff, "%d", &v) == 1) {
                row.push_back(v);
                if ((int)row.size() == n_vq) {
                    forced.push_back(row);
                    row.clear();
                }
            }
            fclose(ff);
        }
        if (dump_stop)
            fprintf(stderr, "DUMPSTOP force_frames=%zu\n", forced.size());
    }

    // #249: PIECE-WISE prompt assembly. BPE is NOT compositional — encode(A+B) !=
    // encode(A)+encode(B) at merge boundaries — so tokenizing the whole prompt as
    // one string drifted ~2 tokens near the text segment. Those interior tokens are
    // weighted ~0 by early-layer attention but amplified by the layer-10 attention
    // sink, drifting the backbone hidden enough to break the 4B binary stop head
    // (runaway). Encode each text segment SEPARATELY and splice the special-token
    // ids in directly, mirroring the reference processor
    // (processing_moss_tts.py `_build_generation_or_voice_clone_codes`).
    std::vector<int32_t> id_vec;
    std::vector<int> ref_row; // per row: index into the reference codes, -1 for text
    // Cloning is only engaged when the reference actually matches this model's
    // codebook count — a mismatched grid would be written into the code channels
    // and read as speaker identity, producing confident nonsense rather than an
    // error. Refuse it loudly instead.
    bool has_ref = sp.ref_codes != nullptr && sp.ref_t_audio > 0 && sp.ref_n_vq > 0;
    if (has_ref && sp.ref_n_vq != n_vq) {
        fprintf(stderr, "moss_tts_local: reference has %d codebooks, model expects %d — ignoring the reference\n",
                sp.ref_n_vq, n_vq);
        has_ref = false;
    }
    auto enc = [&](const std::string& s) {
        int n = 0;
        int32_t* p = moss_tts_local_tokenize(ctx, s.c_str(), &n);
        if (p) {
            id_vec.insert(id_vec.end(), p, p + n);
            ref_row.insert(ref_row.end(), (size_t)n, -1);
            free(p);
        }
    };
    // Every text token is a row with audio_pad in the code channels; only the
    // reference block below carries codes, so the two vectors move in lockstep.
    auto push_tok = [&](uint32_t id) {
        id_vec.push_back((int32_t)id);
        ref_row.push_back(-1);
    };
    {
        const std::string instruction = sp.instruction ? std::string(sp.instruction) : "None";
        const std::string language = sp.language ? std::string(sp.language) : "None";
        const std::string after_ref = "\n- Instruction:\n" + instruction +
                                      "\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone"
                                      "\n- Ambient Sound:\nNone\n- Language:\n" +
                                      language + "\n- Text:\n";
        push_tok(hp.tok_im_start);
        enc("user\n");
        enc("<user_inst>\n- Reference(s):\n");
        // Voice cloning: the reference processor replaces the literal "None"
        // with <audio_start>, one row per reference frame carrying the codes
        // under the USER slot token, then <audio_end>
        // (processing_moss_tts.py::_build_generation_or_voice_clone_codes and
        // _build_audio_rows). Without a reference the "None" text stands.
        if (has_ref) {
            push_tok(hp.tok_audio_start);
            for (int t = 0; t < sp.ref_t_audio; t++) {
                id_vec.push_back((int32_t)hp.tok_audio_user_slot);
                ref_row.push_back(t);
            }
            push_tok(hp.tok_audio_end);
        } else {
            enc("None"); // text-only reference value, encoded alone
        }
        enc(after_ref);
        enc(text ? text : ""); // the user's text, encoded ALONE (this is the boundary that drifted)
        enc("\n</user_inst>");
        push_tok(hp.tok_im_end);
        enc("\n");
        push_tok(hp.tok_im_start);
        enc("assistant\n");
        push_tok(hp.tok_audio_start);
    }
    const int prompt_len = (int)id_vec.size();
    if (prompt_len <= 0)
        return false;
    // #249 confirm: dump the channel-0 prompt ids to diff against the reference
    // processor's input_ids[:,0] (proves piece-wise parity).
    if (const char* pp = getenv("CRISPASR_MOSS_TTS_LOCAL_DUMP_PROMPT_IDS")) {
        if (FILE* pf = fopen(pp, "w")) {
            for (int32_t t : id_vec)
                fprintf(pf, "%d ", t);
            fclose(pf);
        }
    }
    std::vector<int32_t> grid((size_t)prompt_len * stride, (int32_t)hp.audio_pad_code);
    for (int r = 0; r < prompt_len; r++) {
        grid[(size_t)r * stride] = id_vec[r];
        // Reference rows carry the cloned speaker's codes in channels 1..n_vq;
        // every other row keeps audio_pad there.
        if (ref_row[(size_t)r] >= 0)
            for (int q = 0; q < n_vq; q++)
                grid[(size_t)r * stride + 1 + q] = sp.ref_codes[(size_t)q * sp.ref_t_audio + ref_row[(size_t)r]];
    }

    if (!mtl_kv_init(ctx, prompt_len + max_frames + 8))
        return false;

    // Prefill the prompt; backbone hidden at the last position drives frame 0.
    float* emb = mtl_input_embeddings(ctx, grid.data(), prompt_len);
    if (!emb)
        return false;
    float* global_hidden = mtl_run_backbone(ctx, emb, prompt_len, 0);
    free(emb);
    if (!global_hidden)
        return false;
    int pos = prompt_len;

    std::vector<std::vector<int32_t>> frames; // (T, n_vq)
    std::vector<std::vector<int32_t>> history_per_cb(n_vq);

    for (int f = 0; f < max_frames; f++) {
        if (!forced.empty() && f >= (int)forced.size())
            break; // teacher-forcing: dumped our stop logit for every reference frame
        // Local sequence starts with the backbone hidden (position 0).
        std::vector<float> local_seq(global_hidden, global_hidden + d);
        float* lh = mtl_local_forward(ctx, local_seq.data(), 1);
        if (!lh) {
            free(global_hidden);
            return false;
        }
        // CRISPASR_MOSS_TTS_LOCAL_DUMP_HIDDEN=<path>: dump frame-0 backbone hidden
        // (global) + local-transformer output (local) for the #249 per-component
        // diff vs the HF reference — localizes whether the backbone or the depth
        // transformer diverges.
        if (f == 0) {
            if (const char* dhp = getenv("CRISPASR_MOSS_TTS_LOCAL_DUMP_HIDDEN")) {
                if (FILE* hf = fopen(dhp, "w")) {
                    fprintf(hf, "GLOBAL");
                    for (int i = 0; i < d; i++)
                        fprintf(hf, " %.6f", global_hidden[i]);
                    fprintf(hf, "\nLOCAL");
                    for (int i = 0; i < d; i++)
                        fprintf(hf, " %.6f", lh[i]);
                    fprintf(hf, "\n");
                    fclose(hf);
                }
            }
        }
        // Binary continue/stop head: index 0 = assistant_slot (continue), 1 = audio_end.
        float* tl = mtl_apply_head(ctx, ctx->model.local_text_head_w, lh, d, 2);
        // Capture the RAW logits before sample_one softmaxes them in place.
        const float raw_cont = tl ? tl[0] : 0.f;
        const float raw_stop = tl ? tl[1] : 0.f;
        const int stop_idx =
            tl ? sample_one(tl, 2, sp.text_temperature, sp.text_top_p, sp.text_top_k, !text_greedy, rng) : 1;
        if (dump_stop)
            fprintf(stderr, "DUMPSTOP frame=%d cont=%.6f stop=%.6f gap=%.6f\n", f, raw_cont, raw_stop,
                    raw_cont - raw_stop);
        else if (dbg && tl && (f < 8 || f % 64 == 0))
            fprintf(stderr, "moss_tts_local[dbg] frame %d: stop_head continue=%.4f stop=%.4f -> %s\n", f, tl[0], tl[1],
                    stop_idx == 1 ? "STOP" : "cont");
        free(tl);
        // audio_end -> stop (ignored under teacher-forcing). min_audio_frames
        // overrides the stop head: keep generating until the floor is reached —
        // paired with max_audio_frames it yields exact-duration synthesis.
        if (stop_idx == 1 && forced.empty() && (int)frames.size() >= sp.min_audio_frames) {
            free(lh);
            if (dbg)
                fprintf(stderr, "moss_tts_local[dbg] stop head fired at frame %d\n", f);
            break;
        }

        std::vector<int32_t> frame(n_vq, 0);
        for (int k = 0; k < n_vq; k++) {
            float* al = mtl_apply_head(ctx, ctx->model.audio_head[k], lh, d, aud_v);
            if (!al) {
                free(lh);
                free(global_hidden);
                return false;
            }
            if (sp.audio_repetition_penalty != 1.0f)
                apply_repetition_penalty(al, aud_v, history_per_cb[k], sp.audio_repetition_penalty);
            const int32_t code = forced.empty() ? sample_one(al, aud_v, sp.audio_temperature, sp.audio_top_p,
                                                             sp.audio_top_k, !audio_greedy, rng)
                                                : forced[f][k]; // teacher-forced: feed the reference's own code
            free(al);
            frame[k] = code;
            history_per_cb[k].push_back(code);
            if (k + 1 < n_vq) {
                float* ae = mtl_embed_audio_code(ctx, k, code);
                if (!ae) {
                    free(lh);
                    free(global_hidden);
                    return false;
                }
                local_seq.insert(local_seq.end(), ae, ae + d);
                free(ae);
                free(lh);
                lh = mtl_local_forward(ctx, local_seq.data(), (int)(local_seq.size() / d));
                if (!lh) {
                    free(global_hidden);
                    return false;
                }
            }
        }
        free(lh);
        frames.push_back(frame);
        if (sp.max_audio_frames > 0 && (int)frames.size() >= sp.max_audio_frames)
            break;

        // Feed the frame back into the backbone: row = [gen_slot, code_0..code_{n_vq-1}].
        std::vector<int32_t> row(stride, (int32_t)hp.audio_pad_code);
        row[0] = (int32_t)hp.tok_audio_gen_slot;
        for (int k = 0; k < n_vq; k++)
            row[1 + k] = frame[k];
        float* row_emb = mtl_input_embeddings(ctx, row.data(), 1);
        if (!row_emb) {
            free(global_hidden);
            return false;
        }
        free(global_hidden);
        global_hidden = mtl_run_backbone(ctx, row_emb, 1, pos);
        free(row_emb);
        if (!global_hidden)
            return false;
        pos++;
    }
    free(global_hidden);

    out_t_audio = (int)frames.size();
    if (dbg)
        fprintf(stderr, "moss_tts_local[dbg] generated %d frames (max_frames=%d, %s)\n", out_t_audio, max_frames,
                out_t_audio >= max_frames ? "HIT CAP — stop head never fired (runaway)" : "stopped naturally");
    if (out_t_audio <= 0)
        return false;
    out_codes.assign((size_t)n_vq * out_t_audio, 0);
    for (int t = 0; t < out_t_audio; t++)
        for (int k = 0; k < n_vq; k++)
            out_codes[(size_t)k * out_t_audio + t] = frames[t][k];
    return true;
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" moss_tts_local_context_params moss_tts_local_context_default_params(void) {
    moss_tts_local_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.flash_attn = true;
    return p;
}

extern "C" moss_tts_local_synth_params moss_tts_local_synth_default_params(void) {
    moss_tts_local_synth_params p = {};
    // Defaults per the model card (OpenMOSS MOSS-TTS-Local-Transformer-v1.5
    // "Generation Parameters"): audio 1.7 / 0.8 / 25, do_sample=True with the
    // binary stop head SAMPLED at text_temperature 1.0. Using a too-low audio
    // temperature (the old generic 1.0/0.95/50) produces a degenerate acoustic
    // trajectory that never reaches a natural end, so the stop head never fires
    // and generation runs away (P5 run1/run2).
    p.max_new_frames = 4096;
    p.text_temperature = 1.0f; // SAMPLED continue/stop (reference default)
    p.text_top_p = 1.0f;
    p.text_top_k = 50;
    p.audio_temperature = 1.7f;
    p.audio_top_p = 0.8f;
    p.audio_top_k = 25;
    p.audio_repetition_penalty = 1.0f;
    p.min_audio_frames = 0;
    p.max_audio_frames = 0;
    p.seed = 0;
    p.language = nullptr;
    p.instruction = nullptr;
    p.ref_codes = nullptr;
    p.ref_n_vq = 0;
    p.ref_t_audio = 0;
    return p;
}

extern "C" moss_tts_local_context* moss_tts_local_init_from_file(const char* path,
                                                                 moss_tts_local_context_params params) {
    moss_tts_local_context* ctx = new moss_tts_local_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : core_cpu_backend::init();
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    ctx->backend_cpu = core_cpu_backend::init();
    if (ctx->backend_cpu)
        core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (core_cpu_backend::is_cpu(ctx->backend))
        core_cpu_backend::set_n_threads(ctx->backend, ctx->n_threads);
    if (!mtl_load_model(ctx->model, ctx->vocab, path, ctx->backend)) {
        moss_tts_local_free(ctx); // frees the backends this ctx already owns
        return nullptr;
    }
    int n_be = 0;
    ggml_backend_t backends[2];
    backends[n_be++] = ctx->backend;
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        backends[n_be++] = ctx->backend_cpu;
    // The codec (query-chunked attention) reuses this sched and can emit tens of
    // thousands of nodes for long/runaway audio, so its hash-set must be sized for
    // the codec, not just the small per-frame backbone graph (else the codec decode
    // aborts on GGML_ASSERT(hash_set.size >= n_nodes + n_leafs)).
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 262144, false, false);
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    if (params.verbosity >= 1)
        fprintf(stderr, "moss_tts_local: loaded %s (llm %u layers d=%u, local %u layer, n_vq=%u)\n",
                path ? path : "(null)", ctx->model.hparams.llm_layers, ctx->model.hparams.llm_hidden,
                ctx->model.hparams.local_layers, ctx->model.hparams.n_vq);
    return ctx;
}

extern "C" void moss_tts_local_free(moss_tts_local_context* ctx) {
    if (!ctx)
        return;
    if (ctx->codec)
        moss_tts_local_codec::free(ctx->codec);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)
        core_gguf::release_weight_buffer(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" bool moss_tts_local_set_codec_path(moss_tts_local_context* ctx, const char* path_codec) {
    if (!ctx || !path_codec)
        return false;
    ctx->codec_path = path_codec;
    if (ctx->codec) {
        moss_tts_local_codec::free(ctx->codec);
        ctx->codec = nullptr;
    }
    ctx->codec = moss_tts_local_codec::load(path_codec, ctx->backend, ctx->sched, ctx->params.verbosity);
    ctx->codec_loaded = (ctx->codec != nullptr);
    return ctx->codec_loaded;
}

extern "C" int moss_tts_local_n_vq(const moss_tts_local_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_vq : 0;
}
extern "C" int moss_tts_local_hidden_size(const moss_tts_local_context* ctx) {
    return ctx ? (int)ctx->model.hparams.llm_hidden : 0;
}
extern "C" int moss_tts_local_audio_vocab_size(const moss_tts_local_context* ctx) {
    return ctx ? (int)ctx->model.hparams.audio_vocab_size : 0;
}
extern "C" int moss_tts_local_sampling_rate(const moss_tts_local_context* ctx) {
    return ctx ? (int)ctx->model.hparams.sampling_rate : 0;
}
extern "C" bool moss_tts_local_codec_loaded(const moss_tts_local_context* ctx) {
    return ctx && ctx->codec_loaded;
}
extern "C" bool moss_tts_local_can_clone(const moss_tts_local_context* ctx) {
    return ctx && ctx->codec_loaded && ctx->codec && moss_tts_local_codec::encoder_ready(ctx->codec);
}

extern "C" int32_t* moss_tts_local_encode_reference(moss_tts_local_context* ctx, const float* pcm_mono, int n_samples,
                                                    int sample_rate, int* out_n_vq, int* out_t_audio) {
    if (out_n_vq)
        *out_n_vq = 0;
    if (out_t_audio)
        *out_t_audio = 0;
    if (!moss_tts_local_can_clone(ctx) || !pcm_mono || n_samples <= 0 || sample_rate <= 0)
        return nullptr;

    // processing_moss_tts.py::encode_audios_from_wav — resample to the codec
    // rate, loudness-normalise, then duplicate mono across the codec's channels.
    const int target_sr = moss_tts_local_codec::sampling_rate(ctx->codec);
    std::vector<float> mono(pcm_mono, pcm_mono + n_samples);
    if (sample_rate != target_sr) {
        mono = core_audio::resample_polyphase(mono.data(), (int)mono.size(), sample_rate, target_sr);
        if (mono.empty())
            return nullptr;
    }

    // loudness_normalize(target_dbfs=-20, gain_range=(-3,+3)) — RMS power dB.
    // Duplicating mono across channels leaves the mean square unchanged, so the
    // gain is the same computed here or after interleaving.
    double sumsq = 0.0;
    for (float v : mono)
        sumsq += (double)v * (double)v;
    const double cur_dbfs = 10.0 * std::log10(sumsq / (double)mono.size() + 1e-9);
    double gain_db = -20.0 - cur_dbfs;
    gain_db = std::max(-3.0, std::min(gain_db, 3.0));
    const float gain = (float)std::pow(10.0, gain_db / 20.0);
    for (float& v : mono)
        v *= gain;

    const int nch = moss_tts_local_codec::num_channels(ctx->codec);
    std::vector<float> inter((size_t)mono.size() * (size_t)nch);
    for (size_t i = 0; i < mono.size(); i++)
        for (int c = 0; c < nch; c++)
            inter[i * (size_t)nch + (size_t)c] = mono[i];

    int n_vq = 0, t_audio = 0;
    std::vector<int32_t> codes =
        moss_tts_local_codec::encode(ctx->codec, inter.data(), (int64_t)inter.size(), n_vq, t_audio);
    if (codes.empty() || n_vq <= 0 || t_audio <= 0)
        return nullptr;

    int32_t* out = (int32_t*)malloc(codes.size() * sizeof(int32_t));
    if (!out)
        return nullptr;
    memcpy(out, codes.data(), codes.size() * sizeof(int32_t));
    if (out_n_vq)
        *out_n_vq = n_vq;
    if (out_t_audio)
        *out_t_audio = t_audio;
    return out;
}

extern "C" void moss_tts_local_set_seed(moss_tts_local_context* ctx, uint32_t seed) {
    if (ctx)
        ctx->seed = seed;
}

extern "C" int32_t* moss_tts_local_generate_codes(moss_tts_local_context* ctx, const char* text,
                                                  const moss_tts_local_synth_params* sp_in, int* out_n_vq,
                                                  int* out_t_audio) {
    if (out_n_vq)
        *out_n_vq = 0;
    if (out_t_audio)
        *out_t_audio = 0;
    if (!ctx || !text)
        return nullptr;
    moss_tts_local_synth_params sp = sp_in ? *sp_in : moss_tts_local_synth_default_params();
    std::vector<int32_t> codes;
    int t_audio = 0;
    if (!mtl_generate_grid(ctx, text, sp, codes, t_audio))
        return nullptr;
    const int n_vq = (int)ctx->model.hparams.n_vq;
    int32_t* out = (int32_t*)malloc(codes.size() * sizeof(int32_t));
    if (!out)
        return nullptr;
    std::memcpy(out, codes.data(), codes.size() * sizeof(int32_t));
    if (out_n_vq)
        *out_n_vq = n_vq;
    if (out_t_audio)
        *out_t_audio = t_audio;
    return out;
}

extern "C" float* moss_tts_local_synthesize(moss_tts_local_context* ctx, const char* text,
                                            const moss_tts_local_synth_params* sp_in, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx || !text)
        return nullptr;
    if (!ctx->codec_loaded || !ctx->codec) {
        fprintf(stderr, "moss_tts_local: codec not loaded — call moss_tts_local_set_codec_path() with the "
                        "companion MOSS-Audio-Tokenizer-v2 GGUF\n");
        return nullptr;
    }
    moss_tts_local_synth_params sp = sp_in ? *sp_in : moss_tts_local_synth_default_params();

    // 1) Backbone + local transformer -> (n_vq, t_audio) RVQ code grid.
    std::vector<int32_t> codes;
    int t_audio = 0;
    if (!mtl_generate_grid(ctx, text, sp, codes, t_audio) || t_audio <= 0 || codes.empty())
        return nullptr;
    const int n_vq = (int)ctx->model.hparams.n_vq;

    // 2) Codec decode -> channel-interleaved [L0,R0,L1,R1,...] @ 48 kHz.
    std::vector<float> inter = moss_tts_local_codec::decode(ctx->codec, codes.data(), n_vq, t_audio);
    if (inter.empty())
        return nullptr;

    // 3) De-interleave stereo -> mono downmix (the public API is mono).
    const int nch = moss_tts_local_codec::num_channels(ctx->codec);
    const size_t n_mono = (nch > 0) ? inter.size() / (size_t)nch : inter.size();
    float* out = (float*)malloc(n_mono * sizeof(float));
    if (!out)
        return nullptr;
    if (nch == 2) {
        for (size_t k = 0; k < n_mono; k++)
            out[k] = 0.5f * (inter[2 * k] + inter[2 * k + 1]);
    } else {
        std::memcpy(out, inter.data(), n_mono * sizeof(float));
    }
    if (out_n_samples)
        *out_n_samples = (int)n_mono;
    return out;
}

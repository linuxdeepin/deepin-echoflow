// irodori_tts.cpp — native ggml runtime for Aratako/Irodori-TTS.
//
// Architecture:
//   1. TextEncoder: Embedding(102400, 1280) + 14 TextBlock (RoPE self-attn + SwiGLU)
//   2. ReferenceLatentEncoder: Linear(128→1280) + 14 TextBlock (same structure)
//   3. TimestepEmbedding: sinusoidal(512) → Linear(512,2048) → SiLU → Linear(2048,2048) → SiLU → Linear(2048,6144)
//   4. DiT: 24 DiffusionBlock, each:
//      - LowRankAdaLN(rank=256): timestep → shift/scale/gate via low-rank residual MLP
//      - JointAttention: self-KV + text-context-KV + speaker-context-KV, half-RoPE
//      - SwiGLU(2048, 5888)
//   5. DurationPredictor: token_sum with speaker AdaRN-Zero
//   6. Euler RF ODE solver: 40 steps with independent CFG
//   7. DAC-VAE decoder (separate model, handled externally for now)
//
// The implementation builds ggml graphs per inference stage using gallocr.

#include "irodori_tts.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "core/audio_resample.h"
#include "core/dac_decoder.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)
#include "core/sentencepiece.h"
#include "core/spm_bpe.h"
#include "core/tts_ref_cache.h"

#ifdef IRODORI_HAVE_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Debug gating ────────────────────────────────────────────────────

static bool irodori_debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_IRODORI_DEBUG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

#define IRODORI_DBG(...)                                                                                               \
    do {                                                                                                               \
        if (irodori_debug_enabled())                                                                                   \
            std::fprintf(stderr, __VA_ARGS__);                                                                         \
    } while (0)

// ── Hyperparameters ─────────────────────────────────────────────────

struct irodori_hparams {
    int latent_dim = 128;
    int latent_patch_size = 1;
    int model_dim = 2048;
    int num_layers = 24;
    int num_heads = 16;
    float mlp_ratio = 2.875f;
    int text_dim = 1280;
    int text_layers = 14;
    int text_heads = 10;
    float text_mlp_ratio = 2.6f;
    int text_vocab_size = 102400;
    int speaker_dim = 1280;
    int speaker_layers = 14;
    int speaker_heads = 10;
    float speaker_mlp_ratio = 2.6f;
    int speaker_patch_size = 1;
    int timestep_embed_dim = 512;
    int adaln_rank = 256;
    float norm_eps = 1e-5f;
    int use_duration_predictor = 1;
    int duration_aux_dim = 14;
    int duration_hidden_dim = 1024;
    int duration_layers = 3;
    int ode_steps = 40;
    float cfg_scale_text = 3.0f;
    float cfg_scale_speaker = 5.0f;
    int sample_rate = 48000;
    int codec_hop_length = 1920; // Semantic-DACVAE-Japanese-32dim: strides [12,10,8,2] = 1920

    // Caption encoder (VoiceDesign)
    int use_caption_condition = 0;
    int caption_dim = 512;
    int caption_layers = 10;
    int caption_heads = 8;
    float caption_mlp_ratio = 2.6f;
    int caption_vocab_size = 99574;
    float cfg_scale_caption = 3.0f;

    int patched_latent_dim() const { return latent_dim * latent_patch_size; }
    int head_dim() const { return model_dim / num_heads; }
    int text_head_dim() const { return text_dim / text_heads; }
    int speaker_head_dim() const { return speaker_dim / speaker_heads; }
    int caption_head_dim() const { return caption_dim / caption_heads; }
    int ff_dim() const { return (int)(model_dim * mlp_ratio); }
    int text_ff_dim() const { return (int)(text_dim * text_mlp_ratio); }
    int speaker_ff_dim() const { return (int)(speaker_dim * speaker_mlp_ratio); }
    int caption_ff_dim() const { return (int)(caption_dim * caption_mlp_ratio); }
};

// ── Weight structures ───────────────────────────────────────────────

struct irodori_text_block_weights {
    ggml_tensor* attn_norm = nullptr; // (text_dim,)
    ggml_tensor* wq = nullptr;        // (text_dim, text_dim)
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* gate = nullptr;   // gated attention
    ggml_tensor* q_norm = nullptr; // (heads, head_dim)
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* mlp_norm = nullptr; // (text_dim,)
    ggml_tensor* mlp_w1 = nullptr;   // gate (text_dim, ff_dim)
    ggml_tensor* mlp_w2 = nullptr;   // down (ff_dim, text_dim)
    ggml_tensor* mlp_w3 = nullptr;   // up (text_dim, ff_dim)
};

struct irodori_low_rank_adaln_weights {
    // shift/scale/gate each have: down (model_dim, rank) + up (rank, model_dim) + up_bias (model_dim,)
    ggml_tensor* shift_down = nullptr;
    ggml_tensor* shift_up_w = nullptr;
    ggml_tensor* shift_up_b = nullptr;
    ggml_tensor* scale_down = nullptr;
    ggml_tensor* scale_up_w = nullptr;
    ggml_tensor* scale_up_b = nullptr;
    ggml_tensor* gate_down = nullptr;
    ggml_tensor* gate_up_w = nullptr;
    ggml_tensor* gate_up_b = nullptr;
};

struct irodori_dit_block_weights {
    // JointAttention
    ggml_tensor* wq = nullptr; // (model_dim, model_dim)
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* gate = nullptr;
    ggml_tensor* q_norm = nullptr; // (heads, head_dim)
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* wk_text = nullptr; // (text_dim, model_dim)
    ggml_tensor* wv_text = nullptr;
    ggml_tensor* wk_spk = nullptr; // (speaker_dim, model_dim)
    ggml_tensor* wv_spk = nullptr;
    ggml_tensor* wk_cap = nullptr; // (caption_dim, model_dim) — VoiceDesign
    ggml_tensor* wv_cap = nullptr;

    // LowRankAdaLN for attention and MLP
    irodori_low_rank_adaln_weights adaln_attn;
    irodori_low_rank_adaln_weights adaln_mlp;

    // SwiGLU
    ggml_tensor* mlp_w1 = nullptr;
    ggml_tensor* mlp_w2 = nullptr;
    ggml_tensor* mlp_w3 = nullptr;
};

struct irodori_dur_block_weights {
    ggml_tensor* norm = nullptr; // (hidden_dim,)
    ggml_tensor* mlp_w1 = nullptr;
    ggml_tensor* mlp_w2 = nullptr;
    ggml_tensor* mlp_w3 = nullptr;
    ggml_tensor* mod_w = nullptr;     // (speaker_dim, hidden_dim*3)
    ggml_tensor* mod_b = nullptr;     // (hidden_dim*3,)
    ggml_tensor* cap_mod_w = nullptr; // (caption_dim, hidden_dim*3) — VoiceDesign
    ggml_tensor* cap_mod_b = nullptr; // (hidden_dim*3,)
};

struct irodori_weights {
    // Text encoder
    ggml_tensor* text_emb = nullptr; // (vocab_size, text_dim)
    std::vector<irodori_text_block_weights> text_blocks;
    ggml_tensor* text_norm = nullptr; // (text_dim,)

    // Caption encoder (VoiceDesign — same TextEncoder structure)
    ggml_tensor* cap_emb = nullptr; // (caption_vocab_size, caption_dim)
    std::vector<irodori_text_block_weights> cap_blocks;
    ggml_tensor* cap_norm = nullptr; // (caption_dim,)

    // Speaker encoder
    ggml_tensor* spk_in_proj_w = nullptr; // (patched_latent_dim, speaker_dim)
    ggml_tensor* spk_in_proj_b = nullptr; // (speaker_dim,)
    std::vector<irodori_text_block_weights> spk_blocks;
    ggml_tensor* spk_norm = nullptr; // (speaker_dim,)

    // Timestep conditioning MLP
    ggml_tensor* cond_0_w = nullptr; // (timestep_embed_dim, model_dim)
    ggml_tensor* cond_2_w = nullptr; // (model_dim, model_dim)
    ggml_tensor* cond_4_w = nullptr; // (model_dim, model_dim*3)

    // Input/output projections
    ggml_tensor* in_proj_w = nullptr;  // (patched_latent_dim, model_dim)
    ggml_tensor* in_proj_b = nullptr;  // (model_dim,)
    ggml_tensor* out_norm = nullptr;   // (model_dim,)
    ggml_tensor* out_proj_w = nullptr; // (model_dim, patched_latent_dim)
    ggml_tensor* out_proj_b = nullptr; // (patched_latent_dim,)

    // DiT blocks
    std::vector<irodori_dit_block_weights> dit_blocks;

    // Duration predictor
    ggml_tensor* dur_input_proj_w = nullptr;
    ggml_tensor* dur_input_proj_b = nullptr;
    ggml_tensor* dur_null_speaker = nullptr;
    ggml_tensor* dur_null_caption = nullptr; // VoiceDesign
    std::vector<irodori_dur_block_weights> dur_blocks;
    ggml_tensor* dur_out_norm = nullptr;
    ggml_tensor* dur_out_proj_w = nullptr;
    ggml_tensor* dur_out_proj_b = nullptr;
};

// ── DAC-VAE encoder (reference audio → 32-dim latent, voice cloning) ──
// Mirror of the decoder; deterministic encode is
//   z = encoder(pad(audio)); mean = in_proj(z)[:codebook_dim].
struct irodori_dacvae_encoder {
    bool loaded = false;
    int encoder_dim = 64;
    std::vector<int> rates; // downsample strides, e.g. {2,8,10,12}

    ggml_tensor* in_conv_w = nullptr; // Conv1d(1 → encoder_dim, k=7)
    ggml_tensor* in_conv_b = nullptr;

    struct ResUnit { // Snake→Conv(k7,dil)→Snake→Conv(k1)→+x
        ggml_tensor* alpha0 = nullptr;
        ggml_tensor* conv0_w = nullptr;
        ggml_tensor* conv0_b = nullptr;
        ggml_tensor* alpha1 = nullptr;
        ggml_tensor* conv1_w = nullptr;
        ggml_tensor* conv1_b = nullptr;
    };
    struct Block { // 3 ResUnits (dil 1,3,9) → Snake → strided downsample conv
        ResUnit res[3];
        ggml_tensor* down_snake_alpha = nullptr;
        ggml_tensor* down_w = nullptr; // Conv1d(c → 2c, k=2*stride, stride)
        ggml_tensor* down_b = nullptr;
    };
    std::vector<Block> blocks;

    ggml_tensor* out_snake_alpha = nullptr; // final Snake
    ggml_tensor* out_conv_w = nullptr;      // Conv1d(C → C, k=3)
    ggml_tensor* out_conv_b = nullptr;
    ggml_tensor* in_proj_w = nullptr; // VAEBottleneck.in_proj Conv1d(C → 2*codebook_dim, k=1)
    ggml_tensor* in_proj_b = nullptr;
};

// ── Context ─────────────────────────────────────────────────────────

struct irodori_tts_context {
    irodori_hparams hparams;
    irodori_weights weights;

    // ggml state. `backend` runs the encoders/DiT/ODE graphs (GPU when
    // use_gpu); `codec_backend` runs the DAC-VAE decode — same as `backend`
    // except on Vulkan, where conv-heavy codec graphs have a history of
    // gallocr corruption (TADA #192) and default to CPU. All graphs are
    // single-backend gallocr, weights resident on their graph's backend.
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t codec_backend = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;
    ggml_context* w_ctx = nullptr; // weight tensor context (kept alive)

    // §243: persistent cached DiT step graph (CRISPASR_IRODORI_PERSIST_GRAPH).
    // The DiT graph shape is constant across every ODE/CFG run_dit_forward call of
    // one generation, so build + gallocr-alloc ONCE and reuse — stable tensor
    // addresses let the CUDA graph warm up once (Ampere+) instead of re-capturing
    // every step. Rebuilt only when the shape signature changes. Freed at teardown.
    ggml_context* dit_ctx = nullptr;
    ggml_cgraph* dit_gf = nullptr;
    ggml_gallocr_t dit_galloc = nullptr;
    int dit_T_latent = -1, dit_T_text = -1, dit_T_ref = -1, dit_T_cap = -1;
    bool dit_has_spk = false, dit_has_cap = false;

    // Tokenizer (BPE with Metaspace pre-tokenization — matches HF LlamaTokenizer)
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<float> token_scores;
    std::unordered_map<std::string, int32_t> merge_rank;
    bool use_bpe = false; // true when merges are loaded (correct path)
    int bos_token_id = -1;
    int pad_token_id = -1;
    // SentencePiece byte_fallback: byte value → "<0xHH>" token id (-1 if
    // absent). Non-empty when the vocab has byte tokens; lets OOV emoji
    // become byte sequences (Irodori emoji emotion controls) not <unk>.
    std::vector<int32_t> byte_fallback;
#ifdef IRODORI_HAVE_SENTENCEPIECE
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;
#endif

    // Reference latent (speaker conditioning)
    std::vector<float> ref_latent; // (T_ref * latent_dim)
    int ref_latent_frames = 0;

    // DAC-VAE decoder (companion model for audio reconstruction)
    bool has_codec = false;
    ggml_backend_buffer_t codec_buf = nullptr;
    ggml_context* codec_ctx = nullptr;
    // out_proj: Conv1d(codebook_dim → latent_dim, k=1)
    ggml_tensor* codec_out_proj_w = nullptr;
    ggml_tensor* codec_out_proj_b = nullptr;
    // DAC decoder weights (reusing core_dac structure with DACVAE config)
    core_dac::DacWeights dac;
    // FASTCONV (docs/perf-sweep/PLAN.md): baked-F32 decode conv kernels so the
    // per-graph F16→F32 cast becomes a no-op. Gated CRISPASR_IRODORI_FASTCONV.
    core_dac::fastconv_cache dac_fc;
    // DAC-VAE encoder weights (voice cloning: reference audio → latent)
    irodori_dacvae_encoder enc;

    // Caption (VoiceDesign) — set via irodori_tts_set_caption()
    std::string caption_text;

    // Runtime params
    int seed = 0;
    int ode_steps = 40;
    float cfg_scale_text = 3.0f;
    float cfg_scale_speaker = 5.0f;
    float cfg_scale_caption = 3.0f;
    float speed = 1.0f;
    float duration_scale = 1.0f;
    float max_seconds = 30.0f;
    int verbosity = 1;
    int n_threads = 4;
};

// ── RoPE precomputation ─────────────────────────────────────────────

static void precompute_freqs_cis(int dim, int max_len, float theta, std::vector<float>& cos_cache,
                                 std::vector<float>& sin_cache) {
    int half = dim / 2;
    cos_cache.resize(max_len * half);
    sin_cache.resize(max_len * half);
    for (int t = 0; t < max_len; t++) {
        for (int k = 0; k < half; k++) {
            float freq = 1.0f / std::pow(theta, (float)(2 * k) / (float)dim);
            float angle = (float)t * freq;
            cos_cache[t * half + k] = std::cos(angle);
            sin_cache[t * half + k] = std::sin(angle);
        }
    }
}

// ── RMSNorm (graph op) ──────────────────────────────────────────────

static ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    x = ggml_cast(ctx, x, GGML_TYPE_F32);
    x = ggml_rms_norm(ctx, x, eps);
    if (weight->type != GGML_TYPE_F32)
        weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
    return ggml_mul(ctx, x, weight);
}

// mul_mat + cast to F32 (weights may be F16/Q4_K, need F32 for element-wise ops)
static ggml_tensor* mul_mat_f32(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x) {
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    if (y->type != GGML_TYPE_F32)
        y = ggml_cast(ctx, y, GGML_TYPE_F32);
    return y;
}

// ── Timestep embedding (CPU) ────────────────────────────────────────

static void compute_timestep_embedding(float t, int dim, std::vector<float>& out) {
    out.resize(dim);
    int half = dim / 2;
    for (int k = 0; k < half; k++) {
        float freq = 1000.0f * std::exp(-std::log(10000.0f) * (float)k / (float)half);
        float arg = t * freq;
        out[k] = std::cos(arg);
        out[half + k] = std::sin(arg);
    }
}

// ── RoPE is applied via ggml_rope_ext in the graph builders below. ──

// ── Graph builders ──────────────────────────────────────────────────

// Evaluate a ggml graph on the backend using gallocr.
static bool eval_graph(ggml_backend_t backend, ggml_cgraph* gf) {
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        return false;
    }
    ggml_backend_graph_compute(backend, gf);
    ggml_gallocr_free(galloc);
    return true;
}

// ── GGUF loading ────────────────────────────────────────────────────

// Load weights using the core_gguf map-based getter pattern.
template <typename GetFn, typename TryGetFn>
static bool load_weights_from_map(irodori_tts_context* ctx, GetFn get, TryGetFn try_get) {
    auto& w = ctx->weights;
    const auto& hp = ctx->hparams;

    // Text encoder
    w.text_emb = get("irodori.text_enc.emb");
    w.text_norm = get("irodori.text_norm");
    w.text_blocks.resize(hp.text_layers);
    for (int i = 0; i < hp.text_layers; i++) {
        auto& blk = w.text_blocks[i];
        std::string p = "irodori.text_enc.blk." + std::to_string(i) + ".";
        blk.attn_norm = get(p + "attn_norm");
        blk.wq = get(p + "attn.wq");
        blk.wk = get(p + "attn.wk");
        blk.wv = get(p + "attn.wv");
        blk.wo = get(p + "attn.wo");
        blk.gate = get(p + "attn.gate");
        blk.q_norm = get(p + "attn.q_norm");
        blk.k_norm = get(p + "attn.k_norm");
        blk.mlp_norm = get(p + "mlp_norm");
        blk.mlp_w1 = get(p + "mlp.w1");
        blk.mlp_w2 = get(p + "mlp.w2");
        blk.mlp_w3 = get(p + "mlp.w3");
    }

    // Speaker encoder
    w.spk_in_proj_w = get("irodori.spk_enc.in_proj.w");
    w.spk_in_proj_b = get("irodori.spk_enc.in_proj.b");
    w.spk_norm = get("irodori.spk_norm");
    w.spk_blocks.resize(hp.speaker_layers);
    for (int i = 0; i < hp.speaker_layers; i++) {
        auto& blk = w.spk_blocks[i];
        std::string p = "irodori.spk_enc.blk." + std::to_string(i) + ".";
        blk.attn_norm = get(p + "attn_norm");
        blk.wq = get(p + "attn.wq");
        blk.wk = get(p + "attn.wk");
        blk.wv = get(p + "attn.wv");
        blk.wo = get(p + "attn.wo");
        blk.gate = get(p + "attn.gate");
        blk.q_norm = get(p + "attn.q_norm");
        blk.k_norm = get(p + "attn.k_norm");
        blk.mlp_norm = get(p + "mlp_norm");
        blk.mlp_w1 = get(p + "mlp.w1");
        blk.mlp_w2 = get(p + "mlp.w2");
        blk.mlp_w3 = get(p + "mlp.w3");
    }

    // Caption encoder (VoiceDesign)
    if (hp.use_caption_condition) {
        w.cap_emb = get("irodori.cap_enc.emb");
        w.cap_norm = get("irodori.cap_norm");
        w.cap_blocks.resize(hp.caption_layers);
        for (int i = 0; i < hp.caption_layers; i++) {
            auto& blk = w.cap_blocks[i];
            std::string p = "irodori.cap_enc.blk." + std::to_string(i) + ".";
            blk.attn_norm = get(p + "attn_norm");
            blk.wq = get(p + "attn.wq");
            blk.wk = get(p + "attn.wk");
            blk.wv = get(p + "attn.wv");
            blk.wo = get(p + "attn.wo");
            blk.gate = get(p + "attn.gate");
            blk.q_norm = get(p + "attn.q_norm");
            blk.k_norm = get(p + "attn.k_norm");
            blk.mlp_norm = get(p + "mlp_norm");
            blk.mlp_w1 = get(p + "mlp.w1");
            blk.mlp_w2 = get(p + "mlp.w2");
            blk.mlp_w3 = get(p + "mlp.w3");
        }
    }

    // Top-level
    w.cond_0_w = get("irodori.cond.0.w");
    w.cond_2_w = get("irodori.cond.2.w");
    w.cond_4_w = get("irodori.cond.4.w");
    w.in_proj_w = get("irodori.in_proj.w");
    w.in_proj_b = get("irodori.in_proj.b");
    w.out_norm = get("irodori.out_norm");
    w.out_proj_w = get("irodori.out_proj.w");
    w.out_proj_b = get("irodori.out_proj.b");

    // DiT blocks
    w.dit_blocks.resize(hp.num_layers);
    for (int i = 0; i < hp.num_layers; i++) {
        auto& blk = w.dit_blocks[i];
        std::string p = "irodori.dit.blk." + std::to_string(i) + ".";
        blk.wq = get(p + "attn.wq");
        blk.wk = get(p + "attn.wk");
        blk.wv = get(p + "attn.wv");
        blk.wo = get(p + "attn.wo");
        blk.gate = get(p + "attn.gate");
        blk.q_norm = get(p + "attn.q_norm");
        blk.k_norm = get(p + "attn.k_norm");
        blk.wk_text = get(p + "attn.wk_text");
        blk.wv_text = get(p + "attn.wv_text");
        blk.wk_spk = get(p + "attn.wk_spk");
        blk.wv_spk = get(p + "attn.wv_spk");
        if (hp.use_caption_condition) {
            blk.wk_cap = get(p + "attn.wk_cap");
            blk.wv_cap = get(p + "attn.wv_cap");
        }

        // AdaLN attention
        auto load_adaln = [&](irodori_low_rank_adaln_weights& adaln, const std::string& ap) {
            adaln.shift_down = get(ap + "shift_down");
            adaln.shift_up_w = get(ap + "shift_up.w");
            adaln.shift_up_b = get(ap + "shift_up.b");
            adaln.scale_down = get(ap + "scale_down");
            adaln.scale_up_w = get(ap + "scale_up.w");
            adaln.scale_up_b = get(ap + "scale_up.b");
            adaln.gate_down = get(ap + "gate_down");
            adaln.gate_up_w = get(ap + "gate_up.w");
            adaln.gate_up_b = get(ap + "gate_up.b");
        };
        load_adaln(blk.adaln_attn, p + "adaln_attn.");
        load_adaln(blk.adaln_mlp, p + "adaln_mlp.");

        blk.mlp_w1 = get(p + "mlp.w1");
        blk.mlp_w2 = get(p + "mlp.w2");
        blk.mlp_w3 = get(p + "mlp.w3");
    }

    // Duration predictor
    if (hp.use_duration_predictor) {
        w.dur_input_proj_w = get("irodori.dur.input_proj.w");
        w.dur_input_proj_b = get("irodori.dur.input_proj.b");
        w.dur_null_speaker = get("irodori.dur.null_speaker");
        w.dur_null_caption = try_get("irodori.dur.null_caption");
        w.dur_blocks.resize(hp.duration_layers);
        for (int i = 0; i < hp.duration_layers; i++) {
            auto& blk = w.dur_blocks[i];
            std::string p = "irodori.dur.blk." + std::to_string(i) + ".";
            blk.norm = get(p + "norm");
            blk.mlp_w1 = get(p + "mlp.w1");
            blk.mlp_w2 = get(p + "mlp.w2");
            blk.mlp_w3 = get(p + "mlp.w3");
            blk.mod_w = get(p + "mod.w");
            blk.mod_b = get(p + "mod.b");
            blk.cap_mod_w = try_get((p + "cap_mod.w").c_str());
            blk.cap_mod_b = try_get((p + "cap_mod.b").c_str());
        }
        w.dur_out_norm = get("irodori.dur.out_norm");
        w.dur_out_proj_w = get("irodori.dur.out_proj.w");
        w.dur_out_proj_b = get("irodori.dur.out_proj.b");
    }

    return true;
}

// ── Graph: Text encoder forward ─────────────────────────────────────

// Build text encoder graph for a batch of token IDs.
// Input: token_ids (T_text,) I32, pos_ids (T_text,) I32
// Output: text_state (text_dim, T_text) F32
static ggml_tensor* build_text_encoder_graph(ggml_context* ctx, const irodori_tts_context* model,
                                             ggml_tensor* token_ids, ggml_tensor* mask_f, ggml_tensor* pos_ids,
                                             int T_text) {
    const auto& w = model->weights;
    const auto& hp = model->hparams;

    // Embedding lookup
    ggml_tensor* x = ggml_get_rows(ctx, w.text_emb, token_ids);
    // x: (text_dim, T_text) after get_rows
    x = ggml_cast(ctx, x, GGML_TYPE_F32);

    // Apply mask (zero out padding)
    x = ggml_mul(ctx, x, mask_f);

    // Process through transformer blocks
    for (int i = 0; i < hp.text_layers; i++) {
        const auto& blk = w.text_blocks[i];

        // Self-attention with pre-norm
        ggml_tensor* residual = x;
        ggml_tensor* h = rms_norm(ctx, x, blk.attn_norm, hp.norm_eps);

        // Q, K, V projections (F32 output for element-wise ops)
        ggml_tensor* q = mul_mat_f32(ctx, blk.wq, h);
        ggml_tensor* k = mul_mat_f32(ctx, blk.wk, h);
        ggml_tensor* v = mul_mat_f32(ctx, blk.wv, h);

        // Reshape to (head_dim, heads, T)
        int hd = hp.text_head_dim();
        int nh = hp.text_heads;
        q = ggml_reshape_3d(ctx, q, hd, nh, T_text);
        k = ggml_reshape_3d(ctx, k, hd, nh, T_text);
        v = ggml_reshape_3d(ctx, v, hd, nh, T_text);

        // QK norm (per-head RMSNorm) — cast norms to F32 if F16
        q = ggml_rms_norm(ctx, q, hp.norm_eps);
        q = ggml_mul(ctx, q, ggml_cast(ctx, blk.q_norm, GGML_TYPE_F32));
        k = ggml_rms_norm(ctx, k, hp.norm_eps);
        k = ggml_mul(ctx, k, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));

        // RoPE in (hd, nh, T) layout — ne[2]=T matches pos_ids length
        q = ggml_rope_ext(ctx, q, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute to (hd, T, nh) for flash_attn_ext
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_reshape_3d(ctx, v, hd, nh, T_text);
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

        // Flash attention: Q(hd, T, nh), K(hd, T, nh), V(hd, T, nh)
        ggml_tensor* attn_out =
            ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
        attn_out = ggml_cast(ctx, attn_out, GGML_TYPE_F32);
        attn_out = ggml_reshape_2d(ctx, attn_out, hp.text_dim, T_text);

        // Gated attention: y = y * sigmoid(gate(h))
        ggml_tensor* g = mul_mat_f32(ctx, blk.gate, h);
        g = ggml_sigmoid(ctx, g);
        attn_out = ggml_mul(ctx, attn_out, g);

        // Output projection
        attn_out = mul_mat_f32(ctx, blk.wo, attn_out);
        x = ggml_add(ctx, residual, attn_out);

        // FFN with pre-norm (use mul_mat_f32 for F16/Q4K weight safety)
        residual = x;
        h = rms_norm(ctx, x, blk.mlp_norm, hp.norm_eps);
        ggml_tensor* gate_proj = mul_mat_f32(ctx, blk.mlp_w1, h);
        ggml_tensor* up_proj = mul_mat_f32(ctx, blk.mlp_w3, h);
        ggml_tensor* mlp_out = ggml_mul(ctx, ggml_silu(ctx, gate_proj), up_proj);
        mlp_out = mul_mat_f32(ctx, blk.mlp_w2, mlp_out);
        x = ggml_add(ctx, residual, mlp_out);

        // Re-apply mask
        x = ggml_mul(ctx, x, mask_f);
    }

    // Final text norm
    x = rms_norm(ctx, x, w.text_norm, hp.norm_eps);
    return x;
}

// ── Graph: Caption encoder forward (VoiceDesign) ────────────────────
// Same structure as text encoder but uses caption weights/dims.

static ggml_tensor* build_caption_encoder_graph(ggml_context* ctx, const irodori_tts_context* model,
                                                ggml_tensor* token_ids, ggml_tensor* mask_f, ggml_tensor* pos_ids,
                                                int T_cap) {
    const auto& w = model->weights;
    const auto& hp = model->hparams;

    ggml_tensor* x = ggml_get_rows(ctx, w.cap_emb, token_ids);
    x = ggml_cast(ctx, x, GGML_TYPE_F32);
    x = ggml_mul(ctx, x, mask_f);

    for (int i = 0; i < hp.caption_layers; i++) {
        const auto& blk = w.cap_blocks[i];

        ggml_tensor* residual = x;
        ggml_tensor* h = rms_norm(ctx, x, blk.attn_norm, hp.norm_eps);

        ggml_tensor* q = mul_mat_f32(ctx, blk.wq, h);
        ggml_tensor* k = mul_mat_f32(ctx, blk.wk, h);
        ggml_tensor* v = mul_mat_f32(ctx, blk.wv, h);

        int hd = hp.caption_head_dim();
        int nh = hp.caption_heads;
        q = ggml_reshape_3d(ctx, q, hd, nh, T_cap);
        k = ggml_reshape_3d(ctx, k, hd, nh, T_cap);
        v = ggml_reshape_3d(ctx, v, hd, nh, T_cap);

        q = ggml_rms_norm(ctx, q, hp.norm_eps);
        q = ggml_mul(ctx, q, ggml_cast(ctx, blk.q_norm, GGML_TYPE_F32));
        k = ggml_rms_norm(ctx, k, hp.norm_eps);
        k = ggml_mul(ctx, k, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));

        q = ggml_rope_ext(ctx, q, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
        attn_out = ggml_cast(ctx, attn_out, GGML_TYPE_F32);
        attn_out = ggml_reshape_2d(ctx, attn_out, hp.caption_dim, T_cap);

        ggml_tensor* g = mul_mat_f32(ctx, blk.gate, h);
        g = ggml_sigmoid(ctx, g);
        attn_out = ggml_mul(ctx, attn_out, g);
        attn_out = mul_mat_f32(ctx, blk.wo, attn_out);
        x = ggml_add(ctx, residual, attn_out);

        residual = x;
        h = rms_norm(ctx, x, blk.mlp_norm, hp.norm_eps);
        ggml_tensor* gate_proj = mul_mat_f32(ctx, blk.mlp_w1, h);
        ggml_tensor* up_proj = mul_mat_f32(ctx, blk.mlp_w3, h);
        ggml_tensor* mlp_out = ggml_mul(ctx, ggml_silu(ctx, gate_proj), up_proj);
        mlp_out = mul_mat_f32(ctx, blk.mlp_w2, mlp_out);
        x = ggml_add(ctx, residual, mlp_out);
        x = ggml_mul(ctx, x, mask_f);
    }

    x = rms_norm(ctx, x, w.cap_norm, hp.norm_eps);
    return x;
}

// ── Graph: Speaker encoder forward ──────────────────────────────────

static ggml_tensor* build_speaker_encoder_graph(ggml_context* ctx, const irodori_tts_context* model,
                                                ggml_tensor* latent_in, ggml_tensor* mask_f, ggml_tensor* pos_ids,
                                                int T_ref) {
    const auto& w = model->weights;
    const auto& hp = model->hparams;

    // Input projection + scale by 1/6
    ggml_tensor* x = mul_mat_f32(ctx, w.spk_in_proj_w, latent_in);
    x = ggml_add(ctx, x, w.spk_in_proj_b);
    x = ggml_scale(ctx, x, 1.0f / 6.0f);
    x = ggml_mul(ctx, x, mask_f);

    // Same transformer block structure as text encoder (with same fixes)
    for (int i = 0; i < hp.speaker_layers; i++) {
        const auto& blk = w.spk_blocks[i];

        ggml_tensor* residual = x;
        ggml_tensor* h = rms_norm(ctx, x, blk.attn_norm, hp.norm_eps);

        ggml_tensor* q = mul_mat_f32(ctx, blk.wq, h);
        ggml_tensor* k = mul_mat_f32(ctx, blk.wk, h);
        ggml_tensor* v = mul_mat_f32(ctx, blk.wv, h);

        int hd = hp.speaker_head_dim();
        int nh = hp.speaker_heads;
        q = ggml_reshape_3d(ctx, q, hd, nh, T_ref);
        k = ggml_reshape_3d(ctx, k, hd, nh, T_ref);
        v = ggml_reshape_3d(ctx, v, hd, nh, T_ref);

        q = ggml_rms_norm(ctx, q, hp.norm_eps);
        q = ggml_mul(ctx, q, ggml_cast(ctx, blk.q_norm, GGML_TYPE_F32));
        k = ggml_rms_norm(ctx, k, hp.norm_eps);
        k = ggml_mul(ctx, k, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));

        // RoPE in (hd, nh, T) layout then permute to (hd, T, nh) for flash attn
        q = ggml_rope_ext(ctx, q, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
        attn_out = ggml_cast(ctx, attn_out, GGML_TYPE_F32);
        attn_out = ggml_reshape_2d(ctx, attn_out, hp.speaker_dim, T_ref);

        ggml_tensor* g = mul_mat_f32(ctx, blk.gate, h);
        g = ggml_sigmoid(ctx, g);
        attn_out = ggml_mul(ctx, attn_out, g);
        attn_out = mul_mat_f32(ctx, blk.wo, attn_out);
        x = ggml_add(ctx, residual, attn_out);

        residual = x;
        h = rms_norm(ctx, x, blk.mlp_norm, hp.norm_eps);
        ggml_tensor* gate_proj = mul_mat_f32(ctx, blk.mlp_w1, h);
        ggml_tensor* up_proj = mul_mat_f32(ctx, blk.mlp_w3, h);
        ggml_tensor* mlp_out = ggml_mul(ctx, ggml_silu(ctx, gate_proj), up_proj);
        mlp_out = mul_mat_f32(ctx, blk.mlp_w2, mlp_out);
        x = ggml_add(ctx, residual, mlp_out);
        x = ggml_mul(ctx, x, mask_f);
    }

    // Speaker norm
    x = rms_norm(ctx, x, w.spk_norm, hp.norm_eps);
    return x;
}

// ── Graph: LowRankAdaLN ─────────────────────────────────────────────

// Apply LowRank AdaLN: returns (normed_modulated_x, gate)
// cond_embed: (model_dim*3, 1) — pre-chunked into 3 parts of model_dim each
static void apply_low_rank_adaln(ggml_context* ctx, const irodori_low_rank_adaln_weights& adaln, ggml_tensor* x,
                                 ggml_tensor* cond_shift, ggml_tensor* cond_scale, ggml_tensor* cond_gate,
                                 float norm_eps, ggml_tensor** out_x, ggml_tensor** out_gate) {
    // Low-rank residual modulation:
    //   shift = shift_up(shift_down(silu(cond_shift))) + cond_shift
    //   scale = scale_up(scale_down(silu(cond_scale))) + cond_scale
    //   gate  = gate_up(gate_down(silu(cond_gate))) + cond_gate

    auto lr_mod = [&](ggml_tensor* cond, ggml_tensor* down, ggml_tensor* up_w, ggml_tensor* up_b) -> ggml_tensor* {
        ggml_tensor* h = ggml_silu(ctx, cond);
        h = mul_mat_f32(ctx, down, h);
        h = mul_mat_f32(ctx, up_w, h);
        if (up_b)
            h = ggml_add(ctx, h, up_b);
        return ggml_add(ctx, h, cond); // residual connection
    };

    ggml_tensor* shift =
        ggml_cast(ctx, lr_mod(cond_shift, adaln.shift_down, adaln.shift_up_w, adaln.shift_up_b), GGML_TYPE_F32);
    ggml_tensor* scale =
        ggml_cast(ctx, lr_mod(cond_scale, adaln.scale_down, adaln.scale_up_w, adaln.scale_up_b), GGML_TYPE_F32);
    ggml_tensor* gate =
        ggml_cast(ctx, lr_mod(cond_gate, adaln.gate_down, adaln.gate_up_w, adaln.gate_up_b), GGML_TYPE_F32);

    // RMSNorm + modulate: x_normed * (1 + scale) + shift
    ggml_tensor* x_norm = ggml_rms_norm(ctx, x, norm_eps);
    // (1 + scale) = scale + 1: use ggml_scale to add 1 via add_scalar trick
    // Actually: x * (1 + s) = x + x*s, which avoids allocating a scalar tensor
    ggml_tensor* x_scaled = ggml_add(ctx, x_norm, ggml_mul(ctx, x_norm, scale));
    *out_x = ggml_add(ctx, x_scaled, shift);

    // gate = tanh(gate)
    *out_gate = ggml_tanh(ctx, gate);
}

// ── Graph: DiT block forward ────────────────────────────────────────

// Builds one DiffusionBlock.
// x: (model_dim, T_latent)
// cond_embed: (model_dim*3, 1) — timestep conditioning
// text_state: (text_dim, T_text)
// spk_state: (speaker_dim, T_ref) or nullptr
static ggml_tensor* build_dit_block_graph(ggml_context* ctx, const irodori_tts_context* model,
                                          const irodori_dit_block_weights& blk, ggml_tensor* x, ggml_tensor* cond_embed,
                                          ggml_tensor* text_state, int T_text, ggml_tensor* spk_state, int T_ref,
                                          ggml_tensor* cap_state, int T_cap, int T_latent, ggml_tensor* pos_ids,
                                          ggml_tensor* attn_mask) {
    const auto& hp = model->hparams;
    int hd = hp.head_dim();
    int nh = hp.num_heads;

    // Ensure all inputs are F32
    x = ggml_cast(ctx, x, GGML_TYPE_F32);
    cond_embed = ggml_cast(ctx, cond_embed, GGML_TYPE_F32);
    text_state = ggml_cast(ctx, text_state, GGML_TYPE_F32);
    if (spk_state)
        spk_state = ggml_cast(ctx, spk_state, GGML_TYPE_F32);

    // Null check all weight tensors
    if (!blk.wq || !blk.wk || !blk.wv || !blk.wo || !blk.gate || !blk.q_norm || !blk.k_norm || !blk.wk_text ||
        !blk.wv_text || !blk.adaln_attn.shift_down || !blk.adaln_attn.shift_up_w || !blk.adaln_mlp.shift_down ||
        !blk.mlp_w1 || !blk.mlp_w2 || !blk.mlp_w3) {
        std::fprintf(stderr, "[irodori] ERROR: null weight tensor in DiT block\n");
        return x; // bail out
    }

    // Split cond_embed into 3 parts: (model_dim,) each
    int D = hp.model_dim;
    IRODORI_DBG("[irodori]       view cond_embed (D=%d, ne0=%d)...\n", D, (int)cond_embed->ne[0]);
    ggml_tensor* cond_shift = ggml_view_1d(ctx, cond_embed, D, 0);
    ggml_tensor* cond_scale = ggml_view_1d(ctx, cond_embed, D, D * sizeof(float));
    ggml_tensor* cond_gate_raw = ggml_view_1d(ctx, cond_embed, D, 2 * D * sizeof(float));
    IRODORI_DBG("[irodori]       applying AdaLN attn...\n");

    // ── Attention path ──
    ggml_tensor* h_attn = nullptr;
    ggml_tensor* gate_attn = nullptr;
    apply_low_rank_adaln(ctx, blk.adaln_attn, x, cond_shift, cond_scale, cond_gate_raw, hp.norm_eps, &h_attn,
                         &gate_attn);
    IRODORI_DBG("[irodori]       AdaLN attn done\n");

    // Self Q/K/V
    IRODORI_DBG("[irodori]       QKV self proj (wq: %dx%d, h_attn: %dx%d)...\n", (int)blk.wq->ne[0], (int)blk.wq->ne[1],
                (int)h_attn->ne[0], (int)h_attn->ne[1]);
    ggml_tensor* q = mul_mat_f32(ctx, blk.wq, h_attn);
    ggml_tensor* k_self = mul_mat_f32(ctx, blk.wk, h_attn);
    ggml_tensor* v_self = mul_mat_f32(ctx, blk.wv, h_attn);
    IRODORI_DBG("[irodori]       QKV self done\n");

    // Text context K/V
    IRODORI_DBG("[irodori]       text KV proj (wk_text: %dx%d, text_state: %dx%d)...\n", (int)blk.wk_text->ne[0],
                (int)blk.wk_text->ne[1], (int)text_state->ne[0], (int)text_state->ne[1]);
    ggml_tensor* k_text = mul_mat_f32(ctx, blk.wk_text, text_state);
    ggml_tensor* v_text = mul_mat_f32(ctx, blk.wv_text, text_state);
    IRODORI_DBG("[irodori]       text KV done\n");

    // Speaker context K/V
    ggml_tensor* k_spk = nullptr;
    ggml_tensor* v_spk = nullptr;
    int T_kv_total = T_latent + T_text;
    if (spk_state && T_ref > 0) {
        k_spk = mul_mat_f32(ctx, blk.wk_spk, spk_state);
        v_spk = mul_mat_f32(ctx, blk.wv_spk, spk_state);
        T_kv_total += T_ref;
    }

    // Caption context K/V (VoiceDesign)
    ggml_tensor* k_cap = nullptr;
    ggml_tensor* v_cap = nullptr;
    if (cap_state && T_cap > 0 && blk.wk_cap && blk.wv_cap) {
        cap_state = ggml_cast(ctx, cap_state, GGML_TYPE_F32);
        k_cap = mul_mat_f32(ctx, blk.wk_cap, cap_state);
        v_cap = mul_mat_f32(ctx, blk.wv_cap, cap_state);
        T_kv_total += T_cap;
    }

    // All tensors start as (hd*nh, T) from mul_mat → reshape to (hd, nh, T)
    // Apply QK norm and half-RoPE in (hd, nh, T) layout (ne[2]=T matches pos_ids)
    // Then permute to (hd, T, nh) for flash_attn_ext

    q = ggml_reshape_3d(ctx, q, hd, nh, T_latent);
    q = ggml_rms_norm(ctx, q, hp.norm_eps);
    q = ggml_mul(ctx, q, ggml_cast(ctx, blk.q_norm, GGML_TYPE_F32));

    k_self = ggml_reshape_3d(ctx, k_self, hd, nh, T_latent);
    k_self = ggml_rms_norm(ctx, k_self, hp.norm_eps);
    k_self = ggml_mul(ctx, k_self, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));

    v_self = ggml_reshape_3d(ctx, v_self, hd, nh, T_latent);

    // Half-RoPE in (hd, nh, T) layout: split heads dim (ne[1]) in half
    {
        int half_nh = nh / 2;
        size_t nb1 = q->nb[1]; // stride for head dim
        size_t nb2 = q->nb[2]; // stride for T dim

        // Q: first half_nh heads get RoPE
        ggml_tensor* q_rot = ggml_view_3d(ctx, q, hd, half_nh, T_latent, nb1, nb2, 0);
        ggml_tensor* q_pass = ggml_view_3d(ctx, q, hd, nh - half_nh, T_latent, nb1, nb2, half_nh * nb1);
        q_rot = ggml_cont(ctx, q_rot);
        q_rot = ggml_rope_ext(ctx, q_rot, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q = ggml_concat(ctx, q_rot, ggml_cont(ctx, q_pass), 1); // concat on head dim

        // K_self: same half-RoPE
        nb1 = k_self->nb[1];
        nb2 = k_self->nb[2];
        ggml_tensor* ks_rot = ggml_view_3d(ctx, k_self, hd, half_nh, T_latent, nb1, nb2, 0);
        ggml_tensor* ks_pass = ggml_view_3d(ctx, k_self, hd, nh - half_nh, T_latent, nb1, nb2, half_nh * nb1);
        ks_rot = ggml_cont(ctx, ks_rot);
        ks_rot = ggml_rope_ext(ctx, ks_rot, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k_self = ggml_concat(ctx, ks_rot, ggml_cont(ctx, ks_pass), 1);
    }

    // Permute all to (hd, T, nh) for flash_attn_ext
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k_self = ggml_cont(ctx, ggml_permute(ctx, k_self, 0, 2, 1, 3));
    v_self = ggml_cont(ctx, ggml_permute(ctx, v_self, 0, 2, 1, 3));

    // Text K/V: no RoPE (context KV), just norm + permute
    k_text = ggml_reshape_3d(ctx, k_text, hd, nh, T_text);
    k_text = ggml_rms_norm(ctx, k_text, hp.norm_eps);
    k_text = ggml_mul(ctx, k_text, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));
    k_text = ggml_cont(ctx, ggml_permute(ctx, k_text, 0, 2, 1, 3));
    v_text = ggml_reshape_3d(ctx, v_text, hd, nh, T_text);
    v_text = ggml_cont(ctx, ggml_permute(ctx, v_text, 0, 2, 1, 3));

    // Concatenate K and V along sequence dim (dim=1): (hd, T_total, nh)
    ggml_tensor* k_cat = ggml_concat(ctx, k_self, k_text, 1);
    ggml_tensor* v_cat = ggml_concat(ctx, v_self, v_text, 1);

    if (k_spk && v_spk && T_ref > 0) {
        k_spk = ggml_reshape_3d(ctx, k_spk, hd, nh, T_ref);
        k_spk = ggml_rms_norm(ctx, k_spk, hp.norm_eps);
        k_spk = ggml_mul(ctx, k_spk, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));
        k_spk = ggml_cont(ctx, ggml_permute(ctx, k_spk, 0, 2, 1, 3));
        v_spk = ggml_reshape_3d(ctx, v_spk, hd, nh, T_ref);
        v_spk = ggml_cont(ctx, ggml_permute(ctx, v_spk, 0, 2, 1, 3));
        k_cat = ggml_concat(ctx, k_cat, k_spk, 1);
        v_cat = ggml_concat(ctx, v_cat, v_spk, 1);
    }

    // Concat caption KV (VoiceDesign) — after speaker, matching Python order
    if (k_cap && v_cap && T_cap > 0) {
        k_cap = ggml_reshape_3d(ctx, k_cap, hd, nh, T_cap);
        k_cap = ggml_rms_norm(ctx, k_cap, hp.norm_eps);
        k_cap = ggml_mul(ctx, k_cap, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));
        k_cap = ggml_cont(ctx, ggml_permute(ctx, k_cap, 0, 2, 1, 3));
        v_cap = ggml_reshape_3d(ctx, v_cap, hd, nh, T_cap);
        v_cap = ggml_cont(ctx, ggml_permute(ctx, v_cap, 0, 2, 1, 3));
        k_cat = ggml_concat(ctx, k_cat, k_cap, 1);
        v_cat = ggml_concat(ctx, v_cat, v_cap, 1);
    }

    // Flash attention with mask (mask passed from run_dit_forward)
    // mask: (T_kv, T_q) — 0.0 for attend, -inf for masked speaker positions
    ggml_tensor* attn_out =
        ggml_flash_attn_ext(ctx, q, k_cat, v_cat, attn_mask, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
    attn_out = ggml_cast(ctx, attn_out, GGML_TYPE_F32);
    attn_out = ggml_reshape_2d(ctx, attn_out, D, T_latent);

    // Gated attention
    ggml_tensor* g = mul_mat_f32(ctx, blk.gate, h_attn);
    g = ggml_sigmoid(ctx, g);
    attn_out = ggml_mul(ctx, attn_out, g);
    attn_out = mul_mat_f32(ctx, blk.wo, attn_out);

    // gate_attn * attn_out (gate is 1D, attn_out is 2D — larger tensor first for ggml_mul)
    attn_out = ggml_mul(ctx, attn_out, gate_attn);
    x = ggml_add(ctx, x, attn_out);

    // ── MLP path ──
    ggml_tensor* h_mlp = nullptr;
    ggml_tensor* gate_mlp = nullptr;
    apply_low_rank_adaln(ctx, blk.adaln_mlp, x, cond_shift, cond_scale, cond_gate_raw, hp.norm_eps, &h_mlp, &gate_mlp);

    // SwiGLU may produce F16 from quantized weights — need F32 for gate mul
    ggml_tensor* gate_proj = mul_mat_f32(ctx, blk.mlp_w1, h_mlp);
    ggml_tensor* up_proj = mul_mat_f32(ctx, blk.mlp_w3, h_mlp);
    ggml_tensor* mlp_out = ggml_mul(ctx, ggml_silu(ctx, gate_proj), up_proj);
    mlp_out = mul_mat_f32(ctx, blk.mlp_w2, mlp_out);
    mlp_out = ggml_mul(ctx, mlp_out, gate_mlp);
    x = ggml_add(ctx, x, mlp_out);

    return x;
}

// ── Tokenizer ───────────────────────────────────────────────────────
//
// sarashina2.2 uses HuggingFace's LlamaTokenizer (tokenizers BPE with
// Metaspace pre-tokenizer + byte_fallback). The algorithm:
//   1. Metaspace: replace spaces with ▁ (U+2581), no prefix prepend
//   2. BPE: split into UTF-8 characters, iteratively merge lowest-rank pair
//   3. byte_fallback: unknown chars → <0xHH> hex tokens per UTF-8 byte
//
// This matches CrispEmbed's BPETokenizer::bpe_merge() (spm_style path).

static std::vector<int32_t> tokenize_text(const irodori_tts_context* ctx, const char* text) {
    // Override token IDs from env var for parity testing:
    //   CRISPASR_IRODORI_TOKEN_IDS="1,19144,52839,302,275"
    //   CRISPASR_IRODORI_CAPTION_TOKEN_IDS="1,..." (VoiceDesign caption)
    const char* override_ids = std::getenv("CRISPASR_IRODORI_TOKEN_IDS");
    if (override_ids && *override_ids) {
        std::vector<int32_t> tokens;
        std::string s(override_ids);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t end = s.find(',', pos);
            if (end == std::string::npos)
                end = s.size();
            tokens.push_back(std::atoi(s.substr(pos, end - pos).c_str()));
            pos = end + 1;
        }
        IRODORI_DBG("[irodori] using override token IDs (%d tokens)\n", (int)tokens.size());
        return tokens;
    }
    // (dump hook applied to the computed ids below)

    std::vector<int32_t> tokens;

    // Prepend BOS if configured
    if (ctx->bos_token_id >= 0) {
        tokens.push_back(ctx->bos_token_id);
    }

    if (!ctx->token_to_id.empty()) {
        if (ctx->use_bpe && !ctx->merge_rank.empty()) {
            // BPE with Metaspace pre-tokenization (for BPE-family tokenizers)
            // Metaspace: replace spaces with ▁, no prefix prepend (prepend_scheme=never)
            std::string processed;
            for (const char* p = text; *p; p++) {
                if (*p == ' ')
                    processed += "\xe2\x96\x81"; // ▁ (U+2581)
                else
                    processed += *p;
            }
            auto bpe_ids = core_spm_bpe::bpe_merge(processed, ctx->token_to_id, ctx->merge_rank);
            tokens.insert(tokens.end(), bpe_ids.begin(), bpe_ids.end());
#ifdef IRODORI_HAVE_SENTENCEPIECE
        } else if (ctx->sp_processor) {
            // Native SentencePiece processor (when embedded .model is available)
            std::string sp_text = std::string("\xE2\x96\x81") + text; // ▁ + text
            std::vector<int> sp_ids;
            ctx->sp_processor->Encode(sp_text, &sp_ids);
            for (int id : sp_ids)
                tokens.push_back(id);
#endif
        } else if (!ctx->token_scores.empty()) {
            // Unigram (SentencePiece Viterbi) — correct for llm-jp/llm-jp-3 style tokenizers
            core_spm::Config cfg;
            cfg.unk_id = 0;
            cfg.utf8_aligned = true;
            cfg.merge_consecutive_unk = true;
            // byte_fallback: OOV bytes → "<0xHH>" tokens (emoji emotion controls
            // must reach the model as byte sequences, not <unk> noise, #221).
            if (!ctx->byte_fallback.empty())
                cfg.byte_fallback = &ctx->byte_fallback;
            auto sp_tokens = core_spm::tokenize(text, ctx->token_to_id, ctx->token_scores, cfg, true);
            tokens.insert(tokens.end(), sp_tokens.begin(), sp_tokens.end());
        } else {
            IRODORI_DBG("[irodori] WARNING: no tokenizer data loaded\n");
        }
    } else {
        // Byte-level fallback (works but suboptimal)
        IRODORI_DBG("[irodori] WARNING: no vocab loaded, using byte fallback\n");
        const uint8_t* p = (const uint8_t*)text;
        while (*p) {
            tokens.push_back((int32_t)*p);
            p++;
        }
    }

    if (const char* dbg = std::getenv("CRISPASR_IRODORI_DUMP_TOKENS")) {
        if (*dbg && *dbg != '0') {
            std::fprintf(stderr, "[irodori] tokens (%d):", (int)tokens.size());
            for (int32_t t : tokens)
                std::fprintf(stderr, " %d", t);
            std::fprintf(stderr, "\n");
        }
    }
    return tokens;
}

// ── Public API ──────────────────────────────────────────────────────

struct irodori_tts_params irodori_tts_default_params(void) {
    irodori_tts_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.seed = 0;
    p.ode_steps = 40;
    p.cfg_scale_text = 3.0f;
    p.cfg_scale_speaker = 5.0f;
    p.cfg_scale_caption = 3.0f;
    p.speed = 1.0f;
    p.duration_scale = 1.0f;
    p.max_seconds = 30.0f;
    return p;
}

struct irodori_tts_context* irodori_tts_init_from_file(const char* path_model, struct irodori_tts_params params) {
    if (!path_model)
        return nullptr;

    auto* ctx = new irodori_tts_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity = params.verbosity;
    ctx->seed = params.seed;
    ctx->ode_steps = params.ode_steps > 0 ? params.ode_steps : 40;
    ctx->cfg_scale_text = params.cfg_scale_text;
    ctx->cfg_scale_speaker = params.cfg_scale_speaker;
    ctx->cfg_scale_caption = params.cfg_scale_caption;
    ctx->speed = params.speed > 0 ? params.speed : 1.0f;
    ctx->duration_scale = params.duration_scale > 0 ? params.duration_scale : 1.0f;
    ctx->max_seconds = params.max_seconds > 0 ? params.max_seconds : 30.0f;

    // Initialize backends. use_gpu picks the best GPU (Metal/CUDA/Vulkan);
    // graphs are single-backend gallocr, so weights load onto the compute
    // backend directly. CRISPASR_IRODORI_CPU=1 forces CPU regardless.
    ctx->backend_cpu = core_cpu_backend::init();
    if (!ctx->backend_cpu) {
        std::fprintf(stderr, "[irodori] failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);

    const bool force_cpu = [] {
        const char* e = std::getenv("CRISPASR_IRODORI_CPU");
        return e && *e && *e != '0';
    }();
    ctx->backend = (params.use_gpu && !force_cpu) ? crispasr_init_gpu_backend() : nullptr;
    if (ctx->backend && core_cpu_backend::is_cpu(ctx->backend)) {
        ggml_backend_free(ctx->backend); // CPU-only build: keep the threaded instance
        ctx->backend = nullptr;
    }
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    // Codec backend: GPU except on Vulkan, where conv-heavy codec graphs once
    // corrupted via gallocr on native drivers (TADA #192). That corruption no
    // longer reproduces on MoltenVK/M1 with current ggml — the DAC-VAE encoder
    // (cos 0.99996 vs CPU) and decoder (cos 0.99999) run clean on Vulkan there
    // — so the CPU fallback is likely over-conservative on modern Vulkan. Kept
    // as the default pending confirmation on native AMD/RADV (the driver family
    // that originally corrupted); CRISPASR_IRODORI_CODEC_GPU=1 opts into the
    // (validated) GPU codec, CRISPASR_IRODORI_CODEC_CPU=1 forces CPU.
    ctx->codec_backend = ctx->backend;
    if (ctx->backend != ctx->backend_cpu) {
        ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        const bool is_vulkan = reg && std::strcmp(ggml_backend_reg_name(reg), "Vulkan") == 0;
        const char* cg = std::getenv("CRISPASR_IRODORI_CODEC_GPU");
        const char* cc = std::getenv("CRISPASR_IRODORI_CODEC_CPU");
        const bool codec_gpu = (cg && *cg && *cg != '0') || (!is_vulkan && !(cc && *cc && *cc != '0'));
        if (!codec_gpu)
            ctx->codec_backend = ctx->backend_cpu;
    }
    if (ctx->verbosity >= 1 && ctx->backend != ctx->backend_cpu)
        std::fprintf(stderr, "[irodori] backend: %s (codec on %s)\n", ggml_backend_name(ctx->backend),
                     ctx->codec_backend == ctx->backend_cpu ? "CPU" : ggml_backend_name(ctx->codec_backend));

    // Load GGUF
    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] loading model: %s\n", path_model);
    }

    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path_model);
    if (!meta) {
        std::fprintf(stderr, "[irodori] failed to open GGUF metadata: %s\n", path_model);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Read hyperparameters
    auto& hp = ctx->hparams;
    hp.latent_dim = core_gguf::kv_i32(meta, "irodori.latent_dim", hp.latent_dim);
    hp.latent_patch_size = core_gguf::kv_i32(meta, "irodori.latent_patch_size", hp.latent_patch_size);
    hp.model_dim = core_gguf::kv_i32(meta, "irodori.model_dim", hp.model_dim);
    hp.num_layers = core_gguf::kv_i32(meta, "irodori.num_layers", hp.num_layers);
    hp.num_heads = core_gguf::kv_i32(meta, "irodori.num_heads", hp.num_heads);
    hp.mlp_ratio = core_gguf::kv_f32(meta, "irodori.mlp_ratio", hp.mlp_ratio);
    hp.text_dim = core_gguf::kv_i32(meta, "irodori.text_dim", hp.text_dim);
    hp.text_layers = core_gguf::kv_i32(meta, "irodori.text_layers", hp.text_layers);
    hp.text_heads = core_gguf::kv_i32(meta, "irodori.text_heads", hp.text_heads);
    hp.text_mlp_ratio = core_gguf::kv_f32(meta, "irodori.text_mlp_ratio", hp.text_mlp_ratio);
    hp.text_vocab_size = core_gguf::kv_i32(meta, "irodori.text_vocab_size", hp.text_vocab_size);
    hp.speaker_dim = core_gguf::kv_i32(meta, "irodori.speaker_dim", hp.speaker_dim);
    hp.speaker_layers = core_gguf::kv_i32(meta, "irodori.speaker_layers", hp.speaker_layers);
    hp.speaker_heads = core_gguf::kv_i32(meta, "irodori.speaker_heads", hp.speaker_heads);
    hp.speaker_mlp_ratio = core_gguf::kv_f32(meta, "irodori.speaker_mlp_ratio", hp.speaker_mlp_ratio);
    hp.speaker_patch_size = core_gguf::kv_i32(meta, "irodori.speaker_patch_size", hp.speaker_patch_size);
    hp.timestep_embed_dim = core_gguf::kv_i32(meta, "irodori.timestep_embed_dim", hp.timestep_embed_dim);
    hp.adaln_rank = core_gguf::kv_i32(meta, "irodori.adaln_rank", hp.adaln_rank);
    hp.norm_eps = core_gguf::kv_f32(meta, "irodori.norm_eps", hp.norm_eps);
    hp.use_duration_predictor = core_gguf::kv_i32(meta, "irodori.use_duration_predictor", hp.use_duration_predictor);
    hp.duration_aux_dim = core_gguf::kv_i32(meta, "irodori.duration_aux_dim", hp.duration_aux_dim);
    hp.duration_hidden_dim = core_gguf::kv_i32(meta, "irodori.duration_hidden_dim", hp.duration_hidden_dim);
    hp.duration_layers = core_gguf::kv_i32(meta, "irodori.duration_layers", hp.duration_layers);
    hp.ode_steps = core_gguf::kv_i32(meta, "irodori.ode_steps", hp.ode_steps);
    hp.cfg_scale_text = core_gguf::kv_f32(meta, "irodori.cfg_scale_text", hp.cfg_scale_text);
    hp.cfg_scale_speaker = core_gguf::kv_f32(meta, "irodori.cfg_scale_speaker", hp.cfg_scale_speaker);
    hp.sample_rate = core_gguf::kv_i32(meta, "irodori.sample_rate", hp.sample_rate);
    hp.codec_hop_length = core_gguf::kv_i32(meta, "irodori.codec_hop_length", hp.codec_hop_length);

    // Caption (VoiceDesign)
    hp.use_caption_condition = core_gguf::kv_i32(meta, "irodori.use_caption_condition", hp.use_caption_condition);
    if (hp.use_caption_condition) {
        hp.caption_dim = core_gguf::kv_i32(meta, "irodori.caption_dim", hp.caption_dim);
        hp.caption_layers = core_gguf::kv_i32(meta, "irodori.caption_layers", hp.caption_layers);
        hp.caption_heads = core_gguf::kv_i32(meta, "irodori.caption_heads", hp.caption_heads);
        hp.caption_mlp_ratio = core_gguf::kv_f32(meta, "irodori.caption_mlp_ratio", hp.caption_mlp_ratio);
        hp.caption_vocab_size = core_gguf::kv_i32(meta, "irodori.caption_vocab_size", hp.caption_vocab_size);
        hp.cfg_scale_caption = core_gguf::kv_f32(meta, "irodori.cfg_scale_caption", hp.cfg_scale_caption);
    }

    // Load tokenizer from GGUF metadata
    {
        auto vocab_strs = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
        ctx->bos_token_id = core_gguf::kv_i32(meta, "tokenizer.ggml.bos_token_id", -1);
        ctx->pad_token_id = core_gguf::kv_i32(meta, "tokenizer.ggml.pad_token_id", -1);

        if (!vocab_strs.empty()) {
            ctx->vocab.resize(vocab_strs.size());
            for (size_t i = 0; i < vocab_strs.size(); i++) {
                ctx->vocab[i] = vocab_strs[i];
                ctx->token_to_id[vocab_strs[i]] = (int32_t)i;
            }
            // Build the SentencePiece byte_fallback table from "<0xHH>" tokens.
            {
                std::vector<int32_t> bf(256, -1);
                char name[8];
                int found = 0;
                for (int b = 0; b < 256; b++) {
                    std::snprintf(name, sizeof(name), "<0x%02X>", b);
                    auto it = ctx->token_to_id.find(name);
                    if (it != ctx->token_to_id.end()) {
                        bf[b] = it->second;
                        found++;
                    }
                }
                if (found == 256)
                    ctx->byte_fallback = std::move(bf);
            }
            // Load scores (used only by legacy SP Viterbi fallback)
            auto scores = core_gguf::kv_f32_array(meta, "tokenizer.ggml.scores");
            if (!scores.empty()) {
                ctx->token_scores = scores;
            }
            // Load BPE merges (correct tokenization for sarashina2.2)
            auto merges = core_gguf::kv_str_array(meta, "tokenizer.ggml.merges");
            if (!merges.empty()) {
                ctx->merge_rank.reserve(merges.size());
                for (size_t i = 0; i < merges.size(); i++) {
                    ctx->merge_rank[merges[i]] = (int32_t)i;
                }
                ctx->use_bpe = true;
                if (ctx->verbosity >= 1) {
                    std::fprintf(stderr, "[irodori] tokenizer: %zu tokens, %zu BPE merges, bos=%d, pad=%d\n",
                                 ctx->vocab.size(), merges.size(), ctx->bos_token_id, ctx->pad_token_id);
                }
            } else if (!ctx->token_scores.empty()) {
                if (ctx->verbosity >= 1) {
                    std::fprintf(stderr, "[irodori] tokenizer: %zu tokens, %zu scores (unigram), bos=%d, pad=%d\n",
                                 ctx->vocab.size(), ctx->token_scores.size(), ctx->bos_token_id, ctx->pad_token_id);
                }
            } else {
                if (ctx->verbosity >= 1) {
                    std::fprintf(stderr, "[irodori] WARNING: tokenizer has vocab but no scores/merges — "
                                         "reconvert GGUF with updated converter\n");
                }
            }
        } else {
            if (ctx->verbosity >= 1) {
                std::fprintf(stderr, "[irodori] WARNING: no tokenizer in GGUF, using byte fallback\n");
            }
        }
    }

    // SP model will be loaded from the tensor in pass 2 (after weight loading)
    // See below after load_weights_from_map.

    core_gguf::free_metadata(meta);

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr,
                     "[irodori] model: dim=%d, layers=%d, heads=%d, text_dim=%d, "
                     "text_layers=%d, latent_dim=%d\n",
                     hp.model_dim, hp.num_layers, hp.num_heads, hp.text_dim, hp.text_layers, hp.latent_dim);
    }

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "irodori_tts", wl)) {
        std::fprintf(stderr, "[irodori] failed to load weights\n");
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Load weight tensors
    auto& ts = wl.tensors;
    auto get = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::require(ts, name.c_str(), "irodori_tts");
    };
    auto try_get = [&](const std::string& name) -> ggml_tensor* { return core_gguf::try_get(ts, name.c_str()); };

    if (!load_weights_from_map(ctx, get, try_get)) {
        std::fprintf(stderr, "[irodori] failed to resolve weight tensors\n");
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Keep weight context + buffer alive (tensors reference both)
    ctx->w_ctx = wl.ctx;
    ctx->buf_weights = wl.buf;

    // Try to load embedded SentencePiece model from tensor
#ifdef IRODORI_HAVE_SENTENCEPIECE
    {
        ggml_tensor* sp_tensor = core_gguf::try_get(ts, "tokenizer.ggml.raw_model");
        if (sp_tensor && sp_tensor->type == GGML_TYPE_I8) {
            size_t n = ggml_nelements(sp_tensor);
            std::string sp_data(n, '\0');
            ggml_backend_tensor_get(sp_tensor, sp_data.data(), 0, n);
            auto sp = std::make_unique<sentencepiece::SentencePieceProcessor>();
            auto status = sp->LoadFromSerializedProto(sp_data);
            if (status.ok()) {
                ctx->sp_processor = std::move(sp);
                if (ctx->verbosity >= 1) {
                    std::fprintf(stderr, "[irodori] SentencePiece model loaded from GGUF (%zu bytes)\n", n);
                }
            } else if (ctx->verbosity >= 1) {
                std::fprintf(stderr, "[irodori] WARNING: SP model tensor load failed\n");
            }
        }
    }
#endif

    // Set BOS token: prefer GGUF tokenizer value, fall back to sarashina2.2 default (1)
    if (ctx->bos_token_id < 0) {
        ctx->bos_token_id = 1; // sarashina2.2 BOS token ID
    }

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] model loaded successfully\n");
    }

    return ctx;
}

void irodori_tts_free(struct irodori_tts_context* ctx) {
    if (!ctx)
        return;
    ctx->dac_fc.free(); // FASTCONV baked-kernel buffer (on codec_backend)
    if (ctx->codec_buf)
        core_gguf::release_weight_buffer(ctx->codec_buf);
    if (ctx->codec_ctx)
        ggml_free(ctx->codec_ctx);
    if (ctx->buf_weights)
        core_gguf::release_weight_buffer(ctx->buf_weights);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->dit_galloc) // §243 persistent DiT graph
        ggml_gallocr_free(ctx->dit_galloc);
    if (ctx->dit_ctx)
        ggml_free(ctx->dit_ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

int irodori_tts_set_codec_path(struct irodori_tts_context* ctx, const char* codec_gguf_path) {
    if (!ctx || !codec_gguf_path)
        return -1;

    // Load DACVAE decoder GGUF
    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] loading DAC-VAE codec: %s\n", codec_gguf_path);
    }

    // Read metadata
    gguf_context* meta = core_gguf::open_metadata(codec_gguf_path);
    if (!meta) {
        std::fprintf(stderr, "[irodori] failed to open codec GGUF: %s\n", codec_gguf_path);
        return -1;
    }

    int codec_sr = core_gguf::kv_i32(meta, "dacvae.sample_rate", 48000);
    int codec_hop = core_gguf::kv_i32(meta, "dacvae.hop_length", 1920);
    int codec_dim = core_gguf::kv_i32(meta, "dacvae.codebook_dim", 32);
    int codec_latent = core_gguf::kv_i32(meta, "dacvae.latent_dim", 1024);
    int codec_dec_dim = core_gguf::kv_i32(meta, "dacvae.decoder_dim", 1536);
    int n_dec_blocks = core_gguf::kv_i32(meta, "dacvae.n_decoder_blocks", 4);
    core_gguf::free_metadata(meta);

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] codec: sr=%d, hop=%d, codebook_dim=%d, decoder_dim=%d, blocks=%d\n", codec_sr,
                     codec_hop, codec_dim, codec_dec_dim, n_dec_blocks);
    }

    // Configure DAC decoder (reuse core_dac with DACVAE config)
    auto& dac = ctx->dac;
    dac.config.hidden_size = codec_latent;
    dac.config.decoder_hidden_size = codec_dec_dim;
    dac.config.sample_rate = codec_sr;
    dac.config.hop_length = codec_hop;
    dac.config.n_decoder_blocks = std::min(n_dec_blocks, 4);
    // DACVAE strides [12,10,8,2] — read from GGUF or hardcode
    int dacvae_strides[4] = {12, 10, 8, 2};
    int dacvae_channels[5] = {codec_dec_dim, codec_dec_dim / 2, codec_dec_dim / 4, codec_dec_dim / 8,
                              codec_dec_dim / 16};
    for (int i = 0; i < 4; i++)
        dac.config.upsampling_ratios[i] = dacvae_strides[i];
    for (int i = 0; i < 5; i++)
        dac.config.decoder_channels[i] = dacvae_channels[i];

    // Load weights onto the codec's compute backend (see init for the
    // GPU/CPU choice).
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(codec_gguf_path, ctx->codec_backend, "dacvae_codec", wl)) {
        std::fprintf(stderr, "[irodori] failed to load codec weights\n");
        return -1;
    }

    auto& ts = wl.tensors;
    auto get = [&](const char* name) -> ggml_tensor* { return core_gguf::require(ts, name, "dacvae_codec"); };
    auto try_get = [&](const char* name) -> ggml_tensor* { return core_gguf::try_get(ts, name); };

    // out_proj: Conv1d(codebook_dim → latent_dim, k=1)
    ctx->codec_out_proj_w = get("dacvae.out_proj.w");
    ctx->codec_out_proj_b = try_get("dacvae.out_proj.b");

    // in_conv: Conv1d(latent_dim → decoder_dim, k=7)
    dac.in_conv_w = get("dacvae.dec.in_conv.w");
    dac.in_conv_b = try_get("dacvae.dec.in_conv.b");

    // Decoder blocks — the DACVAE GGUF stores all block tensors
    // For the watermark-free path, we need: snake_alpha, up convtranspose, 3 residual units
    for (int b = 0; b < dac.config.n_decoder_blocks; b++) {
        auto& blk = dac.blocks[b];
        char buf[128];

        // Snake alpha (block.0.alpha in the GGUF naming)
        snprintf(buf, sizeof(buf), "dacvae.dec.blk.%d.block.0.alpha", b);
        blk.snake_alpha = try_get(buf);

        // ConvTranspose1d (block.1.weight)
        snprintf(buf, sizeof(buf), "dacvae.dec.blk.%d.block.1.weight", b);
        blk.up_w = try_get(buf);
        snprintf(buf, sizeof(buf), "dacvae.dec.blk.%d.block.1.bias", b);
        blk.up_b = try_get(buf);

        // 3 Residual units (block.4, block.5, block.8 in the DACVAE GGUF)
        // block.4 = ResUnit(d=1), block.5 = ResUnit(d=3), block.8 = ResUnit(d=9)
        int res_indices[3] = {4, 5, 8};
        for (int r = 0; r < 3; r++) {
            auto& ru = blk.res[r];
            int ri = res_indices[r];
            // ResUnit: block[0]=Snake alpha, block[1]=Conv(k=7,dil), block[2]=Snake alpha, block[3]=Conv(k=1)
            snprintf(buf, sizeof(buf), "dacvae.dec.blk.%d.block.%d.block.0.alpha", b, ri);
            ru.alpha0 = try_get(buf);
            snprintf(buf, sizeof(buf), "dacvae.dec.blk.%d.block.%d.block.1.weight", b, ri);
            ru.conv0_w = try_get(buf);
            snprintf(buf, sizeof(buf), "dacvae.dec.blk.%d.block.%d.block.1.bias", b, ri);
            ru.conv0_b = try_get(buf);
            snprintf(buf, sizeof(buf), "dacvae.dec.blk.%d.block.%d.block.2.alpha", b, ri);
            ru.alpha1 = try_get(buf);
            snprintf(buf, sizeof(buf), "dacvae.dec.blk.%d.block.%d.block.3.weight", b, ri);
            ru.conv1_w = try_get(buf);
            snprintf(buf, sizeof(buf), "dacvae.dec.blk.%d.block.%d.block.3.bias", b, ri);
            ru.conv1_b = try_get(buf);
        }
    }

    // Final output: from watermark encoder_block (Snake + Conv → Tanh)
    // wm_model.encoder_block.pre.0.alpha = Snake(96) alpha
    // wm_model.encoder_block.pre.1 = NormConv1d(96→1, k=7)
    dac.out_snake_alpha = try_get("dacvae.wm_model.encoder_block.pre.0.alpha");
    // The Conv(96→1) weight — stored with weight_norm reconstructed in GGUF
    dac.out_conv_w = try_get("dacvae.wm_model.encoder_block.pre.1.weight");
    dac.out_conv_b = try_get("dacvae.wm_model.encoder_block.pre.1.bias");

    // ── Encoder (optional; present in newer GGUFs) — voice cloning ──
    // Reference audio → 32-dim latent. Guarded by try_get so codec GGUFs
    // without encoder tensors still load (cloning just stays disabled).
    if (try_get("dacvae.enc.in_conv.w")) {
        auto& enc = ctx->enc;
        {
            gguf_context* m2 = core_gguf::open_metadata(codec_gguf_path);
            if (m2) {
                enc.encoder_dim = core_gguf::kv_i32(m2, "dacvae.encoder_dim", 64);
                core_gguf::free_metadata(m2);
            }
        }
        // Semantic-DACVAE-Japanese-32dim downsample strides (hop 1920 = ∏).
        enc.rates = {2, 8, 10, 12};

        enc.in_conv_w = get("dacvae.enc.in_conv.w");
        enc.in_conv_b = try_get("dacvae.enc.in_conv.b");
        enc.blocks.resize(enc.rates.size());
        char buf[128];
        for (size_t i = 0; i < enc.rates.size(); i++) {
            auto& blk = enc.blocks[i];
            for (int r = 0; r < 3; r++) {
                auto& ru = blk.res[r];
                snprintf(buf, sizeof(buf), "dacvae.enc.blk%zu.res%d.alpha0", i, r);
                ru.alpha0 = get(buf);
                snprintf(buf, sizeof(buf), "dacvae.enc.blk%zu.res%d.conv0.w", i, r);
                ru.conv0_w = get(buf);
                snprintf(buf, sizeof(buf), "dacvae.enc.blk%zu.res%d.conv0.b", i, r);
                ru.conv0_b = try_get(buf);
                snprintf(buf, sizeof(buf), "dacvae.enc.blk%zu.res%d.alpha1", i, r);
                ru.alpha1 = get(buf);
                snprintf(buf, sizeof(buf), "dacvae.enc.blk%zu.res%d.conv1.w", i, r);
                ru.conv1_w = get(buf);
                snprintf(buf, sizeof(buf), "dacvae.enc.blk%zu.res%d.conv1.b", i, r);
                ru.conv1_b = try_get(buf);
            }
            snprintf(buf, sizeof(buf), "dacvae.enc.blk%zu.down_snake.alpha", i);
            blk.down_snake_alpha = get(buf);
            snprintf(buf, sizeof(buf), "dacvae.enc.blk%zu.down.w", i);
            blk.down_w = get(buf);
            snprintf(buf, sizeof(buf), "dacvae.enc.blk%zu.down.b", i);
            blk.down_b = try_get(buf);
        }
        enc.out_snake_alpha = get("dacvae.enc.out_snake.alpha");
        enc.out_conv_w = get("dacvae.enc.out_conv.w");
        enc.out_conv_b = try_get("dacvae.enc.out_conv.b");
        enc.in_proj_w = get("dacvae.enc.in_proj.w");
        enc.in_proj_b = try_get("dacvae.enc.in_proj.b");
        enc.loaded = true;
        if (ctx->verbosity >= 1) {
            std::fprintf(stderr, "[irodori] DAC-VAE encoder loaded (voice cloning enabled)\n");
        }
    }

    ctx->codec_ctx = wl.ctx;
    ctx->codec_buf = wl.buf;
    ctx->has_codec = true;

    // FASTCONV (docs/perf-sweep/PLAN.md): bake F32 copies of the DECODE conv
    // kernels via the shared core_dac cache so the per-graph F16→F32 cast becomes
    // a no-op + k=1 convs become matmuls. Gated CRISPASR_IRODORI_FASTCONV (default
    // on); =0 restores the legacy cast path (clean A/B). Encode/voice-clone convs
    // stay legacy (rare path).
    {
        const char* e = std::getenv("CRISPASR_IRODORI_FASTCONV");
        const bool on = !(e && e[0] == '0');
        std::vector<ggml_tensor*> convs = {dac.in_conv_w, dac.out_conv_w, ctx->codec_out_proj_w};
        for (auto& blk : dac.blocks) {
            convs.push_back(blk.up_w);
            for (int r = 0; r < 3; r++) {
                convs.push_back(blk.res[r].conv0_w);
                convs.push_back(blk.res[r].conv1_w);
            }
        }
        ctx->dac_fc.bake(ctx->codec_backend, convs, on);
    }

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] DAC-VAE codec loaded (%d decoder blocks)\n", dac.config.n_decoder_blocks);
    }
    return 0;
}

// Strided downsample conv (encoder). x:(Cin,T) w:(K,Cin,Cout) k=2*stride,
// pad=stride/2 (SAME-none). Returns (Cout, T/stride).
static ggml_tensor* enc_downsample_conv(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride) {
    const int Cout = (int)w->ne[2];
    ggml_tensor* y = ggml_cont(g, ggml_transpose(g, x));                         // (T, Cin)
    y = ggml_conv_1d(g, w, y, /*stride=*/stride, /*pad=*/stride / 2, /*dil=*/1); // (T_out, Cout, 1)
    const int Tout = (int)y->ne[0];
    y = ggml_reshape_2d(g, y, Tout, Cout);
    y = ggml_cont(g, ggml_transpose(g, y)); // (Cout, T_out)
    if (b)
        y = ggml_add(g, y, b);
    return y;
}

// Encoder ResidualUnit: Snake→Conv(k7,dil)→Snake→Conv(k1)→+x (all SAME).
static ggml_tensor* enc_res_unit(ggml_context* g, ggml_tensor* x, const irodori_dacvae_encoder::ResUnit& u, int dil) {
    ggml_tensor* y = core_dac::snake(g, x, u.alpha0);
    y = core_dac::conv1d(g, y, u.conv0_w, u.conv0_b, 7, dil);
    y = core_dac::snake(g, y, u.alpha1);
    y = core_dac::conv1d(g, y, u.conv1_w, u.conv1_b, 1);
    return ggml_add(g, x, y);
}

// Run the DAC-VAE encoder on a 48 kHz mono waveform (already loudness-
// normalized and reflect-padded to a multiple of the hop). Produces the
// deterministic latent mean: (T_latent, codebook_dim) row-major.
// Returns 0 on success.
static int run_dacvae_encoder(irodori_tts_context* ctx, const float* wave, int n_samples,
                              std::vector<float>& out_latent, int& out_frames) {
    const auto& enc = ctx->enc;
    if (!enc.loaded)
        return -1;
    const int codebook_dim = ctx->hparams.latent_dim; // 32

    const int n_tensors = 4096;
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(ip);

    // Input waveform as (Cin=1, T)
    ggml_tensor* wav_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, n_samples);
    ggml_set_name(wav_in, "wav_in");
    ggml_set_input(wav_in);

    ggml_tensor* h = core_dac::conv1d(g, wav_in, enc.in_conv_w, enc.in_conv_b, 7); // (64, T)
    h = ggml_cast(g, h, GGML_TYPE_F32);
    for (size_t i = 0; i < enc.blocks.size(); i++) {
        const auto& blk = enc.blocks[i];
        h = enc_res_unit(g, h, blk.res[0], 1);
        h = enc_res_unit(g, h, blk.res[1], 3);
        h = enc_res_unit(g, h, blk.res[2], 9);
        h = core_dac::snake(g, h, blk.down_snake_alpha);
        h = enc_downsample_conv(g, h, blk.down_w, blk.down_b, enc.rates[i]);
        h = ggml_cont(g, h);
        h = ggml_cast(g, h, GGML_TYPE_F32);
    }
    h = core_dac::snake(g, h, enc.out_snake_alpha);
    h = core_dac::conv1d(g, h, enc.out_conv_w, enc.out_conv_b, 3); // (1024, T)
    h = ggml_cast(g, h, GGML_TYPE_F32);
    h = core_dac::conv1d(g, h, enc.in_proj_w, enc.in_proj_b, 1); // (2*codebook_dim, T)
    h = ggml_cast(g, h, GGML_TYPE_F32);

    const int T = (int)h->ne[1];
    // mean = first codebook_dim channels (chunk(2)[0]). Channels are ne[0].
    ggml_tensor* mean = ggml_view_2d(g, h, codebook_dim, T, h->nb[1], 0);
    mean = ggml_cont(g, mean); // (codebook_dim, T) contiguous == (T, codebook_dim) row-major
    ggml_set_name(mean, "ref_mean");
    ggml_set_output(mean);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_tensors, false);
    ggml_build_forward_expand(gf, mean);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->codec_backend));
    if (!ggml_gallocr_reserve(galloc, gf) || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "[irodori] ERROR: encoder graph alloc failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return -1;
    }
    ggml_backend_tensor_set(wav_in, wave, 0, (size_t)n_samples * sizeof(float));
    ggml_backend_graph_compute(ctx->codec_backend, gf);

    out_frames = T;
    out_latent.resize((size_t)T * codebook_dim);
    ggml_backend_tensor_get(mean, out_latent.data(), 0, out_latent.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(g);
    return 0;
}

// ── ITU-R BS.1770-4 integrated loudness (K-weighting + gated) ────────
// Matches audiotools' AudioSignal.loudness() closely enough to reproduce
// the -16 LUFS reference-normalization gain used by irodori_tts/codec.py.
static double bs1770_integrated_loudness(const float* x, int n, int sr) {
    if (n <= 0 || sr <= 0)
        return -70.0;
    // Stage-1 shelving + stage-2 highpass biquads (per BS.1770-4, coeffs
    // are defined at 48 kHz; audiotools resamples filters — we run at the
    // signal's rate using the standard bilinear-derived 48k coefficients,
    // which is what pyloudnorm/audiotools use for 48 kHz reference audio).
    // Pre-filter (high-shelf):
    const double b0s = 1.53512485958697, b1s = -2.69169618940638, b2s = 1.19839281085285;
    const double a1s = -1.69065929318241, a2s = 0.73248077421585;
    // Highpass (RLB):
    const double b0h = 1.0, b1h = -2.0, b2h = 1.0;
    const double a1h = -1.99004745483398, a2h = 0.99007225036621;
    std::vector<double> y(n);
    {
        double z1 = 0, z2 = 0; // shelf
        for (int i = 0; i < n; i++) {
            double in = x[i];
            double out = b0s * in + z1;
            z1 = b1s * in - a1s * out + z2;
            z2 = b2s * in - a2s * out;
            y[i] = out;
        }
        double w1 = 0, w2 = 0; // highpass
        for (int i = 0; i < n; i++) {
            double in = y[i];
            double out = b0h * in + w1;
            w1 = b1h * in - a1h * out + w2;
            w2 = b2h * in - a2h * out;
            y[i] = out;
        }
    }
    // 400 ms blocks, 75% overlap (100 ms hops); mean-square per block.
    const int blk = (int)(0.4 * sr);
    const int hop = (int)(0.1 * sr);
    if (n < blk)
        return -70.0;
    std::vector<double> loud;
    std::vector<double> ms;
    for (int start = 0; start + blk <= n; start += hop) {
        double s = 0;
        for (int i = start; i < start + blk; i++)
            s += y[i] * y[i];
        double meansq = s / blk;
        ms.push_back(meansq);
        loud.push_back(-0.691 + 10.0 * std::log10(meansq + 1e-12));
    }
    // Absolute gate at -70 LUFS, then relative gate at (mean-10 LUFS).
    auto gated_mean = [&](double thr) -> double {
        double acc = 0;
        int cnt = 0;
        for (size_t i = 0; i < loud.size(); i++)
            if (loud[i] > thr) {
                acc += ms[i];
                cnt++;
            }
        return cnt ? acc / cnt : 0.0;
    };
    // First pass: absolute gate.
    double abs_ms = 0;
    int abs_cnt = 0;
    for (size_t i = 0; i < loud.size(); i++)
        if (loud[i] > -70.0) {
            abs_ms += ms[i];
            abs_cnt++;
        }
    if (!abs_cnt)
        return -70.0;
    double rel_thr = -0.691 + 10.0 * std::log10(abs_ms / abs_cnt + 1e-12) - 10.0;
    double final_ms = gated_mean(rel_thr);
    return -0.691 + 10.0 * std::log10(final_ms + 1e-12);
}

int irodori_tts_set_reference(struct irodori_tts_context* ctx, const float* ref_pcm, int n_samples, int sample_rate) {
    if (!ctx || !ref_pcm || n_samples <= 0 || sample_rate <= 0)
        return -1;
    if (!ctx->enc.loaded) {
        std::fprintf(stderr, "[irodori] voice cloning requires an encoder-enabled DAC-VAE codec GGUF "
                             "(re-download the codec: it must contain dacvae.enc.* tensors).\n");
        return -1;
    }

    const int SR = ctx->hparams.sample_rate;                                        // 48000
    const int hop = ctx->hparams.codec_hop_length;                                  // 1920
    const int latent_pd = ctx->hparams.latent_dim * ctx->hparams.latent_patch_size; // 32

    // Debug: feed an already-normalized waveform straight through (isolates
    // the encoder graph from resample/LUFS for parity testing).
    const char* dbg = std::getenv("CRISPASR_IRODORI_ENC_PRENORM");

    // Content-addressed reference cache (runtime, so every caller benefits —
    // CLI, C ABI, server, wrappers). Key = hash of (ref_pcm, sample_rate); skip
    // resample + −16 LUFS + DAC-VAE encode on a hit. Skipped for the debug
    // pre-norm path (its latent is not the normal-pipeline latent).
    const uint64_t cache_key = crispasr_ref_cache::fnv1a(ref_pcm, (size_t)n_samples * sizeof(float)) ^
                               crispasr_ref_cache::fnv1a(&sample_rate, sizeof(sample_rate));
    if (!dbg) {
        std::vector<uint32_t> shape;
        std::vector<float> lat;
        if (crispasr_ref_cache::get_floats("irodori-latent", &cache_key, sizeof(cache_key), shape, lat) &&
            shape.size() == 2 && shape[1] == (uint32_t)latent_pd && shape[0] > 0 &&
            lat.size() == (size_t)shape[0] * shape[1]) {
            ctx->ref_latent = std::move(lat);
            ctx->ref_latent_frames = (int)shape[0];
            if (ctx->verbosity >= 1)
                std::fprintf(stderr, "[irodori] reference from cache (%d latent frames)\n", ctx->ref_latent_frames);
            return 0;
        }
    }

    // 1) Resample to the codec rate.
    std::vector<float> wav;
    if (sample_rate != SR) {
        wav = core_audio::resample_polyphase(ref_pcm, n_samples, sample_rate, SR);
    } else {
        wav.assign(ref_pcm, ref_pcm + n_samples);
    }

    if (!dbg) {
        // 2) Loudness-normalize to -16 LUFS (codec default), then peak-limit.
        double loud = bs1770_integrated_loudness(wav.data(), (int)wav.size(), SR);
        if (loud > -70.0) {
            double gain = std::pow(10.0, (-16.0 - loud) / 20.0);
            for (auto& s : wav)
                s = (float)(s * gain);
        }
        // ensure_max_of_audio: scale down if peak exceeds 1.0.
        float peak = 0.0f;
        for (float s : wav)
            peak = std::max(peak, std::fabs(s));
        if (peak > 1.0f) {
            float inv = 1.0f / peak;
            for (auto& s : wav)
                s *= inv;
        }
        if (ctx->verbosity >= 1)
            std::fprintf(stderr, "[irodori] reference loudness %.2f LUFS → -16 LUFS (peak %.3f)\n", loud, peak);
    }

    // 3) Reflect-pad to a multiple of the hop (DACVAE._pad).
    int rem = (int)wav.size() % hop;
    if (rem != 0) {
        int pad = hop - rem;
        size_t base = wav.size();
        wav.resize(base + pad);
        for (int i = 0; i < pad; i++) // reflect (no edge repeat): x[-2], x[-3], ...
            wav[base + i] = wav[base - 2 - i];
    }

    // 4) Encode → latent mean (T, 32).
    std::vector<float> latent;
    int frames = 0;
    if (run_dacvae_encoder(ctx, wav.data(), (int)wav.size(), latent, frames) != 0 || frames <= 0) {
        std::fprintf(stderr, "[irodori] reference encode failed.\n");
        return -1;
    }
    ctx->ref_latent = std::move(latent);
    ctx->ref_latent_frames = frames;

    // Save to the content-addressed cache for next time (any caller).
    if (!dbg)
        crispasr_ref_cache::put_floats("irodori-latent", &cache_key, sizeof(cache_key),
                                       {(uint32_t)frames, (uint32_t)latent_pd}, ctx->ref_latent.data(),
                                       ctx->ref_latent.size());

    // Optional: dump the latent for parity checks.
    if (const char* out = std::getenv("CRISPASR_IRODORI_ENC_DUMP")) {
        FILE* f = std::fopen(out, "wb");
        if (f) {
            std::fwrite(ctx->ref_latent.data(), sizeof(float), ctx->ref_latent.size(), f);
            std::fclose(f);
            std::fprintf(stderr, "[irodori] dumped ref latent (%d, %d) → %s\n", frames, ctx->hparams.latent_dim, out);
        }
    }

    if (ctx->verbosity >= 1)
        std::fprintf(stderr, "[irodori] reference encoded: %d latent frames (%.2f sec)\n", frames,
                     (float)frames * hop / SR);
    return 0;
}

int irodori_tts_set_reference_latent(struct irodori_tts_context* ctx, const float* latent, int n_frames) {
    if (!ctx || !latent || n_frames <= 0)
        return -1;

    int D = ctx->hparams.latent_dim * ctx->hparams.latent_patch_size;
    ctx->ref_latent.resize(n_frames * D);
    std::memcpy(ctx->ref_latent.data(), latent, n_frames * D * sizeof(float));
    ctx->ref_latent_frames = n_frames;

    IRODORI_DBG("[irodori] set reference latent: %d frames x %d dims\n", n_frames, D);
    return 0;
}

int irodori_tts_get_reference_latent(const struct irodori_tts_context* ctx, const float** out_latent, int* out_frames,
                                     int* out_dim) {
    if (!ctx || ctx->ref_latent_frames <= 0 || ctx->ref_latent.empty())
        return -1;
    if (out_latent)
        *out_latent = ctx->ref_latent.data();
    if (out_frames)
        *out_frames = ctx->ref_latent_frames;
    if (out_dim)
        *out_dim = ctx->hparams.latent_dim * ctx->hparams.latent_patch_size;
    return 0;
}

int irodori_tts_reference_latent_dim(const struct irodori_tts_context* ctx) {
    if (!ctx)
        return 0;
    return ctx->hparams.latent_dim * ctx->hparams.latent_patch_size;
}

void irodori_tts_clear_reference(struct irodori_tts_context* ctx) {
    if (!ctx)
        return;
    ctx->ref_latent.clear();
    ctx->ref_latent_frames = 0;
}

// ── CPU-side forward pass helpers ────────────────────────────────────

// Run text encoder: token_ids → text_state (T_text × text_dim).
// Builds a ggml graph, evaluates on backend, returns CPU-side result.
static std::vector<float> run_text_encoder(irodori_tts_context* ctx, const int32_t* token_ids, int T_text) {
    const auto& hp = ctx->hparams;
    const int D = hp.text_dim;

    // Tensor count: embedding + per-block (Q,K,V,O,gate,norms,mlp,permute,cast) + overhead
    const int n_tensors = 512 + hp.text_layers * 80;
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(ip);

    // Input tensors
    ggml_tensor* ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_text);
    ggml_set_name(ids, "token_ids");
    ggml_set_input(ids);

    ggml_tensor* mask_f = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, T_text);
    ggml_set_name(mask_f, "text_mask_f");
    ggml_set_input(mask_f);

    ggml_tensor* pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_text);
    ggml_set_name(pos, "pos_ids");
    ggml_set_input(pos);

    // Build graph
    ggml_tensor* out = build_text_encoder_graph(g, ctx, ids, mask_f, pos, T_text);
    ggml_set_name(out, "text_state");
    ggml_set_output(out);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_tensors, false);
    ggml_build_forward_expand(gf, out);

    // Allocate and set inputs
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return {};
    }

    ggml_backend_tensor_set(ids, token_ids, 0, T_text * sizeof(int32_t));
    std::vector<float> mask_data(T_text, 1.0f);
    ggml_backend_tensor_set(mask_f, mask_data.data(), 0, T_text * sizeof(float));
    std::vector<int32_t> pos_data(T_text);
    for (int i = 0; i < T_text; i++)
        pos_data[i] = i;
    ggml_backend_tensor_set(pos, pos_data.data(), 0, T_text * sizeof(int32_t));

    // Compute
    ggml_backend_graph_compute(ctx->backend, gf);

    // Read output
    std::vector<float> result(T_text * D);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(g);
    return result;
}

// Run caption encoder: token_ids → caption_state (T_cap × caption_dim).
// VoiceDesign only.
static std::vector<float> run_caption_encoder(irodori_tts_context* ctx, const int32_t* token_ids, int T_cap) {
    const auto& hp = ctx->hparams;
    const int D = hp.caption_dim;

    const int n_tensors = 512 + hp.caption_layers * 80;
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(ip);

    ggml_tensor* ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_cap);
    ggml_set_name(ids, "cap_token_ids");
    ggml_set_input(ids);

    ggml_tensor* mask_f = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, T_cap);
    ggml_set_name(mask_f, "cap_mask_f");
    ggml_set_input(mask_f);

    ggml_tensor* pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_cap);
    ggml_set_name(pos, "cap_pos_ids");
    ggml_set_input(pos);

    ggml_tensor* out = build_caption_encoder_graph(g, ctx, ids, mask_f, pos, T_cap);
    ggml_set_name(out, "cap_state");
    ggml_set_output(out);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_tensors, false);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return {};
    }

    ggml_backend_tensor_set(ids, token_ids, 0, T_cap * sizeof(int32_t));
    std::vector<float> mask_data(T_cap, 1.0f);
    ggml_backend_tensor_set(mask_f, mask_data.data(), 0, T_cap * sizeof(float));
    std::vector<int32_t> pos_data(T_cap);
    for (int i = 0; i < T_cap; i++)
        pos_data[i] = i;
    ggml_backend_tensor_set(pos, pos_data.data(), 0, T_cap * sizeof(int32_t));

    ggml_backend_graph_compute(ctx->backend, gf);

    std::vector<float> result(T_cap * D);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(g);
    return result;
}

// Run the speaker (reference-latent) encoder on ctx->ref_latent → speaker
// state (speaker_dim, T_ref) as a CPU-side array. Empty on failure.
static std::vector<float> run_speaker_encoder(irodori_tts_context* ctx) {
    const auto& hp = ctx->hparams;
    const int T_ref = ctx->ref_latent_frames;
    if (T_ref <= 0 || ctx->ref_latent.empty())
        return {};
    const int latent_pd = hp.patched_latent_dim(); // 32

    const int n_tensors = 512 + hp.speaker_layers * 80;
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(ip);

    // ref_latent is (T_ref, latent_pd) row-major == ggml (latent_pd, T_ref)
    ggml_tensor* latent_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, latent_pd, T_ref);
    ggml_set_name(latent_in, "ref_latent");
    ggml_set_input(latent_in);

    ggml_tensor* mask_f = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, T_ref);
    ggml_set_name(mask_f, "spk_mask_f");
    ggml_set_input(mask_f);

    ggml_tensor* pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_ref);
    ggml_set_name(pos, "spk_pos");
    ggml_set_input(pos);

    ggml_tensor* out = build_speaker_encoder_graph(g, ctx, latent_in, mask_f, pos, T_ref);
    ggml_set_name(out, "spk_state");
    ggml_set_output(out);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_tensors, false);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return {};
    }
    ggml_backend_tensor_set(latent_in, ctx->ref_latent.data(), 0, ctx->ref_latent.size() * sizeof(float));
    std::vector<float> mask_data(T_ref, 1.0f);
    ggml_backend_tensor_set(mask_f, mask_data.data(), 0, T_ref * sizeof(float));
    std::vector<int32_t> pos_data(T_ref);
    for (int i = 0; i < T_ref; i++)
        pos_data[i] = i;
    ggml_backend_tensor_set(pos, pos_data.data(), 0, T_ref * sizeof(int32_t));

    ggml_backend_graph_compute(ctx->backend, gf);

    std::vector<float> result((size_t)T_ref * hp.speaker_dim);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(g);
    return result;
}

// Duration predictor (token_sum_adarn_zero architecture): predicts the total
// number of DAC-VAE latent frames for the utterance. Per token: RMSNorm →
// AdaLN-Zero (shift/scale/gate from silu(speaker_vec)) → SwiGLU → tanh(gate)-
// gated residual, ×N; then out_norm → out_proj → softplus per token, summed.
// The model returns log1p(total_frames), so total_frames == the raw sum.
// spk_vec: speaker_state[:,0] (768) when cloning, else the learned null_speaker.
// Returns total_frames (>0), or -1 on failure / no predictor.
static float run_duration_predictor(irodori_tts_context* ctx, const float* text_state_data, int T_text,
                                    const float* spk_vec_data, const float* cap_vec_data = nullptr) {
    const auto& hp = ctx->hparams;
    const auto& w = ctx->weights;
    if (!hp.use_duration_predictor || !w.dur_input_proj_w || !w.dur_out_proj_w)
        return -1.0f;
    const int Dh = hp.duration_hidden_dim; // 1024
    const int Dt = hp.text_dim;            // 512
    const int Dsp = hp.speaker_dim;        // 768

    const int n_tensors = 512 + hp.duration_layers * 40;
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(ip);

    ggml_tensor* ts_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, Dt, T_text);
    ggml_set_name(ts_in, "dur_text_state");
    ggml_set_input(ts_in);
    ggml_tensor* spk_in = ggml_new_tensor_1d(g, GGML_TYPE_F32, Dsp);
    ggml_set_name(spk_in, "dur_spk_vec");
    ggml_set_input(spk_in);

    // Caption vec input (VoiceDesign dual_adarn_zero)
    const int Dcap = hp.caption_dim;
    const bool has_cap_dur = hp.use_caption_condition && w.dur_null_caption;
    ggml_tensor* cap_dur_in = nullptr;
    if (has_cap_dur) {
        cap_dur_in = ggml_new_tensor_1d(g, GGML_TYPE_F32, Dcap);
        ggml_set_name(cap_dur_in, "dur_cap_vec");
        ggml_set_input(cap_dur_in);
    }

    // h = input_proj(text_state) → (Dh, T)
    ggml_tensor* h = mul_mat_f32(g, w.dur_input_proj_w, ts_in);
    h = ggml_add(g, h, w.dur_input_proj_b);
    ggml_tensor* sm = ggml_silu(g, spk_in); // silu(cond) once; reused per block
    ggml_tensor* cm = has_cap_dur ? ggml_silu(g, cap_dur_in) : nullptr;

    for (int i = 0; i < hp.duration_layers; i++) {
        const auto& blk = w.dur_blocks[i];
        ggml_tensor* hn = rms_norm(g, h, blk.norm, hp.norm_eps); // (Dh, T)
        // Speaker modulation: (shift, scale, gate) = mod(silu(spk)) chunked
        ggml_tensor* mod = mul_mat_f32(g, blk.mod_w, sm); // (3*Dh, 1)
        mod = ggml_add(g, mod, blk.mod_b);
        ggml_tensor* shift = ggml_view_1d(g, mod, Dh, 0);
        ggml_tensor* scale = ggml_view_1d(g, mod, Dh, (size_t)Dh * sizeof(float));
        ggml_tensor* gate = ggml_view_1d(g, mod, Dh, (size_t)2 * Dh * sizeof(float));

        // Caption modulation additive (VoiceDesign dual_adarn_zero):
        //   shift += cap_shift; scale += cap_scale; gate += cap_gate
        if (cm && blk.cap_mod_w) {
            ggml_tensor* cmod = mul_mat_f32(g, blk.cap_mod_w, cm);
            cmod = ggml_add(g, cmod, blk.cap_mod_b);
            ggml_tensor* cap_shift = ggml_view_1d(g, cmod, Dh, 0);
            ggml_tensor* cap_scale = ggml_view_1d(g, cmod, Dh, (size_t)Dh * sizeof(float));
            ggml_tensor* cap_gate = ggml_view_1d(g, cmod, Dh, (size_t)2 * Dh * sizeof(float));
            shift = ggml_add(g, shift, cap_shift);
            scale = ggml_add(g, scale, cap_scale);
            gate = ggml_add(g, gate, cap_gate);
        }

        // hm = hn*(1+scale)+shift  (broadcast (Dh) over (Dh,T))
        ggml_tensor* hm = ggml_add(g, hn, ggml_mul(g, hn, scale));
        hm = ggml_add(g, hm, shift);
        // SwiGLU: w2(silu(w1 hm) * w3 hm)
        ggml_tensor* gate_proj = mul_mat_f32(g, blk.mlp_w1, hm);
        ggml_tensor* up_proj = mul_mat_f32(g, blk.mlp_w3, hm);
        ggml_tensor* mlp = ggml_mul(g, ggml_silu(g, gate_proj), up_proj);
        mlp = mul_mat_f32(g, blk.mlp_w2, mlp);
        // gated residual
        h = ggml_add(g, h, ggml_mul(g, mlp, ggml_tanh(g, gate)));
    }
    h = rms_norm(g, h, w.dur_out_norm, hp.norm_eps);
    ggml_tensor* logits = mul_mat_f32(g, w.dur_out_proj_w, h); // (1, T)
    logits = ggml_add(g, logits, w.dur_out_proj_b);
    ggml_set_name(logits, "dur_logits");
    ggml_set_output(logits);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_tensors, false);
    ggml_build_forward_expand(gf, logits);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return -1.0f;
    }
    ggml_backend_tensor_set(ts_in, text_state_data, 0, (size_t)Dt * T_text * sizeof(float));
    // null_speaker fallback when no reference is attended.
    std::vector<float> spk_tmp;
    const float* spk_src = spk_vec_data;
    if (!spk_src) {
        spk_tmp.resize(Dsp);
        if (w.dur_null_speaker)
            ggml_backend_tensor_get(w.dur_null_speaker, spk_tmp.data(), 0, (size_t)Dsp * sizeof(float));
        spk_src = spk_tmp.data();
    }
    ggml_backend_tensor_set(spk_in, spk_src, 0, (size_t)Dsp * sizeof(float));
    // Caption vec: use provided or null_caption fallback
    if (has_cap_dur && cap_dur_in) {
        std::vector<float> cap_tmp;
        const float* cap_src = cap_vec_data;
        if (!cap_src) {
            cap_tmp.resize(Dcap, 0.0f);
            if (w.dur_null_caption)
                ggml_backend_tensor_get(w.dur_null_caption, cap_tmp.data(), 0, (size_t)Dcap * sizeof(float));
            cap_src = cap_tmp.data();
        }
        ggml_backend_tensor_set(cap_dur_in, cap_src, 0, (size_t)Dcap * sizeof(float));
    }
    ggml_backend_graph_compute(ctx->backend, gf);

    std::vector<float> lg(T_text);
    ggml_backend_tensor_get(logits, lg.data(), 0, (size_t)T_text * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(g);

    // softplus per token, summed (numerically stable).
    double total = 0.0;
    for (int i = 0; i < T_text; i++) {
        float x = lg[i];
        total += (double)std::max(x, 0.0f) + std::log1p(std::exp(-std::fabs(x)));
    }
    return (float)total;
}

// Run one DiT forward pass: x_t + timestep + text_state + spk_state → velocity prediction.
// Returns velocity as CPU-side array (T_latent × patched_latent_dim).
static std::vector<float> run_dit_forward(irodori_tts_context* ctx, const float* x_t_data, int T_latent,
                                          const float* cond_embed_data, // (model_dim*3,)
                                          const float* text_state_data, int T_text, const float* spk_state_data,
                                          int T_ref, bool attend_speaker = false, const float* cap_state_data = nullptr,
                                          int T_cap = 0, bool attend_caption = false) {
    const auto& hp = ctx->hparams;
    const auto& w = ctx->weights;
    const int D = hp.model_dim;
    const int latent_d = hp.patched_latent_dim();

    // §243 STEP-0: split per-DiT-call time into graph construct+alloc (what a
    // persistent cached graph eliminates), input-set, and compute. Gated
    // CRISPASR_IRODORI_DIT_TIMING=1; cumulative, last line = the answer.
    static const bool s_dit_timing = []() {
        const char* e = std::getenv("CRISPASR_IRODORI_DIT_TIMING");
        return e && e[0] && e[0] != '0';
    }();
    static int64_t s_construct_us = 0, s_setinput_us = 0, s_compute_us = 0;
    static int s_dit_calls = 0;
    const int64_t t_dit0 = s_dit_timing ? ggml_time_us() : 0;

    // §243: reuse a persistent cached DiT graph when the shape signature matches
    // (it is constant across every ODE/CFG call of one generation). Env-gated
    // CRISPASR_IRODORI_PERSIST_GRAPH, default OFF. On a cache hit we skip build +
    // gallocr entirely and only re-set inputs below; stable tensor addresses let
    // the CUDA graph warm up once and replay (Ampere+), instead of re-capturing
    // every step. Correctness is identical — only the graph's identity is reused.
    static const bool s_persist = []() {
        const char* e = std::getenv("CRISPASR_IRODORI_PERSIST_GRAPH");
        return e && e[0] && e[0] != '0';
    }();
    const bool has_spk = spk_state_data && T_ref > 0;
    const bool has_cap = cap_state_data && T_cap > 0;
    // KV length (self | text | speaker | caption) — needed both to size the mask
    // tensor at build and to refill it every call, so keep it in the outer scope.
    const int T_kv = T_latent + T_text + (has_spk ? T_ref : 0) + (has_cap ? T_cap : 0);

    ggml_context* g = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor *x_in = nullptr, *cond_in = nullptr, *text_in = nullptr, *pos_latent = nullptr;
    ggml_tensor *spk_in = nullptr, *cap_in = nullptr, *attn_mask_in = nullptr, *x = nullptr;

    const bool cache_hit = s_persist && ctx->dit_gf && ctx->dit_T_latent == T_latent && ctx->dit_T_text == T_text &&
                           ctx->dit_T_ref == T_ref && ctx->dit_T_cap == T_cap && ctx->dit_has_spk == has_spk &&
                           ctx->dit_has_cap == has_cap;

    if (cache_hit) {
        g = ctx->dit_ctx;
        gf = ctx->dit_gf;
        galloc = ctx->dit_galloc;
        x_in = ggml_graph_get_tensor(gf, "x_t");
        cond_in = ggml_graph_get_tensor(gf, "cond_embed");
        text_in = ggml_graph_get_tensor(gf, "text_state");
        pos_latent = ggml_graph_get_tensor(gf, "pos_latent");
        spk_in = ggml_graph_get_tensor(gf, "spk_state"); // null if not in this graph
        cap_in = ggml_graph_get_tensor(gf, "cap_state");
        attn_mask_in = ggml_graph_get_tensor(gf, "attn_mask");
        x = ggml_graph_get_tensor(gf, "v_pred");
    } else {
        // Persist on but shape changed (or first call): free any stale cached graph.
        if (s_persist && ctx->dit_ctx) {
            if (ctx->dit_galloc)
                ggml_gallocr_free(ctx->dit_galloc);
            ggml_free(ctx->dit_ctx);
            ctx->dit_ctx = nullptr;
            ctx->dit_gf = nullptr;
            ctx->dit_galloc = nullptr;
        }

        const int n_tensors = 2048 + hp.num_layers * 200; // extra for half-RoPE views/concats
        size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false);
        ggml_init_params ip = {ctx_size, nullptr, true};
        g = ggml_init(ip);

        // Input tensors
        x_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, latent_d, T_latent);
        ggml_set_name(x_in, "x_t");
        ggml_set_input(x_in);

        cond_in = ggml_new_tensor_1d(g, GGML_TYPE_F32, D * 3);
        ggml_set_name(cond_in, "cond_embed");
        ggml_set_input(cond_in);

        text_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.text_dim, T_text);
        ggml_set_name(text_in, "text_state");
        ggml_set_input(text_in);

        pos_latent = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_latent);
        ggml_set_name(pos_latent, "pos_latent");
        ggml_set_input(pos_latent);

        if (has_spk) {
            spk_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.speaker_dim, T_ref);
            ggml_set_name(spk_in, "spk_state");
            ggml_set_input(spk_in);
        }
        if (has_cap) {
            cap_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.caption_dim, T_cap);
            ggml_set_name(cap_in, "cap_state");
            ggml_set_input(cap_in);
        }

        // Attention mask when speaker or caption needs masking (T_kv from outer scope)
        if (spk_in || cap_in) {
            attn_mask_in = ggml_new_tensor_2d(g, GGML_TYPE_F16, T_kv, T_latent);
            ggml_set_name(attn_mask_in, "attn_mask");
            ggml_set_input(attn_mask_in);
        }

        // Build graph: in_proj → DiT blocks → out_norm → out_proj
        IRODORI_DBG("[irodori]     in_proj: (%d,%d) x (%d,%d)\n", (int)w.in_proj_w->ne[0], (int)w.in_proj_w->ne[1],
                    latent_d, T_latent);
        x = mul_mat_f32(g, w.in_proj_w, x_in);
        x = ggml_add(g, x, w.in_proj_b);
        IRODORI_DBG("[irodori]     after in_proj: (%d,%d)\n", (int)x->ne[0], (int)x->ne[1]);

        // Build just the first layer for debugging, then add more once stable
        int n_build_layers = hp.num_layers;
        const char* env_layers = std::getenv("CRISPASR_IRODORI_LAYERS");
        if (env_layers)
            n_build_layers = std::min(std::atoi(env_layers), hp.num_layers);
        for (int i = 0; i < n_build_layers; i++) {
            IRODORI_DBG("[irodori]     building DiT block %d...\n", i);
            x = build_dit_block_graph(g, ctx, w.dit_blocks[i], x, cond_in, text_in, T_text, spk_in, T_ref, cap_in,
                                      T_cap, T_latent, pos_latent, attn_mask_in);
            IRODORI_DBG("[irodori]     DiT block %d built OK\n", i);
        }

        x = rms_norm(g, x, w.out_norm, hp.norm_eps);
        x = mul_mat_f32(g, w.out_proj_w, x);
        x = ggml_add(g, x, w.out_proj_b);
        ggml_set_name(x, "v_pred");
        ggml_set_output(x);

        IRODORI_DBG("[irodori]     DiT graph built, creating cgraph (%d tensors)...\n", n_tensors);
        gf = ggml_new_graph_custom(g, n_tensors, false);
        ggml_build_forward_expand(gf, x);
        IRODORI_DBG("[irodori]     cgraph has %d nodes, allocating...\n", ggml_graph_n_nodes(gf));

        // Allocate
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        bool res_ok = ggml_gallocr_reserve(galloc, gf);
        IRODORI_DBG("[irodori]     gallocr reserve: %s\n", res_ok ? "OK" : "FAILED");
        bool alloc_ok = ggml_gallocr_alloc_graph(galloc, gf);
        IRODORI_DBG("[irodori]     gallocr alloc: %s\n", alloc_ok ? "OK" : "FAILED");
        if (!res_ok || !alloc_ok) {
            std::fprintf(stderr, "[irodori] ERROR: failed to allocate DiT graph\n");
            ggml_gallocr_free(galloc);
            ggml_free(g);
            return {};
        }

        if (s_persist) {
            ctx->dit_ctx = g;
            ctx->dit_gf = gf;
            ctx->dit_galloc = galloc;
            ctx->dit_T_latent = T_latent;
            ctx->dit_T_text = T_text;
            ctx->dit_T_ref = T_ref;
            ctx->dit_T_cap = T_cap;
            ctx->dit_has_spk = has_spk;
            ctx->dit_has_cap = has_cap;
        }
    }

    const int64_t t_dit1 = s_dit_timing ? ggml_time_us() : 0; // construct+alloc done

    // Set inputs (only if allocated — unused inputs have null buffers)
    if (x_in->buffer)
        ggml_backend_tensor_set(x_in, x_t_data, 0, T_latent * latent_d * sizeof(float));
    if (cond_in->buffer)
        ggml_backend_tensor_set(cond_in, cond_embed_data, 0, D * 3 * sizeof(float));
    if (text_in->buffer)
        ggml_backend_tensor_set(text_in, text_state_data, 0, T_text * hp.text_dim * sizeof(float));
    if (spk_in && spk_in->buffer) {
        ggml_backend_tensor_set(spk_in, spk_state_data, 0, T_ref * hp.speaker_dim * sizeof(float));
    }
    if (cap_in && cap_in->buffer) {
        ggml_backend_tensor_set(cap_in, cap_state_data, 0, (size_t)T_cap * hp.caption_dim * sizeof(float));
    }
    if (pos_latent->buffer) {
        std::vector<int32_t> pos_data(T_latent);
        for (int i = 0; i < T_latent; i++)
            pos_data[i] = i;
        ggml_backend_tensor_set(pos_latent, pos_data.data(), 0, T_latent * sizeof(int32_t));
    }
    if (attn_mask_in && attn_mask_in->buffer) {
        // Build mask (F16): 0.0 for attend, -inf for masked positions.
        // KV order: self | text | speaker | caption (matching Python concat order).
        std::vector<ggml_fp16_t> mask_data((size_t)T_kv * T_latent);
        ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
        ggml_fp16_t neginf_f16 = ggml_fp32_to_fp16(-INFINITY);
        int spk_start = T_latent + T_text;
        int cap_start = spk_start + (spk_in ? T_ref : 0);
        for (int q = 0; q < T_latent; q++) {
            for (int k = 0; k < T_kv; k++) {
                bool is_speaker = spk_in && k >= spk_start && k < cap_start;
                bool is_caption = cap_in && k >= cap_start;
                bool masked = (is_speaker && !attend_speaker) || (is_caption && !attend_caption);
                mask_data[q * T_kv + k] = masked ? neginf_f16 : zero_f16;
            }
        }
        ggml_backend_tensor_set(attn_mask_in, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }

    const int64_t t_dit2 = s_dit_timing ? ggml_time_us() : 0; // input-set done
    ggml_backend_graph_compute(ctx->backend, gf);
    if (s_dit_timing) {
        const int64_t t_dit3 = ggml_time_us();
        s_construct_us += t_dit1 - t_dit0;
        s_setinput_us += t_dit2 - t_dit1;
        s_compute_us += t_dit3 - t_dit2;
        s_dit_calls++;
        const double c = s_construct_us / 1e3, si = s_setinput_us / 1e3, cp = s_compute_us / 1e3;
        const double tot = c + si + cp;
        fprintf(stderr,
                "[irodori dit_timing] calls=%d cum: construct+alloc=%.1fms (%.1f%%) setinput=%.1fms (%.1f%%) "
                "compute=%.1fms (%.1f%%) | persistence removes the construct+alloc share\n",
                s_dit_calls, c, tot > 0 ? 100 * c / tot : 0, si, tot > 0 ? 100 * si / tot : 0, cp,
                tot > 0 ? 100 * cp / tot : 0);
    }

    // Read velocity prediction
    std::vector<float> v_pred(T_latent * latent_d);
    ggml_backend_tensor_get(x, v_pred.data(), 0, v_pred.size() * sizeof(float));

    // §243: keep the graph alive when persisting (freed at teardown); otherwise
    // free per-call as before.
    if (!s_persist) {
        ggml_gallocr_free(galloc);
        ggml_free(g);
    }
    return v_pred;
}

// Compute timestep conditioning: sinusoidal(t) → MLP → cond_embed (model_dim*3,)
static std::vector<float> run_timestep_cond(irodori_tts_context* ctx, float t) {
    const auto& hp = ctx->hparams;
    const auto& w = ctx->weights;
    const int D = hp.model_dim;

    // 1. Sinusoidal timestep embedding (CPU side)
    std::vector<float> t_emb;
    compute_timestep_embedding(t, hp.timestep_embed_dim, t_emb);

    // 2. MLP: Linear → SiLU → Linear → SiLU → Linear
    const int n_tensors = 32;
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(ip);

    ggml_tensor* emb_in = ggml_new_tensor_1d(g, GGML_TYPE_F32, hp.timestep_embed_dim);
    ggml_set_name(emb_in, "t_emb");
    ggml_set_input(emb_in);

    // cond_module: Linear(512,2048) → SiLU → Linear(2048,2048) → SiLU → Linear(2048, 2048*3)
    ggml_tensor* h = mul_mat_f32(g, w.cond_0_w, emb_in);
    h = ggml_silu(g, h);
    h = mul_mat_f32(g, w.cond_2_w, h);
    h = ggml_silu(g, h);
    h = mul_mat_f32(g, w.cond_4_w, h);
    ggml_set_name(h, "cond_embed");
    ggml_set_output(h);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_tensors, false);
    ggml_build_forward_expand(gf, h);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return {};
    }

    ggml_backend_tensor_set(emb_in, t_emb.data(), 0, t_emb.size() * sizeof(float));
    ggml_backend_graph_compute(ctx->backend, gf);

    std::vector<float> cond(D * 3);
    ggml_backend_tensor_get(h, cond.data(), 0, cond.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(g);
    return cond;
}

// Decode one window of DAC-VAE latent (patched_latent_dim, T_win) → PCM
// (T_win * hop samples). Empty on failure. Building one graph per window
// bounds peak codec-decode memory (see irodori_tts_synthesize's overlap-save
// loop). `latent` points at T_win contiguous frames in ggml (D,T) layout.
static std::vector<float> decode_dac_window(irodori_tts_context* ctx, const float* latent, int T_win) {
    const auto& dac = ctx->dac;
    const auto& cfg = dac.config;
    const int latent_d = ctx->hparams.patched_latent_dim();

    const int n_dec_tensors = 4096;
    size_t dec_ctx_size = n_dec_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_dec_tensors, false);
    ggml_init_params ip = {dec_ctx_size, nullptr, true};
    ggml_context* g = ggml_init(ip);

    ggml_tensor* lat_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, latent_d, T_win);
    ggml_set_name(lat_in, "latent_in");
    ggml_set_input(lat_in);

    const core_dac::fastconv_cache* fc = &ctx->dac_fc;
    ggml_tensor* h = lat_in;
    if (ctx->codec_out_proj_w) {
        h = core_dac::conv1d(g, h, ctx->codec_out_proj_w, ctx->codec_out_proj_b, 1, 1, fc);
        h = ggml_cast(g, h, GGML_TYPE_F32);
    }
    if (dac.in_conv_w) {
        h = core_dac::conv1d(g, h, dac.in_conv_w, dac.in_conv_b, 7, 1, fc);
        h = ggml_cast(g, h, GGML_TYPE_F32);
    }
    for (int b = 0; b < cfg.n_decoder_blocks; b++) {
        h = core_dac::dec_block(g, h, dac.blocks[b], cfg.upsampling_ratios[b], fc);
        h = ggml_cont(g, h);
        h = ggml_cast(g, h, GGML_TYPE_F32);
    }
    if (dac.out_snake_alpha)
        h = core_dac::snake(g, h, dac.out_snake_alpha);
    if (dac.out_conv_w) {
        h = core_dac::conv1d(g, h, dac.out_conv_w, dac.out_conv_b, 7, 1, fc);
        h = ggml_cast(g, h, GGML_TYPE_F32);
    }
    h = ggml_tanh(g, h);
    const int T_pcm = (int)h->ne[1];
    h = ggml_reshape_1d(g, h, T_pcm);
    ggml_set_output(h);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_dec_tensors, false);
    ggml_build_forward_expand(gf, h);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->codec_backend));
    std::vector<float> pcm;
    if (ggml_gallocr_reserve(galloc, gf) && ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_backend_tensor_set(lat_in, latent, 0, (size_t)latent_d * T_win * sizeof(float));
        ggml_backend_graph_compute(ctx->codec_backend, gf);
        pcm.resize((size_t)h->ne[0]);
        ggml_backend_tensor_get(h, pcm.data(), 0, pcm.size() * sizeof(float));
    } else {
        std::fprintf(stderr, "[irodori] ERROR: DAC-VAE decode graph alloc failed (T=%d)\n", T_win);
    }
    ggml_gallocr_free(galloc);
    ggml_free(g);
    return pcm;
}

// ── Main synthesize ─────────────────────────────────────────────────

int irodori_tts_synthesize(struct irodori_tts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out) {
    if (!ctx || !text || !pcm_out || !sample_rate_out)
        return 0;

    const auto& hp = ctx->hparams;
    *sample_rate_out = hp.sample_rate;

    // Tokenize
    std::vector<int32_t> token_ids = tokenize_text(ctx, text);
    int T_text = (int)token_ids.size();
    if (T_text == 0)
        return 0;

    IRODORI_DBG("[irodori] text tokens: %d →", T_text);
    for (int i = 0; i < std::min(T_text, 20); i++)
        IRODORI_DBG(" %d", token_ids[i]);
    IRODORI_DBG("\n");

    // Output length is determined after text + speaker encoding, from the
    // model's duration predictor (below) — the old ~6.3-frames/token heuristic
    // under-allocated kanji-heavy text (variable mora per token) and truncated
    // clauses (#221). CRISPASR_IRODORI_T_LATENT still forces an exact count.
    const char* t_lat_override = std::getenv("CRISPASR_IRODORI_T_LATENT");

    // ── Step 1: Encode text ──
    IRODORI_DBG("[irodori] encoding text...\n");
    auto text_state = run_text_encoder(ctx, token_ids.data(), T_text);
    if (text_state.empty()) {
        std::fprintf(stderr, "[irodori] text encoder failed\n");
        return 0;
    }
    IRODORI_DBG("[irodori] text_state: %d x %d (first 4: %.4f %.4f %.4f %.4f)\n", T_text, hp.text_dim, text_state[0],
                text_state[1], text_state[2], text_state[3]);

    // Dump text_state for parity comparison
    {
        const char* dump_ts = std::getenv("CRISPASR_IRODORI_DUMP_TEXT_STATE");
        if (dump_ts && *dump_ts && *dump_ts != '0') {
            std::string path = dump_ts;
            if (path == "1")
                path = "irodori_text_state.bin";
            FILE* f = std::fopen(path.c_str(), "wb");
            if (f) {
                std::fwrite(text_state.data(), sizeof(float), text_state.size(), f);
                std::fclose(f);
                std::fprintf(stderr, "[irodori] text_state dumped to '%s'\n", path.c_str());
            }
        }
    }

    // ── Step 2: Encode speaker reference ──
    // For unconditional (no_ref), pass 1-frame zeroed speaker with all-False mask.
    // The Python always runs speaker encoder on zeros and masks via attn_mask.
    // We pass zeros (speaker encoder output doesn't matter since mask is -inf)
    // but MUST include it in KV so attention shape matches Python.
    int T_ref = 1;
    std::vector<float> spk_state(T_ref * hp.speaker_dim, 0.0f);
    bool attend_speaker = false; // false → unconditional (speaker KV masked)

    // Voice cloning: when a reference latent is set, run the speaker encoder
    // and let the DiT attend to it (see run_dit_forward's attn mask).
    if (ctx->ref_latent_frames > 0 && !ctx->ref_latent.empty()) {
        std::vector<float> spk = run_speaker_encoder(ctx);
        if (!spk.empty()) {
            spk_state = std::move(spk);
            T_ref = ctx->ref_latent_frames;
            attend_speaker = true;
            IRODORI_DBG("[irodori] speaker conditioning active (%d ref frames)\n", T_ref);
        } else {
            std::fprintf(stderr, "[irodori] WARNING: speaker encoder failed; using neutral voice\n");
        }
    }

    // ── Step 2b: Encode caption (VoiceDesign) ──
    std::vector<float> cap_state;
    int T_cap = 0;
    bool attend_caption = false;
    if (hp.use_caption_condition && !ctx->caption_text.empty()) {
        IRODORI_DBG("[irodori] encoding caption: \"%s\"\n", ctx->caption_text.c_str());
        // Caption uses the same tokenizer as text (same vocab in VoiceDesign).
        // Use a separate token override env var so diff harness can control both.
        std::vector<int32_t> cap_ids;
        const char* cap_override = std::getenv("CRISPASR_IRODORI_CAPTION_TOKEN_IDS");
        if (cap_override && *cap_override) {
            std::string s(cap_override);
            size_t pos = 0;
            while (pos < s.size()) {
                size_t end = s.find(',', pos);
                if (end == std::string::npos)
                    end = s.size();
                cap_ids.push_back(std::atoi(s.substr(pos, end - pos).c_str()));
                pos = end + 1;
            }
        } else {
            cap_ids = tokenize_text(ctx, ctx->caption_text.c_str());
        }
        T_cap = (int)cap_ids.size();
        if (T_cap > 0) {
            cap_state = run_caption_encoder(ctx, cap_ids.data(), T_cap);
            if (!cap_state.empty()) {
                attend_caption = true;
                IRODORI_DBG("[irodori] caption conditioning active (%d tokens)\n", T_cap);
            }
        }
    }

    // ── Determine output length ──
    // Priority: explicit override → duration predictor → frames/token fallback.
    // The predictor conditions on the same speaker vector (first speaker-encoder
    // frame when cloning, else null_speaker) as the reference pipeline.
    // For VoiceDesign, caption conditioning uses masked_mean of caption_state.
    int latent_steps = 0;
    const char* dur_src = "duration-predictor";
    if (t_lat_override && std::atoi(t_lat_override) > 0) {
        latent_steps = std::atoi(t_lat_override);
        dur_src = "override";
    } else {
        const float* spk_vec = attend_speaker ? spk_state.data() : nullptr; // spk_state[:,0]
        // Caption vec for duration: masked_mean of caption_state (mean over all T_cap frames)
        std::vector<float> cap_vec_dur;
        const float* cap_vec_ptr = nullptr;
        if (attend_caption && !cap_state.empty()) {
            cap_vec_dur.resize(hp.caption_dim, 0.0f);
            for (int t = 0; t < T_cap; t++)
                for (int d = 0; d < hp.caption_dim; d++)
                    cap_vec_dur[d] += cap_state[t * hp.caption_dim + d];
            float inv_T = 1.0f / (float)T_cap;
            for (int d = 0; d < hp.caption_dim; d++)
                cap_vec_dur[d] *= inv_T;
            cap_vec_ptr = cap_vec_dur.data();
        }
        float frames = run_duration_predictor(ctx, text_state.data(), T_text, spk_vec, cap_vec_ptr);
        if (frames > 0.0f) {
            latent_steps = (int)std::lround(frames * ctx->duration_scale / ctx->speed);
        } else {
            latent_steps = (int)(T_text * 6.3f * ctx->duration_scale / ctx->speed);
            dur_src = "frames/token fallback";
        }
    }
    const int max_steps = (int)(ctx->max_seconds * hp.sample_rate / hp.codec_hop_length);
    latent_steps = std::max(1, std::min(latent_steps, max_steps));
    const int patched_steps = latent_steps / std::max(hp.latent_patch_size, 1);
    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] synthesize: %d tokens → %d latent frames (%.2f sec) [%s]\n", T_text,
                     latent_steps, (float)latent_steps * hp.codec_hop_length / hp.sample_rate, dur_src);
    }

    // ── Step 3: Euler RF ODE solver ──
    IRODORI_DBG("[irodori] ODE solver: %d steps\n", ctx->ode_steps);

    const int latent_d = hp.patched_latent_dim();
    const int n_ode = ctx->ode_steps;

    // Timestep schedule: t goes from 0.999 → 0 (noise → clean)
    std::vector<float> t_schedule(n_ode + 1);
    for (int i = 0; i <= n_ode; i++) {
        float u = (float)i / (float)n_ode;
        t_schedule[i] = (1.0f - u) * 0.999f; // decreasing: 0.999 → 0
    }

    // Initial noise x_t ~ N(0, 1) or loaded from reference file
    std::vector<float> x_t(patched_steps * latent_d);
    {
        const char* ref_noise = std::getenv("CRISPASR_IRODORI_REF_NOISE");
        if (ref_noise && *ref_noise) {
            FILE* f = std::fopen(ref_noise, "rb");
            if (f) {
                size_t n = std::fread(x_t.data(), sizeof(float), x_t.size(), f);
                std::fclose(f);
                std::fprintf(stderr, "[irodori] loaded reference noise from '%s' (%zu floats)\n", ref_noise, n);
            }
        } else {
            unsigned int seed_val = ctx->seed ? (unsigned int)ctx->seed : std::random_device{}();
            std::mt19937 rng(seed_val);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto& v : x_t)
                v = dist(rng);
        }
    }

    // Interval-CFG (opt-in, APPROXIMATE — mirrors OMNIVOICE_CFG_INTERVAL): irodori
    // runs up to THREE uncond DiT forwards per in-window step (text / speaker /
    // caption independent guidance). Recompute those uncond forwards only every K
    // CFG-active steps and reuse the cached uncond velocities in between; the cond
    // forward stays fresh every step; the first CFG-active step always recomputes.
    // This uses slightly stale uncond, so it CHANGES the output and stays gated OFF
    // by default (K=1 = exact). Only active when K>1, so the default recomputes every
    // step and is byte-for-byte the legacy path. Gated CRISPASR_IRODORI_CFG_INTERVAL.
    const int cfg_interval = [] {
        const char* e = std::getenv("CRISPASR_IRODORI_CFG_INTERVAL");
        const int k = e ? std::atoi(e) : 1;
        return k < 1 ? 1 : k;
    }();
    const bool cfg_interval_on = cfg_interval > 1;
    std::vector<float> v_text_cache, v_spk_cache, v_cap_cache; // cached uncond velocities
    int cfg_active_idx = 0;                                    // counts CFG-active steps (for the every-K test)
    if (cfg_interval_on && std::getenv("CRISPASR_IRODORI_CFG_INTERVAL_DEBUG"))
        std::fprintf(stderr,
                     "[irodori] interval-CFG K=%d (uncond recomputed every %d CFG-active steps; first always)\n",
                     cfg_interval, cfg_interval);

    // ODE integration loop
    for (int step = 0; step < n_ode; step++) {
        float t_val = t_schedule[step];
        float t_next = t_schedule[step + 1];
        float dt = t_next - t_val; // negative (moving toward t=0)

        // Compute timestep conditioning
        IRODORI_DBG("[irodori]   step %d: computing timestep cond (t=%.4f)...\n", step, t_val);
        auto cond_embed = run_timestep_cond(ctx, t_val);
        if (cond_embed.empty()) {
            std::fprintf(stderr, "[irodori] timestep cond failed at step %d\n", step);
            return 0;
        }

        IRODORI_DBG("[irodori]   step %d: running DiT forward...\n", step);
        // DiT forward: conditioned pass
        const float* cap_ptr = cap_state.empty() ? nullptr : cap_state.data();
        auto v_cond = run_dit_forward(ctx, x_t.data(), patched_steps, cond_embed.data(), text_state.data(), T_text,
                                      spk_state.empty() ? nullptr : spk_state.data(), T_ref, attend_speaker, cap_ptr,
                                      T_cap, attend_caption);
        if (v_cond.empty()) {
            std::fprintf(stderr, "[irodori] DiT forward failed at step %d\n", step);
            return 0;
        }

        // Dump step-0 v_pred for parity comparison
        if (step == 0) {
            const char* dump_v0 = std::getenv("CRISPASR_IRODORI_DUMP_V_PRED0");
            if (dump_v0 && *dump_v0) {
                FILE* f = std::fopen(dump_v0, "wb");
                if (f) {
                    std::fwrite(v_cond.data(), sizeof(float), v_cond.size(), f);
                    std::fclose(f);
                    std::fprintf(stderr, "[irodori] v_pred step0 dumped to '%s'\n", dump_v0);
                }
            }
        }

        // CFG: independent guidance (matches Irodori rf.py cfg_guidance_mode
        // "independent"):  v = v_cond
        //                    + cfg_text    * (v_cond - v_text_uncond)      [text zeroed]
        //                    + cfg_speaker * (v_cond - v_speaker_uncond)   [speaker masked]
        //                    + cfg_caption * (v_cond - v_caption_uncond)   [caption masked]
        float cfg_text = ctx->cfg_scale_text;
        const char* cfg_env = std::getenv("CRISPASR_IRODORI_CFG_TEXT");
        if (cfg_env)
            cfg_text = (float)std::atof(cfg_env);
        float cfg_speaker = ctx->cfg_scale_speaker;
        const char* cfgsp_env = std::getenv("CRISPASR_IRODORI_CFG_SPEAKER");
        if (cfgsp_env)
            cfg_speaker = (float)std::atof(cfgsp_env);
        float cfg_caption = ctx->cfg_scale_caption;
        const char* cfgcap_env = std::getenv("CRISPASR_IRODORI_CFG_CAPTION");
        if (cfgcap_env)
            cfg_caption = (float)std::atof(cfgcap_env);
        const float cfg_min_t = 0.5f, cfg_max_t = 1.0f;
        bool in_window = (t_val >= cfg_min_t) && (t_val <= cfg_max_t);
        bool do_text_cfg = (cfg_text > 0.0f) && in_window;
        bool do_spk_cfg = attend_speaker && (cfg_speaker > 0.0f) && in_window;
        bool do_cap_cfg = attend_caption && (cfg_caption > 0.0f) && in_window;

        if (do_text_cfg || do_spk_cfg || do_cap_cfg) {
            const float* spk_ptr = spk_state.empty() ? nullptr : spk_state.data();
            // Interval-CFG: recompute the uncond passes on the first CFG-active step
            // and every K-th CFG-active step; otherwise reuse the cached velocities.
            const bool recompute_unc = !cfg_interval_on || (cfg_active_idx % cfg_interval == 0) ||
                                       (do_text_cfg && v_text_cache.empty()) || (do_spk_cfg && v_spk_cache.empty()) ||
                                       (do_cap_cfg && v_cap_cache.empty());
            // Text-unconditional pass: zero text state, keep speaker+caption attended.
            std::vector<float> v_text_uncond;
            if (do_text_cfg) {
                if (recompute_unc) {
                    std::vector<float> text_uncond(T_text * hp.text_dim, 0.0f);
                    v_text_uncond =
                        run_dit_forward(ctx, x_t.data(), patched_steps, cond_embed.data(), text_uncond.data(), T_text,
                                        spk_ptr, T_ref, attend_speaker, cap_ptr, T_cap, attend_caption);
                    if (cfg_interval_on)
                        v_text_cache = v_text_uncond;
                } else {
                    v_text_uncond = v_text_cache;
                }
            }
            // Speaker-unconditional pass: conditioned text+caption, speaker masked out.
            std::vector<float> v_spk_uncond;
            if (do_spk_cfg) {
                if (recompute_unc) {
                    v_spk_uncond =
                        run_dit_forward(ctx, x_t.data(), patched_steps, cond_embed.data(), text_state.data(), T_text,
                                        spk_ptr, T_ref, /*attend_speaker=*/false, cap_ptr, T_cap, attend_caption);
                    if (cfg_interval_on)
                        v_spk_cache = v_spk_uncond;
                } else {
                    v_spk_uncond = v_spk_cache;
                }
            }
            // Caption-unconditional pass: conditioned text+speaker, caption masked out.
            std::vector<float> v_cap_uncond;
            if (do_cap_cfg) {
                if (recompute_unc) {
                    v_cap_uncond =
                        run_dit_forward(ctx, x_t.data(), patched_steps, cond_embed.data(), text_state.data(), T_text,
                                        spk_ptr, T_ref, attend_speaker, cap_ptr, T_cap, /*attend_caption=*/false);
                    if (cfg_interval_on)
                        v_cap_cache = v_cap_uncond;
                } else {
                    v_cap_uncond = v_cap_cache;
                }
            }
            cfg_active_idx++;
            for (size_t i = 0; i < x_t.size(); i++) {
                float v = v_cond[i];
                if (!v_text_uncond.empty())
                    v += cfg_text * (v_cond[i] - v_text_uncond[i]);
                if (!v_spk_uncond.empty())
                    v += cfg_speaker * (v_cond[i] - v_spk_uncond[i]);
                if (!v_cap_uncond.empty())
                    v += cfg_caption * (v_cond[i] - v_cap_uncond[i]);
                x_t[i] += v * dt;
            }
        } else {
            // No CFG for this timestep
            for (size_t i = 0; i < x_t.size(); i++) {
                x_t[i] += v_cond[i] * dt;
            }
        }

        if (ctx->verbosity >= 2 || (ctx->verbosity >= 1 && (step == 0 || step == n_ode - 1))) {
            std::fprintf(stderr, "  ODE step %d/%d  t=%.4f→%.4f\n", step + 1, n_ode, t_val, t_next);
        }
    }

    IRODORI_DBG("[irodori] ODE complete. latent shape: %d x %d\n", patched_steps, latent_d);

    // Dump latent for external DACVAE decode (env-gated)
    {
        const char* dump_env = std::getenv("CRISPASR_IRODORI_DUMP_LATENT");
        if (dump_env && *dump_env && *dump_env != '0') {
            std::string lat_path = dump_env;
            if (lat_path == "1")
                lat_path = "irodori_latent.bin";
            FILE* f = std::fopen(lat_path.c_str(), "wb");
            if (f) {
                std::fwrite(x_t.data(), sizeof(float), x_t.size(), f);
                std::fclose(f);
                std::fprintf(stderr, "[irodori] latent dumped to '%s' (%d frames × %d dims)\n", lat_path.c_str(),
                             patched_steps, latent_d);
            }
        }
    }

    // ── Step 4: DAC-VAE decode (latent → PCM) ──
    if (!ctx->has_codec) {
        // No codec loaded — output silence with warning
        int n_pcm = latent_steps * hp.codec_hop_length;
        float* out = (float*)std::malloc(n_pcm * sizeof(float));
        if (!out)
            return 0;
        std::memset(out, 0, n_pcm * sizeof(float));
        if (ctx->verbosity >= 1) {
            std::fprintf(stderr,
                         "[irodori] output: %d samples (%.2f sec @ %d Hz) "
                         "[DAC-VAE codec not loaded — outputting silence. Use --codec-model]\n",
                         n_pcm, (float)n_pcm / hp.sample_rate, hp.sample_rate);
        }
        *pcm_out = out;
        return n_pcm;
    }

    IRODORI_DBG("[irodori] decoding latent (%d frames) with DAC-VAE...\n", patched_steps);

    // DAC-VAE decode with overlap-save chunking. One graph over the whole
    // latent peaks codec memory at O(T·hop); for long outputs we decode
    // overlapping windows of `chunk` frames with `ctx_frames` of context on
    // each side and keep only the center, bounding peak memory to
    // O((chunk+2·ctx)·hop). The context covers the decoder's receptive field so
    // the seams are ~exact (validated: chunked vs whole cos≈1.0). Short outputs
    // (T ≤ chunk+2·ctx) decode in a single graph — unchanged from before.
    const int hop = hp.codec_hop_length;
    const int T = patched_steps;
    int chunk = 400;     // ~16 s at 1920 hop / 48 kHz
    int ctx_frames = 32; // ≫ the decoder's latent-frame receptive field
    if (const char* e = std::getenv("CRISPASR_IRODORI_DECODE_CHUNK"))
        chunk = std::atoi(e); // 0 → never chunk
    if (const char* e = std::getenv("CRISPASR_IRODORI_DECODE_CTX"))
        ctx_frames = std::max(0, std::atoi(e));
    const bool chunked = chunk > 0 && T > chunk + 2 * ctx_frames;

    std::vector<float> pcm = core_dac::decode_overlap_save(T, hop, chunk, ctx_frames, [&](int start, int n) {
        return decode_dac_window(ctx, x_t.data() + (size_t)start * latent_d, n);
    });

    if (pcm.empty()) {
        // Fallback: silence
        int n_pcm = latent_steps * hop;
        float* out = (float*)std::malloc((size_t)n_pcm * sizeof(float));
        if (out)
            std::memset(out, 0, (size_t)n_pcm * sizeof(float));
        *pcm_out = out;
        return out ? n_pcm : 0;
    }

    int n_pcm = (int)pcm.size();
    float* out = (float*)std::malloc((size_t)n_pcm * sizeof(float));
    if (!out)
        return 0;
    std::memcpy(out, pcm.data(), (size_t)n_pcm * sizeof(float));

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] output: %d samples (%.2f sec @ %d Hz)%s\n", n_pcm,
                     (float)n_pcm / hp.sample_rate, hp.sample_rate, chunked ? " [chunked decode]" : "");
    }
    *pcm_out = out;
    return n_pcm;
}

void irodori_tts_set_seed(struct irodori_tts_context* ctx, int seed) {
    if (ctx)
        ctx->seed = seed;
}

void irodori_tts_set_ode_steps(struct irodori_tts_context* ctx, int steps) {
    if (ctx && steps > 0)
        ctx->ode_steps = steps;
}

void irodori_tts_set_cfg_scale_text(struct irodori_tts_context* ctx, float scale) {
    if (ctx)
        ctx->cfg_scale_text = scale;
}

void irodori_tts_set_cfg_scale_speaker(struct irodori_tts_context* ctx, float scale) {
    if (ctx)
        ctx->cfg_scale_speaker = scale;
}

void irodori_tts_set_cfg_scale_caption(struct irodori_tts_context* ctx, float scale) {
    if (ctx)
        ctx->cfg_scale_caption = scale;
}

void irodori_tts_set_caption(struct irodori_tts_context* ctx, const char* caption) {
    if (!ctx)
        return;
    ctx->caption_text = (caption && *caption) ? caption : "";
}

void irodori_tts_set_speed(struct irodori_tts_context* ctx, float speed) {
    if (ctx && speed > 0)
        ctx->speed = speed;
}

int irodori_tts_sample_rate(const struct irodori_tts_context* ctx) {
    return ctx ? ctx->hparams.sample_rate : 48000;
}

int irodori_tts_vocab_size(const struct irodori_tts_context* ctx) {
    return ctx ? ctx->hparams.text_vocab_size : 0;
}

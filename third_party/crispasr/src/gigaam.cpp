// gigaam.cpp — ai-sage/GigaAM-v3 ggml runtime (Russian ASR)
//
// Rotary Conformer encoder + CTC or RNN-Transducer head. The blueprint is
// `modeling_gigaam.py` in the HF snapshot (trust_remote_code), which is the
// code that actually runs the model — every decision below is traced to it.
//
// Shape of the pipeline:
//
//   log-mel      torchaudio MelSpectrogram(n_mels=64, n_fft=win=320, hop=160,
//                center=False, power=2, htk, norm=None) → log(clamp(x, 1e-9)).
//                NO per-feature z-norm and NO pre-emphasis (unlike the NeMo
//                family). The Hann window and the mel filterbank are copied
//                out of the checkpoint by the converter rather than rebuilt.
//
//   pre-encode   StridingSubsampling(subsampling='conv1d'): two
//                Conv1d(k=5, stride=2, padding=2) + ReLU stages
//                (64→768, 768→768), 4× time downsample. Not NeMo's
//                dw_striding Conv2d stack, so core_conformer::build_pre_encode
//                does not apply.
//
//   16 blocks    FFN1(½) → rotary MHA → conv(dw k=5 + LayerNorm) → FFN2(½)
//                → LayerNorm. The FFN / conv / macaron halves are the same
//                shape as every other Conformer we ship, so the weights live
//                in core_conformer::BlockWeights; only the attention differs
//                and that is why this file has its own block builder rather
//                than calling core_conformer::build_block (which is rel-pos).
//
//   head         CTC: Conv1d(768→C, k=1) + log_softmax + greedy collapse.
//                RNN-T: Embedding(C, 320) + 1-layer LSTM(320) predictor,
//                joint enc(768→320)+pred(320→320) → ReLU → Linear(320→C).
//
// ── The two attention quirks, spelled out ─────────────────────────────────
//
// RotaryPositionMultiHeadAttention.forward calls apply_rotary_pos_emb on
// `query`/`key` BEFORE forward_qkv runs the linear projections, and the
// caller passes `x, x, x`. So the rotation lands on the layer INPUT viewed
// as (T, B, n_heads, head_dim), and the projections consume the ROTATED
// tensor:
//
//     Q = Wq · RoPE(x)      K = Wk · RoPE(x)      V = Wv · x
//
// V is projected from the UNROTATED input. Rotating V, or rotating Q/K after
// projection (the usual arrangement), both give a different model.
//
// And RotaryPositionalEmbedding is constructed as
// `RotaryPositionalEmbedding(d_model // n_heads, pos_emb_max_len)` against
// `PositionalEncoding.__init__(self, dim, base)` — so the rotary base is
// pos_emb_max_len = 5000, NOT the customary 10000. The converter records it
// as `gigaam.rope_base` so the value is not re-derived here.
//
// rtt_half is the NEOX / rotate_half pairing (i, i + n_dims/2) over
// head_dim = 48, which is exactly ggml's GGML_ROPE_TYPE_NEOX.
//
// Batch is always 1 here, and the blueprint builds `att_mask` only when
// `audio_signal.shape[0] > 1` — so full unmasked attention is correct, and
// the conv module's pad_mask is a no-op (masked_fill with an all-false mask).

#include "gigaam.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#include "gguf.h"

#include "core/crispasr_env.h"
#include "core/fastconformer.h" // BlockWeights / BlockParams containers
#include "core/fft.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/mel.h"

#if defined(HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ===========================================================================
// Env gates
// ===========================================================================

static bool gigaam_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_GIGAAM_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

static bool gigaam_debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_GIGAAM_DEBUG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// Flash attention in the encoder. Manual QK^T + soft_max_ext + V is the
// default because it is the path the per-stage diff was validated on; flash
// is bit-different (different accumulation order) and needs its own A/B
// before it can become the default. `use_flash` in the context params turns
// it on too.
static bool gigaam_flash_gate() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_GIGAAM_FLASH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// Scalar fallback for the LSTM / joint loops (A/B against the cblas path).
static bool gigaam_force_scalar() {
    static int v = -1;
    if (v < 0)
        v = (crispasr_env::get("CRISPASR_GIGAAM_FORCE_SCALAR") != nullptr) ? 1 : 0;
    return v != 0;
}

struct gigaam_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit gigaam_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~gigaam_bench_stage() {
        if (!gigaam_bench_enabled())
            return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "  gigaam_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct gigaam_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 64;
    uint32_t n_fft = 320;
    uint32_t win_length = 320;
    uint32_t hop_length = 160;
    uint32_t d_model = 768;
    uint32_t n_layers = 16;
    uint32_t n_heads = 16;
    uint32_t head_dim = 48;
    uint32_t ff_dim = 3072;
    uint32_t subsampling_factor = 4;
    uint32_t subs_kernel = 5;
    uint32_t conv_kernel = 5;
    float rope_base = 5000.0f;
    uint32_t num_classes = 1025; // includes blank
    uint32_t blank_id = 1024;
    uint32_t vocab_size = 1024;
    uint32_t frame_dur_cs = 4;
    bool is_rnnt = true;
    bool is_spm = true;
    uint32_t pred_hidden = 320;
    uint32_t joint_hidden = 320;
};

// ===========================================================================
// Weights
// ===========================================================================

struct gigaam_pre_encode {
    ggml_tensor *conv0_w = nullptr, *conv0_b = nullptr; // (K, 64, 768)
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr; // (K, 768, 768)
};

struct gigaam_enc_layer : core_conformer::BlockWeights {};

struct gigaam_head {
    // CTC
    ggml_tensor *ctc_w = nullptr, *ctc_b = nullptr;
    // RNN-T predictor
    ggml_tensor* embed_w = nullptr;
    ggml_tensor *lstm_w_ih = nullptr, *lstm_w_hh = nullptr;
    ggml_tensor *lstm_b_ih = nullptr, *lstm_b_hh = nullptr;
    // RNN-T joint
    ggml_tensor *joint_enc_w = nullptr, *joint_enc_b = nullptr;
    ggml_tensor *joint_pred_w = nullptr, *joint_pred_b = nullptr;
    ggml_tensor *joint_out_w = nullptr, *joint_out_b = nullptr;
};

// CPU F32 mirrors of the decode-path weights (the transducer steps are tiny
// per-token GEMVs — see the "per-step GPU dispatch is launch-bound" note in
// the dev guide — so they run on cblas, not the sched).
struct gigaam_decode_weights {
    std::vector<float> embed;                  // (C, H)
    std::vector<float> w_ih, w_hh, b_ih, b_hh; // LSTM, 4H gates
    std::vector<float> enc_w, enc_b;           // joint.enc (Jh, d_model)
    std::vector<float> pred_w, pred_b;         // joint.pred (Jh, H)
    std::vector<float> out_w, out_b;           // joint.out (C, Jh)
    std::vector<float> ctc_w, ctc_b;           // CTC head (C, d_model)
    int H = 0, Jh = 0, C = 0, D = 0;
    bool initialised = false;
};

struct gigaam_model {
    gigaam_hparams hparams;

    ggml_tensor* mel_fb = nullptr;
    ggml_tensor* mel_window = nullptr;

    gigaam_pre_encode pre_encode;
    std::vector<gigaam_enc_layer> enc;
    gigaam_head head;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    core_conformer::PwRepackBuf pw_q8;

    std::map<std::string, ggml_tensor*> tensors;
};

struct gigaam_vocab {
    std::vector<std::string> id_to_token;
};

struct gigaam_context {
    gigaam_context_params params;

    gigaam_model model;
    gigaam_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;
    gigaam_decode_weights dec;

    int n_threads = 4;
    int max_symbols = 10; // RNNTGreedyDecoding(max_symbols_per_step=10)
};

// ===========================================================================
// Helpers
// ===========================================================================

static ggml_tensor* require(gigaam_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "gigaam");
}

static std::vector<float> tensor_to_f32(ggml_tensor* t) {
    if (!t)
        return {};
    const int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp((size_t)n);
        ggml_backend_tensor_get(t, tmp.data(), 0, (size_t)n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++)
            out[(size_t)i] = ggml_fp16_to_fp32(tmp[(size_t)i]);
    } else if (ggml_is_quantized(t->type)) {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), out.data(), n);
        } else {
            fprintf(stderr, "gigaam: no dequant for type %d\n", (int)t->type);
            out.assign((size_t)n, 0.0f);
        }
    } else {
        fprintf(stderr, "gigaam: unsupported tensor type %d\n", (int)t->type);
        out.assign((size_t)n, 0.0f);
    }
    return out;
}

// Number of frames the conv1d subsampling produces, mirroring
// StridingSubsampling.calc_output_length: floor((L + 2p - k)/s) + 1 per stage.
static int gigaam_subsampled_len(const gigaam_hparams& hp, int T_mel) {
    const int k = (int)hp.subs_kernel;
    const int p = (k - 1) / 2;
    int len = T_mel;
    for (int i = 0; i < 2; i++) { // subsampling_factor 4 = two stride-2 stages
        len = (len + 2 * p - k) / 2 + 1;
        if (len < 0)
            len = 0;
    }
    return len;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool gigaam_load_model(gigaam_model& model, gigaam_vocab& vocab, const char* path, ggml_backend_t backend) {
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "gigaam.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "gigaam.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "gigaam.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "gigaam.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "gigaam.hop_length", hp.hop_length);
        hp.d_model = core_gguf::kv_u32(gctx, "gigaam.d_model", hp.d_model);
        hp.n_layers = core_gguf::kv_u32(gctx, "gigaam.n_layers", hp.n_layers);
        hp.n_heads = core_gguf::kv_u32(gctx, "gigaam.n_heads", hp.n_heads);
        hp.head_dim = core_gguf::kv_u32(gctx, "gigaam.head_dim", hp.head_dim);
        hp.ff_dim = core_gguf::kv_u32(gctx, "gigaam.ff_dim", hp.ff_dim);
        hp.subsampling_factor = core_gguf::kv_u32(gctx, "gigaam.subsampling_factor", hp.subsampling_factor);
        hp.subs_kernel = core_gguf::kv_u32(gctx, "gigaam.subs_kernel", hp.subs_kernel);
        hp.conv_kernel = core_gguf::kv_u32(gctx, "gigaam.conv_kernel", hp.conv_kernel);
        hp.rope_base = core_gguf::kv_f32(gctx, "gigaam.rope_base", hp.rope_base);
        hp.num_classes = core_gguf::kv_u32(gctx, "gigaam.num_classes", hp.num_classes);
        hp.blank_id = core_gguf::kv_u32(gctx, "gigaam.blank_id", hp.num_classes - 1);
        hp.vocab_size = core_gguf::kv_u32(gctx, "gigaam.vocab_size", hp.vocab_size);
        hp.frame_dur_cs = core_gguf::kv_u32(gctx, "gigaam.frame_dur_cs", hp.frame_dur_cs);
        hp.pred_hidden = core_gguf::kv_u32(gctx, "gigaam.pred_hidden", hp.pred_hidden);
        hp.joint_hidden = core_gguf::kv_u32(gctx, "gigaam.joint_hidden", hp.joint_hidden);
        hp.is_rnnt = core_gguf::kv_str(gctx, "gigaam.head_type", "rnnt") == "rnnt";
        hp.is_spm = core_gguf::kv_str(gctx, "gigaam.tokenizer_type", "spm") == "spm";

        vocab.id_to_token = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");

        core_gguf::free_metadata(gctx);
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "gigaam", wl))
        return false;
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    model.mel_fb = require(model, "preprocessor.fb");
    model.mel_window = require(model, "preprocessor.window");

    model.pre_encode.conv0_w = require(model, "encoder.pre.conv.0.weight");
    model.pre_encode.conv0_b = require(model, "encoder.pre.conv.0.bias");
    model.pre_encode.conv1_w = require(model, "encoder.pre.conv.2.weight");
    model.pre_encode.conv1_b = require(model, "encoder.pre.conv.2.bias");

    model.enc.resize(model.hparams.n_layers);
    for (uint32_t i = 0; i < model.hparams.n_layers; i++) {
        char buf[128];
        auto& e = model.enc[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };

        e.norm_ff1_w = get("norm_ff1.weight");
        e.norm_ff1_b = get("norm_ff1.bias");
        e.ff1_l1_w = get("ff1.linear1.weight");
        e.ff1_l1_b = get("ff1.linear1.bias");
        e.ff1_l2_w = get("ff1.linear2.weight");
        e.ff1_l2_b = get("ff1.linear2.bias");

        e.norm_attn_w = get("norm_attn.weight");
        e.norm_attn_b = get("norm_attn.bias");
        e.attn_q_w = get("attn.q.weight");
        e.attn_q_b = get("attn.q.bias");
        e.attn_k_w = get("attn.k.weight");
        e.attn_k_b = get("attn.k.bias");
        e.attn_v_w = get("attn.v.weight");
        e.attn_v_b = get("attn.v.bias");
        e.attn_out_w = get("attn.out.weight");
        e.attn_out_b = get("attn.out.bias");

        e.norm_conv_w = get("norm_conv.weight");
        e.norm_conv_b = get("norm_conv.bias");
        e.conv_pw1_w = get("conv.pw1.weight");
        e.conv_pw1_b = get("conv.pw1.bias");
        e.conv_dw_w = get("conv.dw.weight");
        e.conv_dw_b = get("conv.dw.bias");
        e.conv_ln_w = get("conv.ln.weight");
        e.conv_ln_b = get("conv.ln.bias");
        e.conv_pw2_w = get("conv.pw2.weight");
        e.conv_pw2_b = get("conv.pw2.bias");

        e.norm_ff2_w = get("norm_ff2.weight");
        e.norm_ff2_b = get("norm_ff2.bias");
        e.ff2_l1_w = get("ff2.linear1.weight");
        e.ff2_l1_b = get("ff2.linear1.bias");
        e.ff2_l2_w = get("ff2.linear2.weight");
        e.ff2_l2_b = get("ff2.linear2.bias");

        e.norm_out_w = get("norm_out.weight");
        e.norm_out_b = get("norm_out.bias");
    }

    auto& h = model.head;
    if (model.hparams.is_rnnt) {
        h.embed_w = require(model, "decoder.embed.weight");
        h.lstm_w_ih = require(model, "decoder.lstm.0.w_ih");
        h.lstm_w_hh = require(model, "decoder.lstm.0.w_hh");
        h.lstm_b_ih = require(model, "decoder.lstm.0.b_ih");
        h.lstm_b_hh = require(model, "decoder.lstm.0.b_hh");
        h.joint_enc_w = require(model, "joint.enc.weight");
        h.joint_enc_b = require(model, "joint.enc.bias");
        h.joint_pred_w = require(model, "joint.pred.weight");
        h.joint_pred_b = require(model, "joint.pred.bias");
        h.joint_out_w = require(model, "joint.out.weight");
        h.joint_out_b = require(model, "joint.out.bias");
    } else {
        h.ctc_w = require(model, "head.ctc.weight");
        h.ctc_b = require(model, "head.ctc.bias");
    }

    fprintf(stderr, "gigaam: d_model=%u layers=%u heads=%u ff=%u head=%s tokenizer=%s classes=%u vocab=%zu\n",
            model.hparams.d_model, model.hparams.n_layers, model.hparams.n_heads, model.hparams.ff_dim,
            model.hparams.is_rnnt ? "rnnt" : "ctc", model.hparams.is_spm ? "spm" : "charwise",
            model.hparams.num_classes, vocab.id_to_token.size());
    return true;
}

// ===========================================================================
// Log-mel — torchaudio MelSpectrogram + log(clamp(x, 1e-9, 1e9))
// ===========================================================================

// n_fft is 320 = 2^6 * 5, so the radix-2 transform does not apply; the shared
// mixed-radix helper recurses down to a 5-point DFT.
static void gigaam_fft_r2c(const float* in, int N, float* out) {
    core_fft::fft_nonpow2_r2c(in, N, out);
}

static std::vector<float> gigaam_compute_mel_impl(gigaam_context* ctx, const float* samples, int n_samples,
                                                  int& T_out) {
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    std::vector<float> window((size_t)win);
    ggml_backend_tensor_get(ctx->model.mel_window, window.data(), 0, (size_t)win * sizeof(float));

    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(ctx->model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = (int)hp.hop_length;
    p.win_length = win;
    p.n_mels = n_mels;
    p.spec_kind = core_mel::SpecKind::Power; // torchaudio power=2.0
    p.log_base = core_mel::LogBase::Ln;      // torch.log
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.log_eps = 1e-9f;                      // x.clamp_(1e-9, 1e9)
    p.norm = core_mel::Normalization::None; // no z-norm, no clip-and-scale
    p.layout = core_mel::Layout::MelsTime;  // (n_mels, T) — conv1d channels
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.center_pad = false; // MelSpectrogram(center=False)
    p.preemph = 0.0f;
    p.drop_last_frame = false;
    p.n_threads = ctx->n_threads;

    return core_mel::compute(samples, n_samples, window.data(), win, mel_fb.data(), n_freqs, gigaam_fft_r2c, p, T_out);
}

// ===========================================================================
// Encoder graph
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

// StridingSubsampling(subsampling='conv1d'): Conv1d(k, s=2, p=(k-1)/2) + ReLU,
// twice. `mel` is (T_mel, n_mels) in ggml terms (ne[0]=T fast) — which is the
// row-major (n_mels, T_mel) buffer core_mel produces. Returns (d_model, T_enc).
static ggml_tensor* gigaam_build_pre_encode(ggml_context* ctx0, ggml_tensor* mel, const gigaam_pre_encode& w, int K,
                                            int* out_T_enc) {
    const int p = (K - 1) / 2;
    auto bias_row = [&](ggml_tensor* b) { return ggml_reshape_2d(ctx0, b, 1, b->ne[0]); };

    ggml_tensor* cur = ggml_conv_1d(ctx0, w.conv0_w, mel, 2, p, 1); // (T1, 768)
    cur = ggml_add(ctx0, cur, bias_row(w.conv0_b));
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_1d(ctx0, w.conv1_w, cur, 2, p, 1); // (T2, 768)
    cur = ggml_add(ctx0, cur, bias_row(w.conv1_b));
    cur = ggml_relu(ctx0, cur);

    if (out_T_enc)
        *out_T_enc = (int)cur->ne[0];

    // (T, d) → (d, T) for the conformer body.
    return ggml_cont(ctx0, ggml_transpose(ctx0, cur));
}

// One rotary Conformer block. `cur` is (d, T); `pos` is an I32 (T,) position
// vector for the RoPE. Returns (d, T).
static ggml_tensor* gigaam_build_block(ggml_context* ctx0, ggml_tensor* cur, ggml_tensor* pos, int T,
                                       const core_conformer::BlockWeights& e, const core_conformer::BlockParams& bp,
                                       float rope_base, bool use_flash) {
    const int d = bp.d;
    const int nh = bp.n_heads;
    const int hd = bp.head_dim;
    const int K = bp.K;
    const float eps = bp.ln_eps;

    auto mm_bias = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx0, w, x);
        return b ? ggml_add(ctx0, y, b) : y;
    };

    // ---- FFN1 (macaron half) ----
    ggml_tensor* inpL = cur;
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, e.norm_ff1_w, e.norm_ff1_b, eps);
    x = mm_bias(e.ff1_l1_w, x, e.ff1_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff1_l2_w, x, e.ff1_l2_b);
    cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, x, 0.5f));

    // ---- Rotary self-attention ----
    ggml_tensor* inpAttn = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_attn_w, e.norm_attn_b, eps);

    // RoPE on the layer input viewed as heads, BEFORE the projections (see
    // the file header). (d, T) → (hd, nh, T) is exactly the (T, B, h, d_k)
    // memory order the blueprint rotates.
    ggml_tensor* xh = ggml_reshape_3d(ctx0, x, hd, nh, T);
    ggml_tensor* xr = ggml_rope_ext(ctx0, xh, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0, rope_base,
                                    /*freq_scale=*/1.0f, /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                                    /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
    ggml_tensor* xrot = ggml_reshape_2d(ctx0, xr, d, T);

    ggml_tensor* Q = mm_bias(e.attn_q_w, xrot, e.attn_q_b);
    ggml_tensor* Kp = mm_bias(e.attn_k_w, xrot, e.attn_k_b);
    ggml_tensor* Vp = mm_bias(e.attn_v_w, x, e.attn_v_b); // NOT rotated

    ggml_tensor* Q3 = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, T), 0, 2, 1, 3);  // (hd, T, nh)
    ggml_tensor* K3 = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kp, hd, nh, T), 0, 2, 1, 3); // (hd, T, nh)
    ggml_tensor* V3 = ggml_reshape_3d(ctx0, Vp, hd, nh, T);                                 // (hd, nh, T)

    const float scale = 1.0f / sqrtf((float)hd);
    ggml_tensor* attn_out;
    if (use_flash) {
        ggml_tensor* V_f = ggml_permute(ctx0, V3, 0, 2, 1, 3); // (hd, T, nh)
        attn_out = ggml_flash_attn_ext(ctx0, Q3, K3, V_f, nullptr, scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(ctx0, attn_out, d, T);
    } else {
        ggml_tensor* scores = ggml_mul_mat(ctx0, ggml_cont(ctx0, K3), ggml_cont(ctx0, Q3)); // (T, T, nh)
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f);
        ggml_tensor* V_hd = ggml_cont(ctx0, ggml_permute(ctx0, V3, 0, 2, 1, 3));  // (hd, T, nh)
        ggml_tensor* V_t = ggml_cont(ctx0, ggml_permute(ctx0, V_hd, 1, 0, 2, 3)); // (T, hd, nh)
        attn_out = ggml_mul_mat(ctx0, V_t, scores);                               // (hd, T, nh)
        attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T);
    }
    attn_out = mm_bias(e.attn_out_w, attn_out, e.attn_out_b);
    cur = ggml_add(ctx0, inpAttn, attn_out);

    // ---- Convolution module ----
    ggml_tensor* inpConv = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_conv_w, e.norm_conv_b, eps);

    ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, e.conv_pw1_w, d, 2 * d);
    ggml_tensor* cnv = mm_bias(pw1_w, x, e.conv_pw1_b);
    cnv = ggml_siglu_swapped(ctx0, cnv); // F.glu(dim=1): first_half * sigmoid(second_half)

    // Depthwise conv, kernel K, symmetric padding (K-1)/2 — the blueprint's
    // nn.Conv1d(padding=(kernel_size-1)//2), not a causal left-pad.
    ggml_tensor* dw_w_f32 = e.conv_dw_w_f32 ? e.conv_dw_w_f32 : ggml_cast(ctx0, e.conv_dw_w, GGML_TYPE_F32);
    ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, K, 1, 1, d);
    cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv)); // (d, T) → (T, d)
    cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1);
    cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (K - 1) / 2, 0, 1, 1);
    cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
    cnv = ggml_reshape_2d(ctx0, cnv, d, T);
    cnv = ggml_add(ctx0, cnv, ggml_reshape_2d(ctx0, e.conv_dw_b, d, 1));

    cnv = ggml_norm_affine(ctx0, cnv, e.conv_ln_w, e.conv_ln_b, eps); // conv_norm_type=layer_norm
    cnv = ggml_silu(ctx0, cnv);

    ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, e.conv_pw2_w, d, d);
    cnv = mm_bias(pw2_w, cnv, e.conv_pw2_b);
    cur = ggml_add(ctx0, inpConv, cnv);

    // ---- FFN2 (macaron half) ----
    ggml_tensor* inpFF2 = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_ff2_w, e.norm_ff2_b, eps);
    x = mm_bias(e.ff2_l1_w, x, e.ff2_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff2_l2_w, x, e.ff2_l2_b);
    cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, x, 0.5f));

    // ---- Block final LN ----
    return ggml_norm_affine(ctx0, cur, e.norm_out_w, e.norm_out_b, eps);
}

// `dump` non-null → a named F32 snapshot per stage is added to the graph
// ("pre_enc", "layer_0" .. "layer_N-1") for the diff harness.
// `out_ctx0` receives the graph's ggml_context; the caller frees it once the
// outputs have been read back (the nodes themselves live in ctx->compute_meta,
// which the context does not own).
static ggml_cgraph* gigaam_build_graph_encoder(gigaam_context* ctx, int T_mel, bool dump, ggml_context** out_ctx0) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // (T_mel, n_mels): ne[0]=T is the fast axis, matching core_mel's
    // MelsTime row-major (n_mels, T_mel) buffer and conv1d's (T, C_in).
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, (int)hp.n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    int T = 0;
    ggml_tensor* cur = gigaam_build_pre_encode(ctx0, mel, m.pre_encode, (int)hp.subs_kernel, &T);
    if (dump) {
        ggml_tensor* snap = ggml_dup(ctx0, cur);
        ggml_set_name(snap, "pre_enc");
        ggml_set_output(snap);
        ggml_build_forward_expand(gf, snap);
    }

    ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    core_conformer::BlockParams bp = {};
    bp.d = (int)hp.d_model;
    bp.n_heads = (int)hp.n_heads;
    bp.head_dim = (int)hp.head_dim;
    bp.K = (int)hp.conv_kernel;
    bp.ln_eps = kLayerNormEps;

    const bool use_flash = ctx->params.use_flash || gigaam_flash_gate();
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = gigaam_build_block(ctx0, cur, pos, T, m.enc[il], bp, hp.rope_base, use_flash);
        if (dump) {
            char nm[32];
            snprintf(nm, sizeof(nm), "layer_%u", il);
            ggml_tensor* snap = ggml_dup(ctx0, cur);
            ggml_set_name(snap, nm);
            ggml_set_output(snap);
            ggml_build_forward_expand(gf, snap);
        }
    }

    ggml_set_name(cur, "enc_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    if (out_ctx0)
        *out_ctx0 = ctx0;
    return gf;
}

static bool gigaam_ensure_sched(gigaam_context* ctx) {
    if (ctx->sched)
        return true;
    ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
    const int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
    return ctx->sched != nullptr;
}

// Run the encoder. `dump_out`/`dump_count` optionally receive the per-stage
// snapshots (slot 0 = pre-encode, 1..n_layers = conformer blocks).
static bool gigaam_run_encoder_impl(gigaam_context* ctx, const float* mel, int n_mels, int T_mel,
                                    std::vector<float>& enc_out, int& T_enc, int& d_model_out, float** dump_out,
                                    int dump_count) {
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "gigaam: mel has %d bands, model expects %u\n", n_mels, ctx->model.hparams.n_mels);
        return false;
    }
    const bool dump = dump_out != nullptr && dump_count > 0;

    ggml_context* ctx0 = nullptr;
    ggml_cgraph* gf = gigaam_build_graph_encoder(ctx, T_mel, dump, &ctx0);
    // Freeing ctx0 does NOT free the graph nodes (they live in
    // ctx->compute_meta), so every early return still has to release it.
    struct Ctx0Guard {
        ggml_context* c;
        ~Ctx0Guard() {
            if (c)
                ggml_free(c);
        }
    } ctx0_guard{ctx0};
    if (!gf)
        return false;
    if (!gigaam_ensure_sched(ctx))
        return false;

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "gigaam: sched alloc encoder graph failed\n");
        return false;
    }

    ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_t, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor* enc_t = ggml_graph_get_tensor(gf, "enc_out");
    T_enc = (int)enc_t->ne[1];
    d_model_out = (int)enc_t->ne[0];

    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "pos");
    std::vector<int32_t> pos((size_t)T_enc);
    for (int i = 0; i < T_enc; i++)
        pos[(size_t)i] = i;
    ggml_backend_tensor_set(pos_t, pos.data(), 0, pos.size() * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "gigaam: encoder graph compute failed\n");
        return false;
    }

    enc_out.resize((size_t)T_enc * d_model_out);
    ggml_backend_tensor_get(enc_t, enc_out.data(), 0, enc_out.size() * sizeof(float));

    if (dump) {
        const size_t nb = (size_t)T_enc * d_model_out * sizeof(float);
        if (dump_out[0]) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, "pre_enc");
            if (t)
                ggml_backend_tensor_get(t, dump_out[0], 0, nb);
        }
        for (uint32_t il = 0; il + 1 < (uint32_t)dump_count && il < ctx->model.hparams.n_layers; il++) {
            if (!dump_out[il + 1])
                continue;
            char nm[32];
            snprintf(nm, sizeof(nm), "layer_%u", il);
            ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
            if (t)
                ggml_backend_tensor_get(t, dump_out[il + 1], 0, nb);
        }
    }

    if (gigaam_debug_enabled()) {
        double mn = 1e30, mx = -1e30, sum = 0.0;
        for (float v : enc_out) {
            mn = std::min(mn, (double)v);
            mx = std::max(mx, (double)v);
            sum += v;
        }
        fprintf(stderr, "gigaam: enc_out T=%d d=%d min=%.4f max=%.4f mean=%.6f\n", T_enc, d_model_out, mn, mx,
                sum / (double)enc_out.size());
    }
    return true;
}

// ===========================================================================
// Decode-path CPU weights
// ===========================================================================

static void gigaam_init_decode_weights(gigaam_context* ctx) {
    auto& W = ctx->dec;
    if (W.initialised)
        return;
    const auto& hp = ctx->model.hparams;
    const auto& h = ctx->model.head;

    W.D = (int)hp.d_model;
    W.C = (int)hp.num_classes;
    if (hp.is_rnnt) {
        W.H = (int)hp.pred_hidden;
        W.Jh = (int)hp.joint_hidden;
        W.embed = tensor_to_f32(h.embed_w);
        W.w_ih = tensor_to_f32(h.lstm_w_ih);
        W.w_hh = tensor_to_f32(h.lstm_w_hh);
        W.b_ih = tensor_to_f32(h.lstm_b_ih);
        W.b_hh = tensor_to_f32(h.lstm_b_hh);
        W.enc_w = tensor_to_f32(h.joint_enc_w);
        W.enc_b = tensor_to_f32(h.joint_enc_b);
        W.pred_w = tensor_to_f32(h.joint_pred_w);
        W.pred_b = tensor_to_f32(h.joint_pred_b);
        W.out_w = tensor_to_f32(h.joint_out_w);
        W.out_b = tensor_to_f32(h.joint_out_b);
        W.C = (int)h.joint_out_b->ne[0];
    } else {
        W.ctc_w = tensor_to_f32(h.ctc_w);
        W.ctc_b = tensor_to_f32(h.ctc_b);
        W.C = (int)h.ctc_b->ne[0];
    }
    W.initialised = true;
}

// y = b + M·x with M row-major (rows, cols).
static void gemv_bias(const float* M, const float* b, const float* x, int rows, int cols, float* y) {
#if defined(HAVE_ACCELERATE)
    if (!gigaam_force_scalar()) {
        std::memcpy(y, b, (size_t)rows * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, M, cols, x, 1, 1.0f, y, 1);
        return;
    }
#endif
    for (int i = 0; i < rows; i++) {
        float s = b[i];
        const float* row = M + (size_t)i * cols;
        for (int k = 0; k < cols; k++)
            s += row[k] * x[k];
        y[i] = s;
    }
}

// ===========================================================================
// RNN-T predictor / joint (CPU)
// ===========================================================================

struct gigaam_lstm_state {
    std::vector<float> h, c;
    void init(int H) {
        h.assign((size_t)H, 0.0f);
        c.assign((size_t)H, 0.0f);
    }
};

// One PyTorch LSTM cell step, gate order [i, f, g, o].
static void gigaam_lstm_step(const gigaam_decode_weights& W, const float* x_in, gigaam_lstm_state& st,
                             std::vector<float>& h_out) {
    const int H = W.H;
    const int H4 = 4 * H;
    std::vector<float> gates((size_t)H4);
    for (int j = 0; j < H4; j++)
        gates[(size_t)j] = W.b_ih[(size_t)j] + W.b_hh[(size_t)j];
#if defined(HAVE_ACCELERATE)
    if (!gigaam_force_scalar()) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, H4, H, 1.0f, W.w_ih.data(), H, x_in, 1, 1.0f, gates.data(), 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, H4, H, 1.0f, W.w_hh.data(), H, st.h.data(), 1, 1.0f, gates.data(), 1);
    } else
#endif
    {
        for (int j = 0; j < H4; j++) {
            float s = gates[(size_t)j];
            const float* ri = W.w_ih.data() + (size_t)j * H;
            const float* rh = W.w_hh.data() + (size_t)j * H;
            for (int k = 0; k < H; k++)
                s += ri[k] * x_in[k] + rh[k] * st.h[(size_t)k];
            gates[(size_t)j] = s;
        }
    }
    auto sig = [](float v) { return 1.0f / (1.0f + expf(-v)); };
    h_out.resize((size_t)H);
    for (int j = 0; j < H; j++) {
        const float i_g = sig(gates[(size_t)(0 * H + j)]);
        const float f_g = sig(gates[(size_t)(1 * H + j)]);
        const float g_g = tanhf(gates[(size_t)(2 * H + j)]);
        const float o_g = sig(gates[(size_t)(3 * H + j)]);
        st.c[(size_t)j] = f_g * st.c[(size_t)j] + i_g * g_g;
        h_out[(size_t)j] = o_g * tanhf(st.c[(size_t)j]);
    }
    st.h = h_out;
}

// RNNTDecoder.predict(x, state): x == nullptr feeds the all-zeros embedding
// (the blueprint's `torch.zeros((batch, 1, pred_hidden))` branch), otherwise
// the embedding row for `token_id`.
static void gigaam_predictor_step(const gigaam_decode_weights& W, const int* token_id, gigaam_lstm_state& st,
                                  std::vector<float>& pred_out) {
    std::vector<float> x((size_t)W.H, 0.0f);
    if (token_id)
        std::memcpy(x.data(), W.embed.data() + (size_t)(*token_id) * W.H, (size_t)W.H * sizeof(float));
    gigaam_lstm_step(W, x.data(), st, pred_out);
}

static void gigaam_joint_proj_enc(const gigaam_decode_weights& W, const float* enc_t, std::vector<float>& out) {
    out.resize((size_t)W.Jh);
    gemv_bias(W.enc_w.data(), W.enc_b.data(), enc_t, W.Jh, W.D, out.data());
}

// joint_net(enc(f) + pred(g)) — RAW logits. The blueprint's log_softmax is
// argmax-invariant, so greedy decode skips it; the diff harness compares the
// same raw values (see gigaam.py's joint_logits_t0).
static void gigaam_joint_step_cpu(const gigaam_decode_weights& W, const float* proj_enc, const float* pred_u,
                                  std::vector<float>& logits) {
    std::vector<float> mid((size_t)W.Jh);
    gemv_bias(W.pred_w.data(), W.pred_b.data(), pred_u, W.Jh, W.H, mid.data());
    for (int i = 0; i < W.Jh; i++) {
        const float v = proj_enc[i] + mid[(size_t)i];
        mid[(size_t)i] = v > 0.0f ? v : 0.0f; // ReLU
    }
    logits.resize((size_t)W.C);
    gemv_bias(W.out_w.data(), W.out_b.data(), mid.data(), W.C, W.Jh, logits.data());
}

// ===========================================================================
// Greedy decoding
// ===========================================================================

struct gigaam_emitted {
    int id;
    int t_start;
    int t_end;
    float p;
};

// RNNTGreedyDecoding._greedy_decode. The blueprint recomputes
// `predict(last_label, dec_state)` at the top of every inner iteration, but
// with (last_label, dec_state) unchanged that call is pure — so carrying the
// cached predictor output forward is equivalent and halves the LSTM work.
static std::vector<gigaam_emitted> gigaam_rnnt_decode(gigaam_context* ctx, const float* enc, int T_enc, int d_model) {
    gigaam_init_decode_weights(ctx);
    const auto& W = ctx->dec;
    const int blank_id = (int)ctx->model.hparams.blank_id;
    const int max_symbols = ctx->max_symbols > 0 ? ctx->max_symbols : 10;

    std::vector<gigaam_emitted> emitted;
    gigaam_lstm_state st;
    st.init(W.H);

    std::vector<float> pred_out;
    gigaam_predictor_step(W, nullptr, st, pred_out); // predict(None, None)

    std::vector<float> proj_e, logits;
    for (int t = 0; t < T_enc; t++) {
        gigaam_joint_proj_enc(W, enc + (size_t)t * d_model, proj_e);

        for (int sym = 0; sym < max_symbols; sym++) {
            gigaam_joint_step_cpu(W, proj_e.data(), pred_out.data(), logits);

            int tok = 0;
            float best = logits[0];
            for (int v = 1; v < W.C; v++) {
                if (logits[(size_t)v] > best) {
                    best = logits[(size_t)v];
                    tok = v;
                }
            }
            if (tok == blank_id)
                break;

            // softmax probability of the emitted token (for the token/word
            // confidence fields; not used by the decode itself)
            float maxl = best, sum = 0.0f;
            for (int v = 0; v < W.C; v++)
                sum += expf(logits[(size_t)v] - maxl);

            emitted.push_back({tok, t, t + 1, 1.0f / sum});
            gigaam_predictor_step(W, &tok, st, pred_out);
        }
    }
    return emitted;
}

// CTCGreedyDecoding.decode: argmax, collapse repeats, drop blanks.
static std::vector<gigaam_emitted> gigaam_ctc_decode(gigaam_context* ctx, const float* enc, int T_enc, int d_model) {
    gigaam_init_decode_weights(ctx);
    const auto& W = ctx->dec;
    const int blank_id = (int)ctx->model.hparams.blank_id;

    std::vector<gigaam_emitted> emitted;
    std::vector<float> logits((size_t)W.C);
    int prev = -1;
    for (int t = 0; t < T_enc; t++) {
        gemv_bias(W.ctc_w.data(), W.ctc_b.data(), enc + (size_t)t * d_model, W.C, W.D, logits.data());
        int tok = 0;
        float best = logits[0];
        for (int v = 1; v < W.C; v++) {
            if (logits[(size_t)v] > best) {
                best = logits[(size_t)v];
                tok = v;
            }
        }
        if (tok != blank_id && tok != prev) {
            float sum = 0.0f;
            for (int v = 0; v < W.C; v++)
                sum += expf(logits[(size_t)v] - best);
            emitted.push_back({tok, t, t + 1, 1.0f / sum});
        }
        prev = tok;
    }
    return emitted;
}

// ===========================================================================
// Detokenization
// ===========================================================================

static std::string gigaam_piece(const gigaam_context* ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return std::string();
    std::string piece = ctx->vocab.id_to_token[(size_t)id];
    if (ctx->model.hparams.is_spm) {
        size_t pos = 0;
        while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
            piece.replace(pos, 3, " ");
            pos += 1;
        }
    }
    return piece;
}

static std::string gigaam_detokenize(const gigaam_context* ctx, const std::vector<gigaam_emitted>& toks) {
    std::string text;
    for (const auto& t : toks)
        text += gigaam_piece(ctx, t.id);
    // SentencePiece emits a leading '▁' on the first word; charwise models
    // never start with a space. Either way, trim the edges.
    size_t b = text.find_first_not_of(' ');
    if (b == std::string::npos)
        return std::string();
    size_t e = text.find_last_not_of(' ');
    return text.substr(b, e - b + 1);
}

static void gigaam_group_words(const gigaam_context* ctx, const std::vector<gigaam_emitted>& toks, int frame_dur_cs,
                               int64_t t_offset_cs, std::vector<gigaam_word_data>& words) {
    std::string cur;
    int64_t t0 = 0, t1 = 0;
    float psum = 0.0f;
    int pn = 0;
    const bool spm = ctx->model.hparams.is_spm;

    auto flush = [&]() {
        if (cur.empty())
            return;
        gigaam_word_data w{};
        snprintf(w.text, sizeof(w.text), "%s", cur.c_str());
        w.t0 = t0;
        w.t1 = t1;
        w.p = pn ? psum / (float)pn : 0.0f;
        words.push_back(w);
        cur.clear();
        psum = 0.0f;
        pn = 0;
    };

    for (const auto& t : toks) {
        std::string piece = gigaam_piece(ctx, t.id);
        const bool starts_word = spm ? (!piece.empty() && piece[0] == ' ') : (piece == " ");
        if (starts_word)
            flush();
        // Charwise models emit the space as its own token; drop it so words
        // carry no leading blank.
        std::string body = piece;
        if (!body.empty() && body[0] == ' ')
            body.erase(0, 1);
        if (body.empty())
            continue;
        if (cur.empty())
            t0 = t_offset_cs + (int64_t)t.t_start * frame_dur_cs;
        cur += body;
        t1 = t_offset_cs + (int64_t)t.t_end * frame_dur_cs;
        psum += t.p;
        pn++;
    }
    flush();
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" struct gigaam_context_params gigaam_context_default_params(void) {
    gigaam_context_params p;
    p.n_threads = 4;
    p.use_flash = false;
    p.verbosity = 1;
    p.use_gpu = false;
    return p;
}

extern "C" struct gigaam_context* gigaam_init_from_file(const char* path_model, struct gigaam_context_params params) {
    auto* ctx = new gigaam_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend_cpu = core_cpu_backend::init();
    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;
    if (params.verbosity > 0)
        fprintf(stderr, "gigaam: backend = %s\n", ggml_backend_name(ctx->backend));

    ctx->compute_meta.resize(16 * 1024 * 1024);

    if (!gigaam_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        fprintf(stderr, "gigaam: failed to load model from '%s'\n", path_model);
        gigaam_free(ctx);
        return nullptr;
    }

    // The conv pointwise weights ship as 3D (1, d, 2d) tensors, which
    // crispasr-quantize's 2D-only rule skips — repack them to Q8_0 when the
    // rest of the model is quantized (issue #81).
    {
        std::vector<core_conformer::BlockWeights*> layers;
        layers.reserve(ctx->model.enc.size());
        for (auto& e : ctx->model.enc)
            layers.push_back(&e);
        const bool quantized = !ctx->model.enc.empty() && ctx->model.enc[0].attn_q_w &&
                               ggml_is_quantized(ctx->model.enc[0].attn_q_w->type);
        core_conformer::repack_conv_pw_q8(layers, ctx->backend, quantized, ctx->model.pw_q8, "gigaam");
    }

    return ctx;
}

extern "C" void gigaam_free(struct gigaam_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    ctx->model.pw_q8.free();
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

extern "C" void gigaam_result_free(struct gigaam_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->tokens);
    free(r->words);
    free(r);
}

extern "C" struct gigaam_result* gigaam_transcribe_ex(struct gigaam_context* ctx, const float* samples, int n_samples,
                                                      int64_t t_offset_cs) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    int T_mel = 0;
    std::vector<float> mel;
    {
        gigaam_bench_stage _b("mel");
        mel = gigaam_compute_mel_impl(ctx, samples, n_samples, T_mel);
    }
    if (mel.empty() || T_mel <= 0)
        return nullptr;

    std::vector<float> enc;
    int T_enc = 0, d_model = 0;
    {
        gigaam_bench_stage _b("encoder");
        if (!gigaam_run_encoder_impl(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, enc, T_enc, d_model,
                                     nullptr, 0))
            return nullptr;
    }
    if (T_enc <= 0)
        return nullptr;

    std::vector<gigaam_emitted> emitted;
    {
        gigaam_bench_stage _b(ctx->model.hparams.is_rnnt ? "rnnt_decode" : "ctc_decode");
        emitted = ctx->model.hparams.is_rnnt ? gigaam_rnnt_decode(ctx, enc.data(), T_enc, d_model)
                                             : gigaam_ctc_decode(ctx, enc.data(), T_enc, d_model);
    }

    auto* r = (gigaam_result*)calloc(1, sizeof(gigaam_result));
    if (!r)
        return nullptr;
    const std::string text = gigaam_detokenize(ctx, emitted);
    r->text = strdup(text.c_str());

    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;
    r->n_tokens = (int)emitted.size();
    if (r->n_tokens > 0) {
        r->tokens = (gigaam_token_data*)calloc((size_t)r->n_tokens, sizeof(gigaam_token_data));
        for (int i = 0; i < r->n_tokens; i++) {
            auto& td = r->tokens[i];
            td.id = emitted[(size_t)i].id;
            td.t0 = t_offset_cs + (int64_t)emitted[(size_t)i].t_start * frame_dur_cs;
            td.t1 = t_offset_cs + (int64_t)emitted[(size_t)i].t_end * frame_dur_cs;
            td.p = emitted[(size_t)i].p;
            snprintf(td.text, sizeof(td.text), "%s", gigaam_piece(ctx, td.id).c_str());
        }
    }

    std::vector<gigaam_word_data> words;
    gigaam_group_words(ctx, emitted, frame_dur_cs, t_offset_cs, words);
    r->n_words = (int)words.size();
    if (r->n_words > 0) {
        r->words = (gigaam_word_data*)malloc((size_t)r->n_words * sizeof(gigaam_word_data));
        memcpy(r->words, words.data(), (size_t)r->n_words * sizeof(gigaam_word_data));
    }
    return r;
}

extern "C" char* gigaam_transcribe(struct gigaam_context* ctx, const float* samples, int n_samples) {
    gigaam_result* r = gigaam_transcribe_ex(ctx, samples, n_samples, 0);
    if (!r)
        return nullptr;
    char* text = r->text;
    r->text = nullptr;
    gigaam_result_free(r);
    return text;
}

extern "C" int gigaam_n_vocab(struct gigaam_context* ctx) {
    return ctx ? (int)ctx->vocab.id_to_token.size() : 0;
}

extern "C" int gigaam_blank_id(struct gigaam_context* ctx) {
    return ctx ? (int)ctx->model.hparams.blank_id : 0;
}

extern "C" const char* gigaam_token_to_str(struct gigaam_context* ctx, int token_id) {
    if (!ctx || token_id < 0 || token_id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[(size_t)token_id].c_str();
}

extern "C" int gigaam_frame_dur_cs(struct gigaam_context* ctx) {
    return ctx ? (int)ctx->model.hparams.frame_dur_cs : 0;
}
extern "C" int gigaam_n_mels(struct gigaam_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_mels : 0;
}
extern "C" int gigaam_sample_rate(struct gigaam_context* ctx) {
    return ctx ? (int)ctx->model.hparams.sample_rate : 0;
}
extern "C" int gigaam_d_model(struct gigaam_context* ctx) {
    return ctx ? (int)ctx->model.hparams.d_model : 0;
}
extern "C" int gigaam_n_layers(struct gigaam_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_layers : 0;
}
extern "C" int gigaam_is_rnnt(struct gigaam_context* ctx) {
    return ctx && ctx->model.hparams.is_rnnt ? 1 : 0;
}
extern "C" int gigaam_is_spm(struct gigaam_context* ctx) {
    return ctx && ctx->model.hparams.is_spm ? 1 : 0;
}

extern "C" int gigaam_est_enc_frames(struct gigaam_context* ctx, int n_samples) {
    if (!ctx || n_samples <= 0)
        return 0;
    const auto& hp = ctx->model.hparams;
    const int T_mel = n_samples >= (int)hp.win_length ? (n_samples - (int)hp.win_length) / (int)hp.hop_length + 1 : 0;
    return gigaam_subsampled_len(hp, T_mel);
}

extern "C" void gigaam_set_max_symbols(struct gigaam_context* ctx, int max_symbols) {
    if (!ctx)
        return;
    ctx->max_symbols = max_symbols > 0 ? max_symbols : 10;
}

// ---- Stage entry points ----

extern "C" float* gigaam_compute_mel(struct gigaam_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                     int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    int T_mel = 0;
    std::vector<float> mel = gigaam_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty() || T_mel <= 0)
        return nullptr;
    auto* out = (float*)malloc(mel.size() * sizeof(float));
    if (!out)
        return nullptr;
    memcpy(out, mel.data(), mel.size() * sizeof(float));
    if (out_n_mels)
        *out_n_mels = (int)ctx->model.hparams.n_mels;
    if (out_T_mel)
        *out_T_mel = T_mel;
    return out;
}

extern "C" float* gigaam_run_encoder(struct gigaam_context* ctx, const float* mel, int n_mels, int T_mel,
                                     int* out_T_enc, int* out_d_model) {
    if (!ctx || !mel || T_mel <= 0)
        return nullptr;
    std::vector<float> enc;
    int T_enc = 0, d_model = 0;
    if (!gigaam_run_encoder_impl(ctx, mel, n_mels, T_mel, enc, T_enc, d_model, nullptr, 0))
        return nullptr;
    auto* out = (float*)malloc(enc.size() * sizeof(float));
    if (!out)
        return nullptr;
    memcpy(out, enc.data(), enc.size() * sizeof(float));
    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d_model;
    return out;
}

extern "C" int gigaam_run_encoder_dump(struct gigaam_context* ctx, const float* mel, int n_mels, int T_mel, float** out,
                                       int out_count, int* out_T_enc, int* out_d_model) {
    if (!ctx || !mel || T_mel <= 0 || !out || out_count <= 0)
        return 1;
    std::vector<float> enc;
    int T_enc = 0, d_model = 0;
    if (!gigaam_run_encoder_impl(ctx, mel, n_mels, T_mel, enc, T_enc, d_model, out, out_count))
        return 1;
    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d_model;
    return 0;
}

extern "C" float* gigaam_ctc_log_probs(struct gigaam_context* ctx, const float* enc_frames, int T_enc, int d_model,
                                       int* out_n_classes) {
    if (!ctx || !enc_frames || T_enc <= 0 || ctx->model.hparams.is_rnnt)
        return nullptr;
    gigaam_init_decode_weights(ctx);
    const auto& W = ctx->dec;
    auto* out = (float*)malloc((size_t)T_enc * W.C * sizeof(float));
    if (!out)
        return nullptr;
    for (int t = 0; t < T_enc; t++) {
        float* row = out + (size_t)t * W.C;
        gemv_bias(W.ctc_w.data(), W.ctc_b.data(), enc_frames + (size_t)t * d_model, W.C, W.D, row);
        // CTCHead ends in log_softmax; reproduce it so the values (not just
        // the argmax) match the reference dump.
        float mx = row[0];
        for (int v = 1; v < W.C; v++)
            mx = std::max(mx, row[v]);
        double sum = 0.0;
        for (int v = 0; v < W.C; v++)
            sum += exp((double)row[v] - mx);
        const float lse = mx + (float)log(sum);
        for (int v = 0; v < W.C; v++)
            row[v] -= lse;
    }
    if (out_n_classes)
        *out_n_classes = W.C;
    return out;
}

extern "C" float* gigaam_joint_project_encoder(struct gigaam_context* ctx, const float* enc_frames, int T_enc,
                                               int d_model, int* out_joint_hidden) {
    if (!ctx || !enc_frames || T_enc <= 0 || !ctx->model.hparams.is_rnnt)
        return nullptr;
    gigaam_init_decode_weights(ctx);
    const auto& W = ctx->dec;
    auto* out = (float*)malloc((size_t)T_enc * W.Jh * sizeof(float));
    if (!out)
        return nullptr;
    for (int t = 0; t < T_enc; t++)
        gemv_bias(W.enc_w.data(), W.enc_b.data(), enc_frames + (size_t)t * d_model, W.Jh, W.D, out + (size_t)t * W.Jh);
    if (out_joint_hidden)
        *out_joint_hidden = W.Jh;
    return out;
}

extern "C" float* gigaam_predictor_initial(struct gigaam_context* ctx, int* out_pred_hidden) {
    if (!ctx || !ctx->model.hparams.is_rnnt)
        return nullptr;
    gigaam_init_decode_weights(ctx);
    const auto& W = ctx->dec;
    gigaam_lstm_state st;
    st.init(W.H);
    std::vector<float> pred_out;
    gigaam_predictor_step(W, nullptr, st, pred_out);
    auto* out = (float*)malloc((size_t)W.H * sizeof(float));
    if (!out)
        return nullptr;
    memcpy(out, pred_out.data(), (size_t)W.H * sizeof(float));
    if (out_pred_hidden)
        *out_pred_hidden = W.H;
    return out;
}

extern "C" float* gigaam_joint_step(struct gigaam_context* ctx, const float* proj_enc, const float* pred_out,
                                    int* out_num_classes) {
    if (!ctx || !proj_enc || !pred_out || !ctx->model.hparams.is_rnnt)
        return nullptr;
    gigaam_init_decode_weights(ctx);
    const auto& W = ctx->dec;
    std::vector<float> logits;
    gigaam_joint_step_cpu(W, proj_enc, pred_out, logits);
    auto* out = (float*)malloc((size_t)W.C * sizeof(float));
    if (!out)
        return nullptr;
    memcpy(out, logits.data(), (size_t)W.C * sizeof(float));
    if (out_num_classes)
        *out_num_classes = W.C;
    return out;
}

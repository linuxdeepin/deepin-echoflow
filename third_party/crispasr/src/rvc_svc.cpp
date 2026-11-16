// rvc_svc.cpp — see rvc_svc.h.
//
// Built against tools/rvc_torch_parity.py, which pins every stage at
// cos 1.00000000 against torch. Where this file looks odd, the numpy spec and
// docs/music-transcription/RVC_BLUEPRINT.md say why — most of the oddities are
// upstream quirks baked into the trained weights, not choices.

#include "rvc_svc.h"

#include "core/gguf_loader.h"
#include "core/hifigan.h"
#include "core/gpu_backend_pref.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------
struct rvc_hparams {
    int content_dim = 768;
    int hidden = 192;
    int inter = 192;
    int n_layers = 6;
    int n_heads = 2;
    int rel_window = 10;
    int n_speakers = 109;
    int gin = 256;
    int sample_rate = 40000;
    int upsample_initial_channel = 512;
    std::vector<int> upsample_rates;
    std::vector<int> upsample_kernel_sizes;
    std::vector<int> resblock_kernel_sizes;
    std::vector<int> resblock_dilations; // flattened n_kernels x n_dilations
    int resblock_n_dilations = 3;
    // Flow geometry is HARDCODED upstream (models.py:624 —
    // ResidualCouplingBlock(inter, hidden, 5, 1, 3)), not config-derived, so
    // these hold for every checkpoint rather than just the one we converted.
    int flow_n_flows = 4;
    int flow_n_layers = 3;
    int flow_kernel = 5;
    int flow_dilation_rate = 1;
    // SineGen. harmonic_num is 0 in GeneratorNSF, which is why the random
    // initial phase is identically zero (rand_ini is one element and the next
    // line zeroes it) — so the ONLY live noise in the source module is additive.
    int harmonic_num = 0;
    float sine_amp = 0.1f;
    float add_noise_std = 0.003f;
    float noise_scale = 0.66666f;

    int upp() const {
        int p = 1;
        for (int r : upsample_rates)
            p *= r;
        return p;
    }
};

struct rvc_svc_context {
    rvc_hparams hp;
    rvc_svc_params params;
    ggml_backend_t backend = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> t; // name -> weight
    // Per-stage capture for the diff harness (HARD RULE #2: intermediates, not
    // just endpoints — endpoints alone cannot localise anything).
    bool capture = false;
    std::map<std::string, ggml_tensor*> caps;
    ggml_tensor* rev_idx_tensor = nullptr; // channel-reversal indices for Flip
};

namespace {

std::vector<int> read_i32_array(gguf_context* g, const char* key) {
    std::vector<int> out;
    const int64_t k = gguf_find_key(g, key);
    if (k < 0)
        return out;
    const int64_t n = gguf_get_arr_n(g, k);
    const auto* d = (const int32_t*)gguf_get_arr_data(g, k);
    out.assign(d, d + n);
    return out;
}

} // namespace

rvc_svc_params rvc_svc_default_params(void) {
    rvc_svc_params p{};
    p.n_threads = 0;
    p.use_gpu = true;
    p.gpu_device = 0;
    return p;
}

void rvc_svc_coarse_pitch(const float* f0_hz, int n, int* out_coarse) {
    // pipeline.py:73-137, exactly. The constants are model-side: emb_pitch is
    // an nn.Embedding(256, hidden) lookup, so an off-by-one here selects a
    // DIFFERENT LEARNED VECTOR rather than producing a small numeric error.
    const double f0_min = 50.0, f0_max = 1100.0;
    const double mel_min = 1127.0 * std::log(1.0 + f0_min / 700.0);
    const double mel_max = 1127.0 * std::log(1.0 + f0_max / 700.0);
    for (int i = 0; i < n; i++) {
        double mel = 1127.0 * std::log(1.0 + (double)f0_hz[i] / 700.0);
        if (mel > 0.0)
            mel = (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0;
        if (mel <= 1.0)
            mel = 1.0;
        if (mel > 255.0)
            mel = 255.0;
        out_coarse[i] = (int)std::lround(mel);
    }
}

int rvc_svc_content_dim(const rvc_svc_context* ctx) {
    return ctx ? ctx->hp.content_dim : 0;
}
int rvc_svc_n_speakers(const rvc_svc_context* ctx) {
    return ctx ? ctx->hp.n_speakers : 0;
}
int rvc_svc_sample_rate(const rvc_svc_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}

rvc_svc_context* rvc_svc_init_from_file(const char* model_path, rvc_svc_params params) {
    gguf_context* meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "rvc: cannot open %s\n", model_path);
        return nullptr;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    if (arch != "rvc") {
        fprintf(stderr, "rvc: '%s' is not an rvc model (arch='%s')\n", model_path, arch.c_str());
        core_gguf::free_metadata(meta);
        return nullptr;
    }

    auto* ctx = new rvc_svc_context();
    ctx->params = params;
    rvc_hparams& hp = ctx->hp;
    hp.content_dim = core_gguf::kv_u32(meta, "rvc.content_dim", 768);
    hp.hidden = core_gguf::kv_u32(meta, "rvc.hidden_channels", 192);
    hp.inter = core_gguf::kv_u32(meta, "rvc.inter_channels", 192);
    hp.n_layers = core_gguf::kv_u32(meta, "rvc.n_layers", 6);
    hp.n_heads = core_gguf::kv_u32(meta, "rvc.n_heads", 2);
    hp.rel_window = core_gguf::kv_u32(meta, "rvc.rel_attn_window", 10);
    hp.n_speakers = core_gguf::kv_u32(meta, "rvc.n_speakers", 109);
    hp.gin = core_gguf::kv_u32(meta, "rvc.gin_channels", 256);
    hp.sample_rate = core_gguf::kv_u32(meta, "rvc.sample_rate", 40000);
    hp.upsample_initial_channel = core_gguf::kv_u32(meta, "rvc.upsample_initial_channel", 512);
    hp.harmonic_num = core_gguf::kv_u32(meta, "rvc.harmonic_num", 0);
    hp.sine_amp = core_gguf::kv_f32(meta, "rvc.sine_amp", 0.1f);
    hp.add_noise_std = core_gguf::kv_f32(meta, "rvc.add_noise_std", 0.003f);
    hp.noise_scale = core_gguf::kv_f32(meta, "rvc.noise_scale", 0.66666f);
    hp.upsample_rates = read_i32_array(meta, "rvc.upsample_rates");
    hp.upsample_kernel_sizes = read_i32_array(meta, "rvc.upsample_kernel_sizes");
    hp.resblock_kernel_sizes = read_i32_array(meta, "rvc.resblock_kernel_sizes");
    hp.resblock_dilations = read_i32_array(meta, "rvc.resblock_dilations");
    hp.resblock_n_dilations = core_gguf::kv_u32(meta, "rvc.resblock_n_dilations", 3);
    core_gguf::free_metadata(meta);

    if (hp.upsample_rates.empty() || hp.upsample_rates.size() != hp.upsample_kernel_sizes.size()) {
        fprintf(stderr, "rvc: bad/missing upsample schedule in %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    // The wire contract promises a 100 Hz feature/F0 rate, and it is derived,
    // not configured: sr / prod(upsample_rates). Refuse a checkpoint that
    // would silently violate it rather than resampling behind the caller.
    const int upp = hp.upp();
    if (upp <= 0 || hp.sample_rate % upp != 0 || hp.sample_rate / upp != 100) {
        fprintf(stderr,
                "rvc: sr/prod(upsample_rates) = %d/%d is not 100 Hz — this checkpoint does not "
                "honour the feature-rate contract (docs/music-transcription/SVC_RECORD_SHAPES.md)\n",
                hp.sample_rate, upp);
        delete ctx;
        return nullptr;
    }

    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, ctx->backend, "rvc", wl)) {
        fprintf(stderr, "rvc: failed to load weights from %s\n", model_path);
        rvc_svc_free(ctx);
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->t = wl.tensors;

    // Cross-check the geometry we were told against the geometry the weights
    // actually have. A GGUF whose KVs disagree with its tensors would otherwise
    // fail much later as an unexplained shape error.
    auto need = [&](const char* n) -> ggml_tensor* {
        auto it = ctx->t.find(n);
        return it == ctx->t.end() ? nullptr : it->second;
    };
    ggml_tensor* emb_phone = need("enc_p.emb_phone.weight");
    ggml_tensor* emb_g = need("emb_g.weight");
    if (!emb_phone || !emb_g) {
        fprintf(stderr, "rvc: missing enc_p.emb_phone.weight / emb_g.weight\n");
        rvc_svc_free(ctx);
        return nullptr;
    }
    if ((int)emb_phone->ne[0] != hp.content_dim) {
        fprintf(stderr, "rvc: content_dim KV says %d but emb_phone is %d — refusing a v1/v2 mismatch\n", hp.content_dim,
                (int)emb_phone->ne[0]);
        rvc_svc_free(ctx);
        return nullptr;
    }

    fprintf(stderr, "rvc: content_dim=%d hidden=%d layers=%d heads=%d speakers=%d sr=%d upp=%d (%d fps)\n",
            hp.content_dim, hp.hidden, hp.n_layers, hp.n_heads, hp.n_speakers, hp.sample_rate, upp,
            hp.sample_rate / upp);
    return ctx;
}

void rvc_svc_free(rvc_svc_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf_w)
        core_gguf::release_weight_buffer(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

void rvc_svc_result_free(rvc_svc_result* r) {
    if (!r)
        return;
    free(r->pcm);
    delete r;
}

// ---------------------------------------------------------------------------
// enc_p — TextEncoder (relative-position transformer)
//
// Validated stage-for-stage against tools/rvc_torch_parity.py (itself cos
// 1.00000000 vs torch). Traps encoded here, all from RVC_BLUEPRINT.md 2b:
//   * LeakyReLU slope 0.1, NOT torch's 0.01 default.
//   * x *= sqrt(hidden) BEFORE the lrelu.
//   * POST-norm residuals: x = norm(x + f(x)).
//   * Relative-position attention over keys AND values (window 10).
//   * FFN is SAME-padded with a plain ReLU.
//   * LayerNorm is over the CHANNEL dim.
// ---------------------------------------------------------------------------

namespace {

// A kernel-1 Conv1d IS a linear, but it is stored with the kernel axis:
// GGUF (out, in, 1) -> ggml ne = [1, in, out]. Drop the leading 1 so mul_mat
// contracts over `in` instead of aborting on a 1-vs-in mismatch.
ggml_tensor* rvc_conv1x1_as_linear(ggml_context* g, ggml_tensor* w) {
    if (w->ne[0] == 1 && w->ne[2] > 1)
        return ggml_reshape_2d(g, w, w->ne[1], w->ne[2]);
    return w;
}

ggml_tensor* rvc_layer_norm(ggml_context* g, ggml_tensor* x, ggml_tensor* gamma, ggml_tensor* beta, float eps) {
    // x: [C, T]. Their LayerNorm transposes so the stats are over C.
    x = ggml_norm(g, x, eps);
    x = ggml_mul(g, x, gamma);
    return ggml_add(g, x, beta);
}

// Slice/pad emb_rel to 2T-1 entries, exactly as _get_relative_embeddings.
// NOTE: the padding is SYMMETRIC (pad_len on BOTH sides). ggml_pad only
// appends, so the front pad is a concat — getting this wrong reads past the
// tensor and trips ggml's view-bounds assert.
ggml_tensor* rvc_rel_embeddings(ggml_context* g, ggml_tensor* emb, int T, int window) {
    const int pad_len = std::max(T - (window + 1), 0);
    const int start = std::max((window + 1) - T, 0);
    // emb: GGUF (1, 2w+1, d) -> ggml ne = [d, 2w+1, 1]
    ggml_tensor* e = ggml_reshape_2d(g, emb, emb->ne[0], emb->ne[1]);
    // Cast to F32 up front. In an f16 GGUF emb_rel_* is F16, and the zero-pad
    // below needs ggml_scale, which is F32-only — so an f16 checkpoint aborted
    // inside ggml_compute_forward_scale. These are small tables ((2w+1) x d)
    // and every consumer of them downstream is F32, so the cast is cheap and
    // removes the whole dtype question.
    if (e->type != GGML_TYPE_F32)
        e = ggml_cast(g, e, GGML_TYPE_F32);
    if (pad_len > 0) {
        // Match the SOURCE dtype: emb_rel_* is F16 in an f16 GGUF and
        // ggml_concat requires both operands to share a type. Hardcoding F32
        // aborted on every f16 checkpoint while f32 worked fine.
        ggml_tensor* pre = ggml_new_tensor_2d(g, GGML_TYPE_F32, e->ne[0], pad_len);
        pre = ggml_scale(g, pre, 0.0f);
        e = ggml_concat(g, pre, e, 1);        // front pad
        e = ggml_pad(g, e, 0, pad_len, 0, 0); // back pad
    }
    return ggml_cont(g, ggml_view_2d(g, e, e->ne[0], 2 * T - 1, e->nb[1], (size_t)start * e->nb[1]));
}

// [2T-1, T, H] -> [T, T, H]: the skew from relative to absolute indexing.
ggml_tensor* rvc_rel_to_abs(ggml_context* g, ggml_tensor* x, int T) {
    x = ggml_pad(g, x, 1, 0, 0, 0); // [2T, T, H]
    const int64_t H = x->ne[2];
    x = ggml_cont(g, ggml_reshape_2d(g, x, 2 * T * T, H)); // flatten
    x = ggml_pad(g, x, T - 1, 0, 0, 0);                    // [2T*T + T-1, H]
    x = ggml_cont(g, ggml_reshape_3d(g, x, 2 * T - 1, T + 1, H));
    return ggml_cont(g, ggml_view_3d(g, x, T, T, H, x->nb[1], x->nb[2], (size_t)(T - 1) * x->nb[0]));
}

// [T, T, H] -> [2T-1, T, H]: the inverse skew, for relative VALUES.
ggml_tensor* rvc_abs_to_rel(ggml_context* g, ggml_tensor* x, int T) {
    const int64_t H = x->ne[2];
    x = ggml_pad(g, x, T - 1, 0, 0, 0); // [2T-1, T, H]
    x = ggml_cont(g, ggml_reshape_2d(g, x, T * (2 * T - 1), H));
    // pad the FRONT by T: ggml_pad only appends, so pad the end of a reversed
    // view is not available either — build it with a zero prefix concat.
    ggml_tensor* pre = ggml_new_tensor_2d(g, GGML_TYPE_F32, T, H);
    pre = ggml_scale(g, pre, 0.0f);
    x = ggml_concat(g, pre, x, 0); // [T*(2T-1)+T, H]
    x = ggml_cont(g, ggml_reshape_3d(g, x, 2 * T, T, H));
    return ggml_cont(g, ggml_view_3d(g, x, 2 * T - 1, T, H, x->nb[1], x->nb[2], (size_t)1 * x->nb[0]));
}

} // namespace

namespace {

// Build the enc_p graph. content: [content_dim, T] f32, pitch: [T] i32.
// Returns stats [2*inter, T]; caller splits into m_p / logs_p.
ggml_tensor* rvc_enc_p_graph(ggml_context* g, rvc_svc_context* c, ggml_tensor* content, ggml_tensor* pitch, int T) {
    const rvc_hparams& hp = c->hp;
    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        if (it == c->t.end()) {
            fprintf(stderr, "rvc: missing tensor %s\n", n.c_str());
            return nullptr;
        }
        return it->second;
    };

    // emb_phone is a Linear: [content_dim, hidden] weight in ggml layout.
    // TAP records a tensor for the diff. `chan_time` marks stages whose
    // REFERENCE is stored as row-major (channels, time) — i.e. time fastest —
    // which is how torch stores a (b, C, T) tensor. Our working layout is
    // [C, T] with CHANNELS fastest, the exact transpose, so those stages must
    // be transposed before comparison or the cosine is meaningless.
    // Getting this wrong reads as a catastrophic FAIL (cos ~0) on a correct
    // graph: `agg` passed and `ctx` "failed" purely because the first is
    // compared against a numpy (H,T,hd) buffer that happens to share our order
    // and the second against a (C,T) one that does not.
    auto TAP = [&](const std::string& nm, ggml_tensor* v, bool chan_time = false) {
        if (c->capture) {
            ggml_tensor* o = chan_time ? ggml_cont(g, ggml_transpose(g, v)) : v;
            ggml_set_output(o);
            c->caps[nm] = o;
        }
        return v;
    };

    ggml_tensor* x = ggml_mul_mat(g, W("enc_p.emb_phone.weight"), content); // [hidden, T]
    x = ggml_add(g, x, W("enc_p.emb_phone.bias"));
    // emb_pitch is an EMBEDDING lookup, not a matmul.
    x = ggml_add(g, x, ggml_get_rows(g, W("enc_p.emb_pitch.weight"), pitch));
    x = ggml_scale(g, x, std::sqrt((float)hp.hidden)); // BEFORE the lrelu
    x = ggml_leaky_relu(g, x, 0.1f, false);            // slope 0.1, not 0.01
    TAP("encp_lrelu", x);

    const int H = hp.n_heads;
    const int hd = hp.hidden / H;
    const float scale = 1.0f / std::sqrt((float)hd);

    for (int l = 0; l < hp.n_layers; l++) {
        const std::string p = "enc_p.encoder.attn_layers." + std::to_string(l) + ".";
        // k=1 convs are plain linears over the channel dim.
        ggml_tensor* q =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "conv_q.weight")), x), W(p + "conv_q.bias"));
        ggml_tensor* k =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "conv_k.weight")), x), W(p + "conv_k.bias"));
        ggml_tensor* v =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "conv_v.weight")), x), W(p + "conv_v.bias"));

        // [hidden, T] -> [hd, T, H] -> heads on ne2
        q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, q, hd, H, T), 0, 2, 1, 3));
        k = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, k, hd, H, T), 0, 2, 1, 3));
        v = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, v, hd, H, T), 0, 2, 1, 3));

        ggml_tensor* qs = ggml_scale(g, q, scale);
        ggml_tensor* scores = ggml_mul_mat(g, k, qs); // [T_k, T_q, H]

        ggml_tensor* rel_k = rvc_rel_embeddings(g, W(p + "emb_rel_k"), T, hp.rel_window); // [hd, 2T-1]
        ggml_tensor* rl = ggml_mul_mat(g, rel_k, qs);                                     // [2T-1, T, H]
        scores = ggml_add(g, scores, rvc_rel_to_abs(g, rl, T));

        ggml_tensor* attn = ggml_soft_max(g, scores); // [T_k, T_q, H]
        if (l == 0)
            TAP("encp_L0_attn_w", attn);
        if (l == 0)
            TAP("encp_L0_v", v);
        ggml_tensor* out = ggml_mul_mat(g, ggml_cont(g, ggml_transpose(g, v)), attn); // [hd, T_q, H]
        if (l == 0)
            TAP("encp_L0_agg", out);

        // relative VALUES — omitting this still "works" and merely degrades.
        ggml_tensor* rel_v = rvc_rel_embeddings(g, W(p + "emb_rel_v"), T, hp.rel_window); // [hd, 2T-1]
        ggml_tensor* ar = rvc_abs_to_rel(g, attn, T);                                     // [2T-1, T, H]
        out = ggml_add(g, out, ggml_mul_mat(g, ggml_cont(g, ggml_transpose(g, rel_v)), ar));

        out = ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3)); // [hd, H, T]
        out = ggml_reshape_2d(g, out, hp.hidden, T);
        if (l == 0)
            TAP("encp_L0_ctx", out, /*chan_time=*/true);
        ggml_tensor* y =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "conv_o.weight")), out), W(p + "conv_o.bias"));
        TAP("encp_L" + std::to_string(l) + "_attn", y, /*chan_time=*/true);

        const std::string n1 = "enc_p.encoder.norm_layers_1." + std::to_string(l) + ".";
        x = rvc_layer_norm(g, ggml_add(g, x, y), W(n1 + "gamma"), W(n1 + "beta"), 1e-5f); // POST-norm
        TAP("encp_L" + std::to_string(l) + "_norm1", x, /*chan_time=*/true);

        const std::string f = "enc_p.encoder.ffn_layers." + std::to_string(l) + ".";
        // ggml_conv_1d takes input as [length, channels]; our working layout
        // is [channels, time], so transpose in and back out. The bias is added
        // AFTER transposing back so it broadcasts over time, not over channels.
        ggml_tensor* w1 = W(f + "conv_1.weight");
        const int k1 = (int)w1->ne[0];
        ggml_tensor* h = ggml_conv_1d(g, w1, ggml_cont(g, ggml_transpose(g, x)), 1, (k1 - 1) / 2, 1);
        h = ggml_add(g, ggml_cont(g, ggml_transpose(g, h)), W(f + "conv_1.bias"));
        h = ggml_relu(g, h); // plain ReLU: activation != "gelu" here
        ggml_tensor* w2 = W(f + "conv_2.weight");
        const int k2 = (int)w2->ne[0];
        ggml_tensor* y2 = ggml_conv_1d(g, w2, ggml_cont(g, ggml_transpose(g, h)), 1, (k2 - 1) / 2, 1);
        y2 = ggml_add(g, ggml_cont(g, ggml_transpose(g, y2)), W(f + "conv_2.bias"));

        const std::string n2 = "enc_p.encoder.norm_layers_2." + std::to_string(l) + ".";
        TAP("encp_L" + std::to_string(l) + "_ffn", y2, /*chan_time=*/true);
        x = rvc_layer_norm(g, ggml_add(g, x, y2), W(n2 + "gamma"), W(n2 + "beta"), 1e-5f);
        TAP("encp_L" + std::to_string(l) + "_norm2", x, /*chan_time=*/true);
    }

    ggml_tensor* stats =
        ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W("enc_p.proj.weight")), x), W("enc_p.proj.bias"));
    return stats; // [2*inter, T]
}

} // namespace

// ---------------------------------------------------------------------------
// SineGen — the NSF source signal.
//
// Computed on the HOST, not in ggml: it is signal generation (cumsum, linear
// interpolation, modulo, wrap detection), not a learned layer — the same call
// as BTC's CQT front end. Doing it in ggml would mean expressing a running
// phase accumulation with no matching op, for no benefit.
//
// The phase logic CANNOT be paraphrased. A plausible rewrite scored cos -0.04
// against torch (right amplitude, uncorrelated phase). The real sequence
// (models.py:329-351):
//   1. rad = (f0/sr) % 1                       at FRAME rate
//   2. tmp = cumsum(rad) * upp                 still frame rate
//   3. tmp -> LINEAR interpolate to out rate, align_corners=True
//   4. rad -> NEAREST interpolate to out rate
//   5. tmp %= 1; wrap points are where diff(tmp) < 0
//   6. phase = cumsum(rad_up + shift), shift = -1 at each wrap
//   7. sine = sin(phase * 2*pi)
// The linear-interpolated cumsum locates the WRAPS only; the phase accumulates
// over the NEAREST-upsampled values. The two interpolations use DIFFERENT
// modes. Accumulate in double: the running sum grows without bound while only
// its fraction matters.
//
// harmonic_num is 0, so the random initial phase is identically zero and the
// only live noise here is the ADDITIVE term (voicing-dependent).
// ---------------------------------------------------------------------------

namespace {

void rvc_sine_gen(const float* f0_hz, int T, int upp, int sr, const float* noise, float sine_amp, float noise_std,
                  std::vector<float>& out_sine, std::vector<float>& out_uv) {
    const int64_t N = (int64_t)T * upp;
    std::vector<double> rad((size_t)T), tmp((size_t)T);
    double acc = 0.0;
    for (int t = 0; t < T; t++) {
        rad[(size_t)t] = std::fmod((double)f0_hz[t] / (double)sr, 1.0);
        acc += rad[(size_t)t];
        tmp[(size_t)t] = acc * upp;
    }
    // linear interpolate `tmp` to the output rate, align_corners=True
    std::vector<double> tmp_up((size_t)N);
    for (int64_t i = 0; i < N; i++) {
        const double x = (T > 1) ? (double)i * (double)(T - 1) / (double)(N - 1) : 0.0;
        const int64_t i0 = (int64_t)std::floor(x);
        const int64_t i1 = std::min<int64_t>(i0 + 1, T - 1);
        const double fr = x - (double)i0;
        tmp_up[(size_t)i] = tmp[(size_t)i0] * (1.0 - fr) + tmp[(size_t)i1] * fr;
        tmp_up[(size_t)i] = tmp_up[(size_t)i] - std::floor(tmp_up[(size_t)i]); // %= 1
    }
    out_sine.assign((size_t)N, 0.0f);
    out_uv.assign((size_t)N, 0.0f);
    double phase = 0.0;
    for (int64_t i = 0; i < N; i++) {
        const double r = rad[(size_t)(i / upp)]; // NEAREST upsample
        const double shift = (i > 0 && (tmp_up[(size_t)i] - tmp_up[(size_t)i - 1]) < 0.0) ? -1.0 : 0.0;
        phase += r + shift;
        const double uv = (f0_hz[i / upp] > 0.0f) ? 1.0 : 0.0;
        const double na = uv * noise_std + (1.0 - uv) * sine_amp / 3.0;
        const double s = std::sin(phase * 2.0 * M_PI) * sine_amp;
        out_sine[(size_t)i] = (float)(s * uv + na * (noise ? (double)noise[i] : 0.0));
        out_uv[(size_t)i] = (float)uv;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// flow — ResidualCouplingBlock, REVERSE pass
//
// Traps (RVC_BLUEPRINT.md 2c):
//   * mean_only=True, so `logs` is ZERO and the coupling is purely ADDITIVE:
//     the reverse is x1 = x1 - m, NOT (x1 - m) * exp(-logs).
//   * flows interleaves [Coupling, Flip] x 4 and the reverse walks it
//     backwards, so Flip comes FIRST.
//   * Flip reverses the CHANNEL axis.
//   * The WaveNet is gated: tanh(first half) * sigmoid(second half) of
//     (x_in + g_l), with the speaker conditioning projected once then sliced
//     per layer.
//   * kernel 5 / dilation rate 1 / 3 layers are hardcoded at models.py:624.
// ---------------------------------------------------------------------------

namespace {

// Reverse the CHANNEL axis. ggml has no flip, and a negative-stride view is not
// expressible, so transpose -> get_rows(reversed index) -> transpose back.
ggml_tensor* rvc_flip_channels(ggml_context* g, ggml_tensor* x, ggml_tensor* rev_idx) {
    ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, x)); // [T, C]
    ggml_tensor* f = ggml_get_rows(g, xt, rev_idx);       // [T, C] channels reversed
    return ggml_cont(g, ggml_transpose(g, f));            // [C, T]
}

ggml_tensor* rvc_wn(ggml_context* g, rvc_svc_context* c, const std::string& pre, ggml_tensor* x, ggml_tensor* gcond,
                    int hidden, int n_layers, int kernel, int dil_rate) {
    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        return it == c->t.end() ? nullptr : it->second;
    };
    ggml_tensor* output = nullptr;
    for (int i = 0; i < n_layers; i++) {
        const int d = (int)std::pow((double)dil_rate, (double)i);
        const int pad = (kernel * d - d) / 2;
        ggml_tensor* w = W(pre + "in_layers." + std::to_string(i) + ".weight");
        // conv_1d wants [length, channels]; our layout is [channels, time].
        ggml_tensor* xin = ggml_conv_1d(g, w, ggml_cont(g, ggml_transpose(g, x)), 1, pad, d);
        xin = ggml_add(g, ggml_cont(g, ggml_transpose(g, xin)),
                       W(pre + "in_layers." + std::to_string(i) + ".bias")); // [2*hidden, T]

        // speaker conditioning: one projection, sliced per layer
        ggml_tensor* gl = ggml_cont(g, ggml_view_1d(g, gcond, 2 * hidden, (size_t)i * 2 * hidden * gcond->nb[0]));
        xin = ggml_add(g, xin, gl);

        const int64_t T = xin->ne[1];
        ggml_tensor* ta = ggml_cont(g, ggml_view_2d(g, xin, hidden, T, xin->nb[1], 0));
        ggml_tensor* sa = ggml_cont(g, ggml_view_2d(g, xin, hidden, T, xin->nb[1], (size_t)hidden * xin->nb[0]));
        ggml_tensor* acts = ggml_mul(g, ggml_tanh(g, ta), ggml_sigmoid(g, sa)); // gated

        ggml_tensor* rw = W(pre + "res_skip_layers." + std::to_string(i) + ".weight");
        ggml_tensor* rs = ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, rw), acts),
                                   W(pre + "res_skip_layers." + std::to_string(i) + ".bias"));
        if (i < n_layers - 1) {
            ggml_tensor* res = ggml_cont(g, ggml_view_2d(g, rs, hidden, T, rs->nb[1], 0));
            ggml_tensor* skp = ggml_cont(g, ggml_view_2d(g, rs, hidden, T, rs->nb[1], (size_t)hidden * rs->nb[0]));
            x = ggml_add(g, x, res);
            output = output ? ggml_add(g, output, skp) : skp;
        } else {
            output = output ? ggml_add(g, output, rs) : rs;
        }
    }
    return output;
}

ggml_tensor* rvc_flow_graph_conds(ggml_context* g, rvc_svc_context* c, ggml_tensor* z_p,
                                  const std::vector<ggml_tensor*>& conds, ggml_tensor* rev_idx, int T) {
    const rvc_hparams& hp = c->hp;
    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        return it == c->t.end() ? nullptr : it->second;
    };
    ggml_tensor* x = z_p;
    const int half = hp.inter / 2;
    for (int idx = hp.flow_n_flows - 1; idx >= 0; idx--) {
        x = rvc_flip_channels(g, x, rev_idx); // Flip comes FIRST in reverse
        const std::string p = "flow.flows." + std::to_string(idx * 2) + ".";
        ggml_tensor* x0 = ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], 0));
        ggml_tensor* x1 = ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], (size_t)half * x->nb[0]));
        ggml_tensor* h =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "pre.weight")), x0), W(p + "pre.bias"));
        h = rvc_wn(g, c, p + "enc.", h, conds[idx], hp.hidden, hp.flow_n_layers, hp.flow_kernel, hp.flow_dilation_rate);
        ggml_tensor* m =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "post.weight")), h), W(p + "post.bias"));
        x1 = ggml_sub(g, x1, m); // mean_only => no exp(-logs)
        x = ggml_concat(g, x0, x1, 0);
        if (c->capture) {
            ggml_tensor* cap = ggml_cont(g, ggml_transpose(g, x));
            ggml_set_output(cap);
            c->caps["flow_c" + std::to_string(idx)] = cap;
        }
    }
    return x;
}

ggml_tensor* rvc_flow_graph(ggml_context* g, rvc_svc_context* c, ggml_tensor* z_p, ggml_tensor* gcond,
                            ggml_tensor* rev_idx, int T) {
    const rvc_hparams& hp = c->hp;
    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        return it == c->t.end() ? nullptr : it->second;
    };
    ggml_tensor* x = z_p;
    const int half = hp.inter / 2;
    for (int idx = hp.flow_n_flows - 1; idx >= 0; idx--) {
        x = rvc_flip_channels(g, x, rev_idx); // Flip comes FIRST in reverse
        const std::string p = "flow.flows." + std::to_string(idx * 2) + ".";
        ggml_tensor* x0 = ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], 0));
        ggml_tensor* x1 = ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], (size_t)half * x->nb[0]));

        ggml_tensor* h =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "pre.weight")), x0), W(p + "pre.bias"));
        h = rvc_wn(g, c, p + "enc.", h, gcond, hp.hidden, hp.flow_n_layers, hp.flow_kernel, hp.flow_dilation_rate);
        ggml_tensor* m =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "post.weight")), h), W(p + "post.bias"));
        x1 = ggml_sub(g, x1, m); // mean_only => no exp(-logs)
        x = ggml_concat(g, x0, x1, 0);
    }
    return x;
}

} // namespace

// ---------------------------------------------------------------------------
// dec — GeneratorNSF (NSF-HiFi-GAN)
//
// Traps (RVC_BLUEPRINT.md 2d):
//   * TWO different LeakyReLU slopes in ONE function: per-stage and ResBlock use
//     LRELU_SLOPE = 0.1, but the FINAL pre-conv_post call is a bare
//     F.leaky_relu(x) -> torch's 0.01 default (models.py:529).
//   * dec.cond HAS a bias (nn.Conv1d default). Omitting it is a constant
//     per-channel offset, invisible structurally, and cost cos 0.998 in numpy.
//   * conv_post is bias=False (models.py:484).
//   * The source is added AFTER the transpose-conv, via noise_convs[i] whose
//     stride is prod(rates[i+1:]) and padding stride/2.
//   * Each stage sums num_kernels ResBlocks and DIVIDES by num_kernels.
// ---------------------------------------------------------------------------

namespace {

ggml_tensor* rvc_dec_graph(ggml_context* g, rvc_svc_context* c, ggml_tensor* z, ggml_tensor* har, ggml_tensor* gemb,
                           int T) {
    const rvc_hparams& hp = c->hp;
    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        if (it == c->t.end()) {
            fprintf(stderr, "rvc: missing tensor %s\n", n.c_str());
            return nullptr;
        }
        return it->second;
    };
    auto TAPD = [&](const std::string& nm, ggml_tensor* v) {
        if (c->capture) {
            ggml_tensor* o = ggml_cont(g, ggml_transpose(g, v));
            ggml_set_output(o);
            c->caps[nm] = o;
        }
        return v;
    };

    // m_source: tanh(l_linear(sine)). harmonic_num=0 so this is 1 -> 1.
    ggml_tensor* hs = ggml_mul_mat(g, W("dec.m_source.l_linear.weight"), har);
    hs = ggml_tanh(g, ggml_add(g, hs, W("dec.m_source.l_linear.bias"))); // [1, T*upp]
    TAPD("dec_har_source", hs);

    // conv_pre + speaker cond (WITH its bias)
    ggml_tensor* cpw = W("dec.conv_pre.weight");
    ggml_tensor* x = ggml_conv_1d(g, cpw, ggml_cont(g, ggml_transpose(g, z)), 1, ((int)cpw->ne[0] - 1) / 2, 1);
    x = ggml_add(g, ggml_cont(g, ggml_transpose(g, x)), W("dec.conv_pre.bias"));
    TAPD("dec_conv_pre", x);
    ggml_tensor* cond =
        ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W("dec.cond.weight")), gemb), W("dec.cond.bias"));
    x = ggml_add(g, x, cond);

    const int nk = (int)hp.resblock_kernel_sizes.size();
    for (int i = 0; i < (int)hp.upsample_rates.size(); i++) {
        const int u = hp.upsample_rates[i];
        const int k = hp.upsample_kernel_sizes[i];
        x = ggml_leaky_relu(g, x, 0.1f, false); // LRELU_SLOPE
        // ConvTranspose1d(stride=u, padding=(k-u)/2). ggml_conv_transpose_1d
        // ASSERTS p0 == 0 — it has no padding support — so the torch padding is
        // expressed as a symmetric CROP of the unpadded output, which is what
        // core_convt::convt1d_crop does (and it takes/returns [C, T] directly).
        ggml_tensor* uw = W("dec.ups." + std::to_string(i) + ".weight");
        const int crop = (k - u) / 2;
        x = core_convt::convt1d_crop(g, x, uw, W("dec.ups." + std::to_string(i) + ".bias"), u, crop, crop);
        TAPD("dec_ups" + std::to_string(i), x);

        // source injection: strided conv over har
        int stride = 1;
        for (int j = i + 1; j < (int)hp.upsample_rates.size(); j++)
            stride *= hp.upsample_rates[j];
        ggml_tensor* nw = W("dec.noise_convs." + std::to_string(i) + ".weight");
        ggml_tensor* xs =
            ggml_conv_1d(g, nw, ggml_cont(g, ggml_transpose(g, hs)), stride, stride > 1 ? stride / 2 : 0, 1);
        xs = ggml_add(g, ggml_cont(g, ggml_transpose(g, xs)), W("dec.noise_convs." + std::to_string(i) + ".bias"));
        TAPD("dec_nc" + std::to_string(i), xs);
        x = ggml_add(g, x, xs);

        // core_hifigan::resblock_forward works TIME-MAJOR (T, C) while this
        // graph is channel-major, so transpose around it rather than
        // reimplementing the MRF block.
        ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, x)); // (T, C)
        ggml_tensor* acc = nullptr;
        for (int j = 0; j < nk; j++) {
            const int idx = i * nk + j;
            std::vector<int> dils(hp.resblock_dilations.begin() + (size_t)j * hp.resblock_n_dilations,
                                  hp.resblock_dilations.begin() + (size_t)(j + 1) * hp.resblock_n_dilations);
            ggml_tensor* rb = core_hifigan::resblock_forward(g, xt, c->t, "dec.resblocks." + std::to_string(idx),
                                                             hp.resblock_kernel_sizes[j], dils, 0.1f);
            acc = acc ? ggml_add(g, acc, rb) : rb;
        }
        // divide by num_kernels, then back to channel-major
        x = ggml_cont(g, ggml_transpose(g, ggml_scale(g, acc, 1.0f / (float)nk)));
    }

    x = ggml_leaky_relu(g, x, 0.01f, false); // BARE F.leaky_relu -> 0.01, NOT 0.1
    ggml_tensor* pw = W("dec.conv_post.weight");
    x = ggml_conv_1d(g, pw, ggml_cont(g, ggml_transpose(g, x)), 1, ((int)pw->ne[0] - 1) / 2, 1);
    x = ggml_cont(g, ggml_transpose(g, x)); // conv_post is bias=False
    return ggml_tanh(g, x);
}

} // namespace

// ---------------------------------------------------------------------------
// Per-stage diff. Input-aligned: the reference carries input_phone/input_pitch
// AND both noise buffers, which we replay, so the comparison is deterministic
// even though the model is stochastic.
// ---------------------------------------------------------------------------

namespace {

double rvc_cos(const float* a, const float* b, int64_t n) {
    double d = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        d += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    const double den = std::sqrt(na) * std::sqrt(nb);
    return den > 0 ? d / den : (na == 0 && nb == 0 ? 1.0 : 0.0);
}

double rvc_max_abs(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

bool rvc_ref_get(core_gguf::WeightLoad& rw, const char* name, std::vector<float>& out) {
    auto it = rw.tensors.find(name);
    if (it == rw.tensors.end() || !it->second)
        return false;
    const int64_t n = ggml_nelements(it->second);
    out.resize((size_t)n);
    ggml_backend_tensor_get(it->second, out.data(), 0, (size_t)n * sizeof(float));
    return true;
}

} // namespace

int rvc_svc_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    rvc_svc_params p = rvc_svc_default_params();
    p.use_gpu = false; // structural diff on CPU first
    rvc_svc_context* c = rvc_svc_init_from_file(model_gguf, p);
    if (!c) {
        fprintf(stderr, "rvc_diff: failed to load %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, c->backend, "rvc_ref", rw)) {
        fprintf(stderr, "rvc_diff: failed to load reference %s\n", ref_gguf);
        rvc_svc_free(c);
        return 2;
    }

    std::vector<float> in_phone, in_pitch, ref_mp, ref_logs;
    if (!rvc_ref_get(rw, "input_phone", in_phone) || !rvc_ref_get(rw, "input_pitch", in_pitch)) {
        fprintf(stderr, "rvc_diff: reference lacks input_phone/input_pitch — re-dump with the current spec\n");
        core_gguf::free_weights(rw);
        rvc_svc_free(c);
        return 2;
    }
    const rvc_hparams& hp = c->hp;
    const int T = (int)(in_phone.size() / (size_t)hp.content_dim);

    // Graph
    const size_t mem = (size_t)512 * 1024 * 1024;
    std::vector<uint8_t> buf(mem);
    ggml_init_params ip{mem, buf.data(), true};
    ggml_context* g = ggml_init(ip);
    ggml_tensor* content = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.content_dim, T);
    ggml_tensor* pitch = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_input(content);
    ggml_set_input(pitch);
    c->capture = true;
    ggml_tensor* stats = rvc_enc_p_graph(g, c, content, pitch, T);

    // flow is INPUT-ALIGNED on the reference's z_p: it is drawn with the
    // model's own noise, so recomputing it here would diverge by construction.
    std::vector<float> ref_zp;
    ggml_tensor* z_out = nullptr;
    ggml_tensor* zp_in = nullptr;
    ggml_tensor* z_in = nullptr;
    ggml_tensor* har_in = nullptr;
    ggml_tensor* audio_out = nullptr;
    if (rvc_ref_get(rw, "z_p", ref_zp)) {
        // The reference z_p is (inter, T) row-major -> TIME fastest, whereas
        // our working layout is [inter, T] with CHANNELS fastest. Declare the
        // input in the reference's order and transpose in-graph, rather than
        // uploading a transposed buffer and getting cos ~0 on a correct graph
        // (the same trap that made enc_p look broken).
        zp_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, T, hp.inter); // [T, inter]
        ggml_set_input(zp_in);
        ggml_tensor* zp_ct = ggml_cont(g, ggml_transpose(g, zp_in)); // [inter, T]
        // speaker embedding -> cond projection, done once and sliced per layer
        ggml_tensor* gemb = ggml_cont(g, ggml_view_1d(g, c->t["emb_g.weight"], hp.gin, 0)); // sid 0
        ggml_tensor* rev = ggml_new_tensor_1d(g, GGML_TYPE_I32, hp.inter);
        ggml_set_input(rev);
        c->rev_idx_tensor = rev;
        // one cond_layer per coupling block; project inside rvc_wn's caller
        std::vector<ggml_tensor*> conds;
        for (int i = 0; i < hp.flow_n_flows; i++) {
            const std::string cp = "flow.flows." + std::to_string(i * 2) + ".enc.cond_layer.weight";
            ggml_tensor* cw = rvc_conv1x1_as_linear(g, c->t[cp]);
            ggml_tensor* cb = c->t["flow.flows." + std::to_string(i * 2) + ".enc.cond_layer.bias"];
            conds.push_back(ggml_add(g, ggml_mul_mat(g, cw, gemb), cb));
        }
        z_out = rvc_flow_graph_conds(g, c, zp_ct, conds, rev, T);
        ggml_set_output(z_out);

        // dec: input-aligned on the reference z AND the host SineGen output,
        // so a flow difference cannot masquerade as a vocoder failure.
        std::vector<float> ref_z, ref_f0, ref_noise;
        if (rvc_ref_get(rw, "z", ref_z) && rvc_ref_get(rw, "input_f0", ref_f0) &&
            rvc_ref_get(rw, "noise_sine", ref_noise)) {
            z_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, T, hp.inter); // ref order
            ggml_set_input(z_in);
            har_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, (int64_t)T * hp.upp());
            ggml_set_input(har_in);
            ggml_tensor* z_ct = ggml_cont(g, ggml_transpose(g, z_in)); // [inter, T]
            audio_out = rvc_dec_graph(g, c, z_ct, har_in, gemb, T);
            ggml_set_output(audio_out);
        }
    }
    if (!stats) {
        ggml_free(g);
        core_gguf::free_weights(rw);
        rvc_svc_free(c);
        return 2;
    }
    // m_p/logs_p in the reference are (inter, T) row-major -> transpose.
    stats = ggml_cont(g, ggml_transpose(g, stats)); // [T, 2*inter]
    ggml_set_output(stats);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 8192, false);
    ggml_build_forward_expand(gf, stats);
    // The transposed tap copies are not reachable from `stats`, so expand the
    // graph over them too — otherwise gallocr never allocates their buffers and
    // ggml_backend_tensor_get aborts with "tensor buffer not set".
    for (auto& kv : c->caps)
        ggml_build_forward_expand(gf, kv.second);
    if (z_out)
        ggml_build_forward_expand(gf, z_out);
    if (audio_out)
        ggml_build_forward_expand(gf, audio_out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "rvc_diff: graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        core_gguf::free_weights(rw);
        rvc_svc_free(c);
        return 2;
    }
    ggml_backend_tensor_set(content, in_phone.data(), 0, in_phone.size() * sizeof(float));
    std::vector<int32_t> pi((size_t)T);
    for (int i = 0; i < T; i++)
        pi[(size_t)i] = (int32_t)std::lround(in_pitch[(size_t)i]);
    ggml_backend_tensor_set(pitch, pi.data(), 0, pi.size() * sizeof(int32_t));
    if (zp_in) {
        ggml_backend_tensor_set(zp_in, ref_zp.data(), 0, ref_zp.size() * sizeof(float));
        std::vector<int32_t> rev((size_t)hp.inter);
        for (int i = 0; i < hp.inter; i++)
            rev[(size_t)i] = hp.inter - 1 - i;
        ggml_backend_tensor_set(c->rev_idx_tensor, rev.data(), 0, rev.size() * sizeof(int32_t));
    }
    if (z_in && har_in) {
        std::vector<float> rz, rf0, rn;
        rvc_ref_get(rw, "z", rz);
        rvc_ref_get(rw, "input_f0", rf0);
        rvc_ref_get(rw, "noise_sine", rn);
        ggml_backend_tensor_set(z_in, rz.data(), 0, rz.size() * sizeof(float));
        std::vector<float> sine, uv;
        rvc_sine_gen(rf0.data(), T, hp.upp(), hp.sample_rate, rn.data(), hp.sine_amp, hp.add_noise_std, sine, uv);
        ggml_backend_tensor_set(har_in, sine.data(), 0, sine.size() * sizeof(float));
    }
    ggml_backend_graph_compute(c->backend, gf);

    std::vector<float> out((size_t)ggml_nelements(stats));
    ggml_backend_tensor_get(stats, out.data(), 0, out.size() * sizeof(float));

    int n_fail = 0;
    const double COS_MIN = 0.9999;
    auto report = [&](const char* stage, const float* mine, const std::vector<float>& ref) {
        const int64_t n = (int64_t)ref.size();
        const double cs = rvc_cos(mine, ref.data(), n);
        const bool ok = cs >= COS_MIN;
        if (!ok)
            n_fail++;
        if (verbosity >= 1 || !ok)
            fprintf(stderr, "  %-10s %s cos=%.8f max_abs=%.3e\n", stage, ok ? "PASS" : "FAIL", cs,
                    rvc_max_abs(mine, ref.data(), n));
    };

    fprintf(stderr, "rvc diff (T=%d, content_dim=%d, inter=%d):\n", T, hp.content_dim, hp.inter);

    // Per-stage, EARLIEST FIRST — the first FAIL is the bug (HARD RULE #2).
    {
        std::vector<std::string> order;
        order.push_back("encp_lrelu");
        // inside layer 0's attention, earliest first
        order.push_back("encp_L0_q");
        order.push_back("encp_L0_scores_norel");
        order.push_back("encp_L0_rl");
        order.push_back("encp_L0_scores");
        order.push_back("encp_L0_attn_w");
        order.push_back("encp_L0_v");
        order.push_back("encp_L0_agg");
        order.push_back("encp_L0_ctx");
        for (int l = 0; l < hp.n_layers; l++) {
            const std::string L = "encp_L" + std::to_string(l);
            order.push_back(L + "_attn");
            order.push_back(L + "_norm1");
            order.push_back(L + "_ffn");
            order.push_back(L + "_norm2");
        }
        for (int i = hp.flow_n_flows - 1; i >= 0; i--)
            order.push_back("flow_c" + std::to_string(i));
        bool first = true;
        for (const auto& nm : order) {
            auto it = c->caps.find(nm);
            std::vector<float> ref;
            if (it == c->caps.end() || !rvc_ref_get(rw, nm.c_str(), ref))
                continue;
            std::vector<float> mine((size_t)ggml_nelements(it->second));
            ggml_backend_tensor_get(it->second, mine.data(), 0, mine.size() * sizeof(float));
            const int64_t n = (int64_t)std::min(mine.size(), ref.size());
            const double cs = rvc_cos(mine.data(), ref.data(), n);
            const bool ok = cs >= COS_MIN && mine.size() == ref.size();
            if (!ok)
                n_fail++;
            if (verbosity >= 1 || !ok) {
                fprintf(stderr, "  %-16s %s cos=%.8f max_abs=%.3e (mine=%zu ref=%zu)%s\n", nm.c_str(),
                        ok ? "PASS" : "FAIL", cs, rvc_max_abs(mine.data(), ref.data(), n), mine.size(), ref.size(),
                        (!ok && first) ? "  <-- FIRST DIVERGENCE" : "");
            }
            if (!ok)
                first = false;
        }
    }
    // stats is now [T, 2*inter] with T fastest: rows 0..inter-1 are m_p,
    // inter..2*inter-1 are logs_p, each T-contiguous — matching the reference.
    if (rvc_ref_get(rw, "m_p", ref_mp))
        report("m_p", out.data(), ref_mp);
    if (rvc_ref_get(rw, "logs_p", ref_logs))
        report("logs_p", out.data() + (size_t)hp.inter * T, ref_logs);

    // dec: validate the HOST SineGen before anything is built on it.
    {
        std::vector<float> f0, noise, ref_sine;
        if (rvc_ref_get(rw, "input_f0", f0) && rvc_ref_get(rw, "noise_sine", noise) &&
            rvc_ref_get(rw, "dec_sine_raw", ref_sine)) {
            std::vector<float> sine, uv;
            rvc_sine_gen(f0.data(), T, hp.upp(), hp.sample_rate, noise.data(), hp.sine_amp, hp.add_noise_std, sine, uv);
            const int64_t n = (int64_t)std::min(sine.size(), ref_sine.size());
            const double cs = rvc_cos(sine.data(), ref_sine.data(), n);
            const bool ok = cs >= COS_MIN && sine.size() == ref_sine.size();
            if (!ok)
                n_fail++;
            fprintf(stderr, "  %-16s %s cos=%.8f max_abs=%.3e (mine=%zu ref=%zu)\n", "dec_sine_raw",
                    ok ? "PASS" : "FAIL", cs, rvc_max_abs(sine.data(), ref_sine.data(), n), sine.size(),
                    ref_sine.size());
        }
    }

    // dec stages, earliest first
    {
        std::vector<std::string> dorder = {"dec_har_source", "dec_conv_pre"};
        for (int i = 0; i < (int)hp.upsample_rates.size(); i++) {
            dorder.push_back("dec_ups" + std::to_string(i));
            dorder.push_back("dec_nc" + std::to_string(i));
        }
        for (const auto& nm : dorder) {
            auto it = c->caps.find(nm);
            std::vector<float> ref;
            if (it == c->caps.end() || !rvc_ref_get(rw, nm.c_str(), ref))
                continue;
            std::vector<float> mine((size_t)ggml_nelements(it->second));
            ggml_backend_tensor_get(it->second, mine.data(), 0, mine.size() * sizeof(float));
            const int64_t n = (int64_t)std::min(mine.size(), ref.size());
            const double cs = rvc_cos(mine.data(), ref.data(), n);
            const bool ok = cs >= COS_MIN && mine.size() == ref.size();
            if (!ok)
                n_fail++;
            fprintf(stderr, "  %-16s %s cos=%.8f max_abs=%.3e (mine=%zu ref=%zu)\n", nm.c_str(), ok ? "PASS" : "FAIL",
                    cs, rvc_max_abs(mine.data(), ref.data(), n), mine.size(), ref.size());
        }
    }
    // END-TO-END through the REAL entry point. The per-stage checks above are
    // all input-aligned (they feed the reference's z_p / z / har), so none of
    // them exercises the CHAINING. This runs rvc_svc_convert() with the
    // reference's own noise buffers: if the chaining is right it must reproduce
    // the reference audio.
    {
        std::vector<float> ph, f0, nzp, nsine, ref_audio;
        if (rvc_ref_get(rw, "input_phone", ph) && rvc_ref_get(rw, "input_f0", f0) && rvc_ref_get(rw, "noise_zp", nzp) &&
            rvc_ref_get(rw, "noise_sine", nsine) && rvc_ref_get(rw, "output_audio", ref_audio)) {
            rvc_svc_result* res = rvc_svc_convert(c, ph.data(), T, f0.data(), 0, nzp.data(), nsine.data());
            if (!res) {
                fprintf(stderr, "  %-16s FAIL (convert returned null)\n", "convert_e2e");
                n_fail++;
            } else {
                const int64_t n = (int64_t)std::min((size_t)res->n_samples, ref_audio.size());
                const double cs = rvc_cos(res->pcm, ref_audio.data(), n);
                const bool ok = cs >= COS_MIN && (size_t)res->n_samples == ref_audio.size();
                if (!ok)
                    n_fail++;
                fprintf(stderr, "  %-16s %s cos=%.8f max_abs=%.3e (mine=%d ref=%zu)\n", "convert_e2e",
                        ok ? "PASS" : "FAIL", cs, rvc_max_abs(res->pcm, ref_audio.data(), n), res->n_samples,
                        ref_audio.size());
                rvc_svc_result_free(res);
            }
        }
    }

    if (audio_out) {
        std::vector<float> ref_audio;
        if (rvc_ref_get(rw, "output_audio", ref_audio)) {
            std::vector<float> mine((size_t)ggml_nelements(audio_out));
            ggml_backend_tensor_get(audio_out, mine.data(), 0, mine.size() * sizeof(float));
            report("output_audio", mine.data(), ref_audio);
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(g);
    core_gguf::free_weights(rw);
    rvc_svc_free(c);
    fprintf(stderr, "rvc diff: %d stage(s) FAILED.\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// convert() — the real entry point: enc_p -> sample z_p -> flow -> dec.
//
// One graph, one compute. The z_p sample happens IN-GRAPH from an uploaded
// noise tensor, so passing the reference's buffers reproduces the reference
// audio bit-for-bit; passing NULL draws fresh noise, which is what production
// wants (the model is stochastic by design).
// ---------------------------------------------------------------------------

rvc_svc_result* rvc_svc_convert(rvc_svc_context* c, const float* content, int n_frames, const float* f0_hz,
                                int speaker_id, const float* noise_zp, const float* noise_sine) {
    if (!c || !content || !f0_hz || n_frames <= 0)
        return nullptr;
    const rvc_hparams& hp = c->hp;
    if (speaker_id < 0 || speaker_id >= hp.n_speakers) {
        fprintf(stderr, "rvc: speaker_id %d out of range (0..%d)\n", speaker_id, hp.n_speakers - 1);
        return nullptr;
    }
    const int T = n_frames;
    const int upp = hp.upp();

    // Noise: supplied (deterministic replay) or drawn.
    std::vector<float> zp_noise((size_t)hp.inter * T), sine_noise((size_t)T * upp);
    if (noise_zp) {
        std::memcpy(zp_noise.data(), noise_zp, zp_noise.size() * sizeof(float));
    } else {
        std::mt19937 rng{std::random_device{}()};
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : zp_noise)
            v = nd(rng);
    }
    if (noise_sine) {
        std::memcpy(sine_noise.data(), noise_sine, sine_noise.size() * sizeof(float));
    } else {
        std::mt19937 rng{std::random_device{}()};
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : sine_noise)
            v = nd(rng);
    }

    // SineGen on the host (see rvc_sine_gen).
    std::vector<float> sine, uv;
    rvc_sine_gen(f0_hz, T, upp, hp.sample_rate, sine_noise.data(), hp.sine_amp, hp.add_noise_std, sine, uv);

    std::vector<int> coarse((size_t)T);
    rvc_svc_coarse_pitch(f0_hz, T, coarse.data());

    const size_t mem = (size_t)1024 * 1024 * 1024;
    std::vector<uint8_t> buf(mem);
    ggml_init_params ip{mem, buf.data(), true};
    ggml_context* g = ggml_init(ip);
    if (!g)
        return nullptr;

    ggml_tensor* t_content = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.content_dim, T);
    ggml_tensor* t_pitch = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    // The z_p noise is (inter, T) row-major on the wire (TIME fastest) — that
    // is how the reference dumps it and how a caller naturally lays it out —
    // whereas our graph layout is channels-fastest. Declare it in wire order
    // and transpose in-graph. Feeding it raw gave cos 0.40 end-to-end while
    // every per-stage check still passed, because the stage checks are all
    // input-aligned and never exercise this path.
    ggml_tensor* t_noise = ggml_new_tensor_2d(g, GGML_TYPE_F32, T, hp.inter);
    ggml_tensor* t_har = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, (int64_t)T * upp);
    ggml_tensor* t_rev = ggml_new_tensor_1d(g, GGML_TYPE_I32, hp.inter);
    for (ggml_tensor* t : {t_content, t_pitch, t_noise, t_har, t_rev})
        ggml_set_input(t);

    c->capture = false; // no taps on the production path
    ggml_tensor* stats = rvc_enc_p_graph(g, c, t_content, t_pitch, T);
    if (!stats) {
        ggml_free(g);
        return nullptr;
    }
    // stats is [2*inter, T]; split channel-wise.
    ggml_tensor* m_p = ggml_cont(g, ggml_view_2d(g, stats, hp.inter, T, stats->nb[1], 0));
    ggml_tensor* logs_p =
        ggml_cont(g, ggml_view_2d(g, stats, hp.inter, T, stats->nb[1], (size_t)hp.inter * stats->nb[0]));
    // z_p = m_p + exp(logs_p) * noise * noise_scale   (models.py:684)
    ggml_tensor* noise_ct = ggml_cont(g, ggml_transpose(g, t_noise)); // [inter, T]
    ggml_tensor* z_p = ggml_add(g, m_p, ggml_scale(g, ggml_mul(g, ggml_exp(g, logs_p), noise_ct), hp.noise_scale));

    ggml_tensor* gemb = ggml_cont(
        g, ggml_view_1d(g, c->t["emb_g.weight"], hp.gin, (size_t)speaker_id * hp.gin * c->t["emb_g.weight"]->nb[0]));
    std::vector<ggml_tensor*> conds;
    for (int i = 0; i < hp.flow_n_flows; i++) {
        const std::string b = "flow.flows." + std::to_string(i * 2) + ".enc.cond_layer.";
        conds.push_back(
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, c->t[b + "weight"]), gemb), c->t[b + "bias"]));
    }
    ggml_tensor* z = rvc_flow_graph_conds(g, c, z_p, conds, t_rev, T);
    ggml_tensor* audio = rvc_dec_graph(g, c, z, t_har, gemb, T);
    ggml_set_output(audio);

    ggml_cgraph* gf = ggml_new_graph_custom(g, 16384, false);
    ggml_build_forward_expand(gf, audio);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "rvc: graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return nullptr;
    }

    ggml_backend_tensor_set(t_content, content, 0, (size_t)hp.content_dim * T * sizeof(float));
    std::vector<int32_t> pi(coarse.begin(), coarse.end());
    ggml_backend_tensor_set(t_pitch, pi.data(), 0, pi.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_noise, zp_noise.data(), 0, zp_noise.size() * sizeof(float));
    ggml_backend_tensor_set(t_har, sine.data(), 0, sine.size() * sizeof(float));
    std::vector<int32_t> rev((size_t)hp.inter);
    for (int i = 0; i < hp.inter; i++)
        rev[(size_t)i] = hp.inter - 1 - i;
    ggml_backend_tensor_set(t_rev, rev.data(), 0, rev.size() * sizeof(int32_t));

    ggml_backend_graph_compute(c->backend, gf);

    const int64_t n_out = ggml_nelements(audio);
    auto* r = new rvc_svc_result();
    r->n_samples = (int)n_out;
    r->pcm = (float*)malloc((size_t)n_out * sizeof(float));
    ggml_backend_tensor_get(audio, r->pcm, 0, (size_t)n_out * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(g);
    return r;
}

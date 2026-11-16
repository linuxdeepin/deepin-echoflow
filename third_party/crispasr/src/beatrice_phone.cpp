// beatrice_phone.cpp — see beatrice_phone.h.
//
// Built against tools/beatrice_torch_parity.py --component phone_extractor.
// Layout convention and the shared ConvNeXt primitives are in
// core/beatrice_ops.h; read that first.

#include "beatrice_phone.h"

#include "core/beatrice_ops.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

using namespace beatrice_ops;

struct beatrice_phone_hparams {
    int phone_channels = 128;
    int hidden = 128;
    int n_blocks = 20;
    int n_heads = 4;
    int sample_rate = 16000;
    int hop_length = 160;
    // ConvNeXtStack(embed_kernel_size=9, kernel_size=17, delay=0)
    int embed_padding = 8, embed_trim = 8;
    int dw_padding = 16, dw_trim = 16;
};

struct beatrice_phone_context {
    beatrice_phone_hparams hp;
    beatrice_phone_params params;
    ggml_backend_t backend = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> t;
    bool capture = false;
    std::map<std::string, ggml_tensor*> caps;
};

namespace {

// One head-split self-attention over a single [C, T] subsequence.
//
// Written out rather than routed through core/attention.h's encoder_self_attn
// because that helper uses ggml_flash_attn_ext, which is 2-D only and carries
// mask-padding requirements; here there are four independent subsequences per
// block and a plain F32 causal mask is easier to validate stage by stage.
//
// `in_proj` is torch's FUSED [C, 3C] MultiheadAttention weight, split by view.
// The scale is the ordinary 1/sqrt(head_dim): the `.beatrice` dump format
// pre-folds sqrt(head_dim) into BOTH q and k, but we convert from the .pt
// state_dict, so applying it again would scale logits by 1/head_dim.
ggml_tensor* mha_subsequence(ggml_context* ctx, ggml_tensor* x /*[C,T]*/, ggml_tensor* in_w, ggml_tensor* in_b,
                             ggml_tensor* out_w, ggml_tensor* out_b, ggml_tensor* mask, int C, int n_heads) {
    const int hd = C / n_heads;
    const int T = (int)x->ne[1];

    auto proj = [&](int idx) {
        ggml_tensor* w = ggml_view_2d(ctx, in_w, C, C, in_w->nb[1], (size_t)idx * C * in_w->nb[1]);
        ggml_tensor* y = ggml_mul_mat(ctx, w, x); // [C, T]
        if (in_b) {
            ggml_tensor* b = ggml_view_1d(ctx, in_b, C, (size_t)idx * C * in_b->nb[0]);
            y = ggml_add(ctx, y, b);
        }
        return y;
    };
    ggml_tensor* Q = proj(0);
    ggml_tensor* K = proj(1);
    ggml_tensor* V = proj(2);

    // [C, T] -> [hd, n_heads, T] -> [hd, T, n_heads]
    Q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, Q, hd, n_heads, T), 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, K, hd, n_heads, T), 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, V, hd, n_heads, T), 0, 2, 1, 3));

    // scores[k, q, head] — mul_mat contracts over ne0 = hd
    ggml_tensor* scores = ggml_mul_mat(ctx, K, Q);
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f / std::sqrt((float)hd), 0.0f);

    // out[hd, q, head] = sum_k V[k, hd] * scores[k, q]
    ggml_tensor* Vt = ggml_cont(ctx, ggml_transpose(ctx, V)); // [T, hd, n_heads]
    ggml_tensor* out = ggml_mul_mat(ctx, Vt, scores);         // [hd, T, n_heads]

    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)); // [hd, n_heads, T]
    out = ggml_reshape_2d(ctx, out, C, T);
    out = ggml_mul_mat(ctx, out_w, out);
    if (out_b)
        out = ggml_add(ctx, out, out_b);
    return out; // [C, T]
}

struct graph_io {
    ggml_tensor* wav = nullptr;   // [n_samples, 1]
    ggml_tensor* mask = nullptr;  // [T4, T4] causal
    ggml_tensor* units = nullptr; // [T, phone_channels]
};

ggml_cgraph* build_graph(beatrice_phone_context* c, ggml_context* ctx, int n_samples, int& T_out, int& T4_out,
                         graph_io& io) {
    const beatrice_phone_hparams& hp = c->hp;
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);

    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        return it == c->t.end() ? nullptr : it->second;
    };
    auto cap = [&](const std::string& name, ggml_tensor* t) {
        if (c->capture) {
            ggml_set_name(t, name.c_str());
            ggml_set_output(t); // REQUIRED — else the allocator recycles it
            c->caps[name] = t;
            ggml_build_forward_expand(gf, t);
        }
        return t;
    };

    io.wav = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_samples, 1);
    ggml_set_input(io.wav);

    // --- FeatureExtractor: 6 strided convs, no bias, /160 overall.
    // The reference pads the RAW WAVEFORM by (40, 40) once, before conv0.
    ggml_tensor* x = ggml_pad(ctx, io.wav, 40, 0, 0, 0);
    // ggml_pad only appends, so shift the content right by 40 to get symmetric
    // padding: build [40 zeros | wav | 40 zeros] explicitly.
    {
        ggml_tensor* padded = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_samples + 80, 1);
        ggml_set_input(padded);
        io.wav = padded; // the host writes the already-padded buffer
        x = padded;
    }
    const int strides[6] = {5, 2, 2, 2, 2, 2};
    for (int i = 0; i < 6; i++) {
        ggml_tensor* w = W("feature_extractor.conv" + std::to_string(i) + ".weight");
        x = ggml_conv_1d(ctx, w, x, strides[i], 0, 1); // bias=False upstream
        x = ggml_gelu(ctx, x);
        cap("fe_conv" + std::to_string(i), x);
    }

    // --- FeatureProjection: LayerNorm WITH affine, kept explicit (not folded)
    x = layernorm_tc(ctx, x, W("feature_projection.norm.weight"), W("feature_projection.norm.bias"));
    cap("feature_projection", x);

    // --- ConvNeXtStack
    x = ggml_conv_1d(ctx, W("backbone.embed.weight"), x, 1, hp.embed_padding, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, W("backbone.embed.bias"), 1, hp.hidden));
    x = trim_tail(ctx, x, hp.embed_trim);
    cap("backbone_embed", x);
    x = layernorm_tc(ctx, x, W("backbone.norm.weight"), W("backbone.norm.bias"));
    cap("backbone_norm", x);

    const int T = (int)x->ne[0];
    T_out = T;
    const int pad_len = (4 - (T % 4)) % 4;
    if (pad_len)
        x = ggml_pad(ctx, x, pad_len, 0, 0, 0); // append zeros on the time axis
    const int Tp = T + pad_len;
    const int T4 = Tp / 4;
    T4_out = T4;

    io.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T4, T4);
    ggml_set_input(io.mask);

    for (int i = 0; i < hp.n_blocks; i++) {
        const std::string p = "backbone.convnext." + std::to_string(i) + ".";
        ggml_tensor* identity = x;

        // STRIDED interleave: frame t -> subsequence t%4 at position t//4.
        // In this layout ne0 is time, so reshaping [Tp, C] to [4, T4, C] puts
        // s = t%4 in ne0 for free -- t = 4*j + s means s varies fastest. A
        // CHUNKED split (contiguous quarters) is a different function: it
        // scores cos 0.693 against the reference.
        ggml_tensor* x3 = ggml_reshape_3d(ctx, x, 4, T4, hp.hidden);
        // -> [C, T4, 4]: send c to dim0, j to dim1, s to dim2
        ggml_tensor* sub = ggml_cont(ctx, ggml_permute(ctx, x3, 2, 1, 0, 3));

        std::vector<ggml_tensor*> outs(4);
        for (int s = 0; s < 4; s++) {
            ggml_tensor* xs =
                ggml_cont(ctx, ggml_view_2d(ctx, sub, hp.hidden, T4, sub->nb[1], (size_t)s * sub->nb[2])); // [C, T4]
            // attn_norm's affine was folded into in_proj by merge_weights and is
            // absent from the GGUF, so this normalises only.
            ggml_tensor* h = ggml_norm(ctx, xs, kLayerNormEps);
            outs[(size_t)s] = mha_subsequence(ctx, h, W(p + "mha.in_proj_weight"), W(p + "mha.in_proj_bias"),
                                              W(p + "mha.out_proj.weight"), W(p + "mha.out_proj.bias"), io.mask,
                                              hp.hidden, hp.n_heads);
        }
        // re-interleave: [C, T4] x4 -> [C, T4, 4] -> [4, T4, C] -> [Tp, C]
        ggml_tensor* cat = ggml_concat(ctx, ggml_concat(ctx, outs[0], outs[1], 2),
                                       ggml_concat(ctx, outs[2], outs[3], 2), 2); // [C, T4, 4]
        ggml_tensor* back = ggml_cont(ctx, ggml_permute(ctx, cat, 2, 1, 0, 3));   // [4, T4, C]
        back = ggml_reshape_2d(ctx, back, Tp, hp.hidden);
        // Capture the BRANCH before the residual. The post-residual sum is
        // residual-dominated and nearly blind to this branch -- a broken
        // interleave scored cos 0.99999986 on it while breaking 41 later
        // stages -- so the delta is the stage that actually tests the MHA.
        // ggml_cont is REQUIRED here, not tidiness: `back` is a reshape VIEW,
        // and ggml_set_output on a view does not stop the allocator recycling
        // the parent's buffer. Capturing the view read recycled memory and
        // reported cos -0.012 on a branch that was in fact correct -- while the
        // post-residual stage it feeds passed at cos 0.9999999. Two stages that
        // cannot both be true is the tell.
        cap("pblock" + std::to_string(i) + "_attn_delta", ggml_cont(ctx, back));
        x = ggml_add(ctx, back, identity);
        cap("pblock" + std::to_string(i) + "_attn", x);

        ggml_tensor* dwout = nullptr;
        x = convnext_conv_branch(ctx, x, W(p + "dwconv.weight"), W(p + "dwconv.bias"), W(p + "pwconv1.weight"),
                                 W(p + "pwconv1.bias"), W(p + "pwconv2.weight"), W(p + "pwconv2.bias"), hp.dw_padding,
                                 hp.dw_trim, hp.hidden, &dwout);
        cap("pblock" + std::to_string(i) + "_out", x);
    }

    if (pad_len)
        x = trim_tail(ctx, x, pad_len);
    x = layernorm_tc(ctx, x, W("backbone.final_layer_norm.weight"), W("backbone.final_layer_norm.bias"));
    cap("backbone_final_norm", x);

    // head is applied AFTER a gelu, unlike the pitch estimator's
    x = ggml_gelu(ctx, x);
    ggml_tensor* hw = W("head.weight"); // [1, C, phone] after weight-norm removal
    io.units = pointwise(ctx, x, ggml_reshape_2d(ctx, hw, hw->ne[1], hw->ne[2]), W("head.bias"));
    cap("phone", io.units);
    ggml_build_forward_expand(gf, io.units);
    return gf;
}

} // namespace

beatrice_phone_params beatrice_phone_default_params(void) {
    beatrice_phone_params p{};
    p.n_threads = 0;
    p.use_gpu = true;
    p.gpu_device = 0;
    return p;
}

int beatrice_phone_channels(const beatrice_phone_context* c) {
    return c ? c->hp.phone_channels : 0;
}
int beatrice_phone_sample_rate(const beatrice_phone_context* c) {
    return c ? c->hp.sample_rate : 0;
}
int beatrice_phone_hop_length(const beatrice_phone_context* c) {
    return c ? c->hp.hop_length : 0;
}

beatrice_phone_context* beatrice_phone_init_from_file(const char* model_path, beatrice_phone_params params) {
    gguf_context* meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "beatrice-phone: cannot open %s\n", model_path);
        return nullptr;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    const std::string comp = core_gguf::kv_str(meta, "beatrice.component", "");
    if (arch != "beatrice" || comp != "phone_extractor") {
        fprintf(stderr, "beatrice-phone: '%s' is arch='%s' component='%s', need beatrice/phone_extractor\n", model_path,
                arch.c_str(), comp.c_str());
        core_gguf::free_metadata(meta);
        return nullptr;
    }
    auto* c = new beatrice_phone_context();
    c->params = params;
    beatrice_phone_hparams& hp = c->hp;
    hp.phone_channels = core_gguf::kv_u32(meta, "beatrice.phone_extractor.phone_channels", hp.phone_channels);
    hp.hidden = core_gguf::kv_u32(meta, "beatrice.phone_extractor.hidden_channels", hp.hidden);
    hp.n_blocks = core_gguf::kv_u32(meta, "beatrice.phone_extractor.n_blocks", hp.n_blocks);
    hp.n_heads = core_gguf::kv_u32(meta, "beatrice.phone_extractor.n_heads", hp.n_heads);
    hp.sample_rate = core_gguf::kv_u32(meta, "beatrice.in_sample_rate", hp.sample_rate);
    core_gguf::free_metadata(meta);

    if (hp.hidden % hp.n_heads != 0) {
        fprintf(stderr, "beatrice-phone: hidden %d not divisible by n_heads %d\n", hp.hidden, hp.n_heads);
        delete c;
        return nullptr;
    }

    c->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!c->backend)
        c->backend = core_cpu_backend::init();

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, c->backend, "beatrice", wl)) {
        fprintf(stderr, "beatrice-phone: failed to load weights from %s\n", model_path);
        beatrice_phone_free(c);
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->t = wl.tensors;

    fprintf(stderr, "beatrice/phone_extractor: phone=%d hidden=%d blocks=%d heads=%d sr=%d (%d fps)\n",
            hp.phone_channels, hp.hidden, hp.n_blocks, hp.n_heads, hp.sample_rate, hp.sample_rate / hp.hop_length);
    return c;
}

void beatrice_phone_free(beatrice_phone_context* c) {
    if (!c)
        return;
    if (c->buf_w)
        core_gguf::release_weight_buffer(c->buf_w);
    if (c->ctx_w)
        ggml_free(c->ctx_w);
    if (c->backend)
        ggml_backend_free(c->backend);
    delete c;
}

namespace {

// Causal mask for soft_max_ext: 0 where attendable, -INF above the diagonal.
// The reference builds ones(t4, t4).triu(1) as a BOOL mask, i.e. position q may
// attend to k <= q.
std::vector<float> causal_mask(int T4) {
    std::vector<float> m((size_t)T4 * T4, 0.0f);
    for (int q = 0; q < T4; q++)
        for (int k = 0; k < T4; k++)
            if (k > q)
                m[(size_t)q * T4 + k] = -INFINITY;
    return m;
}

std::vector<float> pad_wav(const float* pcm, int n) {
    std::vector<float> v((size_t)n + 80, 0.0f);
    std::memcpy(v.data() + 40, pcm, (size_t)n * sizeof(float));
    return v;
}

} // namespace

beatrice_phone_result* beatrice_phone_extract(beatrice_phone_context* c, const float* pcm, int n_samples) {
    if (!c || !pcm || n_samples < 160)
        return nullptr;
    const int n160 = (n_samples / 160) * 160; // partial frames are not defined upstream

    const size_t mem = (size_t)3 * 1024 * 1024 * 1024;
    ggml_init_params ip{mem, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    graph_io io;
    int T = 0, T4 = 0;
    ggml_cgraph* gf = build_graph(c, ctx, n160, T, T4, io);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    const std::vector<float> wav = pad_wav(pcm, n160);
    ggml_backend_tensor_set(io.wav, wav.data(), 0, wav.size() * sizeof(float));
    const std::vector<float> mask = causal_mask(T4);
    ggml_backend_tensor_set(io.mask, mask.data(), 0, mask.size() * sizeof(float));
    if (c->params.n_threads > 0)
        core_cpu_backend::set_n_threads(c->backend, c->params.n_threads);
    ggml_backend_graph_compute(c->backend, gf);

    auto* r = new beatrice_phone_result();
    r->n_frames = (int)io.units->ne[0];
    r->n_channels = c->hp.phone_channels;
    r->units = (float*)malloc((size_t)r->n_frames * r->n_channels * sizeof(float));
    ggml_backend_tensor_get(io.units, r->units, 0, (size_t)r->n_frames * r->n_channels * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return r;
}

void beatrice_phone_result_free(beatrice_phone_result* r) {
    if (!r)
        return;
    free(r->units);
    delete r;
}

// ---------------------------------------------------------------------------
// Parity harness
// ---------------------------------------------------------------------------
namespace {

bool ref_get(ggml_context* rctx, const char* name, std::vector<float>& out) {
    ggml_tensor* t = ggml_get_tensor(rctx, (std::string("ref.") + name).c_str());
    if (!t)
        return false;
    out.resize((size_t)ggml_nelements(t));
    std::memcpy(out.data(), t->data, out.size() * sizeof(float));
    return true;
}

double cos_sim(const float* a, const float* b, int64_t n) {
    double d = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        d += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return d / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

double max_abs(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

} // namespace

int beatrice_phone_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    beatrice_phone_params p = beatrice_phone_default_params();
    p.use_gpu = false;
    beatrice_phone_context* c = beatrice_phone_init_from_file(model_gguf, p);
    if (!c)
        return 2;

    ggml_context* rctx = nullptr;
    gguf_init_params gp{false, &rctx};
    gguf_context* rg = gguf_init_from_file(ref_gguf, gp);
    if (!rg) {
        fprintf(stderr, "beatrice-phone: cannot open reference %s\n", ref_gguf);
        beatrice_phone_free(c);
        return 2;
    }
    std::vector<float> ref_wav;
    if (!ref_get(rctx, "input_wav", ref_wav)) {
        fprintf(stderr, "beatrice-phone: reference lacks input_wav\n");
        gguf_free(rg);
        beatrice_phone_free(c);
        return 2;
    }

    int n_fail = 0;
    bool flagged_first = false;
    // The gate is cosine; `rel` is REPORTED but deliberately NOT gated on.
    //
    // A tighter relative gate was tried and reverted: at 1e-4 the CONTROL fails
    // 65 stages, because f32 noise on the waveform convolutions legitimately
    // reaches rel 2.7e-04 (fe_conv0) and 1.3e-03 end to end. No single global
    // threshold separates signal from noise here -- a broken interleave carries
    // 6.3e-04 at pblock*_attn_delta, which sits BELOW the control's own noise at
    // other stages.
    //
    // So read the numbers, not just PASS/FAIL. On the attn_delta stages -- the
    // sensitive probe for the MHA -- the control runs 4e-07 (block 0) to ~3e-05
    // (deeper blocks), while a chunked-instead-of-strided interleave or a
    // missing causal mask jumps to 6e-04 and trips 60 stages downstream. Both
    // negative controls ARE caught; they simply are not caught at block 0.
    auto report = [&](const char* nm, const float* mine, const std::vector<float>& ref, size_t n_mine) {
        const int64_t n = (int64_t)std::min(n_mine, ref.size());
        const double cs = cos_sim(mine, ref.data(), n);
        const double ma = max_abs(mine, ref.data(), n);
        double peak = 0.0;
        for (int64_t i = 0; i < n; i++)
            peak = std::max(peak, std::fabs((double)ref[(size_t)i]));
        const double rel = peak > 0 ? ma / peak : 0.0;
        const bool ok = cs >= 0.9999 && n_mine == ref.size();
        if (!ok)
            n_fail++;
        if (verbosity >= 1 || !ok) {
            const bool first = !ok && !flagged_first;
            fprintf(stderr, "  %-24s %s cos=%.8f max_abs=%.3e rel=%.2e (mine=%zu ref=%zu)%s\n", nm,
                    ok ? "PASS" : "FAIL", cs, ma, rel, n_mine, ref.size(), first ? "  <-- FIRST DIVERGENCE" : "");
            if (first)
                flagged_first = true;
        }
    };

    c->capture = true;
    const size_t mem = (size_t)3 * 1024 * 1024 * 1024;
    ggml_init_params ip{mem, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    graph_io io;
    int T = 0, T4 = 0;
    const int n160 = ((int)ref_wav.size() / 160) * 160;
    ggml_cgraph* gf = build_graph(c, ctx, n160, T, T4, io);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    const std::vector<float> wav = pad_wav(ref_wav.data(), n160);
    ggml_backend_tensor_set(io.wav, wav.data(), 0, wav.size() * sizeof(float));
    const std::vector<float> mask = causal_mask(T4);
    ggml_backend_tensor_set(io.mask, mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_graph_compute(c->backend, gf);

    // Earliest first — the FIRST failure is the bug (HARD RULE #2).
    std::vector<std::string> order;
    for (int i = 0; i < 6; i++)
        order.push_back("fe_conv" + std::to_string(i));
    order.push_back("feature_projection");
    order.push_back("backbone_embed");
    order.push_back("backbone_norm");
    for (int i = 0; i < c->hp.n_blocks; i++) {
        order.push_back("pblock" + std::to_string(i) + "_attn_delta");
        order.push_back("pblock" + std::to_string(i) + "_attn");
        order.push_back("pblock" + std::to_string(i) + "_out");
    }
    order.push_back("backbone_final_norm");
    order.push_back("phone");

    for (const auto& nm : order) {
        auto it = c->caps.find(nm);
        std::vector<float> ref;
        if (it == c->caps.end() || !ref_get(rctx, nm.c_str(), ref))
            continue;
        std::vector<float> mine((size_t)ggml_nelements(it->second));
        ggml_backend_tensor_get(it->second, mine.data(), 0, mine.size() * sizeof(float));
        report(nm.c_str(), mine.data(), ref, mine.size());
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    c->capture = false;

    // END TO END through the public API — the per-stage checks above are all
    // input-aligned and never test the wiring between them.
    {
        std::vector<float> ref_phone;
        beatrice_phone_result* r = beatrice_phone_extract(c, ref_wav.data(), (int)ref_wav.size());
        if (!r) {
            fprintf(stderr, "  %-24s FAIL (extract returned null)\n", "extract_e2e");
            n_fail++;
        } else {
            if (ref_get(rctx, "phone", ref_phone))
                report("extract_e2e_phone", r->units, ref_phone, (size_t)r->n_frames * r->n_channels);
            beatrice_phone_result_free(r);
        }
    }

    fprintf(stderr, "beatrice-phone parity: %d stage(s) FAILED\n", n_fail);
    gguf_free(rg);
    beatrice_phone_free(c);
    return n_fail ? 1 : 0;
}

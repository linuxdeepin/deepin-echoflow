// beatrice_pitch.cpp — see beatrice_pitch.h.
//
// Built against tools/beatrice_torch_parity.py. Where this file looks odd, the
// blueprint (docs/music-transcription/BEATRICE_BLUEPRINT.md) says why.
//
// LAYOUT. The canonical layout here is ggml ne = [time, channels] (time
// fastest), which is byte-for-byte what torch's [batch, channels, time] dumps
// to. Every captured stage is in that layout, so the diff harness compares flat
// buffers with NO transpose on either side. Three separate RVC-port bugs came
// from transposing at this boundary; each produced ~0 cosine on a graph that was
// in fact correct.
//
// The blocks transpose to [channels, time] internally for LayerNorm and the
// pointwise convs (ggml_norm normalises over ne0, and mul_mat contracts over
// ne0) and back again — exactly the transpose(1, 2) dance the torch source
// does, for the same reason.

#include "beatrice_pitch.h"

#include "core/beatrice_ops.h"
#include "core/fft.h"
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

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------
struct beatrice_pitch_hparams {
    int pitch_bins = 448;
    int channels = 192;
    int n_blocks = 9;
    int bins_per_octave = 96;
    int sample_rate = 16000;
    // extract_pitch_features constants. win = max_corr_period + corr_win_length
    // is an invariant the reference itself asserts.
    int hop_length = 160;
    int win_length = 560;
    int max_corr_period = 256;
    int corr_win_length = 304;
    int instfreq_cutoff_bin = 64;
    // CausalConv1d geometry. padding != trim whenever delay > 0, so both are
    // carried rather than re-derived.
    int embed_kernel_size = 3, embed_padding = 1, embed_trim = 0;
    int dw_kernel_size = 33, dw_padding = 32, dw_trim = 32;
};

struct beatrice_pitch_context {
    beatrice_pitch_hparams hp;
    beatrice_pitch_params params;
    ggml_backend_t backend = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> t;
    // Per-stage capture for the diff harness (HARD RULE #2: intermediates, not
    // just endpoints — an endpoint alone cannot localise anything).
    bool capture = false;
    std::map<std::string, ggml_tensor*> caps;
};

namespace {

// ---------------------------------------------------------------------------
// DSP front end (extract_pitch_features), host side
// ---------------------------------------------------------------------------
//
// Host side rather than in the graph because it is pure DSP over short frames:
// an rFFT of 560 (NOT a power of two — 560 = 2^4 * 35, so the mixed-radix path
// in core/fft.h is required; the radix-2 one would overrun) plus a direct
// correlation. Same call as rvc_sine_gen.
//
// The reference computes the correlation as irfft(rfft(flip(y)) * rfft(y[-304:]))
// and slices [304:]. That reduces exactly to a plain time-domain correlation:
//
//     corr[t] = sum_{j=0}^{303} y[256+j] * y[256+j-(t+1)]
//
// verified against torch at 1e-15 relative (double round-off). So no inverse
// FFT is needed anywhere, and the difference function below is formed directly
// as sum (y[a+j] - y[a-lag+j])^2 rather than as energy0 + energy - 2*corr. The
// two are algebraically identical, but the direct form cannot suffer the
// catastrophic cancellation that the reference's `clamp(min=0)` exists to paper
// over.
// LAYOUT: channel-major, TIME FASTEST -- element (c, t) lives at c*n_frames + t,
// which is ggml ne = [T, C] and byte-identical to the reference dump. Writing
// these frame-major (t*C + c) is the natural way to fill them frame by frame and
// is WRONG; it cost a debugging round here even with the warning at the top of
// this file. The tell was that dsp_energy passed while the other two failed --
// energy has one channel, so it is the only layout-invariant stage.
struct pitch_features {
    std::vector<float> instfreq;  // [192][n_frames]
    std::vector<float> corr_diff; // [256][n_frames]
    std::vector<float> energy;    // [n_frames]
    int n_frames = 0;
};

void extract_pitch_features(const beatrice_pitch_hparams& hp, const float* pcm, int n_samples, pitch_features& out) {
    const int win = hp.win_length, hop = hp.hop_length;
    const int cutoff = hp.instfreq_cutoff_bin;
    const int max_corr = hp.max_corr_period, corr_win = hp.corr_win_length;
    const int pad = (win - hop) / 2;

    // Reflectionless zero padding, matching F.pad(y, (pad, pad)) default mode.
    std::vector<float> y((size_t)n_samples + 2 * (size_t)pad, 0.0f);
    std::memcpy(y.data() + pad, pcm, (size_t)n_samples * sizeof(float));

    const int total = (int)y.size();
    const int n_frames = total >= win ? (total - win) / hop + 1 : 0;
    out.n_frames = n_frames;
    if (n_frames <= 0)
        return;

    out.instfreq.assign((size_t)n_frames * 3 * cutoff, 0.0f);
    out.corr_diff.assign((size_t)n_frames * max_corr, 0.0f);
    out.energy.assign((size_t)n_frames, 0.0f);

    std::vector<float> spec(2 * (size_t)win);
    std::vector<float> prev_re(cutoff, 0.0f), prev_im(cutoff, 0.0f);

    // Precompute the cosine window: sin(pi*(n+0.5)/N), verified against
    // torch.signal.windows.cosine.
    std::vector<float> wcos((size_t)win);
    for (int n = 0; n < win; n++)
        wcos[(size_t)n] = std::sin((float)M_PI * ((float)n + 0.5f) / (float)win);

    for (int f = 0; f < n_frames; f++) {
        const float* fr = y.data() + (size_t)f * hop;

        // --- instantaneous-frequency branch
        core_fft::fft_nonpow2_r2c(fr, win, spec.data());
        float* if_out = out.instfreq.data();
        const size_t nf = (size_t)n_frames;
        for (int k = 0; k < cutoff; k++) {
            const float re = spec[2 * (size_t)k], im = spec[2 * (size_t)k + 1];
            const float mag = std::sqrt(re * re + im * im);
            if_out[(size_t)k * nf + (size_t)f] = std::log10(mag + 1e-5f);
            if (f == 0) {
                // The reference prepends zeros, so frame 0's phase delta is 0.
                if_out[(size_t)(cutoff + k) * nf + (size_t)f] = 0.0f;
                if_out[(size_t)(2 * cutoff + k) * nf + (size_t)f] = 0.0f;
            } else {
                // spec[t] * conj(spec[t-1]), normalised by |.| + 1e-5.
                const float dre = re * prev_re[(size_t)k] + im * prev_im[(size_t)k];
                const float dim = im * prev_re[(size_t)k] - re * prev_im[(size_t)k];
                const float den = std::sqrt(dre * dre + dim * dim) + 1e-5f;
                if_out[(size_t)(cutoff + k) * nf + (size_t)f] = dre / den;
                if_out[(size_t)(2 * cutoff + k) * nf + (size_t)f] = dim / den;
            }
            prev_re[(size_t)k] = re;
            prev_im[(size_t)k] = im;
        }

        // --- difference function (YIN-style), direct form
        for (int t = 0; t < max_corr; t++) {
            const int lag = t + 1;
            double acc = 0.0;
            const float* a = fr + max_corr;
            const float* b = fr + max_corr - lag;
            for (int j = 0; j < corr_win; j++) {
                const double d = (double)a[j] - (double)b[j];
                acc += d * d;
            }
            acc *= 2.0 / (double)corr_win;
            out.corr_diff[(size_t)t * (size_t)n_frames + (size_t)f] = (float)std::sqrt(acc < 0.0 ? 0.0 : acc);
        }

        // --- energy for the converter: cosine-windowed, log10, halved
        double e = 0.0;
        for (int n = 0; n < win; n++) {
            const double v = (double)fr[n] * (double)wcos[(size_t)n];
            e += v * v;
        }
        if (e < 1e-3)
            e = 1e-3;
        out.energy[(size_t)f] = (float)(std::log10(e) * 0.5);
    }
}

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------


} // namespace

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

beatrice_pitch_params beatrice_pitch_default_params(void) {
    beatrice_pitch_params p{};
    p.n_threads = 0;
    p.use_gpu = true;
    p.gpu_device = 0;
    return p;
}

int beatrice_pitch_n_bins(const beatrice_pitch_context* ctx) {
    return ctx ? ctx->hp.pitch_bins : 0;
}
int beatrice_pitch_bins_per_octave(const beatrice_pitch_context* ctx) {
    return ctx ? ctx->hp.bins_per_octave : 0;
}
int beatrice_pitch_sample_rate(const beatrice_pitch_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}
int beatrice_pitch_hop_length(const beatrice_pitch_context* ctx) {
    return ctx ? ctx->hp.hop_length : 0;
}

beatrice_pitch_context* beatrice_pitch_init_from_file(const char* model_path, beatrice_pitch_params params) {
    gguf_context* meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "beatrice: cannot open %s\n", model_path);
        return nullptr;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    if (arch != "beatrice") {
        fprintf(stderr, "beatrice: '%s' is not a beatrice model (arch='%s')\n", model_path, arch.c_str());
        core_gguf::free_metadata(meta);
        return nullptr;
    }
    // Beatrice ships one component per checkpoint. Refuse the wrong one loudly:
    // phone_extractor and pitch_estimator have different tensors entirely, so
    // the alternative is an unexplained missing-tensor error much later.
    const std::string comp = core_gguf::kv_str(meta, "beatrice.component", "");
    if (comp != "pitch_estimator") {
        fprintf(stderr, "beatrice: '%s' holds component '%s', not pitch_estimator\n", model_path, comp.c_str());
        core_gguf::free_metadata(meta);
        return nullptr;
    }

    auto* ctx = new beatrice_pitch_context();
    ctx->params = params;
    beatrice_pitch_hparams& hp = ctx->hp;
    hp.pitch_bins = core_gguf::kv_u32(meta, "beatrice.pitch_estimator.pitch_bins", hp.pitch_bins);
    hp.channels = core_gguf::kv_u32(meta, "beatrice.pitch_estimator.channels", hp.channels);
    hp.n_blocks = core_gguf::kv_u32(meta, "beatrice.pitch_estimator.n_blocks", hp.n_blocks);
    hp.bins_per_octave = core_gguf::kv_u32(meta, "beatrice.pitch_estimator.pitch_bins_per_octave", hp.bins_per_octave);
    hp.sample_rate = core_gguf::kv_u32(meta, "beatrice.in_sample_rate", hp.sample_rate);
    hp.hop_length = core_gguf::kv_u32(meta, "beatrice.pitch.hop_length", hp.hop_length);
    hp.win_length = core_gguf::kv_u32(meta, "beatrice.pitch.win_length", hp.win_length);
    hp.max_corr_period = core_gguf::kv_u32(meta, "beatrice.pitch.max_corr_period", hp.max_corr_period);
    hp.corr_win_length = core_gguf::kv_u32(meta, "beatrice.pitch.corr_win_length", hp.corr_win_length);
    hp.instfreq_cutoff_bin = core_gguf::kv_u32(meta, "beatrice.pitch.instfreq_cutoff_bin", hp.instfreq_cutoff_bin);
    hp.embed_kernel_size = core_gguf::kv_u32(meta, "beatrice.pitch.embed_kernel_size", hp.embed_kernel_size);
    hp.embed_padding = core_gguf::kv_u32(meta, "beatrice.pitch.embed_padding", hp.embed_padding);
    hp.embed_trim = core_gguf::kv_u32(meta, "beatrice.pitch.embed_trim", hp.embed_trim);
    hp.dw_kernel_size = core_gguf::kv_u32(meta, "beatrice.pitch.dw_kernel_size", hp.dw_kernel_size);
    hp.dw_padding = core_gguf::kv_u32(meta, "beatrice.pitch.dw_padding", hp.dw_padding);
    hp.dw_trim = core_gguf::kv_u32(meta, "beatrice.pitch.dw_trim", hp.dw_trim);
    core_gguf::free_metadata(meta);

    if (hp.max_corr_period + hp.corr_win_length != hp.win_length) {
        fprintf(stderr,
                "beatrice: max_corr_period + corr_win_length = %d + %d != win_length %d — "
                "extract_pitch_features asserts this invariant\n",
                hp.max_corr_period, hp.corr_win_length, hp.win_length);
        delete ctx;
        return nullptr;
    }

    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, ctx->backend, "beatrice", wl)) {
        fprintf(stderr, "beatrice: failed to load weights from %s\n", model_path);
        beatrice_pitch_free(ctx);
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->t = wl.tensors;

    // Cross-check declared geometry against the weights themselves.
    auto it = ctx->t.find("head.weight");
    if (it == ctx->t.end()) {
        fprintf(stderr, "beatrice: missing head.weight\n");
        beatrice_pitch_free(ctx);
        return nullptr;
    }
    // head is Conv1d(channels, pitch_bins, 1) -> ggml ne = [1, channels, bins]
    if ((int)it->second->ne[2] != hp.pitch_bins || (int)it->second->ne[1] != hp.channels) {
        fprintf(stderr, "beatrice: KVs say bins=%d channels=%d but head.weight is [%d, %d] — refusing\n", hp.pitch_bins,
                hp.channels, (int)it->second->ne[1], (int)it->second->ne[2]);
        beatrice_pitch_free(ctx);
        return nullptr;
    }

    fprintf(stderr, "beatrice/pitch_estimator: bins=%d channels=%d blocks=%d sr=%d hop=%d (%d fps)\n", hp.pitch_bins,
            hp.channels, hp.n_blocks, hp.sample_rate, hp.hop_length, hp.sample_rate / hp.hop_length);
    return ctx;
}

void beatrice_pitch_free(beatrice_pitch_context* ctx) {
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

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------
namespace {

struct graph_io {
    ggml_tensor* instfreq = nullptr;  // [T, 192]
    ggml_tensor* corr_diff = nullptr; // [T, 256]
    ggml_tensor* logits = nullptr;    // [T, bins]
};

ggml_cgraph* build_graph(beatrice_pitch_context* c, ggml_context* ctx, int T, graph_io& io) {
    const beatrice_pitch_hparams& hp = c->hp;
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);

    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        return it == c->t.end() ? nullptr : it->second;
    };
    auto cap = [&](const char* name, ggml_tensor* t) {
        if (c->capture) {
            ggml_set_name(t, name);
            // ggml_set_output is REQUIRED, not decorative: without it the graph
            // allocator recycles an intermediate's buffer as soon as its last
            // consumer has run, and reading it afterwards yields whatever tensor
            // was allocated over it. The symptom is diagnostic and misleading --
            // every intermediate "fails" while the final stages pass, which
            // reads like a port that is broken everywhere except the end.
            ggml_set_output(t);
            c->caps[name] = t;
            ggml_build_forward_expand(gf, t);
        }
        return t;
    };

    io.instfreq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, 3 * hp.instfreq_cutoff_bin);
    ggml_set_input(io.instfreq);
    io.corr_diff = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, hp.max_corr_period);
    ggml_set_input(io.corr_diff);

    // --- two embedding branches. The conv weights are 1x1, so they reshape to
    // a plain [Cin, Cout] matrix.
    auto conv1x1_w = [&](const std::string& n) -> ggml_tensor* {
        ggml_tensor* w = W(n); // [1, Cin, Cout]
        return w ? ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]) : nullptr;
    };

    ggml_tensor* xi =
        beatrice_ops::pointwise(ctx, io.instfreq, conv1x1_w("instfreq_embed_0.weight"), W("instfreq_embed_0.bias"));
    xi = ggml_gelu(ctx, xi); // tanh approximation — matches approximate="tanh"
    cap("instfreq_embed_0_gelu", xi);
    xi = beatrice_ops::pointwise(ctx, xi, conv1x1_w("instfreq_embed_1.weight"), W("instfreq_embed_1.bias"));
    cap("instfreq_embed_1", xi);

    ggml_tensor* xc =
        beatrice_ops::pointwise(ctx, io.corr_diff, conv1x1_w("corr_embed_0.weight"), W("corr_embed_0.bias"));
    xc = ggml_gelu(ctx, xc);
    cap("corr_embed_0_gelu", xc);
    xc = beatrice_ops::pointwise(ctx, xc, conv1x1_w("corr_embed_1.weight"), W("corr_embed_1.bias"));
    cap("corr_embed_1", xc);

    ggml_tensor* x = ggml_gelu(ctx, ggml_add(ctx, xi, xc));
    cap("branch_sum_gelu", x);

    // --- ConvNeXtStack: embed -> norm(with affine) -> blocks -> final norm
    x = ggml_conv_1d(ctx, W("backbone.embed.weight"), x, 1, hp.embed_padding, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, W("backbone.embed.bias"), 1, hp.channels));
    x = beatrice_ops::trim_tail(ctx, x, hp.embed_trim);
    cap("backbone_embed", x);

    x = beatrice_ops::layernorm_tc(ctx, x, W("backbone.norm.weight"), W("backbone.norm.bias"));
    cap("backbone_norm", x);

    for (int i = 0; i < hp.n_blocks; i++) {
        const std::string p = "backbone.convnext." + std::to_string(i) + ".";
        ggml_tensor* identity = x;

        // Depthwise, strictly causal: left-pad dw_padding then trim dw_trim off
        // the tail. They are equal only because delay == 0 here; the converter
        // emits both so this never silently re-derives one from the other.
        ggml_tensor* h = ggml_conv_1d_dw(ctx, W(p + "dwconv.weight"), x, 1, hp.dw_padding, 1);
        h = ggml_add(ctx, h, ggml_reshape_2d(ctx, W(p + "dwconv.bias"), 1, hp.channels));
        h = beatrice_ops::trim_tail(ctx, h, hp.dw_trim);
        cap(("block" + std::to_string(i) + "_dwconv").c_str(), h);

        // Per-block LayerNorm: affine already folded into pwconv1, so
        // normalise only. Passing the (identity) affine here would be harmless
        // but it is not stored at all.
        h = beatrice_ops::layernorm_tc(ctx, h, nullptr, nullptr);
        h = beatrice_ops::pointwise(ctx, h, W(p + "pwconv1.weight"), W(p + "pwconv1.bias"));
        h = ggml_gelu(ctx, h);
        h = beatrice_ops::pointwise(ctx, h, W(p + "pwconv2.weight"), W(p + "pwconv2.bias"));
        // gamma / pre_scale / post_scale / post_scale_weight are identically 1
        // after merge_weights and are not present in the GGUF.
        x = ggml_add(ctx, h, identity);
        cap(("block" + std::to_string(i) + "_out").c_str(), x);
    }

    x = beatrice_ops::layernorm_tc(ctx, x, W("backbone.final_layer_norm.weight"), W("backbone.final_layer_norm.bias"));
    cap("backbone_final_norm", x);

    io.logits = beatrice_ops::pointwise(ctx, x, conv1x1_w("head.weight"), W("head.bias"));
    cap("logits", io.logits);
    ggml_build_forward_expand(gf, io.logits);
    return gf;
}

// sample_pitch: softmax over bins, force bin 0 (the unvoiced class) out of the
// running, box-filter of width 4, argmax over the filtered sequence, then a
// masked argmax within the winning band. A plain argmax over the logits is a
// DIFFERENT function that agrees except on ambiguous frames — i.e. it looks
// almost right.
// `logits` is BIN-MAJOR, TIME FASTEST: element (bin, frame) at bin*n_frames +
// frame. That is the ggml-native layout of the [T, bins] graph output, and the
// reference dump's layout too. Indexing it frame-major (f*n_bins + k) is the
// natural-looking thing to write and is wrong -- it made all 400 frames differ.
// `features`, when non-null, receives the 3 pitch features in CHANNEL-MAJOR,
// TIME-FASTEST order: [unvoiced_proba | half_pitch_proba | double_pitch_proba],
// each n_frames long.
//
// unvoiced_proba is the ONLY voicing signal this model produces. The quantised
// bin never returns 0 (bin 0 is forced to -100 before the argmax), so without
// this a consumer cannot distinguish speech from silence -- measured, that gap
// turns a 9.3-cent agreement with CREPE on voiced frames into 461.9 cents
// overall. ConverterNetwork prepends energy to make the 4 channels
// embed_pitch_features consumes.
void sample_pitch(const float* logits, int n_bins, int n_frames, int band_width, int bins_per_octave, int* out,
                  float* features) {
    std::vector<double> p((size_t)n_bins);
    std::vector<double> band;
    for (int f = 0; f < n_frames; f++) {
        auto at = [&](int k) { return (double)logits[(size_t)k * (size_t)n_frames + (size_t)f]; };
        double mx = at(0);
        for (int k = 1; k < n_bins; k++)
            mx = std::max(mx, at(k));
        double sum = 0.0;
        for (int k = 0; k < n_bins; k++) {
            p[(size_t)k] = std::exp(at(k) - mx);
            sum += p[(size_t)k];
        }
        for (int k = 0; k < n_bins; k++)
            p[(size_t)k] /= sum;
        // Captured BEFORE bin 0 is suppressed -- it is the real softmax
        // probability of the unvoiced class, not the -100 sentinel.
        const double unvoiced = p[0];
        p[0] = -100.0; // unvoiced class excluded from the pitch argmax

        const int n_band = n_bins - band_width + 1;
        band.assign((size_t)n_band, 0.0);
        int best = 0;
        double best_v = -1e300;
        for (int k = 0; k < n_band; k++) {
            double acc = 0.0;
            for (int j = 0; j < band_width; j++)
                acc += p[(size_t)(k + j)];
            band[(size_t)k] = acc;
            if (acc > best_v) {
                best_v = acc;
                best = k;
            }
        }
        int arg = best;
        double arg_v = -1e300;
        for (int k = best; k < best + band_width && k < n_bins; k++) {
            if (p[(size_t)k] > arg_v) {
                arg_v = p[(size_t)k];
                arg = k;
            }
        }
        out[f] = arg;

        if (features) {
            // The two clamps are ASYMMETRIC in the reference and copying one to
            // the other is a silent bug: half clamps the index to min 1, double
            // clamps to max (n_bins - band_width).
            const double band_proba = band[(size_t)best];
            double half = 0.0, dbl = 0.0;
            if (best > bins_per_octave) {
                int idx = best - bins_per_octave;
                if (idx < 1)
                    idx = 1;
                half = band[(size_t)idx] / (band_proba + 1e-6);
            }
            if (best <= n_bins - band_width - bins_per_octave) {
                int idx = best + bins_per_octave;
                if (idx > n_bins - band_width)
                    idx = n_bins - band_width;
                dbl = band[(size_t)idx] / (band_proba + 1e-6);
            }
            features[(size_t)0 * n_frames + (size_t)f] = (float)unvoiced;
            features[(size_t)1 * n_frames + (size_t)f] = (float)half;
            features[(size_t)2 * n_frames + (size_t)f] = (float)dbl;
        }
    }
}

// Masked bin probabilities at one frame: softmax over bins, then bin 0 (the
// unvoiced class) forced to -100 exactly as sample_pitch does.
void bin_probs(const float* logits, int n_bins, int n_frames, int frame, std::vector<double>& p) {
    p.assign((size_t)n_bins, 0.0);
    auto at = [&](int k) { return (double)logits[(size_t)k * (size_t)n_frames + (size_t)frame]; };
    double mx = at(0);
    for (int k = 1; k < n_bins; k++)
        mx = std::max(mx, at(k));
    double sum = 0.0;
    for (int k = 0; k < n_bins; k++) {
        p[(size_t)k] = std::exp(at(k) - mx);
        sum += p[(size_t)k];
    }
    for (int k = 0; k < n_bins; k++)
        p[(size_t)k] /= sum;
    p[0] = -100.0;
}

// Band scores (the width-4 box filter over the softmaxed bins) at one frame.
// The harness needs these from BOTH logit sets to judge a quantised-pitch
// mismatch honestly.
void band_scores(const float* logits, int n_bins, int n_frames, int band_width, int frame, std::vector<double>& out) {
    std::vector<double> p;
    bin_probs(logits, n_bins, n_frames, frame, p);
    out.assign((size_t)(n_bins - band_width + 1), 0.0);
    for (int k = 0; k + band_width <= n_bins; k++) {
        double acc = 0.0;
        for (int j = 0; j < band_width; j++)
            acc += p[(size_t)(k + j)];
        out[(size_t)k] = acc;
    }
}

// Is a quantised-pitch mismatch at `frame` explained by numerics rather than by
// a logic bug?
//
// SELF-CALIBRATING, deliberately. A fixed margin cannot work here: 113 of 400
// frames on jfk.wav sit inside 1e-3 (min 3.4e-06), so exact-match is a
// permanently red test, while any hand-picked tolerance is just a number chosen
// to make the test green. Instead measure the perturbation the f32 logit
// difference actually induces in band-score space (`delta`), and accept the
// mismatch only if the reference's own preference between the two candidates is
// no larger than that. A frame the reference decides by more than the noise
// could explain is a REAL failure and stays red.
// sample_pitch decides in TWO stages -- which band, then which bin inside it --
// and a bug in either produces a mismatch. Checking only the band choice makes
// this criterion vacuous for the second stage: a plain argmax substituted for
// the within-band argmax leaves the band identical (gap 0) and was waved through
// as "numeric" until this checked both. Verified by negative control.
bool mismatch_is_numeric(const float* mine, const float* ref, int n_bins, int n_frames, int band_width, int frame,
                         int my_bin, int ref_bin, double& gap, double& delta) {
    std::vector<double> bm, br;
    band_scores(mine, n_bins, n_frames, band_width, frame, bm);
    band_scores(ref, n_bins, n_frames, band_width, frame, br);
    delta = 0.0;
    for (size_t k = 0; k < bm.size(); k++)
        delta = std::max(delta, std::fabs(bm[k] - br[k]));
    const size_t best_mine = (size_t)(std::max_element(bm.begin(), bm.end()) - bm.begin());
    const size_t best_ref = (size_t)(std::max_element(br.begin(), br.end()) - br.begin());
    gap = br[best_ref] - br[best_mine];
    if (gap > 2.0 * delta)
        return false; // stage 1: the reference picks its band by more than noise

    // Stage 2. Under the REFERENCE's own probabilities, the two candidate bins
    // must be near-tied. The comparison is ABSOLUTE on purpose: if my bin scores
    // *higher* than the reference's, that is not a tie, it is my code ignoring
    // the band mask -- exactly what a plain argmax does.
    if (my_bin < 0 || my_bin >= n_bins || ref_bin < 0 || ref_bin >= n_bins)
        return false;
    std::vector<double> pm, pr;
    bin_probs(mine, n_bins, n_frames, frame, pm);
    bin_probs(ref, n_bins, n_frames, frame, pr);
    double dp = 0.0;
    for (int k = 0; k < n_bins; k++)
        dp = std::max(dp, std::fabs(pm[(size_t)k] - pr[(size_t)k]));
    const double pgap = std::fabs(pr[(size_t)ref_bin] - pr[(size_t)my_bin]);
    gap = std::max(gap, pgap);
    delta = std::max(delta, dp);
    return pgap <= 2.0 * dp;
}

} // namespace

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

beatrice_pitch_result* beatrice_pitch_estimate(beatrice_pitch_context* c, const float* pcm, int n_samples) {
    if (!c || !pcm || n_samples <= 0)
        return nullptr;

    pitch_features feat;
    extract_pitch_features(c->hp, pcm, n_samples, feat);
    if (feat.n_frames <= 0)
        return nullptr;
    const int T = feat.n_frames;

    const size_t mem = (size_t)1024 * 1024 * 1024;
    ggml_init_params ip{mem, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    graph_io io;
    ggml_cgraph* gf = build_graph(c, ctx, T, io);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_tensor_set(io.instfreq, feat.instfreq.data(), 0, feat.instfreq.size() * sizeof(float));
    ggml_backend_tensor_set(io.corr_diff, feat.corr_diff.data(), 0, feat.corr_diff.size() * sizeof(float));
    if (c->params.n_threads > 0)
        core_cpu_backend::set_n_threads(c->backend, c->params.n_threads);
    ggml_backend_graph_compute(c->backend, gf);

    auto* r = new beatrice_pitch_result();
    r->n_frames = T;
    r->n_bins = c->hp.pitch_bins;
    r->logits = (float*)malloc((size_t)T * c->hp.pitch_bins * sizeof(float));
    r->quantized = (int*)malloc((size_t)T * sizeof(int));
    r->energy = (float*)malloc((size_t)T * sizeof(float));
    r->pitch_features = (float*)malloc((size_t)3 * T * sizeof(float));
    ggml_backend_tensor_get(io.logits, r->logits, 0, (size_t)T * c->hp.pitch_bins * sizeof(float));
    std::memcpy(r->energy, feat.energy.data(), (size_t)T * sizeof(float));
    sample_pitch(r->logits, c->hp.pitch_bins, T, 4, c->hp.bins_per_octave, r->quantized, r->pitch_features);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return r;
}

float beatrice_pitch_bin_to_hz(int bin) {
    return 55.0f * std::pow(2.0f, (float)bin / 96.0f);
}

void beatrice_pitch_to_f0_hz(const beatrice_pitch_result* r, float energy_threshold, float* out_f0_hz) {
    if (!r || !out_f0_hz)
        return;
    for (int i = 0; i < r->n_frames; i++) {
        // See the header: unvoiced_proba is inert on this checkpoint (max
        // 8.9e-07 measured), so energy is the gate that actually works.
        const bool unvoiced = r->energy && r->energy[i] < energy_threshold;
        out_f0_hz[i] = unvoiced ? 0.0f : beatrice_pitch_bin_to_hz(r->quantized[i]);
    }
}

void beatrice_pitch_result_free(beatrice_pitch_result* r) {
    if (!r)
        return;
    free(r->logits);
    free(r->quantized);
    free(r->energy);
    free(r->pitch_features);
    delete r;
}

// ---------------------------------------------------------------------------
// Parity harness
// ---------------------------------------------------------------------------
namespace {

struct ref_weights {
    gguf_context* g = nullptr;
    ggml_context* ctx = nullptr;
};

bool ref_get(ref_weights& rw, const char* name, std::vector<float>& out) {
    const std::string full = std::string("ref.") + name;
    ggml_tensor* t = ggml_get_tensor(rw.ctx, full.c_str());
    if (!t)
        return false;
    out.resize((size_t)ggml_nelements(t));
    std::memcpy(out.data(), t->data, out.size() * sizeof(float));
    return true;
}

double cos_sim(const float* a, const float* b, int64_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int64_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

double max_abs(const float* a, const float* b, int64_t n) {
    double m = 0.0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

} // namespace

int beatrice_pitch_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    beatrice_pitch_params p = beatrice_pitch_default_params();
    p.use_gpu = false; // parity is a CPU-reference comparison
    beatrice_pitch_context* c = beatrice_pitch_init_from_file(model_gguf, p);
    if (!c)
        return 2;

    ref_weights rw;
    ggml_context* rctx = nullptr;
    gguf_init_params gp{false, &rctx};
    rw.g = gguf_init_from_file(ref_gguf, gp);
    if (!rw.g) {
        fprintf(stderr, "beatrice: cannot open reference %s\n", ref_gguf);
        beatrice_pitch_free(c);
        return 2;
    }
    rw.ctx = rctx;

    // Drive the graph from the reference's own DSP output where available, so a
    // front-end bug does not masquerade as a network bug (and vice versa).
    std::vector<float> ref_if, ref_cd, ref_energy, ref_wav;
    const bool have_dsp = ref_get(rw, "dsp_instfreq_features", ref_if) && ref_get(rw, "dsp_corr_diff", ref_cd);
    if (!have_dsp) {
        fprintf(stderr, "beatrice: reference lacks dsp_* stages\n");
        gguf_free(rw.g);
        beatrice_pitch_free(c);
        return 2;
    }
    const int T = (int)(ref_cd.size() / (size_t)c->hp.max_corr_period);

    int n_fail = 0;
    auto report = [&](const char* nm, const float* mine, const std::vector<float>& ref, size_t n_mine) {
        const int64_t n = (int64_t)std::min(n_mine, ref.size());
        const double cs = cos_sim(mine, ref.data(), n);
        const double ma = max_abs(mine, ref.data(), n);
        const bool ok = cs >= 0.9999 && n_mine == ref.size();
        if (!ok)
            n_fail++;
        if (verbosity >= 1 || !ok)
            fprintf(stderr, "  %-24s %s cos=%.8f max_abs=%.3e (mine=%zu ref=%zu)\n", nm, ok ? "PASS" : "FAIL", cs, ma,
                    n_mine, ref.size());
    };

    // --- HOST DSP FIRST. It feeds everything else, so a failure here makes
    // every later stage meaningless.
    if (ref_get(rw, "input_wav", ref_wav)) {
        pitch_features feat;
        extract_pitch_features(c->hp, ref_wav.data(), (int)ref_wav.size(), feat);
        std::vector<float> tmp;
        if (ref_get(rw, "dsp_instfreq_features", tmp))
            report("dsp_instfreq_features", feat.instfreq.data(), tmp, feat.instfreq.size());
        if (ref_get(rw, "dsp_corr_diff", tmp))
            report("dsp_corr_diff", feat.corr_diff.data(), tmp, feat.corr_diff.size());
        if (ref_get(rw, "dsp_energy", tmp))
            report("dsp_energy", feat.energy.data(), tmp, feat.energy.size());
    }

    // --- network, fed the REFERENCE features
    c->capture = true;
    const size_t mem = (size_t)1024 * 1024 * 1024;
    ggml_init_params ip{mem, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    graph_io io;
    ggml_cgraph* gf = build_graph(c, ctx, T, io);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_tensor_set(io.instfreq, ref_if.data(), 0, ref_if.size() * sizeof(float));
    ggml_backend_tensor_set(io.corr_diff, ref_cd.data(), 0, ref_cd.size() * sizeof(float));
    ggml_backend_graph_compute(c->backend, gf);

    // Earliest first — the FIRST failure is the bug (HARD RULE #2).
    std::vector<std::string> order = {
        "instfreq_embed_0_gelu", "instfreq_embed_1", "corr_embed_0_gelu", "corr_embed_1",
        "branch_sum_gelu",       "backbone_embed",   "backbone_norm",
    };
    for (int i = 0; i < c->hp.n_blocks; i++) {
        order.push_back("block" + std::to_string(i) + "_dwconv");
        order.push_back("block" + std::to_string(i) + "_out");
    }
    order.push_back("backbone_final_norm");
    order.push_back("logits");

    for (const auto& nm : order) {
        auto it = c->caps.find(nm);
        std::vector<float> ref;
        if (it == c->caps.end() || !ref_get(rw, nm.c_str(), ref))
            continue;
        std::vector<float> mine((size_t)ggml_nelements(it->second));
        ggml_backend_tensor_get(it->second, mine.data(), 0, mine.size() * sizeof(float));
        report(nm.c_str(), mine.data(), ref, mine.size());
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    c->capture = false;

    // --- END TO END. Everything above is INPUT-ALIGNED: each stage is fed the
    // reference's own inputs, so every one of them passes even when the code
    // wiring the stages together is broken. In the RVC port all 47 per-stage
    // checks were green while the assembled call sat at cos 0.40. This runs the
    // real public API on the real waveform and is the only check that covers the
    // host DSP -> graph handoff and sample_pitch.
    if (!ref_wav.empty()) {
        std::vector<float> ref_logits, ref_q, ref_energy_e2e;
        beatrice_pitch_result* r = beatrice_pitch_estimate(c, ref_wav.data(), (int)ref_wav.size());
        if (!r) {
            fprintf(stderr, "  %-24s FAIL (estimate returned null)\n", "estimate_e2e");
            n_fail++;
        } else {
            if (ref_get(rw, "logits", ref_logits))
                report("estimate_e2e_logits", r->logits, ref_logits, (size_t)r->n_frames * r->n_bins);
            if (ref_get(rw, "energy", ref_energy_e2e))
                report("estimate_e2e_energy", r->energy, ref_energy_e2e, (size_t)r->n_frames);
            std::vector<float> ref_feats;
            if (ref_get(rw, "pitch_features", ref_feats))
                report("estimate_e2e_pitch_features", r->pitch_features, ref_feats, (size_t)3 * r->n_frames);
            // sample_pitch is a banded argmax, not an argmax -- a distinct code
            // path that no tensor comparison above touches. Compare as EXACT
            // integers: a bin index that is off by one is a different learned
            // pitch, not a small numeric error.
            if (ref_get(rw, "quantized_pitch", ref_q) && !ref_logits.empty()) {
                // Every mismatch must be explained by a near-tie in the
                // REFERENCE's own band scores. A mismatch on a frame the
                // reference decides confidently is a real bug and fails.
                const size_t n = std::min((size_t)r->n_frames, ref_q.size());
                size_t bad = 0, numeric = 0;
                double worst_gap = 0.0, worst_delta = 0.0;
                for (size_t i = 0; i < n; i++) {
                    if (r->quantized[i] == (int)std::lround(ref_q[i]))
                        continue;
                    bad++;
                    double gap = 0.0, delta = 0.0;
                    if (mismatch_is_numeric(r->logits, ref_logits.data(), r->n_bins, r->n_frames, 4, (int)i,
                                            r->quantized[i], (int)std::lround(ref_q[i]), gap, delta)) {
                        numeric++;
                    } else {
                        worst_gap = std::max(worst_gap, gap);
                        worst_delta = std::max(worst_delta, delta);
                    }
                }
                const bool ok = bad == numeric && (size_t)r->n_frames == ref_q.size();
                if (!ok)
                    n_fail++;
                fprintf(stderr, "  %-24s %s %zu/%zu frames differ, %zu within f32 band-score noise%s\n",
                        "estimate_e2e_quantized", ok ? "PASS" : "FAIL", bad, n, numeric,
                        ok ? "" : " <-- decided by more than noise");
                if (!ok)
                    fprintf(stderr, "        worst unexplained: ref prefers by %.3e, noise only %.3e\n", worst_gap,
                            worst_delta);
            }
            beatrice_pitch_result_free(r);
        }
    }

    fprintf(stderr, "beatrice-pitch parity: %d stage(s) FAILED\n", n_fail);
    gguf_free(rw.g);
    beatrice_pitch_free(c);
    return n_fail ? 1 : 0;
}

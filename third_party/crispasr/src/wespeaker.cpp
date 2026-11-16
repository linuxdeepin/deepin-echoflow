// wespeaker.cpp — WeSpeaker ResNet34-LM speaker-embedding ggml runtime.
//
// Clean-room implementation from the architecture, not a translation of the
// upstream source. The blueprint (wenet-e2e/wespeaker `models/resnet.py` +
// `cli/speaker.py`, Apache-2.0) was read to establish the spec below and is
// used as an oracle by tools/reference_backends/wespeaker.py; no expression
// from it is reproduced here. The WEIGHTS are CC-BY-4.0 — see wespeaker.h.
//
//   features   Kaldi fbank(80 mel, 25/10 ms, hamming) on an INT16-SCALE
//              waveform, then per-utterance CMN (subtract each bin's mean
//              over time). `wavform_norm` defaults to False upstream, which
//              is why the waveform is scaled by 32768 before framing.
//
//   input map  (T, 80) -> a 2-D map with HEIGHT = freq(80), WIDTH = time(T),
//              1 channel. In ggml that is ne = [W=T, H=80, C=1, N=1].
//              This orientation is not cosmetic: TSTP reduces over time and
//              then flattens (channel, freq) with freq fastest, and seg_1's
//              5120 columns are ordered c*10 + f. Transposing the map would
//              silently permute the stats vector against seg_1's weights.
//
//   stem       Conv2d(1->32, 3x3, s1, p1) -> BN -> ReLU
//   layer1..4  BasicBlock x [3,4,6,3] at 32/64/128/256 channels; the first
//              block of stages 2-4 strides 2 (both axes) and carries a 1x1
//              projection shortcut. Block: relu(bn1(conv1(x))) ->
//              bn2(conv2(.)) -> += shortcut(x) -> relu.
//              Every BN is folded into its conv by the converter, so each
//              conv here is a plain weight + bias.
//
//   TSTP       over the time axis of (256, 10, T'):
//                mean, and std = sqrt(var + 1e-7) with UNBIASED (n-1) var.
//              stats = concat(mean, std) -> 5120.
//
//   seg1       Linear(5120 -> 256) -> the embedding. No ReLU, no BN, and
//              deliberately NO L2 normalisation (two_emb_layer=False).
//
// Windows are short in practice — the diarizer embeds ~1.2 s at a time — so
// the graph is rebuilt per call rather than cached. Caching a fixed-shape
// graph across sched allocations is the §215 use-after-free trap; if this
// ever shows up in a profile, do it with the gallocr/persistent pattern, not
// by holding a cgraph across ggml_backend_sched_reset().

#include "wespeaker.h"

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
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/kaldi_fbank.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ===========================================================================
// Env gates
// ===========================================================================

static bool wespeaker_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_WESPEAKER_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

static bool wespeaker_debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_WESPEAKER_DEBUG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct wespeaker_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit wespeaker_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~wespeaker_bench_stage() {
        if (!wespeaker_bench_enabled())
            return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "  wespeaker_bench: %-14s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Model
// ===========================================================================

struct wespeaker_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 80;
    uint32_t frame_length_ms = 25;
    uint32_t frame_shift_ms = 10;
    uint32_t m_channels = 32;
    uint32_t embed_dim = 256;
    uint32_t stats_dim = 2560;
    float tstp_eps = 1e-7f;
    bool tstp_unbiased_var = true;
    bool int16_scale = true;
    bool cmn = true;
    std::string window_type = "hamming";
    std::vector<int32_t> num_blocks = {3, 4, 6, 3};
};

struct wespeaker_block {
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
    ggml_tensor *shortcut_w = nullptr, *shortcut_b = nullptr; // null when identity
    int stride = 1;
};

struct wespeaker_model {
    wespeaker_hparams hparams;

    ggml_tensor *stem_w = nullptr, *stem_b = nullptr;
    std::vector<std::vector<wespeaker_block>> layers; // [stage][block]
    ggml_tensor *seg1_w = nullptr, *seg1_b = nullptr;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

struct wespeaker_context {
    wespeaker_context_params params;
    wespeaker_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    // Accelerate/OpenBLAS backend, created lazily and ONLY in im2col conv
    // mode — the direct-conv graph has no MUL_MAT the BLAS backend could
    // take (its GEMM is internal to the CPU CONV_2D op), so offering it
    // would change nothing but sched behaviour.
    ggml_backend_t backend_blas = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;
    int n_threads = 4;
    // False on a worker clone: it borrows another context's weights and must
    // not free them. See wespeaker_init_worker.
    bool owns_model = true;
};

static ggml_tensor* require(wespeaker_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "wespeaker");
}

static ggml_tensor* try_get(wespeaker_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

// ===========================================================================
// Loading
// ===========================================================================

static bool wespeaker_load_model(wespeaker_model& model, const char* path, ggml_backend_t backend) {
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;
        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "wespeaker.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "wespeaker.n_mels", hp.n_mels);
        hp.frame_length_ms = core_gguf::kv_u32(gctx, "wespeaker.frame_length_ms", hp.frame_length_ms);
        hp.frame_shift_ms = core_gguf::kv_u32(gctx, "wespeaker.frame_shift_ms", hp.frame_shift_ms);
        hp.m_channels = core_gguf::kv_u32(gctx, "wespeaker.m_channels", hp.m_channels);
        hp.embed_dim = core_gguf::kv_u32(gctx, "wespeaker.embed_dim", hp.embed_dim);
        hp.stats_dim = core_gguf::kv_u32(gctx, "wespeaker.stats_dim", hp.stats_dim);
        hp.tstp_eps = core_gguf::kv_f32(gctx, "wespeaker.tstp_eps", hp.tstp_eps);
        hp.tstp_unbiased_var = core_gguf::kv_bool(gctx, "wespeaker.tstp_unbiased_var", hp.tstp_unbiased_var);
        hp.int16_scale = core_gguf::kv_bool(gctx, "wespeaker.int16_scale", hp.int16_scale);
        hp.cmn = core_gguf::kv_bool(gctx, "wespeaker.cmn", hp.cmn);
        hp.window_type = core_gguf::kv_str(gctx, "wespeaker.window_type", hp.window_type.c_str());
        core_gguf::free_metadata(gctx);
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "wespeaker", wl))
        return false;
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    model.stem_w = require(model, "stem.conv.weight");
    model.stem_b = require(model, "stem.conv.bias");

    model.layers.resize(model.hparams.num_blocks.size());
    for (size_t s = 0; s < model.hparams.num_blocks.size(); s++) {
        const int n_blocks = model.hparams.num_blocks[s];
        model.layers[s].resize((size_t)n_blocks);
        for (int b = 0; b < n_blocks; b++) {
            char nm[128];
            auto get = [&](const char* suf) {
                snprintf(nm, sizeof(nm), "layers.%zu.%d.%s", s, b, suf);
                return require(model, nm);
            };
            auto opt = [&](const char* suf) {
                snprintf(nm, sizeof(nm), "layers.%zu.%d.%s", s, b, suf);
                return try_get(model, nm);
            };
            auto& blk = model.layers[s][(size_t)b];
            blk.conv1_w = get("conv1.weight");
            blk.conv1_b = get("conv1.bias");
            blk.conv2_w = get("conv2.weight");
            blk.conv2_b = get("conv2.bias");
            blk.shortcut_w = opt("shortcut.weight");
            blk.shortcut_b = opt("shortcut.bias");
            // Stages 2-4 halve both axes in their first block; stage 1 never
            // strides. The projection shortcut appears exactly where the shape
            // changes, so its presence and the stride agree by construction.
            blk.stride = (s > 0 && b == 0) ? 2 : 1;
            if ((blk.shortcut_w != nullptr) != (blk.stride == 2)) {
                fprintf(stderr, "wespeaker: stage %zu block %d — shortcut/stride mismatch\n", s, b);
                return false;
            }
        }
    }

    model.seg1_w = require(model, "seg1.weight");
    model.seg1_b = require(model, "seg1.bias");

    fprintf(stderr, "wespeaker: resnet34 m_channels=%u embed=%u stats=%u mels=%u (%s window%s%s)\n",
            model.hparams.m_channels, model.hparams.embed_dim, model.hparams.stats_dim, model.hparams.n_mels,
            model.hparams.window_type.c_str(), model.hparams.cmn ? ", cmn" : "",
            model.hparams.int16_scale ? ", int16-scale" : "");
    return true;
}

// ===========================================================================
// Features — Kaldi fbank + per-utterance CMN
// ===========================================================================

static std::vector<float> wespeaker_fbank(wespeaker_context* ctx, const float* samples, int n_samples, int& T_out) {
    const auto& hp = ctx->model.hparams;

    core_kaldi::FbankParams p;
    p.sample_rate = (int)hp.sample_rate;
    p.n_mels = (int)hp.n_mels;
    p.frame_length_ms = (int)hp.frame_length_ms;
    p.frame_shift_ms = (int)hp.frame_shift_ms;
    p.window_type = (hp.window_type == "hamming") ? core_kaldi::WindowType::Hamming : core_kaldi::WindowType::Povey;
    // torchaudio.load(normalize=False) upstream: kaldi.fbank sees an
    // int16-scale waveform, so the log-mel floor sits where the reference
    // put it.
    p.int16_scale = hp.int16_scale;

    T_out = 0;
    std::vector<float> feat = core_kaldi::compute_fbank(samples, n_samples, p, T_out);
    if (feat.empty() || T_out <= 0)
        return {};

    if (hp.cmn) {
        // feat = feat - feat.mean(dim=0): subtract each mel bin's mean over
        // time. Accumulate in double — 1000+ frames of log-mel in float32
        // loses digits the reference (float64 in numpy's mean) keeps.
        const int F = (int)hp.n_mels;
        std::vector<double> mean((size_t)F, 0.0);
        for (int t = 0; t < T_out; t++)
            for (int f = 0; f < F; f++)
                mean[(size_t)f] += feat[(size_t)t * F + f];
        for (int f = 0; f < F; f++)
            mean[(size_t)f] /= (double)T_out;
        for (int t = 0; t < T_out; t++)
            for (int f = 0; f < F; f++)
                feat[(size_t)t * F + f] -= (float)mean[(size_t)f];
    }
    return feat;
}

// ===========================================================================
// Graph
// ===========================================================================

// Conv lowering (#324 perf). "im2col" (DEFAULT) lowers each conv to explicit
// IM2COL + MUL_MAT graph nodes — same convolution, but the GEMMs become
// schedulable ops, which lets them reach the Accelerate BLAS backend on CPU
// and the simdgroup mul_mm kernels on Metal. "direct" restores GGML_OP_CONV_2D,
// whose CPU path im2cols into scratch and GEMMs via llamafile internally but
// is a naive scalar kernel on Metal.
//
// Measured before the default flip (esrit.wav 215 s, -t 8, M-series):
// diarization delta 9.8 s -> 5.9 s wall (~1.6x); embeddings cosine 1.0 vs
// direct (test_wespeaker_live.cpp "im2col conv matches direct conv"); DER on
// the 8-file VoxConverse shard identical per file, mean 7.32%.
static bool wespeaker_conv_im2col() {
    // Read per call (it is once per graph build, so free) so tests can flip
    // modes with setenv() inside one process.
    const char* e = crispasr_env::get("CRISPASR_WESPEAKER_CONV");
    return !(e && strcmp(e, "direct") == 0);
}

static ggml_tensor* ws_conv_2d(ggml_context* ctx0, ggml_tensor* w, ggml_tensor* x, int s0, int s1, int p0, int p1,
                               int d0, int d1) {
    if (wespeaker_conv_im2col())
        return ggml_conv_2d(ctx0, w, x, s0, s1, p0, p1, d0, d1);
    return ggml_conv_2d_direct(ctx0, w, x, s0, s1, p0, p1, d0, d1);
}

// bias is (OC,) — broadcast it across width and height.
static ggml_tensor* add_conv_bias(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* b) {
    return ggml_add(ctx0, x, ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1));
}

static ggml_tensor* build_block(ggml_context* ctx0, ggml_tensor* x, const wespeaker_block& blk) {
    const int s = blk.stride;
    // conv1: 3x3, stride s, pad 1 -> relu
    ggml_tensor* h = ws_conv_2d(ctx0, blk.conv1_w, x, s, s, 1, 1, 1, 1);
    h = ggml_relu(ctx0, add_conv_bias(ctx0, h, blk.conv1_b));
    // conv2: 3x3, stride 1, pad 1
    h = ws_conv_2d(ctx0, blk.conv2_w, h, 1, 1, 1, 1, 1, 1);
    h = add_conv_bias(ctx0, h, blk.conv2_b);
    // shortcut: identity, or 1x1 stride-s projection with NO padding
    ggml_tensor* sc = x;
    if (blk.shortcut_w) {
        sc = ws_conv_2d(ctx0, blk.shortcut_w, x, s, s, 0, 0, 1, 1);
        sc = add_conv_bias(ctx0, sc, blk.shortcut_b);
    }
    return ggml_relu(ctx0, ggml_add(ctx0, h, sc));
}

// TSTP: reduce (C, F, T) over time into concat(mean, std), 2*C*F long.
static ggml_tensor* build_tstp(ggml_context* ctx0, ggml_tensor* x, float eps, bool unbiased, int T) {
    // ggml_mean reduces ne[0] (= time in this layout) and keeps ne[0] = 1,
    // which is exactly what the subtraction below needs to broadcast.
    ggml_tensor* mean = ggml_mean(ctx0, x); // (1, F, C, 1)

    ggml_tensor* d = ggml_sub(ctx0, x, mean);
    ggml_tensor* var = ggml_mean(ctx0, ggml_sqr(ctx0, d)); // biased (÷N)
    if (unbiased && T > 1) {
        // torch.var defaults to the unbiased estimator; rescale N -> N-1.
        var = ggml_scale(ctx0, var, (float)T / (float)(T - 1));
    }
    // sqrt(var + eps). NOT ggml_add1(ggml_new_f32(...)): ggml_new_f32 writes
    // into tensor->data at build time, which is a null pointer in a no_alloc
    // graph context. ggml_scale_bias folds the constant into the op instead.
    ggml_tensor* stdv = ggml_sqrt(ctx0, ggml_scale_bias(ctx0, var, 1.0f, eps));

    // Flatten (1, F, C, N) -> (F*C, N) with freq fastest, matching torch's
    // flatten(start_dim=1) over (C, F): index c*F + f. N is the window batch
    // (ne[3]); for the unbatched paths N == 1 and this is the old 1-D flatten
    // with a trailing singleton, same memory, same arithmetic.
    const int64_t n = mean->ne[1] * mean->ne[2];
    const int64_t N = mean->ne[3];
    mean = ggml_reshape_2d(ctx0, ggml_cont(ctx0, mean), n, N);
    stdv = ggml_reshape_2d(ctx0, ggml_cont(ctx0, stdv), n, N);
    return ggml_concat(ctx0, mean, stdv, 0); // cat((mean, std)) -> (2*F*C, N)
}

struct wespeaker_graph {
    ggml_context* ctx0 = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* input = nullptr;
};

// `wins`, when non-empty, asks for ONE embedding per (start, end) frame range
// instead of one over the whole input. The trunk runs once and each range is
// then sliced out of its output — see wespeaker_embed_windows.
//
// `n_batch` > 1 instead batches that many INDEPENDENT same-T inputs along
// ne[3] — one forward pass, one (embed_dim, n_batch) output, no arithmetic
// shared between them. Mutually exclusive with `wins` and `with_stages`; see
// wespeaker_embed_batch.
static wespeaker_graph build_graph(wespeaker_context* ctx, int T, bool with_stages,
                                   const std::vector<std::pair<int, int>>& wins = {}, int n_batch = 1) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    wespeaker_graph g;
    g.ctx0 = ggml_init(ip);
    // The trunk is ~250 nodes; each extra window adds ~12 for its slice, TSTP
    // and projection.
    g.gf = ggml_new_graph_custom(g.ctx0, 1024 + 16 * wins.size(), false);

    // ne = [W=time, H=freq, C=1, N=batch]
    g.input = ggml_new_tensor_4d(g.ctx0, GGML_TYPE_F32, T, (int64_t)hp.n_mels, 1, (int64_t)n_batch);
    ggml_set_name(g.input, "fbank");
    ggml_set_input(g.input);

    auto snap = [&](ggml_tensor* t, const char* name) {
        if (!with_stages)
            return;
        ggml_tensor* d = ggml_dup(ctx->sched ? g.ctx0 : g.ctx0, t);
        ggml_set_name(d, name);
        ggml_set_output(d);
        ggml_build_forward_expand(g.gf, d);
    };

    ggml_tensor* h = ws_conv_2d(g.ctx0, m.stem_w, g.input, 1, 1, 1, 1, 1, 1);
    h = ggml_relu(g.ctx0, add_conv_bias(g.ctx0, h, m.stem_b));
    snap(h, "stem_out");

    for (size_t s = 0; s < m.layers.size(); s++) {
        for (const auto& blk : m.layers[s])
            h = build_block(g.ctx0, h, blk);
        char nm[32];
        snprintf(nm, sizeof(nm), "layer%zu_out", s + 1);
        snap(h, nm);
    }

    auto project = [&](ggml_tensor* x, int n_time) {
        ggml_tensor* st = build_tstp(g.ctx0, x, hp.tstp_eps, hp.tstp_unbiased_var, n_time);
        return std::make_pair(st, ggml_add(g.ctx0, ggml_mul_mat(g.ctx0, m.seg1_w, st), m.seg1_b));
    };

    if (wins.empty()) {
        auto pr = project(h, (int)h->ne[0]);
        snap(pr.first, "stats");
        ggml_set_name(pr.second, "embedding");
        ggml_set_output(pr.second);
        ggml_build_forward_expand(g.gf, pr.second);
        return g;
    }

    // Windowed: the trunk downsamples time by the same factor the strides
    // imply, so map frame ranges onto trunk columns by proportion rather than
    // hard-coding 8 — that keeps this correct if the stride pattern changes.
    const int64_t hT = h->ne[0];
    for (size_t i = 0; i < wins.size(); i++) {
        int64_t a = (int64_t)wins[i].first * hT / std::max(1, T);
        int64_t b = (int64_t)wins[i].second * hT / std::max(1, T);
        a = std::min(std::max<int64_t>(a, 0), hT - 1);
        b = std::min(std::max(b, a + 1), hT);
        // ggml_mean wants a contiguous operand, and the slice is small
        // (~15 x 10 x 256), so materialise it.
        ggml_tensor* v = ggml_cont(
            g.ctx0, ggml_view_3d(g.ctx0, h, b - a, h->ne[1], h->ne[2], h->nb[1], h->nb[2], (size_t)a * h->nb[0]));
        ggml_tensor* e = project(v, (int)(b - a)).second;
        char nm[32];
        snprintf(nm, sizeof(nm), "embedding%zu", i);
        ggml_set_name(e, nm);
        ggml_set_output(e);
        ggml_build_forward_expand(g.gf, e);
    }
    return g;
}

static bool ensure_sched(wespeaker_context* ctx) {
    if (ctx->sched)
        return true;
    if (wespeaker_conv_im2col() && !ctx->backend_blas)
        ctx->backend_blas = ggml_backend_init_by_name("BLAS", nullptr); // null on non-BLAS builds — fine
    ggml_backend_t backends[3];
    int n_be = 0;
    if (ctx->backend != ctx->backend_cpu)
        backends[n_be++] = ctx->backend;
    if (ctx->backend_blas)
        backends[n_be++] = ctx->backend_blas;
    backends[n_be++] = ctx->backend_cpu;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 4096, false, false);
    return ctx->sched != nullptr;
}

static int run_graph(wespeaker_context* ctx, const float* feat, int T, wespeaker_stage_cb cb, void* ud,
                     float* out_emb) {
    const auto& hp = ctx->model.hparams;
    if (!ensure_sched(ctx))
        return 1;

    wespeaker_graph g = build_graph(ctx, T, cb != nullptr);
    struct Guard {
        ggml_context* c;
        ~Guard() {
            if (c)
                ggml_free(c);
        }
    } guard{g.ctx0};
    if (!g.gf)
        return 1;

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, g.gf)) {
        fprintf(stderr, "wespeaker: sched alloc failed (T=%d)\n", T);
        return 1;
    }

    // The fbank arrives as (T, n_mels) row-major but the graph wants
    // ne=[T, n_mels] i.e. time fastest — transpose on upload.
    {
        const int F = (int)hp.n_mels;
        std::vector<float> in((size_t)T * F);
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++)
                in[(size_t)f * T + t] = feat[(size_t)t * F + f];
        ggml_backend_tensor_set(g.input, in.data(), 0, in.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "wespeaker: graph compute failed\n");
        return 1;
    }

    if (cb) {
        for (int i = 0; i < ggml_graph_n_nodes(g.gf); i++) {
            ggml_tensor* t = ggml_graph_node(g.gf, i);
            const char* nm = ggml_get_name(t);
            if (!nm || !*nm || strcmp(nm, "fbank") == 0)
                continue;
            const bool wanted = strcmp(nm, "stats") == 0 || strcmp(nm, "embedding") == 0 ||
                                strncmp(nm, "stem_out", 8) == 0 || strncmp(nm, "layer", 5) == 0;
            if (!wanted)
                continue;
            std::vector<float> buf((size_t)ggml_nelements(t));
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
            cb(nm, buf.data(), (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], ud);
        }
    }

    if (out_emb) {
        ggml_tensor* e = ggml_graph_get_tensor(g.gf, "embedding");
        if (!e)
            return 1;
        ggml_backend_tensor_get(e, out_emb, 0, (size_t)hp.embed_dim * sizeof(float));
    }
    return 0;
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" struct wespeaker_context_params wespeaker_context_default_params(void) {
    wespeaker_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    return p;
}

extern "C" struct wespeaker_context* wespeaker_init_from_file(const char* path_model,
                                                              struct wespeaker_context_params params) {
    if (!path_model || !*path_model)
        return nullptr;
    auto* ctx = new wespeaker_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend_cpu = core_cpu_backend::init();
    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;
    // n_threads used to be stored and never applied, so every context silently
    // ran at ggml's default no matter what the caller asked for — and any
    // attempt to run several contexts at once oversubscribed the machine by
    // that default factor.
    core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (params.verbosity > 0)
        fprintf(stderr, "wespeaker: backend = %s\n", ggml_backend_name(ctx->backend));

    ctx->compute_meta.resize(8 * 1024 * 1024);

    if (!wespeaker_load_model(ctx->model, path_model, ctx->backend)) {
        fprintf(stderr, "wespeaker: failed to load model from '%s'\n", path_model);
        wespeaker_free(ctx);
        return nullptr;
    }
    return ctx;
}

// A second context over the SAME weights, for embedding several windows at
// once. Only the per-call machinery is duplicated — backend, scheduler and
// graph scratch, none of which is re-entrant — while model.ctx/buf/tensors are
// borrowed. Loading the GGUF once per worker instead would cost a re-read per
// worker (painful when the model sits on slow storage) and N times the RAM,
// for weights that are read-only during a forward pass.
extern "C" struct wespeaker_context* wespeaker_init_worker(struct wespeaker_context* src) {
    if (!src)
        return nullptr;
    auto* ctx = new wespeaker_context();
    ctx->params = src->params;
    ctx->n_threads = src->n_threads;
    ctx->model = src->model; // shares ctx/buf/tensors — see owns_model
    ctx->owns_model = false;
    ctx->backend_cpu = core_cpu_backend::init();
    if (!ctx->backend_cpu) {
        delete ctx;
        return nullptr;
    }
    ctx->backend = ctx->backend_cpu;
    core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);
    ctx->compute_meta.resize(src->compute_meta.size());
    return ctx;
}

extern "C" void wespeaker_free(struct wespeaker_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->backend_blas)
        ggml_backend_free(ctx->backend_blas);
    if (ctx->owns_model && ctx->model.buf)
        core_gguf::release_weight_buffer(ctx->model.buf);
    if (ctx->owns_model && ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" int wespeaker_embed_dim(struct wespeaker_context* ctx) {
    return ctx ? (int)ctx->model.hparams.embed_dim : 0;
}
extern "C" int wespeaker_sample_rate(struct wespeaker_context* ctx) {
    return ctx ? (int)ctx->model.hparams.sample_rate : 0;
}
extern "C" int wespeaker_n_mels(struct wespeaker_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_mels : 0;
}

extern "C" int wespeaker_min_samples(struct wespeaker_context* ctx) {
    if (!ctx)
        return 0;
    const auto& hp = ctx->model.hparams;
    const int win = (int)(hp.sample_rate * hp.frame_length_ms / 1000);
    const int hop = (int)(hp.sample_rate * hp.frame_shift_ms / 1000);
    // Time is halved three times, so the final map needs >= 1 column; 8
    // frames is the smallest input that survives and still gives TSTP a
    // non-degenerate variance.
    return win + 7 * hop;
}

extern "C" float* wespeaker_compute_fbank(struct wespeaker_context* ctx, const float* samples, int n_samples,
                                          int* out_T, int* out_n_mels) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    int T = 0;
    std::vector<float> feat = wespeaker_fbank(ctx, samples, n_samples, T);
    if (feat.empty())
        return nullptr;
    auto* out = (float*)malloc(feat.size() * sizeof(float));
    if (!out)
        return nullptr;
    memcpy(out, feat.data(), feat.size() * sizeof(float));
    if (out_T)
        *out_T = T;
    if (out_n_mels)
        *out_n_mels = (int)ctx->model.hparams.n_mels;
    return out;
}

extern "C" int wespeaker_embed_staged(struct wespeaker_context* ctx, const float* samples, int n_samples,
                                      wespeaker_stage_cb cb, void* userdata, float* out_embedding) {
    if (!ctx || !samples || n_samples <= 0)
        return 1;
    if (n_samples < wespeaker_min_samples(ctx)) {
        if (wespeaker_debug_enabled())
            fprintf(stderr, "wespeaker: %d samples is below the %d-sample minimum\n", n_samples,
                    wespeaker_min_samples(ctx));
        return 2;
    }

    int T = 0;
    std::vector<float> feat;
    {
        wespeaker_bench_stage _b("fbank");
        feat = wespeaker_fbank(ctx, samples, n_samples, T);
    }
    if (feat.empty() || T <= 0)
        return 1;

    wespeaker_bench_stage _b("resnet");
    return run_graph(ctx, feat.data(), T, cb, userdata, out_embedding);
}

// Embed several windows of ONE contiguous span with a single trunk pass.
//
// The sliding window is 1.2 s at a 0.6 s hop, so embedding each window
// separately pushes every sample through ResNet34 TWICE. Here the fbank and the
// whole convolutional trunk run once over the span and each window is a slice
// of the trunk's output, which is where the 2x goes.
//
// This is NOT bit-identical to calling wespeaker_embed() per window, and cannot
// be: cepstral mean normalisation is now computed over the span rather than the
// window, and interior windows see their neighbours' audio through the convs'
// receptive field instead of zero padding. Both arguably give the model MORE
// context, but "arguably better" is not a licence to skip the check — gate any
// change here on DER, not on cosine against the per-window path.
extern "C" int wespeaker_embed_windows(struct wespeaker_context* ctx, const float* samples, int n_samples,
                                       const int* win_start, const int* win_end, int n_win, float* out_embeddings) {
    if (!ctx || !samples || n_samples <= 0 || !win_start || !win_end || n_win <= 0 || !out_embeddings)
        return 1;
    if (n_samples < wespeaker_min_samples(ctx))
        return 2;

    int T = 0;
    std::vector<float> feat;
    {
        wespeaker_bench_stage _b("fbank");
        feat = wespeaker_fbank(ctx, samples, n_samples, T);
    }
    if (feat.empty() || T <= 0)
        return 1;

    // Sample offsets -> fbank frames. frame f starts at f * frame_shift.
    const int hop = (int)(ctx->model.hparams.sample_rate / 1000 * ctx->model.hparams.frame_shift_ms);
    std::vector<std::pair<int, int>> wins;
    wins.reserve((size_t)n_win);
    for (int i = 0; i < n_win; i++) {
        int a = hop > 0 ? win_start[i] / hop : 0;
        int b = hop > 0 ? (win_end[i] + hop - 1) / hop : T;
        a = std::min(std::max(a, 0), T - 1);
        b = std::min(std::max(b, a + 1), T);
        wins.emplace_back(a, b);
    }

    wespeaker_bench_stage _b("resnet_windows");
    if (!ensure_sched(ctx))
        return 1;
    wespeaker_graph g = build_graph(ctx, T, false, wins);
    if (!g.ctx0)
        return 1;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, g.gf)) {
        ggml_free(g.ctx0);
        return 1;
    }
    {
        const int F = (int)ctx->model.hparams.n_mels;
        std::vector<float> in((size_t)T * F);
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++)
                in[(size_t)f * T + t] = feat[(size_t)t * F + f];
        ggml_backend_tensor_set(g.input, in.data(), 0, in.size() * sizeof(float));
    }
    int rc = 0;
    if (ggml_backend_sched_graph_compute(ctx->sched, g.gf) != GGML_STATUS_SUCCESS) {
        rc = 1;
    } else {
        const int dim = (int)ctx->model.hparams.embed_dim;
        for (int i = 0; i < n_win && rc == 0; i++) {
            char nm[32];
            snprintf(nm, sizeof(nm), "embedding%d", i);
            ggml_tensor* e = ggml_graph_get_tensor(g.gf, nm);
            if (!e) {
                rc = 1;
                break;
            }
            ggml_backend_tensor_get(e, out_embeddings + (size_t)i * dim, 0, (size_t)dim * sizeof(float));
        }
    }
    ggml_free(g.ctx0);
    return rc;
}

extern "C" int wespeaker_embed(struct wespeaker_context* ctx, const float* samples, int n_samples,
                               float* out_embedding) {
    if (!out_embedding)
        return 1;
    return wespeaker_embed_staged(ctx, samples, n_samples, nullptr, nullptr, out_embedding);
}

// ---------------------------------------------------------------------------
// Batched embedding (#324 perf): N independent same-T windows, one graph.
// ---------------------------------------------------------------------------

// Max windows per graph. Bigger batches amortise dispatch further but grow the
// activation buffers ~N-fold. On the GPU the dispatch overhead dominates, so
// the cap defaults to the ceiling (measured on esrit.wav: batch 32 8.5 s vs
// batch 16 9.9 s wall); on CPU it stays at 16, though the CPU default doesn't
// batch at all — see CRISPASR_DIARIZE_BATCH_EMBED.
static int wespeaker_max_batch(const wespeaker_context* ctx) {
    // Read per call so tests can vary the cap with setenv() in-process.
    int v = (ctx && ctx->backend != ctx->backend_cpu) ? 32 : 16;
    if (const char* e = crispasr_env::get("CRISPASR_WESPEAKER_BATCH")) {
        const int x = atoi(e);
        if (x > 0)
            v = std::min(x, 32);
    }
    return v;
}

// One forward pass over `n` fbanks that all have exactly T frames, writing
// n * embed_dim floats to `out`. Purely a fused dispatch of the per-window
// graph — same weights, same ops, same per-window inputs.
static int run_graph_batched(wespeaker_context* ctx, const float* const* feats, int T, int n, float* out) {
    const auto& hp = ctx->model.hparams;
    if (!ensure_sched(ctx))
        return 1;

    wespeaker_graph g = build_graph(ctx, T, false, {}, n);
    struct Guard {
        ggml_context* c;
        ~Guard() {
            if (c)
                ggml_free(c);
        }
    } guard{g.ctx0};
    if (!g.gf)
        return 1;

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, g.gf)) {
        // Activation buffers grow ~n-fold with the batch; report and let the
        // caller halve the batch rather than failing the windows outright.
        fprintf(stderr, "wespeaker: sched alloc failed (T=%d, batch=%d)\n", T, n);
        return 1;
    }

    // Same transpose-on-upload as run_graph, once per batch slot: slot w of
    // the (T, n_mels, 1, n) input starts at w * T * n_mels.
    {
        const int F = (int)hp.n_mels;
        std::vector<float> in((size_t)n * T * F);
        for (int w = 0; w < n; w++) {
            const float* feat = feats[w];
            float* dst = in.data() + (size_t)w * T * F;
            for (int t = 0; t < T; t++)
                for (int f = 0; f < F; f++)
                    dst[(size_t)f * T + t] = feat[(size_t)t * F + f];
        }
        ggml_backend_tensor_set(g.input, in.data(), 0, in.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "wespeaker: batched graph compute failed (T=%d, batch=%d)\n", T, n);
        return 1;
    }

    ggml_tensor* e = ggml_graph_get_tensor(g.gf, "embedding");
    if (!e)
        return 1;
    ggml_backend_tensor_get(e, out, 0, (size_t)n * hp.embed_dim * sizeof(float));
    return 0;
}

// Run one same-T chunk, halving on alloc/compute failure so an oversized
// batch degrades to smaller batches (and ultimately the per-window graph)
// instead of failing the windows.
static int wespeaker_run_chunk(wespeaker_context* ctx, const std::vector<std::vector<float>>& feats, const int* idx,
                               int cnt, int T, int dim, float* out_embeddings) {
    if (cnt == 1) {
        // The existing per-window graph — bit-identical to wespeaker_embed().
        return run_graph(ctx, feats[(size_t)idx[0]].data(), T, nullptr, nullptr, out_embeddings + (size_t)idx[0] * dim);
    }
    std::vector<float> chunk_out((size_t)cnt * dim);
    std::vector<const float*> ptrs((size_t)cnt);
    for (int k = 0; k < cnt; k++)
        ptrs[(size_t)k] = feats[(size_t)idx[k]].data();
    if (run_graph_batched(ctx, ptrs.data(), T, cnt, chunk_out.data()) == 0) {
        for (int k = 0; k < cnt; k++)
            memcpy(out_embeddings + (size_t)idx[k] * dim, chunk_out.data() + (size_t)k * dim,
                   (size_t)dim * sizeof(float));
        return 0;
    }
    const int half = cnt / 2;
    if (wespeaker_run_chunk(ctx, feats, idx, half, T, dim, out_embeddings) != 0)
        return 1;
    return wespeaker_run_chunk(ctx, feats, idx + half, cnt - half, T, dim, out_embeddings);
}

extern "C" int wespeaker_embed_batch(struct wespeaker_context* ctx, const float* samples, int64_t n_samples,
                                     const int64_t* offsets, const int* lengths, int n_win, float* out_embeddings) {
    if (!ctx || !samples || n_samples <= 0 || !offsets || !lengths || n_win <= 0 || !out_embeddings)
        return 1;
    const int min_n = wespeaker_min_samples(ctx);
    for (int i = 0; i < n_win; i++) {
        if (offsets[i] < 0 || lengths[i] <= 0 || offsets[i] + lengths[i] > n_samples)
            return 1;
        if (lengths[i] < min_n)
            return 2; // caller falls back per-window to isolate the short one
    }

    // Per-window fbank + per-window CMN — the arithmetic-identity guarantee.
    std::vector<std::vector<float>> feats((size_t)n_win);
    std::vector<int> Ts((size_t)n_win);
    {
        wespeaker_bench_stage _b("fbank");
        for (int i = 0; i < n_win; i++) {
            int T = 0;
            feats[(size_t)i] = wespeaker_fbank(ctx, samples + offsets[i], lengths[i], T);
            if (feats[(size_t)i].empty() || T <= 0)
                return 1;
            Ts[(size_t)i] = T;
        }
    }

    // Group by exact frame count — same T means same graph shape, and only
    // same-shape windows share a batch. No padding, ever.
    std::map<int, std::vector<int>> by_T;
    for (int i = 0; i < n_win; i++)
        by_T[Ts[(size_t)i]].push_back(i);

    const int dim = (int)ctx->model.hparams.embed_dim;
    const int max_batch = wespeaker_max_batch(ctx);
    wespeaker_bench_stage _b("resnet_batch");
    for (auto& kv : by_T) {
        std::vector<int>& idx = kv.second;
        for (size_t at = 0; at < idx.size(); at += (size_t)max_batch) {
            const int cnt = (int)std::min((size_t)max_batch, idx.size() - at);
            if (wespeaker_debug_enabled() && cnt > 1)
                fprintf(stderr, "wespeaker: batch T=%d n=%d\n", kv.first, cnt);
            if (wespeaker_run_chunk(ctx, feats, idx.data() + (int)at, cnt, kv.first, dim, out_embeddings) != 0)
                return 1;
        }
    }
    return 0;
}

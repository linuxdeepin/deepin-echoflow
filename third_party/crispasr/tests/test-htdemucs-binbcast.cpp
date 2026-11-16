// test-htdemucs-binbcast.cpp — issue #398 guard: no F16 weight may reach a
// broadcast binary op as src1 in the htdemucs ggml graphs.
//
// Hermetic: no model, no network, no GPU. The CUDA binbcast kernels cast
// src1 to the src0/dst element type — for an F32 activation with an F16
// affine weight that means `GGML_ASSERT(nb10 % sizeof(src1_t) == 0)` fails
// (2 % 4) and the process aborts. The CPU path converts per element, so the
// only way to catch this off-GPU is structurally: walk the graph and check
// every ADD/SUB/MUL/DIV src1 type. That is what htd_first_bad_binbcast()
// does, and what these tests pin down:
//
//   1. the guard DETECTS the pre-fix graph shape (F16 GroupNorm affine fed
//      straight into ggml_mul/ggml_add — the exact shape the shipped
//      htdemucs-f16.gguf produces via `*.dconv.layers.N.4.weight`),
//   2. htd_bcast_f32() removes the hazard and the guard passes,
//   3. the casted graph computes the same values as a scalar reference,
//   4. the CRISPASR_HTDEMUCS_NO_BCAST_CAST bisection gate leaves the
//      tensor untouched when disabled.

#include "htdemucs_ggml_util.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

namespace {

constexpr int kT = 4; // time
constexpr int kF = 2; // frequency bands
constexpr int kC = 3; // channels (the affine axis)

struct AffineGraph {
    ggml_context* ctx_w = nullptr; // weight storage (data allocated)
    ggml_context* ctx_g = nullptr; // graph (no_alloc)
    ggml_cgraph* gf = nullptr;
    ggml_tensor* x = nullptr;
    ggml_tensor* out = nullptr;

    ~AffineGraph() {
        if (ctx_g)
            ggml_free(ctx_g);
        if (ctx_w)
            ggml_free(ctx_w);
    }
};

// Mirror of g_dconv_groupnorm's affine tail: y = x * w + b with w/b
// broadcast over ne[0]/ne[2]. `cast` mirrors the issue-#398 fix being
// on (true) or the pre-fix graph (false).
void build_affine(AffineGraph& ag, const std::vector<float>& w_vals, const std::vector<float>& b_vals, bool cast) {
    {
        ggml_init_params ip{2 * ggml_tensor_overhead() + 2 * kC * sizeof(float), nullptr, false};
        ag.ctx_w = ggml_init(ip);
        REQUIRE(ag.ctx_w != nullptr);
    }
    // F16 weights, exactly like the shipped htdemucs-f16.gguf stores the
    // dconv GroupNorm affines.
    ggml_tensor* w = ggml_new_tensor_1d(ag.ctx_w, GGML_TYPE_F16, kC);
    ggml_tensor* b = ggml_new_tensor_1d(ag.ctx_w, GGML_TYPE_F16, kC);
    for (int i = 0; i < kC; i++) {
        ((ggml_fp16_t*)w->data)[i] = ggml_fp32_to_fp16(w_vals[i]);
        ((ggml_fp16_t*)b->data)[i] = ggml_fp32_to_fp16(b_vals[i]);
    }

    {
        const size_t n_nodes = 64;
        ggml_init_params ip{ggml_tensor_overhead() * n_nodes + ggml_graph_overhead(), nullptr, true};
        ag.ctx_g = ggml_init(ip);
        REQUIRE(ag.ctx_g != nullptr);
    }
    ggml_context* g = ag.ctx_g;
    ag.x = ggml_new_tensor_3d(g, GGML_TYPE_F32, kT, kC, kF);
    ggml_set_input(ag.x);

    ggml_tensor* wg = cast ? htd_bcast_f32(g, w, true) : w;
    ggml_tensor* bg = cast ? htd_bcast_f32(g, b, true) : b;
    ggml_tensor* y = ggml_mul(g, ag.x, ggml_reshape_3d(g, wg, 1, kC, 1));
    y = ggml_add(g, y, ggml_reshape_3d(g, bg, 1, kC, 1));
    ggml_set_output(y);
    ag.out = y;

    ag.gf = ggml_new_graph(g);
    ggml_build_forward_expand(ag.gf, y);
}

} // namespace

TEST_CASE("htdemucs binbcast guard detects the pre-fix F16-affine graph", "[unit][htdemucs][binbcast]") {
    AffineGraph ag;
    build_affine(ag, {1.0f, 2.0f, 3.0f}, {0.5f, -0.5f, 0.0f}, /*cast*/ false);

    // This is the shape that aborted the server in issue #398. The guard
    // must flag it — a guard that passes the buggy graph is not a guard.
    const ggml_tensor* bad = htd_first_bad_binbcast(ag.gf);
    REQUIRE(bad != nullptr);
    REQUIRE(bad->op == GGML_OP_MUL);
    REQUIRE(bad->src[1]->type == GGML_TYPE_F16);
}

TEST_CASE("htd_bcast_f32 clears the hazard and preserves values", "[unit][htdemucs][binbcast]") {
    const std::vector<float> w_vals{1.0f, 2.0f, 3.0f};
    const std::vector<float> b_vals{0.5f, -0.5f, 0.0f};
    AffineGraph ag;
    build_affine(ag, w_vals, b_vals, /*cast*/ true);

    REQUIRE(htd_first_bad_binbcast(ag.gf) == nullptr);

    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    REQUIRE(ggml_gallocr_alloc_graph(alloc, ag.gf));

    std::vector<float> x(kT * kC * kF);
    for (size_t i = 0; i < x.size(); i++)
        x[i] = 0.25f * (float)i - 1.0f;
    ggml_backend_tensor_set(ag.x, x.data(), 0, x.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, ag.gf) == GGML_STATUS_SUCCESS);

    std::vector<float> got(x.size());
    ggml_backend_tensor_get(ag.out, got.data(), 0, got.size() * sizeof(float));

    // Scalar reference of the same affine, through the same F16 rounding.
    for (int f = 0; f < kF; f++) {
        for (int c = 0; c < kC; c++) {
            const float wc = ggml_fp16_to_fp32(ggml_fp32_to_fp16(w_vals[c]));
            const float bc = ggml_fp16_to_fp32(ggml_fp32_to_fp16(b_vals[c]));
            for (int t = 0; t < kT; t++) {
                const size_t i = (size_t)f * kC * kT + (size_t)c * kT + t;
                REQUIRE_THAT(got[i], Catch::Matchers::WithinRel(x[i] * wc + bc, 1e-6f));
            }
        }
    }

    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
}

TEST_CASE("htd_bcast_f32 pass-through cases", "[unit][htdemucs][binbcast]") {
    ggml_init_params ip{4 * ggml_tensor_overhead() + 256, nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);

    ggml_tensor* w16 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 4);
    ggml_tensor* w32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

    // F32 weights are returned untouched — no graph noise for the common case.
    REQUIRE(htd_bcast_f32(ctx, w32, true) == w32);
    // nullptr stays nullptr (optional biases).
    REQUIRE(htd_bcast_f32(ctx, (ggml_tensor*)nullptr, true) == nullptr);
    // The bisection gate (CRISPASR_HTDEMUCS_NO_BCAST_CAST=1) reproduces the
    // pre-fix graph: the F16 tensor passes through unconverted.
    REQUIRE(htd_bcast_f32(ctx, w16, false) == w16);
    // Enabled: a fresh F32 node is inserted.
    ggml_tensor* c = htd_bcast_f32(ctx, w16, true);
    REQUIRE(c != w16);
    REQUIRE(c->type == GGML_TYPE_F32);

    ggml_free(ctx);
}

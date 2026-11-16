// test-fastconv.cpp — unit test for core_dac::fastconv_cache + conv1d fast path.
//
// FASTCONV (docs/perf-sweep/PLAN.md item 1) bakes F32 copies of F16 conv kernels
// at load so the fork's per-graph F16→F32 cast becomes a no-op, and routes k=1
// convs through a channel matmul instead of a pure-copy im2col. This test proves
// — with no model, on a CPU backend — that the fast path is:
//   (a) numerically equivalent to the legacy conv1d for K>1 (reduction-order drift
//       only, ~F16-codec level), and
//   (b) numerically equivalent for k=1 (matmul == im2col+matmul), and
//   (c) BYTE-identical to legacy when the cache is disabled (clean A/B gate).

#include <catch2/catch_test_macros.hpp>

#include "core/dac_decoder.h"
#include "core/hifigan.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <vector>

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint32_t nx() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (uint32_t)(s >> 11);
    }
    float sym() { return (float)((nx() % 20001) / 10000.0 - 1.0); }
};

// Persistent weights (F16 conv kernels + F32 bias + F32 input) on a CPU backend,
// so fastconv_cache::bake can read the F16 kernels at "load".
struct Weights {
    ggml_backend_t backend = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *w7 = nullptr, *w1 = nullptr, *b = nullptr, *x = nullptr;
    int Cin, Cout, T;

    Weights(int cin, int cout, int t, Rng& rng) : Cin(cin), Cout(cout), T(t) {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{4 * ggml_tensor_overhead(), nullptr, /*.no_alloc=*/true};
        ctx = ggml_init(ip);
        w7 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, Cin, Cout); // (K,Cin,Cout)
        w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, Cin, Cout);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Cout);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Cin, T); // (Cin,T)
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        set_f16(w7, rng);
        set_f16(w1, rng);
        set_f32(b, rng);
        set_f32(x, rng);
    }
    ~Weights() {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    void set_f16(ggml_tensor* t, Rng& rng) {
        size_t n = ggml_nelements(t);
        std::vector<ggml_fp16_t> h(n);
        for (size_t i = 0; i < n; i++)
            h[i] = ggml_fp32_to_fp16(rng.sym());
        ggml_backend_tensor_set(t, h.data(), 0, n * sizeof(ggml_fp16_t));
    }
    void set_f32(ggml_tensor* t, Rng& rng) {
        size_t n = ggml_nelements(t);
        std::vector<float> f(n);
        for (size_t i = 0; i < n; i++)
            f[i] = rng.sym();
        ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
    }
};

// Run conv1d(x, w, b, K, dil) — legacy if fc==nullptr, else the fast overload.
std::vector<float> run(Weights& W, ggml_tensor* w, int K, const core_dac::fastconv_cache* fc) {
    std::vector<uint8_t> meta(2 * 1024 * 1024);
    ggml_init_params ip{meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* y = fc ? core_dac::conv1d(ctx, W.x, w, W.b, K, 1, fc) : core_dac::conv1d(ctx, W.x, w, W.b, K, 1);
    ggml_set_output(y);
    ggml_set_name(y, "y");
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_graph_compute(W.backend, gf);
    ggml_tensor* yt = ggml_graph_get_tensor(gf, "y");
    std::vector<float> out(ggml_nelements(yt));
    ggml_backend_tensor_get(yt, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return out;
}

void cos_and_max(const std::vector<float>& a, const std::vector<float>& b, double& cos, double& max_abs) {
    REQUIRE(a.size() == b.size());
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
        mx = std::max(mx, std::fabs((double)a[i] - b[i]));
    }
    cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    max_abs = mx;
}

} // namespace

TEST_CASE("core_dac::fastconv — numerically equivalent to legacy conv1d", "[unit][fastconv]") {
    Rng rng(0xFA57C0Full);
    Weights W(6, 10, 40, rng); // Cin=6, Cout=10, T=40

    core_dac::fastconv_cache fc;
    fc.bake(W.backend, {W.w7, W.w1}, /*on=*/true);
    REQUIRE(fc.enabled);
    REQUIRE(fc.get(W.w7) != W.w7); // baked F32 replaces the F16 kernel
    REQUIRE(fc.get(W.w7)->type == GGML_TYPE_F32);

    SECTION("K=7 conv: fast ≈ legacy") {
        auto legacy = run(W, W.w7, 7, nullptr);
        auto fast = run(W, W.w7, 7, &fc);
        double cos, mx;
        cos_and_max(legacy, fast, cos, mx);
        REQUIRE(cos > 0.99999);
        REQUIRE(mx < 1e-2);
    }
    SECTION("k=1 conv (matmul path): fast ≈ legacy") {
        auto legacy = run(W, W.w1, 1, nullptr);
        auto fast = run(W, W.w1, 1, &fc);
        double cos, mx;
        cos_and_max(legacy, fast, cos, mx);
        REQUIRE(cos > 0.99999);
        REQUIRE(mx < 1e-2);
    }
}

TEST_CASE("core_dac::fastconv — disabled cache is byte-identical to legacy", "[unit][fastconv]") {
    Rng rng(0xD15AB1Eull);
    Weights W(4, 8, 24, rng);
    core_dac::fastconv_cache fc; // never baked → enabled=false
    REQUIRE_FALSE(fc.enabled);

    auto legacy = run(W, W.w7, 7, nullptr);
    auto gated = run(W, W.w7, 7, &fc); // routes to legacy since disabled
    REQUIRE(legacy.size() == gated.size());
    for (size_t i = 0; i < legacy.size(); i++)
        REQUIRE(legacy[i] == gated[i]); // exact
}

TEST_CASE("core_hifigan::conv1d — FASTCONV overload equivalent to legacy", "[unit][fastconv]") {
    // HiFi-GAN conv is time-major (x is (T, C_in)); reuse the shared cache to bake
    // the F16 kernel and confirm the baked-F32 path matches the legacy F16-cast path.
    Rng rng(0x41F1ull);
    const int Cin = 5, Cout = 9, T = 32, K = 7;
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_init_params ip{4 * ggml_tensor_overhead(), nullptr, true};
    ggml_context* wctx = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_3d(wctx, GGML_TYPE_F16, K, Cin, Cout);
    ggml_tensor* b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, Cout);
    ggml_tensor* x = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, T, Cin); // time-major (T, Cin)
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    {
        size_t n = ggml_nelements(w);
        std::vector<ggml_fp16_t> h(n);
        for (size_t i = 0; i < n; i++)
            h[i] = ggml_fp32_to_fp16(rng.sym());
        ggml_backend_tensor_set(w, h.data(), 0, n * sizeof(ggml_fp16_t));
        std::vector<float> bb(Cout), xx((size_t)T * Cin);
        for (auto& v : bb)
            v = rng.sym();
        for (auto& v : xx)
            v = rng.sym();
        ggml_backend_tensor_set(b, bb.data(), 0, bb.size() * sizeof(float));
        ggml_backend_tensor_set(x, xx.data(), 0, xx.size() * sizeof(float));
    }
    core_dac::fastconv_cache fc;
    fc.bake(backend, {w}, true);
    REQUIRE(fc.get(w)->type == GGML_TYPE_F32);

    auto run_hg = [&](const core_dac::fastconv_cache* c) {
        std::vector<uint8_t> meta(1 << 20);
        ggml_init_params gip{meta.size(), meta.data(), true};
        ggml_context* g = ggml_init(gip);
        ggml_tensor* y =
            c ? core_hifigan::conv1d(g, x, w, b, 1, K / 2, 1, c) : core_hifigan::conv1d(g, x, w, b, 1, K / 2, 1);
        ggml_set_output(y);
        ggml_set_name(y, "y");
        ggml_cgraph* gf = ggml_new_graph(g);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_graph_compute(backend, gf);
        std::vector<float> out(ggml_nelements(ggml_graph_get_tensor(gf, "y")));
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "y"), out.data(), 0, out.size() * sizeof(float));
        ggml_gallocr_free(ga);
        ggml_free(g);
        return out;
    };
    auto legacy = run_hg(nullptr);
    auto fast = run_hg(&fc);
    double cos, mx;
    cos_and_max(legacy, fast, cos, mx);
    REQUIRE(cos > 0.99999);
    REQUIRE(mx < 1e-2);

    fc.free();
    ggml_backend_buffer_free(wbuf);
    ggml_backend_free(backend);
    ggml_free(wctx);
}

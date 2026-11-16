// test-cpu-backend-shim.cpp — src/core/ggml_cpu_backend.h (#355).
//
// The shim exists so a GGML_BACKEND_DL build can reach the CPU backend, which
// is a dlopen'd module there rather than a linked library. Its contract is that
// both branches behave the same: without DL each wrapper is the direct call it
// replaces, with DL it goes through the registry.
//
// This suite compiles unchanged in either configuration and asserts the
// behaviour rather than the implementation, so `cmake -DGGML_BACKEND_DL=ON`
// runs exactly these cases against the registry path. That is the only way the
// two branches can be held equivalent — a test that only ever runs in the
// default build would say nothing about the one being introduced.
//
// Pure CPU, no model, no GPU.

#include "core/ggml_cpu_backend.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

namespace {

// A graph is needed to test compute(); build the smallest real one: c = a + b.
struct AddGraph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* a = nullptr;
    ggml_tensor* b = nullptr;
    ggml_tensor* c = nullptr;

    explicit AddGraph(int n) {
        // No no_alloc: the tensors own their data so the CPU path can read and
        // write them directly, which is what these call sites do.
        ggml_init_params ip = {};
        ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + (size_t)n * sizeof(float) * 4 + 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc = false;
        ctx = ggml_init(ip);
        REQUIRE(ctx != nullptr);
        a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        c = ggml_add(ctx, a, b);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);
    }
    ~AddGraph() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

} // namespace

TEST_CASE("cpu shim: init returns a CPU backend that reports itself as one", "[unit][cpu-shim][issue-355]") {
    ggml_backend_t b = core_cpu_backend::init();
    REQUIRE(b != nullptr);
    // The whole point of is_cpu() across ~77 call sites is "should this run the
    // CPU path"; a shim that answered false would silently reroute all of them.
    REQUIRE(core_cpu_backend::is_cpu(b));
    ggml_backend_free(b);
}

TEST_CASE("cpu shim: a null backend is not a CPU backend", "[unit][cpu-shim][issue-355]") {
    // Call sites pass a possibly-null backend; under DL the registry lookup can
    // legitimately fail, and neither branch may dereference it.
    REQUIRE_FALSE(core_cpu_backend::is_cpu(nullptr));
}

TEST_CASE("cpu shim: set_n_threads is accepted for any sane count", "[unit][cpu-shim][issue-355]") {
    ggml_backend_t b = core_cpu_backend::init();
    REQUIRE(b != nullptr);
    // Under DL this resolves a named proc; a missing one must be a no-op rather
    // than a crash, so the assertion is that the call is survivable.
    core_cpu_backend::set_n_threads(b, 1);
    core_cpu_backend::set_n_threads(b, 4);
    core_cpu_backend::set_n_threads(nullptr, 4);
    ggml_backend_free(b);
}

TEST_CASE("cpu shim: buffer_type and reg are available", "[unit][cpu-shim][issue-355]") {
    REQUIRE(core_cpu_backend::buffer_type() != nullptr);
    REQUIRE(core_cpu_backend::reg() != nullptr);
}

TEST_CASE("cpu shim: compute() actually computes", "[unit][cpu-shim][issue-355]") {
    // The behavioural core. 18 call sites moved from ggml_graph_compute_with_ctx
    // to this wrapper; if the DL branch dispatched to a backend that never ran
    // the graph, every one of them would silently read stale output rather than
    // fail — so assert the arithmetic, not the return code alone.
    const int n = 8;
    AddGraph g(n);
    auto* pa = (float*)g.a->data;
    auto* pb = (float*)g.b->data;
    for (int i = 0; i < n; i++) {
        pa[i] = (float)i;
        pb[i] = (float)(10 * i);
    }
    REQUIRE(core_cpu_backend::compute(g.ctx, g.gf, 2) == GGML_STATUS_SUCCESS);
    const auto* pc = (const float*)g.c->data;
    for (int i = 0; i < n; i++) {
        INFO("element " << i);
        REQUIRE(pc[i] == (float)(11 * i));
    }
}

TEST_CASE("cpu shim: plan + compute_planned computes the same thing", "[unit][cpu-shim][issue-355]") {
    // The hot-loop form: the VAD per-frame and wav2vec2 per-layer paths size a
    // work buffer from plan() once and reuse it. Under DL plan() reports a zero
    // work_size and the backend plans internally — the result must be identical
    // either way, which is what this pins.
    const int n = 8;
    AddGraph g(n);
    auto* pa = (float*)g.a->data;
    auto* pb = (float*)g.b->data;
    for (int i = 0; i < n; i++) {
        pa[i] = (float)(i + 1);
        pb[i] = 0.5f;
    }
    ggml_cplan plan = core_cpu_backend::plan(g.gf, 2);
    std::vector<uint8_t> work;
    if (plan.work_size > 0) {
        work.resize(plan.work_size);
        plan.work_data = work.data();
    }
    REQUIRE(core_cpu_backend::compute_planned(g.gf, &plan, 2) == GGML_STATUS_SUCCESS);
    const auto* pc = (const float*)g.c->data;
    for (int i = 0; i < n; i++) {
        INFO("element " << i);
        REQUIRE(pc[i] == (float)(i + 1) + 0.5f);
    }
}

TEST_CASE("cpu shim: a threadpool is created or cleanly declined", "[unit][cpu-shim][issue-355]") {
    // Under DL this returns nullptr on purpose — ggml's CPU registry does not
    // expose the threadpool setter, so there is nothing to attach one to. Every
    // caller guards on the pointer, so both answers must be safe, and freeing
    // whatever came back (including nullptr) must not crash.
    ggml_threadpool_t p = core_cpu_backend::threadpool_new(2);
    core_cpu_backend::threadpool_free(p);
    core_cpu_backend::threadpool_free(nullptr);
    SUCCEED("threadpool_new/_free are survivable in both branches");
}

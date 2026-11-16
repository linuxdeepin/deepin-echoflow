// #486: execute a graph on the CPU handle owned by each model, rather than
// checking that a setter was called. The linker hook observes that handle just
// before destruction; it forwards to the real free after the worker probe.
#include "core/ggml_cpu_backend.h"
#include "dia_tts.h"
#include "nemotron.h"
#include "paraformer.h"
#include "gguf.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

namespace {
struct WorkerProbe {
    std::atomic<int> calls{0};
    std::atomic<int> workers{0};
};
struct Observation {
    ggml_status status;
    int calls;
    int workers;
    bool arithmetic;
};
thread_local bool observe = false;
thread_local std::vector<Observation> observations;

void probe_op(ggml_tensor* dst, const ggml_tensor* src, int ith, int nth, void* data) {
    auto& p = *static_cast<WorkerProbe*>(data);
    p.calls.fetch_add(1);
    p.workers.store(nth);
    for (int64_t i = ith; i < ggml_nelements(src); i += nth)
        static_cast<float*>(dst->data)[i] = 2 * static_cast<const float*>(src->data)[i];
}

Observation probe(ggml_backend_t backend) {
    ggml_init_params ip{ggml_tensor_overhead() * 4 + ggml_graph_overhead() + 4096, nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    WorkerProbe p;
    ggml_tensor* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    for (int i = 0; i < 16; ++i)
        static_cast<float*>(input->data)[i] = static_cast<float>(i + 1);
    ggml_tensor* output = ggml_map_custom1(ctx, input, probe_op, GGML_N_TASKS_MAX, &p);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    Observation result{ggml_backend_graph_compute(backend, graph), p.calls.load(), p.workers.load(), true};
    for (int i = 0; i < 16; ++i)
        result.arithmetic &= static_cast<float*>(output->data)[i] == 2 * (i + 1);
    ggml_free(ctx);
    return result;
}

void check_observation(int expected) {
    REQUIRE(observations.size() == 1);
    const auto& result = observations.front();
    INFO("expected " << expected << " workers, observed " << result.workers << " and " << result.calls << " calls");
    REQUIRE(result.status == GGML_STATUS_SUCCESS);
    REQUIRE(result.arithmetic);
    REQUIRE(result.workers == expected);
    REQUIRE(result.calls == expected);
}

// Tiny loadable contexts exercise initialization and Dia's runtime setter
// without downloading speech weights. They are never used for inference.
struct ModelFixture {
    std::string path;
    explicit ModelFixture(const char* tag) : path(std::string("thread-count-") + tag + ".tmp.gguf") {
        ggml_init_params ip{ggml_tensor_overhead() + 1024, nullptr, false};
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(t, "thread_probe_weight");
        static_cast<float*>(t->data)[0] = 1;
        gguf_context* meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", "dia");
        gguf_set_val_u32(meta, "dia.encoder.layers", 1);
        gguf_set_val_u32(meta, "dia.encoder.max_context_length", 4);
        gguf_set_val_u32(meta, "dia.attn_head_size", 2);
        gguf_set_val_u32(meta, "dia.decoder.layers", 1);
        gguf_set_val_u32(meta, "dia.decoder.output_heads", 1);
        gguf_set_val_u32(meta, "dia.decoder.attn_heads", 1);
        gguf_set_val_u32(meta, "dia.decoder.max_generation_size", 4);
        gguf_set_val_u32(meta, "paraformer.n_enc_blocks0", 0);
        gguf_set_val_u32(meta, "paraformer.n_enc_blocks", 0);
        gguf_set_val_u32(meta, "paraformer.n_dec_blocks", 0);
        gguf_add_tensor(meta, t);
        const bool written = gguf_write_to_file(meta, path.c_str(), false);
        gguf_free(meta);
        ggml_free(ctx);
        REQUIRE(written);
    }
    ~ModelFixture() { std::remove(path.c_str()); }
};
struct Capture {
    Capture() {
        observations.clear();
        observe = true;
    }
    ~Capture() { observe = false; }
};
} // namespace

extern "C" void __real_ggml_backend_free(ggml_backend_t backend);
extern "C" void __wrap_ggml_backend_free(ggml_backend_t backend) {
    if (observe && core_cpu_backend::is_cpu(backend))
        observations.push_back(probe(backend));
    __real_ggml_backend_free(backend);
}

TEST_CASE("nemotron CPU graph honors requested threads", "[unit][model-threads][issue-486]") {
    const int threads = GENERATE(1, 3, 8, 0, -2);
    Capture capture;
    auto params = nemotron_context_default_params();
    params.n_threads = threads;
    params.use_gpu = false;
    // Model parsing fails after CPU initialization; the destruction hook runs
    // the graph on that real CPU handle before it is released.
    REQUIRE(nemotron_init_from_file("thread-count-missing-model.gguf", params) == nullptr);
    check_observation(threads > 0 ? threads : 4);
}

TEST_CASE("paraformer CPU graph honors requested threads", "[unit][model-threads][issue-486]") {
    const int threads = GENERATE(1, 3, 8, 0, -2);
    ModelFixture fixture("paraformer");
    Capture capture;
    auto params = paraformer_context_default_params();
    params.n_threads = threads;
    params.use_gpu = false;
    auto* ctx = paraformer_init_from_file(fixture.path.c_str(), params);
    REQUIRE(ctx != nullptr);
    paraformer_free(ctx);
    check_observation(threads > 0 ? threads : 4);
}

TEST_CASE("dia CPU graph honors requested threads", "[unit][model-threads][issue-486]") {
    const int threads = GENERATE(1, 3, 8, 0, -2);
    ModelFixture fixture("init");
    Capture capture;
    auto params = dia_tts_context_default_params();
    params.n_threads = threads;
    params.use_gpu = false;
    auto* ctx = dia_tts_init_from_file(fixture.path.c_str(), params);
    REQUIRE(ctx != nullptr);
    dia_tts_free(ctx);
    check_observation(threads > 0 ? threads : 4);
}

TEST_CASE("dia runtime setter changes CPU graph workers", "[unit][model-threads][issue-486]") {
    const int threads = GENERATE(1, 8, 0, -2);
    ModelFixture fixture("setter");
    Capture capture;
    auto params = dia_tts_context_default_params();
    params.use_gpu = false;
    auto* ctx = dia_tts_init_from_file(fixture.path.c_str(), params);
    REQUIRE(ctx != nullptr);
    dia_tts_set_n_threads(ctx, threads);
    dia_tts_free(ctx);
    check_observation(threads > 0 ? threads : 4);
}

TEST_CASE("paraformer failed model load releases its CPU handle", "[unit][model-load][issue-486]") {
    Capture capture;
    auto params = paraformer_context_default_params();
    params.n_threads = 3;
    params.use_gpu = false;
    REQUIRE(paraformer_init_from_file("thread-count-missing-model.gguf", params) == nullptr);
    check_observation(3);
}

TEST_CASE("paraformer rejects null and empty paths", "[unit][model-load][issue-486]") {
    auto params = paraformer_context_default_params();
    REQUIRE(paraformer_init_from_file(nullptr, params) == nullptr);
    REQUIRE(paraformer_init_from_file("", params) == nullptr);
}

// test-gguf-split-alloc.cpp — unit tests for load_weights_split() chunked
// allocation (issue #276).
//
// Verifies that:
// 1. load_weights_split partitions tensors by the is_gpu predicate correctly.
// 2. Both GPU and CPU partitions are independently addressable.
// 3. Tensor data is correctly loaded into the right partition.
// 4. free_weights leaves no buffer handle behind and is idempotent.
// 5. A partition above the chunk limit really does overflow into split_bufs,
//    and releasing the partition releases those chunks with it.
//    CRISPASR_GGUF_MAX_ALLOC_CHUNK lowers the limit so this needs no
//    multi-gigabyte allocation and no Vulkan hardware.

#include <catch2/catch_test_macros.hpp>

#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Write a GGUF with multiple named F32 tensors.
void write_multi_tensor_gguf(const std::string& path, int n_tensors, int elems_per_tensor) {
    const size_t mem = (size_t)n_tensors * ((size_t)elems_per_tensor * sizeof(float) + ggml_tensor_overhead()) + 4096;
    ggml_init_params ip = {/*mem_size=*/mem, /*mem_buffer=*/nullptr, /*no_alloc=*/false};
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);

    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "test_split");

    for (int i = 0; i < n_tensors; i++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.weight", i);
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elems_per_tensor);
        ggml_set_name(t, name);
        // Fill with recognizable data: each tensor gets value = (float)(i+1).
        float* d = (float*)t->data;
        for (int j = 0; j < elems_per_tensor; j++)
            d[j] = (float)(i + 1);
        gguf_add_tensor(g, t);
    }

    REQUIRE(gguf_write_to_file(g, path.c_str(), /*only_meta=*/false));
    gguf_free(g);
    ggml_free(ctx);
}

// Portable env helpers (Windows has no POSIX setenv/unsetenv).
void test_setenv(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    ::setenv(k, v, 1);
#endif
}
void test_unsetenv(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    ::unsetenv(k);
#endif
}

// Predicate: layers 0..threshold-1 go to GPU, rest to CPU.
struct SplitCtx {
    int threshold;
};
bool test_is_gpu(const char* name, void* user) {
    auto* sc = static_cast<SplitCtx*>(user);
    // Parse layer index from "blk.<N>.weight"
    int il = core_gguf::blk_layer_of(name);
    if (il < 0)
        return true; // non-layered → GPU
    return il < sc->threshold;
}

} // namespace

TEST_CASE("load_weights_split partitions tensors by predicate", "[unit][gguf-split]") {
    const std::string path = "crispasr_test_split_alloc.gguf";
    const int n_tensors = 8;
    const int elems = 32; // small tensors, 128 bytes each

    write_multi_tensor_gguf(path, n_tensors, elems);

    ggml_backend_t gpu_be = ggml_backend_cpu_init(); // use CPU as "GPU" for test
    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    REQUIRE(gpu_be);
    REQUIRE(cpu_be);

    SECTION("all tensors to GPU (threshold = n_tensors)") {
        SplitCtx sc{n_tensors};
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights_split(path.c_str(), gpu_be, cpu_be, test_is_gpu, &sc, "test-split", wl));

        // All 8 tensors should be in the map.
        REQUIRE(wl.tensors.size() == (size_t)n_tensors);
        // GPU buffer must exist (all tensors there), CPU buffer should be null.
        REQUIRE(wl.buf != nullptr);
        REQUIRE(wl.buf_cpu == nullptr);

        // Verify data integrity: blk.3.weight should contain all 4.0f.
        ggml_tensor* t3 = core_gguf::try_get(wl.tensors, "blk.3.weight");
        REQUIRE(t3 != nullptr);
        std::vector<float> data(elems);
        ggml_backend_tensor_get(t3, data.data(), 0, elems * sizeof(float));
        for (int j = 0; j < elems; j++)
            REQUIRE(data[j] == 4.0f);

        core_gguf::free_weights(wl);
    }

    SECTION("all tensors to CPU (threshold = 0)") {
        SplitCtx sc{0};
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights_split(path.c_str(), gpu_be, cpu_be, test_is_gpu, &sc, "test-split", wl));

        REQUIRE(wl.tensors.size() == (size_t)n_tensors);
        REQUIRE(wl.buf == nullptr);
        REQUIRE(wl.buf_cpu != nullptr);

        // Verify data: blk.7.weight should contain all 8.0f.
        ggml_tensor* t7 = core_gguf::try_get(wl.tensors, "blk.7.weight");
        REQUIRE(t7 != nullptr);
        std::vector<float> data(elems);
        ggml_backend_tensor_get(t7, data.data(), 0, elems * sizeof(float));
        for (int j = 0; j < elems; j++)
            REQUIRE(data[j] == 8.0f);

        core_gguf::free_weights(wl);
    }

    SECTION("mixed split: 4 GPU + 4 CPU") {
        SplitCtx sc{4}; // blk.0..3 → GPU, blk.4..7 → CPU
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights_split(path.c_str(), gpu_be, cpu_be, test_is_gpu, &sc, "test-split", wl));

        REQUIRE(wl.tensors.size() == (size_t)n_tensors);
        REQUIRE(wl.buf != nullptr);
        REQUIRE(wl.buf_cpu != nullptr);

        // GPU tensor: blk.1.weight → value 2.0f
        ggml_tensor* t1 = core_gguf::try_get(wl.tensors, "blk.1.weight");
        REQUIRE(t1 != nullptr);
        REQUIRE(t1->buffer == wl.buf);
        std::vector<float> d1(elems);
        ggml_backend_tensor_get(t1, d1.data(), 0, elems * sizeof(float));
        for (int j = 0; j < elems; j++)
            REQUIRE(d1[j] == 2.0f);

        // CPU tensor: blk.5.weight → value 6.0f
        ggml_tensor* t5 = core_gguf::try_get(wl.tensors, "blk.5.weight");
        REQUIRE(t5 != nullptr);
        REQUIRE(t5->buffer == wl.buf_cpu);
        std::vector<float> d5(elems);
        ggml_backend_tensor_get(t5, d5.data(), 0, elems * sizeof(float));
        for (int j = 0; j < elems; j++)
            REQUIRE(d5[j] == 6.0f);

        core_gguf::free_weights(wl);
    }

    SECTION("split_bufs is empty for small models (no chunking needed)") {
        SplitCtx sc{4};
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights_split(path.c_str(), gpu_be, cpu_be, test_is_gpu, &sc, "test-split", wl));

        // With 8 tiny tensors (128 bytes each), everything fits in one buffer
        // per partition — no overflow into split_bufs.
        REQUIRE(wl.split_bufs.empty());

        core_gguf::free_weights(wl);
    }

    std::remove(path.c_str());
    ggml_backend_free(gpu_be);
    ggml_backend_free(cpu_be);
}

TEST_CASE("load_weights_split rejects null backends/predicate", "[unit][gguf-split]") {
    const std::string path = "crispasr_test_split_reject.gguf";
    write_multi_tensor_gguf(path, 2, 16);

    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be);

    core_gguf::WeightLoad wl;
    SplitCtx sc{1};

    SECTION("null gpu_backend") {
        REQUIRE_FALSE(core_gguf::load_weights_split(path.c_str(), nullptr, be, test_is_gpu, &sc, "test", wl));
    }
    SECTION("null cpu_backend") {
        REQUIRE_FALSE(core_gguf::load_weights_split(path.c_str(), be, nullptr, test_is_gpu, &sc, "test", wl));
    }
    SECTION("null predicate") {
        REQUIRE_FALSE(core_gguf::load_weights_split(path.c_str(), be, be, nullptr, &sc, "test", wl));
    }

    std::remove(path.c_str());
    ggml_backend_free(be);
}

TEST_CASE("a chunked partition's overflow buffers are released with it", "[unit][gguf-split]") {
    // The overflow chunks are owned by the loader and released with the first
    // buffer of their own partition. Reaching that branch normally needs a
    // partition above 1.5 GiB, so CRISPASR_GGUF_MAX_ALLOC_CHUNK lowers the
    // limit far enough for a synthetic model to chunk.
    //
    // What this proves: the chunked path runs, the overflow buffers are
    // recorded, and releasing the partition once — or twice — neither
    // double-frees nor leaves a stale handle. What it cannot prove is that
    // the memory came back, because ggml exposes no per-backend allocation
    // counter to assert against; that half is the code review's.
    test_setenv("CRISPASR_GGUF_MAX_ALLOC_CHUNK", "1024");

    ggml_backend_t gpu_be = ggml_backend_cpu_init();
    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    REQUIRE(gpu_be);
    REQUIRE(cpu_be);

    const std::string path = "crispasr_test_split_chunked.gguf";
    // 8 tensors of 1 KiB each: several chunks per partition at a 1 KiB limit.
    write_multi_tensor_gguf(path, 8, 256);

    SplitCtx sc{4};
    core_gguf::WeightLoad wl;
    REQUIRE(core_gguf::load_weights_split(path.c_str(), gpu_be, cpu_be, test_is_gpu, &sc, "test-chunked", wl));
    REQUIRE(wl.buf != nullptr);
    REQUIRE(wl.buf_cpu != nullptr);
    // Positive control: without this the case would pass having never chunked.
    REQUIRE_FALSE(wl.split_bufs.empty());

    core_gguf::free_weights(wl);
    REQUIRE(wl.buf == nullptr);
    REQUIRE(wl.buf_cpu == nullptr);
    REQUIRE(wl.split_bufs.empty());

    // Idempotent: the loader's record was taken and erased, not just read, so
    // a second release cannot free the same overflow chunks again.
    core_gguf::free_weights(wl);
    REQUIRE(wl.split_bufs.empty());

    test_unsetenv("CRISPASR_GGUF_MAX_ALLOC_CHUNK");
    std::remove(path.c_str());
    ggml_backend_free(gpu_be);
    ggml_backend_free(cpu_be);
}

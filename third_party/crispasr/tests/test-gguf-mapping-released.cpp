// test-gguf-mapping-released.cpp — the weight mmap must be gone after the
// weights are freed.
//
// core_gguf::load_weights maps the whole GGUF and hands the region to the
// backend. On a device advertising `buffer_from_host_ptr` (Apple-Silicon
// Metal) the backend does not own those pages — `buffer_from_host_ptr` has no
// deallocator parameter, and Metal passes `deallocator:nil` — so freeing the
// backend buffer alone left the file mapped for the life of the process. The
// mapping is `MAP_PRIVATE | PROT_READ|PROT_WRITE`, so every page privatizes on
// first read: the resident pages are dirty and anonymous, and can only be
// compressed or swapped, never dropped. A process that loaded several models
// therefore held all of them at once.
//
// The oracle is exact rather than a footprint threshold: after the free, no
// region of this process may name the weight file. There is nothing to settle
// and nothing to poll, and the assertion survives a change of release
// mechanism because its subject is the mapping, not munmap.
//
// Two paths are covered. The CPU mmap path runs everywhere and releases
// through the buffer's own free callback. The zero-copy GPU path is the one
// that leaked; it self-skips where no device advertises the capability.

#include <catch2/catch_test_macros.hpp>

#include "test-region-probe.h"

#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

// Portable env helper (Windows has no POSIX setenv).
void test_setenv(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    ::setenv(k, v, 1);
#endif
}

using test_region::absolute_path_of;
using test_region::count_regions_backed_by;
using test_region::region_probe_available;

// Write a GGUF several megabytes wide, so its mapping is a region of its own
// rather than something the kernel might fold into a neighbour.
void write_gguf(const std::string& path, int n_tensors, int elems) {
    const size_t mem = (size_t)n_tensors * ((size_t)elems * sizeof(float) + ggml_tensor_overhead()) + 4096;
    ggml_init_params ip = {/*mem_size=*/mem, /*mem_buffer=*/nullptr, /*no_alloc=*/false};
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);

    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "test_release");
    for (int i = 0; i < n_tensors; i++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.weight", i);
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elems);
        ggml_set_name(t, name);
        float* d = (float*)t->data;
        for (int j = 0; j < elems; j++)
            d[j] = (float)(i + 1);
        gguf_add_tensor(g, t);
    }
    REQUIRE(gguf_write_to_file(g, path.c_str(), /*only_meta=*/false));
    gguf_free(g);
    ggml_free(ctx);
}

// A GPU backend that hands host pointers to the device, or nullptr when this
// machine has none. That capability is exactly what selects the leaking path.
ggml_backend_t init_host_ptr_gpu_backend() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev)
        return nullptr;
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.buffer_from_host_ptr)
        return nullptr;
    return ggml_backend_dev_init(dev, nullptr);
}

struct Fixture {
    std::string rel;
    std::string abs;
    explicit Fixture(const char* name) : rel(name) {
        write_gguf(rel, /*n_tensors=*/4, /*elems=*/262144); // 4 MiB of weights
        abs = absolute_path_of(rel);
    }
    ~Fixture() { std::remove(rel.c_str()); }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
};

} // namespace

TEST_CASE("freeing weights unmaps the CPU mmap path's region", "[unit][gguf-mapping]") {
    if (!region_probe_available()) {
        SUCCEED("region enumeration unavailable on this platform");
        return;
    }
    test_setenv("CRISPASR_GGUF_MMAP", "1");

    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    Fixture fx("crispasr_test_mapping_cpu.gguf");
    REQUIRE(count_regions_backed_by(fx.abs) == 0);

    core_gguf::WeightLoad wl;
    REQUIRE(core_gguf::load_weights(fx.rel.c_str(), backend, "test-map-cpu", wl));
    // Positive control: the mmap path was taken and the file is mapped. The
    // count is not pinned to 1 because a kernel may report one mapping as
    // several adjacent regions; what this case asserts exactly is the zero
    // below, and the repeated-cycles case asserts the per-load accumulation.
    REQUIRE(count_regions_backed_by(fx.abs) >= 1);

    core_gguf::free_weights(wl);
    REQUIRE(count_regions_backed_by(fx.abs) == 0);

    ggml_backend_free(backend);
}

TEST_CASE("freeing weights unmaps the zero-copy GPU path's region", "[unit][gguf-mapping]") {
    if (!region_probe_available()) {
        SUCCEED("region enumeration unavailable on this platform");
        return;
    }
    test_setenv("CRISPASR_GGUF_MMAP", "1");

    ggml_backend_t backend = init_host_ptr_gpu_backend();
    if (!backend) {
        SUCCEED("no GPU device advertising buffer_from_host_ptr — the leaking path does not exist here");
        return;
    }
    Fixture fx("crispasr_test_mapping_gpu.gguf");
    REQUIRE(count_regions_backed_by(fx.abs) == 0);

    core_gguf::WeightLoad wl;
    REQUIRE(core_gguf::load_weights(fx.rel.c_str(), backend, "test-map-gpu", wl));
    // Positive control: the zero-copy path was actually taken. Without it a
    // fall-through to the legacy alloc+copy loader would satisfy the absence
    // assertion below having never created the mapping under test. Not pinned
    // to 1 — see the CPU case.
    REQUIRE(count_regions_backed_by(fx.abs) >= 1);

    core_gguf::free_weights(wl);
    REQUIRE(count_regions_backed_by(fx.abs) == 0);

    ggml_backend_free(backend);
}

TEST_CASE("repeated load/free cycles leave no mapping behind", "[unit][gguf-mapping]") {
    if (!region_probe_available()) {
        SUCCEED("region enumeration unavailable on this platform");
        return;
    }
    test_setenv("CRISPASR_GGUF_MMAP", "1");

    // Each load maps the file again and records a separate region, so a
    // release that handled only one of them accumulates the rest. Twenty
    // cycles make that a count of twenty rather than an ambiguous one.
    ggml_backend_t gpu = init_host_ptr_gpu_backend();
    ggml_backend_t backend = gpu ? gpu : ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    Fixture fx("crispasr_test_mapping_loop.gguf");

    for (int i = 0; i < 20; i++) {
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights(fx.rel.c_str(), backend, "test-map-loop", wl));
        core_gguf::free_weights(wl);
    }
    REQUIRE(count_regions_backed_by(fx.abs) == 0);

    ggml_backend_free(backend);
}

// test-gguf-release.cpp — contract of core_gguf::release_weight_buffer().
//
// load_weights() can hand back a backend buffer that owns a host mmap the
// backend itself does not own: on a device advertising `buffer_from_host_ptr`
// (Apple-Silicon Metal) the weight file is mapped and passed to
// `newBufferWithBytesNoCopy:…deallocator:nil`, so ggml_backend_buffer_free()
// releases the device-side view and leaves the mapping behind.
// release_weight_buffer() is the entry point that releases both.
//
// This file pins the entry point's contract on paths that need no GPU, so it
// runs everywhere: the CPU mmap path, the legacy alloc+copy path (which
// registers no mapping at all, and must therefore be an ordinary buffer free),
// repeat calls, and the null handle. test-gguf-mapping-released.cpp covers the
// mapping's actual disappearance.

#include <catch2/catch_test_macros.hpp>

#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

// Write a minimal valid GGUF with one F32 tensor of `n` elements.
void write_gguf(const std::string& path, int n) {
    ggml_init_params ip = {/*mem_size=*/(size_t)n * sizeof(float) + ggml_tensor_overhead() + 1024,
                           /*mem_buffer=*/nullptr, /*no_alloc=*/false};
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(t, "test.weight");
    for (int i = 0; i < n; i++)
        ((float*)t->data)[i] = (float)i;

    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "test");
    gguf_add_tensor(g, t);
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

// RAII for CRISPASR_GGUF_MMAP, which load_weights reads on every call. Set
// rather than assumed: an inherited `CRISPASR_GGUF_MMAP=0` would otherwise
// silently send both halves of the loop below down the same path.
struct MmapEnv {
    std::string saved;
    bool had = false;
    explicit MmapEnv(const char* value) {
        if (const char* v = std::getenv("CRISPASR_GGUF_MMAP")) {
            saved = v;
            had = true;
        }
        test_setenv("CRISPASR_GGUF_MMAP", value);
    }
    ~MmapEnv() {
        if (had)
            test_setenv("CRISPASR_GGUF_MMAP", saved.c_str());
        else
            test_unsetenv("CRISPASR_GGUF_MMAP");
    }
    MmapEnv(const MmapEnv&) = delete;
    MmapEnv& operator=(const MmapEnv&) = delete;
};

} // namespace

TEST_CASE("release_weight_buffer releases a loaded weight buffer and nulls the handle", "[unit][gguf-release]") {
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);

    const std::string path = "crispasr_test_gguf_release.gguf";
    write_gguf(path, 4096);

    // Both loader paths reach the same release call, and they differ in
    // exactly the way that matters: `=1` takes the mmap path, `=0` the legacy
    // alloc+copy path that registers no mapping. A release that only worked
    // when an entry existed would pass one and fail the other.
    for (const char* mmap_mode : {"1", "0"}) {
        MmapEnv env(mmap_mode);
        INFO("CRISPASR_GGUF_MMAP=" << mmap_mode);

        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights(path.c_str(), backend, "test-release", wl));
        REQUIRE(wl.buf != nullptr);
        // The weights are readable before the release — otherwise "released"
        // would be indistinguishable from "never loaded".
        ggml_tensor* t = core_gguf::require(wl.tensors, "test.weight", "test-release");
        REQUIRE(t != nullptr);
        float first = 0.0f;
        ggml_backend_tensor_get(t, &first, 0, sizeof(float));
        REQUIRE(first == 0.0f);

        core_gguf::release_weight_buffer(wl.buf);
        REQUIRE(wl.buf == nullptr);

        // A second call through the same handle must not double-free. This is
        // the double-release case a caller can actually write; releasing a
        // saved copy of the raw pointer is not tested because reading a freed
        // ggml_backend_buffer is undefined behaviour whatever the side map does.
        core_gguf::release_weight_buffer(wl.buf);
        REQUIRE(wl.buf == nullptr);

        ggml_free(wl.ctx);
        wl.ctx = nullptr;
        wl.tensors.clear();
    }

    std::remove(path.c_str());
    ggml_backend_free(backend);
}

TEST_CASE("release_weight_buffer accepts a null handle", "[unit][gguf-release]") {
    ggml_backend_buffer_t buf = nullptr;
    core_gguf::release_weight_buffer(buf);
    REQUIRE(buf == nullptr);
}

TEST_CASE("free_weights releases every buffer it owns", "[unit][gguf-release]") {
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);

    const std::string path = "crispasr_test_gguf_release_free.gguf";
    write_gguf(path, 4096);

    MmapEnv env("1");
    core_gguf::WeightLoad wl;
    REQUIRE(core_gguf::load_weights(path.c_str(), backend, "test-free", wl));
    REQUIRE(wl.buf != nullptr);
    REQUIRE(wl.ctx != nullptr);
    REQUIRE(wl.tensors.count("test.weight") == 1);

    core_gguf::free_weights(wl);
    REQUIRE(wl.buf == nullptr);
    REQUIRE(wl.buf_cpu == nullptr);
    REQUIRE(wl.split_bufs.empty());
    REQUIRE(wl.ctx == nullptr);
    REQUIRE(wl.tensors.empty());

    std::remove(path.c_str());
    ggml_backend_free(backend);
}

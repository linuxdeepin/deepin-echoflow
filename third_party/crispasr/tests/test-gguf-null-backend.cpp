// test-gguf-null-backend.cpp — issue #405 guard: a null backend handed to
// core_gguf::load_weights must FAIL the load, not abort the process.
//
// Hermetic: no model download, no GPU. Under GGML_BACKEND_DL every backend
// init can legitimately return null — when the host CPU is below the ISA
// floor of every shipped libggml-cpu variant, ggml_backend_score() refuses
// them all and the registry has no CPU device. `parakeet_init_from_file`
// then handed that null straight to load_weights, which walked into
// ggml_backend_get_device(nullptr) and killed the whole server with
// `GGML_ASSERT(backend) failed` (ggml-backend.cpp:471 — the #405 stack).
// Before the fix this very test died with that abort instead of failing an
// assertion; that is the red-verify.

#include "core/gguf_loader.h"
#include "core/ggml_cpu_backend.h"

#include "ggml.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <string>

namespace {

// Write a minimal but valid GGUF (one F32 tensor) so the null-backend guard
// is what the load trips on, not a parse error.
std::string write_tiny_gguf(const char* tag) {
    // Per-case filename: ctest -j runs the two cases of this binary
    // CONCURRENTLY (catch_discover_tests registers each TEST_CASE as its own
    // ctest entry), and a shared name let one case delete the file while the
    // other was reading it — a flake seen on the first -j2 run after landing.
    std::string path = std::string("test-gguf-null-backend.") + tag + ".tmp.gguf";
    ggml_init_params ip{ggml_tensor_overhead() + 64 * sizeof(float), nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_set_name(t, "w");
    for (int i = 0; i < 16; i++)
        ((float*)t->data)[i] = (float)i;

    gguf_context* g = gguf_init_empty();
    REQUIRE(g != nullptr);
    gguf_set_val_str(g, "general.architecture", "test");
    gguf_add_tensor(g, t);
    gguf_write_to_file(g, path.c_str(), /*only_meta*/ false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

} // namespace

TEST_CASE("load_weights with a null backend fails cleanly instead of aborting", "[unit][gguf][null-backend]") {
    const std::string path = write_tiny_gguf("null");

    core_gguf::WeightLoad wl;
    // Pre-fix: GGML_ASSERT(backend) abort inside ggml_backend_get_device().
    // Post-fix: a clean `false` and an error on stderr.
    REQUIRE_FALSE(core_gguf::load_weights(path.c_str(), /*backend*/ nullptr, "test405", wl));
    REQUIRE(wl.ctx == nullptr);

    std::remove(path.c_str());
}

TEST_CASE("load_weights with a real CPU backend still succeeds on the same file", "[unit][gguf][null-backend]") {
    // Control arm: the guard must reject ONLY the null backend, not break
    // normal loading of the identical file.
    const std::string path = write_tiny_gguf("cpu");

    ggml_backend_t cpu = core_cpu_backend::init();
    REQUIRE(cpu != nullptr);

    core_gguf::WeightLoad wl;
    REQUIRE(core_gguf::load_weights(path.c_str(), cpu, "test405", wl));
    REQUIRE(wl.ctx != nullptr);
    REQUIRE(wl.tensors.count("w") == 1);

    if (wl.buf)
        core_gguf::release_weight_buffer(wl.buf);
    if (wl.ctx)
        ggml_free(wl.ctx);
    ggml_backend_free(cpu);
    std::remove(path.c_str());
}

// test-gguf-bounds.cpp — deterministic regression guard for the GGUF
// load_weights out-of-bounds hardening (untrusted-input security work).
//
// A GGUF model file is untrusted (users download/convert models). A crafted or
// truncated GGUF can declare a tensor whose data range exceeds the file — the
// mmap zero-copy loader would then dereference past the mapping → SIGBUS. The
// fix added subtractive, overflow-safe bounds checks in
// core_gguf::load_weights (gguf_loader.cpp): `data_off > size || off > size -
// data_off || nbytes > size - data_off - off` → reject, don't crash.
//
// This test builds a VALID one-tensor GGUF, then truncates it to the start of
// the tensor-data section so metadata still parses but the declared tensor
// overruns the file, and asserts load_weights returns false (safe reject) — and
// that the intact file still loads. It runs with no model, deterministically.

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
#include <unistd.h> // truncate

namespace {

// Write a minimal valid GGUF with one F32 tensor.
void write_valid_gguf(const std::string& path, int n) {
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

// Portable env helper (Windows has no POSIX setenv).
void test_setenv(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    ::setenv(k, v, 1);
#endif
}

// Copy `src` and truncate the copy 8 bytes short of the declared tensor data,
// so metadata parses fully but the tensor overruns the file.
void write_truncated_copy(const std::string& src_path, const std::string& dst_path, size_t keep_bytes) {
    FILE* src = std::fopen(src_path.c_str(), "rb");
    FILE* dst = std::fopen(dst_path.c_str(), "wb");
    REQUIRE(src);
    REQUIRE(dst);
    char buf[4096];
    size_t r;
    while ((r = std::fread(buf, 1, sizeof(buf), src)) > 0)
        std::fwrite(buf, 1, r, dst);
    std::fclose(src);
    std::fclose(dst);
    REQUIRE(::truncate(dst_path.c_str(), (off_t)keep_bytes) == 0);
}

// A GPU backend that hands host pointers to the device, or nullptr when this
// machine has none. That capability selects the zero-copy load path, which is
// the only one that maps the file into a buffer the backend does not own.
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

// Re-read the written file to get the true data-section offset (the write
// context's gguf_get_data_offset is not populated until write).
size_t read_data_offset(const std::string& path) {
    gguf_init_params gp = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    REQUIRE(g != nullptr);
    size_t off = gguf_get_data_offset(g);
    gguf_free(g);
    return off;
}

} // namespace

TEST_CASE("core_gguf::load_weights rejects a truncated GGUF without crashing", "[unit][gguf-bounds]") {
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);

    const std::string good = "crispasr_test_gguf_bounds_ok.gguf";
    const std::string bad = "crispasr_test_gguf_bounds_trunc.gguf";

    const int n = 64; // 64 f32 = 256 tensor bytes
    write_valid_gguf(good, n);
    const size_t data_off = read_data_offset(good);
    const size_t nbytes = (size_t)n * sizeof(float);

    // Positive control: the intact file must load with the tensor present.
    {
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights(good.c_str(), backend, "test-ok", wl));
        REQUIRE(wl.tensors.count("test.weight") == 1);
        core_gguf::free_weights(wl);
    }

    // Craft the malicious file: copy the good one, then truncate to just SHORT
    // of the full tensor data (metadata fully parses; the declared 256-byte
    // tensor overruns the file by 8 bytes). This forces control into the
    // subtractive bounds check in load_weights (nbytes > size - data_off - off),
    // not the earlier magic/metadata rejection — i.e. the exact hardened path.
    write_truncated_copy(good, bad, data_off + nbytes - 8);

    // The load must fail gracefully (false), not SIGBUS. Reaching this REQUIRE at
    // all means no crash; the value check confirms it was rejected.
    {
        core_gguf::WeightLoad wl;
        REQUIRE_FALSE(core_gguf::load_weights(bad.c_str(), backend, "test-trunc", wl));
    }

    std::remove(good.c_str());
    std::remove(bad.c_str());
    ggml_backend_free(backend);
}

TEST_CASE("a rejected GGUF leaves no mapping behind on the zero-copy path", "[unit][gguf-bounds]") {
    // The zero-copy path maps the whole file and hands it to the device before
    // it validates tensor bounds, so the rejection this file's first case
    // covers happens with a mapping already registered against the backend
    // buffer. Returning false there looks safe and is not: without an explicit
    // release, both the buffer and the whole-file mapping are abandoned — and
    // this branch is reached by a truncated or crafted GGUF, which is
    // attacker-supplied input, so the leak is reachable on demand.
    if (!test_region::region_probe_available()) {
        SUCCEED("region enumeration unavailable on this platform");
        return;
    }
    test_setenv("CRISPASR_GGUF_MMAP", "1");

    ggml_backend_t backend = init_host_ptr_gpu_backend();
    if (!backend) {
        SUCCEED("no GPU device advertising buffer_from_host_ptr — this path does not exist here");
        return;
    }

    const std::string good = "crispasr_test_gguf_reject_ok.gguf";
    const std::string bad = "crispasr_test_gguf_reject_trunc.gguf";
    const int n = 65536; // 256 KiB of tensor data
    write_valid_gguf(good, n);
    const size_t data_off = read_data_offset(good);
    write_truncated_copy(good, bad, data_off + (size_t)n * sizeof(float) - 8);

    const std::string bad_abs = test_region::absolute_path_of(bad);
    REQUIRE(test_region::count_regions_backed_by(bad_abs) == 0);

    core_gguf::WeightLoad wl;
    REQUIRE_FALSE(core_gguf::load_weights(bad.c_str(), backend, "test-reject", wl));
    REQUIRE(test_region::count_regions_backed_by(bad_abs) == 0);

    std::remove(good.c_str());
    std::remove(bad.c_str());
    ggml_backend_free(backend);
}

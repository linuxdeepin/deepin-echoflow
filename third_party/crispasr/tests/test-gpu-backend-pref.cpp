// test-gpu-backend-pref.cpp — unit tests for the `--gpu-backend cpu`
// short-circuit in core/gpu_backend_pref.h (CrispEmbed T18 sync) and the
// Metal pipeline-cache cap policy (PLAN #88 follow-up).
//
// Each TEST_CASE runs as its own ctest process (catch_discover_tests), so the
// process-global preference and the once-per-process policy latch are fresh
// per case.
//
// The legacy-arm case doubles as the proof the gate can go red: with
// CRISPASR_GPU_PREF_CPU_LEGACY=1 the helper reproduces the pre-fix behaviour
// (fall through to ggml_backend_init_best), and on any box with a GPU backend
// compiled in that returns a non-CPU backend — exactly what the fix forbids
// by default.

#include "core/gpu_backend_pref.h"
#include "core/metal_pipeline_cache_policy.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include "portable_env.h"

namespace {

// Is any non-CPU device registered? (Statically-linked GPU backends
// self-register; no ggml_backend_load_all() needed for them.)
bool have_gpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        enum ggml_backend_dev_type dt = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU)
            return true;
    }
    return false;
}

} // namespace

TEST_CASE("gpu-backend-pref: cpu short-circuits to the CPU backend", "[unit]") {
    unsetenv("CRISPASR_GPU_PREF_CPU_LEGACY");
    crispasr_set_gpu_backend_pref("cpu");
    ggml_backend_t b = crispasr_init_gpu_backend();
    REQUIRE(b != nullptr);
    // The whole point of the flag: no GPU backend, ever. Pre-fix this
    // returned ggml_backend_init_best() (Metal on Apple Silicon).
    REQUIRE(ggml_backend_is_cpu(b));
    ggml_backend_free(b);
}

TEST_CASE("gpu-backend-pref: legacy gate restores the pre-fix fall-through", "[unit]") {
    setenv("CRISPASR_GPU_PREF_CPU_LEGACY", "1", 1);
    crispasr_set_gpu_backend_pref("cpu");
    ggml_backend_t b = crispasr_init_gpu_backend();
    unsetenv("CRISPASR_GPU_PREF_CPU_LEGACY");
    REQUIRE(b != nullptr);
    if (have_gpu_device()) {
        // Red-gate proof: the legacy path is the defect the fix removes.
        REQUIRE(!ggml_backend_is_cpu(b));
    }
    ggml_backend_free(b);
}

TEST_CASE("gpu-backend-pref: CPU_LEGACY=0 still short-circuits (value-parsed)", "[unit]") {
    setenv("CRISPASR_GPU_PREF_CPU_LEGACY", "0", 1);
    crispasr_set_gpu_backend_pref("cpu");
    ggml_backend_t b = crispasr_init_gpu_backend();
    unsetenv("CRISPASR_GPU_PREF_CPU_LEGACY");
    REQUIRE(b != nullptr);
    REQUIRE(ggml_backend_is_cpu(b));
    ggml_backend_free(b);
}

TEST_CASE("gpu-backend-pref: empty pref still returns a backend", "[unit]") {
    crispasr_set_gpu_backend_pref("");
    ggml_backend_t b = crispasr_init_gpu_backend();
    REQUIRE(b != nullptr);
    ggml_backend_free(b);
}

TEST_CASE("metal-pipeline-cache-policy: oversized archive disables the cache", "[unit]") {
#if defined(__APPLE__)
    // Point the policy at a scratch dir holding a 2 MB fake archive with a
    // 1 MB cap; apply() must set ggml's own kill switch and say so.
    std::string dir = "/tmp/crispasr-test-metal-cache";
    std::string cmd = "mkdir -p " + dir;
    REQUIRE(std::system(cmd.c_str()) == 0);
    {
        std::ofstream f(dir + "/Fake_Device.archive", std::ios::binary);
        std::string mb(1024 * 1024, 'x');
        f << mb << mb;
    }
    unsetenv("GGML_METAL_PIPELINE_CACHE_DISABLE");
    setenv("GGML_METAL_PIPELINE_CACHE", dir.c_str(), 1);
    setenv("CRISPASR_METAL_PIPELINE_CACHE_MAX_MB", "1", 1);
    REQUIRE(core_metal_cache::apply());
    const char* dis = std::getenv("GGML_METAL_PIPELINE_CACHE_DISABLE");
    REQUIRE(dis != nullptr);
    REQUIRE(std::string(dis) == "1");
#else
    REQUIRE(!core_metal_cache::apply());
#endif
}

TEST_CASE("metal-pipeline-cache-policy: MAX_MB=0 means uncapped", "[unit]") {
#if defined(__APPLE__)
    std::string dir = "/tmp/crispasr-test-metal-cache";
    std::string cmd = "mkdir -p " + dir;
    REQUIRE(std::system(cmd.c_str()) == 0);
    {
        std::ofstream f(dir + "/Fake_Device.archive", std::ios::binary);
        std::string mb(1024 * 1024, 'x');
        f << mb << mb;
    }
    unsetenv("GGML_METAL_PIPELINE_CACHE_DISABLE");
    setenv("GGML_METAL_PIPELINE_CACHE", dir.c_str(), 1);
    setenv("CRISPASR_METAL_PIPELINE_CACHE_MAX_MB", "0", 1);
    REQUIRE(!core_metal_cache::apply());
    REQUIRE(std::getenv("GGML_METAL_PIPELINE_CACHE_DISABLE") == nullptr);
#endif
}

// test-vibevoice-enc-policy.cpp — issue #418: σ-VAE tokenizer-encoder
// backend decision table.
//
// Hermetic: no model, no GPU, no Vulkan. The policy routes the VibeVoice-ASR
// acoustic/semantic encoders to CPU on Vulkan devices whose
// maxComputeWorkGroupCount is 65535 (Intel Arc/Iris/UHD, llvmpipe) — the
// conv dispatches over long 24 kHz audio overflow there and ggml aborts in
// ggml_vk_dispatch_pipeline. A wrong default is invisible on NVIDIA/AMD
// boxes (both paths work), so the table gets its own test — same reasoning
// as the #402 chatterbox attention policy.

#include "vibevoice_backend_policy.h"

#include <catch2/catch_test_macros.hpp>

using vibevoice_policy::enc_use_cpu;
using vibevoice_policy::low_workgroup_limit_device;

TEST_CASE("enc policy: low-workgroup-limit Vulkan devices divert to CPU", "[unit][vibevoice][enc-policy]") {
    // The reporter's card (#418) and the issue-#52 family.
    CHECK(enc_use_cpu(nullptr, "Vulkan0", "Intel(R) Arc(TM) B580 Graphics"));
    CHECK(enc_use_cpu(nullptr, "Vulkan0", "Intel(R) Iris(R) Xe Graphics"));
    CHECK(enc_use_cpu(nullptr, "Vulkan0", "Intel(R) UHD Graphics"));
    // Mesa's software rasterizer reports the same 65535 limit — it is also
    // the only reproduction venue we have without Intel hardware.
    CHECK(enc_use_cpu(nullptr, "Vulkan0", "llvmpipe (LLVM 20.1.2, 256 bits)"));
}

TEST_CASE("enc policy: big-limit devices and non-Vulkan backends stay put", "[unit][vibevoice][enc-policy]") {
    CHECK_FALSE(enc_use_cpu(nullptr, "Vulkan0", "NVIDIA GeForce RTX 4060"));
    CHECK_FALSE(enc_use_cpu(nullptr, "Vulkan0", "AMD Radeon RX 7900 XTX (RADV NAVI31)"));
    // Same device names on a NON-Vulkan backend never divert — the limit is
    // a Vulkan-dispatch property, not a device property in general.
    CHECK_FALSE(enc_use_cpu(nullptr, "CUDA0", "NVIDIA GeForce RTX 4060"));
    CHECK_FALSE(enc_use_cpu(nullptr, "Metal", "Apple M1"));
    CHECK_FALSE(enc_use_cpu(nullptr, "CPU", nullptr));
    CHECK_FALSE(enc_use_cpu(nullptr, nullptr, nullptr));
}

TEST_CASE("enc policy: env overrides win in both directions", "[unit][vibevoice][enc-policy]") {
    // gpu: the repro arm — run on the active backend even where known to abort.
    CHECK_FALSE(enc_use_cpu("gpu", "Vulkan0", "Intel(R) Arc(TM) B580 Graphics"));
    CHECK_FALSE(enc_use_cpu("gpu", "Vulkan0", "llvmpipe (LLVM 20.1.2, 256 bits)"));
    // cpu: force the fallback anywhere.
    CHECK(enc_use_cpu("cpu", "Vulkan0", "NVIDIA GeForce RTX 4060"));
    CHECK(enc_use_cpu("cpu", "CUDA0", "NVIDIA GeForce RTX 4060"));
    // Unrecognised values behave like auto.
    CHECK(enc_use_cpu("auto", "Vulkan0", "Intel(R) Arc(TM) B580 Graphics"));
    CHECK_FALSE(enc_use_cpu("auto", "Vulkan0", "NVIDIA GeForce RTX 4060"));
}

TEST_CASE("enc policy: name matcher shared with the #52 decoder fallback", "[unit][vibevoice][enc-policy]") {
    CHECK(low_workgroup_limit_device("Intel(R) Arc(TM) Graphics"));
    CHECK(low_workgroup_limit_device("llvmpipe (LLVM 15.0.7, 256 bits)"));
    CHECK_FALSE(low_workgroup_limit_device("NVIDIA GeForce RTX 4060"));
    CHECK_FALSE(low_workgroup_limit_device(nullptr));
}

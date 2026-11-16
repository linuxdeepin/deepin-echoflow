// VoxCPM2 AudioVAE upscaler parameter and null-guard tests. No GGUF required.
#include <catch2/catch_test_macros.hpp>

#include "voxcpm2_vae.h"

TEST_CASE("voxcpm2_vae defaults are sensible", "[unit][voxcpm2-vae]") {
    const voxcpm2_vae_context_params p = voxcpm2_vae_context_default_params();
    CHECK(p.n_threads >= 1);
    CHECK(p.verbosity >= 0);
}

TEST_CASE("voxcpm2_vae rejects invalid initialization", "[unit][voxcpm2-vae]") {
    const voxcpm2_vae_context_params p = voxcpm2_vae_context_default_params();
    CHECK(voxcpm2_vae_init_from_file(nullptr, p) == nullptr);
    CHECK(voxcpm2_vae_init_from_file("", p) == nullptr);
}

TEST_CASE("voxcpm2_vae null operations are safe", "[unit][voxcpm2-vae]") {
    int n = 123;
    CHECK(voxcpm2_vae_upscale(nullptr, nullptr, 0, &n) == nullptr);
    CHECK(n == 0);
    voxcpm2_vae_pcm_free(nullptr);
    voxcpm2_vae_free(nullptr);
}

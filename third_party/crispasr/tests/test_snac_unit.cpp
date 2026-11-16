// Unit + live tests for core/snac.h — SNAC 24kHz decoder.
//
// Unit tests verify API surface compiles and links (no model needed).
// Live tests require CRISPASR_MODEL_SNAC pointing to snac-24khz.gguf.

#include <catch2/catch_test_macros.hpp>

#include "core/snac.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Unit tests — no model needed, verify API surface
// ---------------------------------------------------------------------------

TEST_CASE("snac: default_params returns sane defaults", "[unit][snac]") {
    snac_decoder_params p = snac_decoder_default_params();
    CHECK(p.n_threads > 0);
    CHECK(p.verbosity == 1);
    CHECK(p.use_gpu == false);
}

TEST_CASE("snac: init_from_file returns null for missing file", "[unit][snac]") {
    snac_decoder_params p = snac_decoder_default_params();
    p.verbosity = 0;
    snac_decoder_ctx* ctx = snac_decoder_init_from_file("/nonexistent/snac.gguf", p);
    CHECK(ctx == nullptr);
}

TEST_CASE("snac: free accepts null", "[unit][snac]") {
    snac_decoder_free(nullptr); // must not crash
}

// ---------------------------------------------------------------------------
// Live tests — need CRISPASR_MODEL_SNAC
// ---------------------------------------------------------------------------

TEST_CASE("snac: load + query metadata", "[live][snac]") {
    const char* path = std::getenv("CRISPASR_MODEL_SNAC");
    if (!path || !*path)
        SKIP("CRISPASR_MODEL_SNAC not set");

    snac_decoder_params p = snac_decoder_default_params();
    p.verbosity = 0;
    snac_decoder_ctx* ctx = snac_decoder_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    CHECK(snac_decoder_sample_rate(ctx) == 24000);
    CHECK(snac_decoder_n_codebooks(ctx) == 3);
    CHECK(snac_decoder_hop_length(ctx) == 512);

    uint32_t strides[3] = {};
    snac_decoder_vq_strides(ctx, strides);
    CHECK(strides[0] == 4);
    CHECK(strides[1] == 2);
    CHECK(strides[2] == 1);

    snac_decoder_free(ctx);
}

TEST_CASE("snac: decode zeros produces non-null output", "[live][snac]") {
    const char* path = std::getenv("CRISPASR_MODEL_SNAC");
    if (!path || !*path)
        SKIP("CRISPASR_MODEL_SNAC not set");

    snac_decoder_params p = snac_decoder_default_params();
    p.verbosity = 0;
    snac_decoder_ctx* ctx = snac_decoder_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    // 1 super-frame: c0=1, c1=2, c2=4 codes (all zeros = silence-ish)
    int32_t c0[] = {0};
    int32_t c1[] = {0, 0};
    int32_t c2[] = {0, 0, 0, 0};
    int n_samples = 0;

    float* pcm = snac_decoder_decode(ctx, c0, 1, c1, 2, c2, 4, &n_samples);
    REQUIRE(pcm != nullptr);
    CHECK(n_samples == 2048); // 1 super-frame × 512 hop × 4 stride = 2048

    // Output should be finite
    bool all_finite = true;
    for (int i = 0; i < n_samples; i++) {
        if (!std::isfinite(pcm[i])) {
            all_finite = false;
            break;
        }
    }
    CHECK(all_finite);

    free(pcm);
    snac_decoder_free(ctx);
}

TEST_CASE("snac: decode rejects mismatched lengths", "[live][snac]") {
    const char* path = std::getenv("CRISPASR_MODEL_SNAC");
    if (!path || !*path)
        SKIP("CRISPASR_MODEL_SNAC not set");

    snac_decoder_params p = snac_decoder_default_params();
    p.verbosity = 0;
    snac_decoder_ctx* ctx = snac_decoder_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    // Wrong ratios: c0=1 but c1=1 (should be 2)
    int32_t c0[] = {0};
    int32_t c1[] = {0};
    int32_t c2[] = {0, 0, 0, 0};
    int n_samples = 0;

    float* pcm = snac_decoder_decode(ctx, c0, 1, c1, 1, c2, 4, &n_samples);
    CHECK(pcm == nullptr); // should reject

    snac_decoder_free(ctx);
}

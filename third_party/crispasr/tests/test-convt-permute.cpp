// core_convt::permute_convt1d_weight must equal the plain [K, OC, IC] ->
// [IC, K*OC] permutation byte for byte, for F32 and F16 sources and for
// shapes that are not multiples of its 32x32 transpose tile (#461: the tiled
// version replaced a naive loop that cost ~1.1 s of every voxcpm2 process
// start; 22 backends share this helper, so any index slip would corrupt all
// their transposed convs).

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <vector>

#include "core/conv.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

// The pre-#461 loop, kept verbatim as the reference.
std::vector<float> reference(const std::vector<float>& tmp, int K, int OC, int IC) {
    std::vector<float> dp((size_t)IC * K * OC);
    for (int ic = 0; ic < IC; ic++)
        for (int oc = 0; oc < OC; oc++)
            for (int k = 0; k < K; k++)
                dp[(size_t)(oc * K + k) * IC + ic] = tmp[(size_t)ic * OC * K + oc * K + k];
    return dp;
}

void check(ggml_type type, int K, int OC, int IC) {
    ggml_init_params ip = {ggml_tensor_overhead() * 4, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_3d(ctx, type, K, OC, IC);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    REQUIRE(buf != nullptr);

    const size_t n = (size_t)K * OC * IC;
    std::vector<float> vals(n);
    for (size_t i = 0; i < n; i++)
        vals[i] = (float)((i * 2654435761u) % 4099) * 0.25f - 512.0f; // exact in F16 too
    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(w, vals.data(), 0, n * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> h(n);
        for (size_t i = 0; i < n; i++)
            h[i] = ggml_fp32_to_fp16(vals[i]);
        ggml_backend_tensor_set(w, h.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++)
            vals[i] = ggml_fp16_to_fp32(h[i]);
    }

    auto got = core_convt::permute_convt1d_weight(w);
    const std::vector<float> want = reference(vals, K, OC, IC);
    CHECK(std::memcmp(got.get(), want.data(), n * sizeof(float)) == 0);
    // Positive control: unless IC or OC*K is 1 (then it IS the identity), the
    // permutation moves data, so a helper that skipped it fails the memcmp above.
    if (IC > 1 && OC * K > 1)
        CHECK(std::memcmp(vals.data(), want.data(), n * sizeof(float)) != 0);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

} // namespace

TEST_CASE("permute_convt1d_weight == naive permutation (F32)", "[unit][core][convt]") {
    check(GGML_TYPE_F32, 16, 64, 128); // tile-aligned
    check(GGML_TYPE_F32, 3, 5, 37);    // ragged in both tile dimensions
    check(GGML_TYPE_F32, 10, 33, 70);
    check(GGML_TYPE_F32, 4, 1, 1);
}

TEST_CASE("permute_convt1d_weight == naive permutation (F16)", "[unit][core][convt]") {
    check(GGML_TYPE_F16, 16, 64, 128);
    check(GGML_TYPE_F16, 7, 9, 45);
}

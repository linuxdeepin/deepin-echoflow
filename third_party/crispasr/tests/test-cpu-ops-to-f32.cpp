// test-cpu-ops-to-f32.cpp — unit tests for core_cpu::to_f32.
//
// to_f32 is the quantized-safe CPU weight reader: it must dequantize F16 and any
// block-quantized type correctly, not just F32. A raw
// `ggml_backend_tensor_get(t, buf, 0, n*sizeof(float))` over-reads a quantized
// tensor (nbytes << n*4) and an F16-only reader leaves quantized weights as
// garbage — the bug class this helper closes (paraformer CIF, piper emb,
// parakeet ctc, and the CrispEmbed pcs/DeBERTa/MLM-head crashes).

#include <catch2/catch_test_macros.hpp>

#include "core/cpu_ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

// Build a 2-D [n0,n1] tensor of `type` from f32 `src`, read it back through
// core_cpu::to_f32, and return the dequantized result.
std::vector<float> roundtrip(ggml_type type, const std::vector<float>& src, int64_t n0, int64_t n1) {
    ggml_init_params ip{ggml_tensor_overhead() + 512, nullptr, /*no_alloc=*/true};
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* t = ggml_new_tensor_2d(ctx, type, n0, n1);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<uint8_t> raw(ggml_nbytes(t));
    if (type == GGML_TYPE_F32) {
        std::memcpy(raw.data(), src.data(), raw.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t*)raw.data(), (int64_t)src.size());
    } else {
        ggml_quantize_chunk(type, src.data(), raw.data(), 0, n1, n0, nullptr);
    }
    ggml_backend_tensor_set(t, raw.data(), 0, raw.size());

    std::vector<float> out = core_cpu::to_f32(t);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return out;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

// Smooth data in [-0.9, 0.9]; n0 = 256 keeps rows block-aligned for Q4_K/Q8_0.
std::vector<float> make_src(int64_t n0, int64_t n1) {
    std::vector<float> v((size_t)(n0 * n1));
    for (size_t i = 0; i < v.size(); i++)
        v[i] = 0.9f * std::sin(0.013 * (double)i);
    return v;
}

} // namespace

TEST_CASE("to_f32: null tensor returns empty", "[unit]") {
    REQUIRE(core_cpu::to_f32(nullptr).empty());
}

TEST_CASE("to_f32: F32 is exact", "[unit]") {
    auto src = make_src(256, 4);
    auto out = roundtrip(GGML_TYPE_F32, src, 256, 4);
    REQUIRE(out.size() == src.size());
    REQUIRE(max_abs(out, src) == 0.0);
}

TEST_CASE("to_f32: F16 within precision", "[unit]") {
    auto src = make_src(256, 4);
    auto out = roundtrip(GGML_TYPE_F16, src, 256, 4);
    REQUIRE(out.size() == src.size());
    REQUIRE(max_abs(out, src) < 1e-3);
}

TEST_CASE("to_f32: Q8_0 dequantizes to the original (not garbage)", "[unit]") {
    auto src = make_src(256, 4);
    auto out = roundtrip(GGML_TYPE_Q8_0, src, 256, 4);
    REQUIRE(out.size() == src.size());
    REQUIRE(cosine(out, src) > 0.999);
    REQUIRE(max_abs(out, src) < 0.05);
}

TEST_CASE("to_f32: Q4_K dequantizes to the original (not garbage)", "[unit]") {
    auto src = make_src(256, 8);
    auto out = roundtrip(GGML_TYPE_Q4_K, src, 256, 8);
    REQUIRE(out.size() == src.size());
    REQUIRE(cosine(out, src) > 0.99);
}

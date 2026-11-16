// Regression for PR #410: flash_attn_ext consumes a layer slice of the
// Chatterbox GPT-2 KV cache directly. Exercise a non-zero layer offset and
// compare it with the old ggml_cont materialization path.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <vector>

TEST_CASE("flash attention accepts an offset KV-cache layer view", "[unit][chatterbox][attention]") {
    constexpr int64_t hd = 32;
    constexpr int64_t max_ctx = 32;
    constexpr int64_t lk = 17;
    constexpr int64_t heads = 2;
    constexpr int64_t layers = 3;
    constexpr int64_t layer = 1;

    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);

    ggml_init_params cache_params{ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    ggml_context* cache_ctx = ggml_init(cache_params);
    REQUIRE(cache_ctx != nullptr);
    ggml_tensor* cache_k = ggml_new_tensor_4d(cache_ctx, GGML_TYPE_F32, hd, max_ctx, heads, layers);
    ggml_tensor* cache_v = ggml_new_tensor_4d(cache_ctx, GGML_TYPE_F32, hd, max_ctx, heads, layers);
    ggml_backend_buffer_t cache_buffer = ggml_backend_alloc_ctx_tensors(cache_ctx, backend);
    REQUIRE(cache_buffer != nullptr);

    const size_t cache_elements = (size_t)hd * max_ctx * heads * layers;
    std::vector<float> k_data(cache_elements);
    std::vector<float> v_data(cache_elements);
    for (size_t i = 0; i < cache_elements; ++i) {
        k_data[i] = 0.15f * std::sin((float)i * 0.013f);
        v_data[i] = 0.20f * std::cos((float)i * 0.017f);
    }
    ggml_backend_tensor_set(cache_k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(cache_v, v_data.data(), 0, v_data.size() * sizeof(float));

    std::vector<unsigned char> meta(4u * 1024 * 1024);
    ggml_init_params graph_params{meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(graph_params);
    REQUIRE(ctx != nullptr);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 128, false);

    ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, heads);
    ggml_set_input(q);
    const size_t layer_offset = (size_t)layer * cache_k->nb[3];
    ggml_tensor* k_view = ggml_view_3d(ctx, cache_k, hd, lk, heads, cache_k->nb[1], cache_k->nb[2], layer_offset);
    ggml_tensor* v_view = ggml_view_3d(ctx, cache_v, hd, lk, heads, cache_v->nb[1], cache_v->nb[2], layer_offset);

    ggml_tensor* direct = ggml_flash_attn_ext(ctx, q, k_view, v_view, nullptr, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
    ggml_tensor* copied = ggml_flash_attn_ext(ctx, q, ggml_cont(ctx, k_view), ggml_cont(ctx, v_view), nullptr,
                                              1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
    ggml_set_output(direct);
    ggml_set_output(copied);
    ggml_build_forward_expand(graph, direct);
    ggml_build_forward_expand(graph, copied);

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    REQUIRE(allocator != nullptr);
    REQUIRE(ggml_gallocr_alloc_graph(allocator, graph));
    std::vector<float> q_data((size_t)hd * heads);
    for (size_t i = 0; i < q_data.size(); ++i)
        q_data[i] = 0.10f * std::sin((float)i * 0.071f);
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> direct_data((size_t)hd * heads);
    std::vector<float> copied_data(direct_data.size());
    ggml_backend_tensor_get(direct, direct_data.data(), 0, direct_data.size() * sizeof(float));
    ggml_backend_tensor_get(copied, copied_data.data(), 0, copied_data.size() * sizeof(float));
    for (size_t i = 0; i < direct_data.size(); ++i)
        REQUIRE(direct_data[i] == Catch::Approx(copied_data[i]).margin(1e-6f));

    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    ggml_backend_buffer_free(cache_buffer);
    ggml_free(cache_ctx);
    ggml_backend_free(backend);
}

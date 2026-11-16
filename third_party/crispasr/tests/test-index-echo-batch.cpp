#include <catch2/catch_test_macros.hpp>
#include "core/index_echo_batch.h"
#include "llama-batch.h"
#include "llama-vocab.h"
#include "qwen35-norm.h"
#include "ggml-cpu.h"
#include <cmath>
#include <memory>

TEST_CASE("Index-Echo audio positions survive llama M-RoPE microbatch splitting", "[unit][index-echo]") {
    llama_batch batch = llama_batch_init(5, 8, 1);
    batch.n_tokens = 5;
    REQUIRE(core_index_echo::positions(batch, 137, 4));
    for (int i = 0; i < batch.n_tokens; ++i) {
        for (int d = 0; d < 8; ++d)
            batch.embd[i * 8 + d] = 0;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == 4;
    }
    llama_vocab vocab;
    llama_batch_allocr allocator(4);
    REQUIRE(allocator.init(batch, vocab, nullptr, 8, 1, false));
    allocator.split_reset();
    int consumed = 0;
    while (consumed < batch.n_tokens) {
        auto microbatch = allocator.split_simple(2);
        REQUIRE(microbatch.n_tokens > 0);
        REQUIRE(microbatch.n_pos == 4);
        for (unsigned axis = 0; axis < microbatch.n_pos; ++axis)
            for (unsigned i = 0; i < microbatch.n_tokens; ++i)
                REQUIRE(microbatch.pos[axis * microbatch.n_tokens + i] == 137 + consumed + (int)i);
        consumed += microbatch.n_tokens;
    }
    llama_batch_free(batch);
}

TEST_CASE("Qwen3.5 query/key norm adds epsilon inside the squared norm", "[unit][index-echo]") {
    const ggml_init_params params = {1024 * 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    REQUIRE(ctx);
    auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 2);
    auto* values = (float*)input->data;
    double sum = 0;
    for (int i = 0; i < 128; ++i) {
        values[i] = i % 2 ? 2e-4f : 1e-4f;
        sum += (double)values[i] * values[i];
        values[128 + i] = 0;
    }
    auto* output = llama_qwen35::l2_norm(ctx.get(), input);
    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    REQUIRE(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS);
    const auto* actual = (const float*)output->data;
    // Independent scalar oracle for the pinned Transformers formula. Small
    // heads expose the clamp/add difference; a zero head must remain finite.
    for (int i = 0; i < 128; ++i) {
        CHECK(std::abs(actual[i] - values[i] / std::sqrt(sum + 1e-6)) < 1e-6);
        CHECK(actual[128 + i] == 0);
    }
}

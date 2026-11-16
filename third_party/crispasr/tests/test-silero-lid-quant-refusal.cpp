// test-silero-lid-quant-refusal.cpp — the #409 tail: silero-lid must REFUSE
// quantized weights instead of misdetecting.
//
// Hermetic: no model download, no network. The published q8_0/q5_0
// quantizations of silero-lid-95 were broken by construction — the ggml
// graph deflated every logit into junk (clean English read pa-IN at p≈0.02)
// and the legacy path dereferenced tensor->data as f32 and produced NaN
// confidences. Worse, the #409 evidence floor did NOT catch it: the junk
// top-logits sat above the floor while the softmax mass was noise. The only
// safe behaviour for a 16 MB f32 classifier is to refuse quantized weights
// loudly (silero_lid.cpp lid_load), and this test pins that refusal:
//
//   1. a model whose weights include ONE quantized tensor → init nullptr,
//   2. the identical stub with f32 weights → init succeeds (the refusal
//      keys on quantization, not on the stub-ness of the fixture).
//
// The fixture carries exactly the two tensors lid_load names critical
// (lid.lang.weight, lid.pool.weight); every other tensor is optional at
// load time, so the control arm exercises the same accept path the real
// f32 file takes.

#include "silero_lid.h"

#include "ggml.h"
#include "gguf.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ne0 x ne1 tensor of `type` with deterministic contents. Quantized types go
// through ggml_quantize_chunk so the on-disk blocks are genuine.
ggml_tensor* make_tensor(ggml_context* ctx, const char* name, ggml_type type, int ne0, int ne1) {
    ggml_tensor* t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_name(t, name);
    std::vector<float> src((size_t)ne0 * ne1);
    for (size_t i = 0; i < src.size(); i++)
        src[i] = 0.01f * (float)(i % 97) - 0.5f;
    if (type == GGML_TYPE_F32) {
        memcpy(t->data, src.data(), src.size() * sizeof(float));
    } else {
        ggml_quantize_chunk(type, src.data(), t->data, 0, ne1, ne0, nullptr);
    }
    return t;
}

std::string write_stub(const char* tag, ggml_type lang_type) {
    std::string path = std::string("test-silero-quant-refusal.") + tag + ".tmp.gguf";
    ggml_init_params ip{2 * ggml_tensor_overhead() + (size_t)(1 << 20), nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);

    // 256 is a multiple of every quant block size in play (q8_0 = 32).
    ggml_tensor* lang = make_tensor(ctx, "lid.lang.weight", lang_type, 256, 8);
    ggml_tensor* pool = make_tensor(ctx, "lid.pool.weight", GGML_TYPE_F32, 256, 4);

    gguf_context* g = gguf_init_empty();
    REQUIRE(g != nullptr);
    gguf_set_val_str(g, "general.architecture", "silero_lid");
    gguf_add_tensor(g, lang);
    gguf_add_tensor(g, pool);
    gguf_write_to_file(g, path.c_str(), /*only_meta*/ false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

} // namespace

TEST_CASE("silero-lid refuses a model with quantized weights", "[unit][lid][silero-quant]") {
    const std::string path = write_stub("q8", GGML_TYPE_Q8_0);
    silero_lid_context* ctx = silero_lid_init(path.c_str(), 2);
    // Pre-guard, this loaded and then misdetected (pa-IN / NaN). The guard
    // must turn it into a clean init failure.
    REQUIRE(ctx == nullptr);
    std::remove(path.c_str());
}

TEST_CASE("silero-lid still accepts f32 weights (refusal does not over-fire)", "[unit][lid][silero-quant]") {
    const std::string path = write_stub("f32", GGML_TYPE_F32);
    silero_lid_context* ctx = silero_lid_init(path.c_str(), 2);
    REQUIRE(ctx != nullptr);
    silero_lid_free(ctx);
    std::remove(path.c_str());
}

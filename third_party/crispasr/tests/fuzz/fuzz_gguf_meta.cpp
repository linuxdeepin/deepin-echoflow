// fuzz_gguf_meta.cpp — libFuzzer harness over GGUF metadata parsing.
//
// A GGUF model file is untrusted input (users download/convert models). This
// fuzzes core_gguf::open_metadata() plus the array/KV accessors that read
// counts + string lengths straight out of the file (kv_str_array is the vocab
// path — counts and per-string lengths from the file), i.e. the code that would
// be reached by a malicious model's tokenizer arrays and KV table.
//
// GGUF has a magic + structured header, so seed from a real (small) .gguf for
// useful coverage:
//   cmake -B build-fuzz -DCRISPASR_FUZZ=ON -DCRISPASR_SANITIZE_ADDRESS=ON \
//         -DCRISPASR_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
//   cmake --build build-fuzz --target crispasr-fuzz-gguf
//   mkdir -p corpus && cp some-small-model.gguf corpus/
//   ./build-fuzz/bin/crispasr-fuzz-gguf -max_len=2097152 corpus

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "core/gguf_loader.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size > 8u * 1024u * 1024u)
        return 0;

    const char* path = "crispasr_fuzz_gguf.bin";
    FILE* f = std::fopen(path, "wb");
    if (!f)
        return 0;
    if (size)
        std::fwrite(data, 1, size, f);
    std::fclose(f);

    gguf_context* g = core_gguf::open_metadata(path);
    if (g) {
        // Scalar KV accessors (default-returning; exercise the type-dispatch).
        (void)core_gguf::kv_str(g, "general.architecture", "");
        (void)core_gguf::kv_u32(g, "general.file_type", 0);
        (void)core_gguf::kv_i32(g, "general.quantization_version", 0);
        (void)core_gguf::kv_f32(g, "general.rope_freq_base", 0.0f);
        (void)core_gguf::kv_bool(g, "general.use_parallel_residual", false);
        // Array accessors — the vocab path: count + per-string lengths from file.
        (void)core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
        (void)core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
        (void)core_gguf::kv_f32_array(g, "tokenizer.ggml.scores");
        core_gguf::free_metadata(g);
    }
    return 0;
}

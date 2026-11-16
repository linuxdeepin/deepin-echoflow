// test_audioseal_cosine.cpp — compare ggml AudioSeal output against PyTorch reference.
//
// Usage:
//   1. python3 -c "..." to generate /tmp/audioseal_ref_input.npy + /tmp/audioseal_ref_watermarked.npy
//   2. AUDIOSEAL_GGUF=/tmp/audioseal.gguf build/bin/test_audioseal_cosine
//
// Reports per-stage cosine similarity.

#include "audioseal.h"
#include "core/crispasr_env.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Minimal .npy reader for float32 1D arrays
static std::vector<float> load_npy(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return {};
    }
    // Skip header: starts with \x93NUMPY, then version, then header_len, then dict
    char magic[6];
    fread(magic, 1, 6, f);
    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);
    uint16_t header_len;
    fread(&header_len, 2, 1, f);
    // Skip the header dict
    fseek(f, 6 + 2 + 2 + header_len, SEEK_SET);
    // Read rest as float32
    long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, pos, SEEK_SET);
    int n = (int)((end - pos) / sizeof(float));
    std::vector<float> data(n);
    fread(data.data(), sizeof(float), n, f);
    fclose(f);
    return data;
}

static double cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na < 1e-12 || nb < 1e-12)
        return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

int main() {
    const char* gguf_path = crispasr_env::get("CRISPASR_AUDIOSEAL_GGUF");
    if (!gguf_path) {
        fprintf(stderr, "Set AUDIOSEAL_GGUF=/tmp/audioseal.gguf\n");
        return 1;
    }

    // Load reference
    auto input = load_npy("/tmp/audioseal_ref_input.npy");
    auto ref_output = load_npy("/tmp/audioseal_ref_watermarked.npy");
    if (input.empty() || ref_output.empty()) {
        fprintf(stderr, "Run the Python reference generator first\n");
        return 1;
    }
    printf("Input: %d samples\n", (int)input.size());
    printf("Reference output: %d samples\n", (int)ref_output.size());

    // Load AudioSeal
    auto params = audioseal_default_params();
    params.verbosity = 0;
    auto* ctx = audioseal_init_from_file(gguf_path, params);
    if (!ctx) {
        fprintf(stderr, "Failed to load AudioSeal GGUF\n");
        return 2;
    }

    // Embed with default message (all ones)
    uint8_t msg[16];
    memset(msg, 1, 16);
    float* our_output = audioseal_embed(ctx, input.data(), (int)input.size(), msg);
    if (!our_output) {
        fprintf(stderr, "audioseal_embed returned null\n");
        audioseal_free(ctx);
        return 3;
    }

    // Compare
    int n = (int)std::min(input.size(), ref_output.size());
    double cos_full = cosine_similarity(our_output, ref_output.data(), n);
    printf("\nCosine similarity (full output): %.6f\n", cos_full);

    // Compare watermark only (output - input)
    std::vector<float> our_wm(n), ref_wm(n);
    for (int i = 0; i < n; i++) {
        our_wm[i] = our_output[i] - input[i];
        ref_wm[i] = ref_output[i] - input[i];
    }
    double cos_wm = cosine_similarity(our_wm.data(), ref_wm.data(), n);
    printf("Cosine similarity (watermark only): %.6f\n", cos_wm);

    // RMS of watermark
    double our_rms = 0, ref_rms = 0;
    for (int i = 0; i < n; i++) {
        our_rms += our_wm[i] * our_wm[i];
        ref_rms += ref_wm[i] * ref_wm[i];
    }
    our_rms = std::sqrt(our_rms / n);
    ref_rms = std::sqrt(ref_rms / n);
    printf("Watermark RMS: ours=%.6f  ref=%.6f  ratio=%.3f\n", our_rms, ref_rms, our_rms / (ref_rms + 1e-12));

    // Max absolute error
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float err = std::abs(our_output[i] - ref_output[i]);
        if (err > max_err)
            max_err = err;
    }
    printf("Max absolute error: %.6f\n", max_err);

    printf("\nResult: %s\n", cos_full > 0.95 ? "PASS (>95%% cosine)" : "FAIL (<95%% cosine)");

    free(our_output);
    audioseal_free(ctx);
    return cos_full > 0.95 ? 0 : 4;
}

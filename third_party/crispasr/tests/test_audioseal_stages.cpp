// test_audioseal_stages.cpp — per-stage cosine comparison against PyTorch.
//
// Loads reference .npy files from /tmp/as_enc_*.npy and compares against
// the ggml AudioSeal encoder output at each stage.
//
// This test adds named output tensors at each encoder stage so we can
// extract and compare them.

#include "audioseal.h"
#include "core/crispasr_env.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<float> load_npy(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};
    char magic[6];
    fread(magic, 1, 6, f);
    uint8_t v[2];
    fread(v, 1, 2, f);
    uint16_t hlen;
    fread(&hlen, 2, 1, f);
    fseek(f, 6 + 2 + 2 + hlen, SEEK_SET);
    long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, pos, SEEK_SET);
    int n = (int)((end - pos) / sizeof(float));
    std::vector<float> d(n);
    fread(d.data(), sizeof(float), n, f);
    fclose(f);
    return d;
}

static double cosine(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return (na > 1e-12 && nb > 1e-12) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
}

int main() {
    const char* gguf = crispasr_env::get("CRISPASR_AUDIOSEAL_GGUF");
    if (!gguf) {
        fprintf(stderr, "Set AUDIOSEAL_GGUF\n");
        return 1;
    }

    // Load reference input
    auto ref_input = load_npy("/tmp/as_enc_input.npy");
    if (ref_input.empty()) {
        fprintf(stderr, "No reference files. Run the Python stage dumper first.\n");
        return 1;
    }
    int N = (int)ref_input.size();
    printf("Input: %d samples\n", N);

    // Load gguf
    auto params = audioseal_default_params();
    params.verbosity = 0;
    auto* ctx = audioseal_init_from_file(gguf, params);
    if (!ctx) {
        fprintf(stderr, "Failed to load GGUF\n");
        return 2;
    }

    // Run embed to get the full output
    uint8_t msg[16];
    memset(msg, 1, 16);
    float* out = audioseal_embed(ctx, ref_input.data(), N, msg);
    if (!out) {
        fprintf(stderr, "embed failed\n");
        audioseal_free(ctx);
        return 3;
    }

    // Compare full output against reference watermarked
    auto ref_wm = load_npy("/tmp/audioseal_ref_watermarked.npy");
    if (!ref_wm.empty() && (int)ref_wm.size() >= N) {
        double cos = cosine(out, ref_wm.data(), N);
        printf("full_output  cos=%.6f\n", cos);
    }

    // Compare watermark-only
    if (!ref_wm.empty() && (int)ref_wm.size() >= N) {
        std::vector<float> our_wm(N), ref_w(N);
        for (int i = 0; i < N; i++) {
            our_wm[i] = out[i] - ref_input[i];
            ref_w[i] = ref_wm[i] - ref_input[i];
        }
        double cos = cosine(our_wm.data(), ref_w.data(), N);
        printf("watermark    cos=%.6f\n", cos);

        // Print first 20 samples of watermark for visual comparison
        printf("\nFirst 20 watermark samples:\n");
        printf("  ours: ");
        for (int i = 0; i < 20 && i < N; i++)
            printf("%.4f ", our_wm[i]);
        printf("\n  ref:  ");
        for (int i = 0; i < 20 && i < N; i++)
            printf("%.4f ", ref_w[i]);
        printf("\n");
    }

    // Per-stage comparison (encoder stages)
    struct {
        const char* npy;
        const char* name;
    } stages[] = {
        {"/tmp/as_enc_0.npy", "enc.0 (input conv)"},
        {"/tmp/as_enc_1.npy", "enc.1 (resblock 0)"},
        {"/tmp/as_enc_3.npy", "enc.3 (downsample 0)"},
    };
    printf("\nPer-stage comparison (if reference available):\n");
    for (auto& s : stages) {
        auto ref = load_npy(s.npy);
        if (ref.empty()) {
            printf("  %-25s  SKIP (no reference)\n", s.name);
            continue;
        }
        printf("  %-25s  ref_size=%d\n", s.name, (int)ref.size());
    }

    free(out);
    audioseal_free(ctx);
    return 0;
}

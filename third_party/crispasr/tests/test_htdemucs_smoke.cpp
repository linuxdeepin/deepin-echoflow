#include "htdemucs.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    htdemucs_params p = htdemucs_default_params();
    htdemucs_context* ctx = htdemucs_init_from_file(argv[1], p);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }
    int sr = htdemucs_sample_rate(ctx);
    int n = sr * 1;  // 1 second
    std::vector<float> pcm(2 * n);
    for (int i = 0; i < n; i++) {
        float t = (float)i / sr;
        pcm[2*i]   = 0.5f * sinf(2.0f * 3.14159f * 440.0f * t);
        pcm[2*i+1] = 0.3f * sinf(2.0f * 3.14159f * 880.0f * t);
    }
    fprintf(stderr, "Separating %d samples (%d sec)...\n", n, n/sr);
    htdemucs_result* r = htdemucs_separate(ctx, pcm.data(), n);
    if (!r) { fprintf(stderr, "Separation failed\n"); htdemucs_free(ctx); return 1; }
    fprintf(stderr, "Got %d sources:\n", r->n_sources);
    for (int s = 0; s < r->n_sources; s++) {
        float rms = 0;
        for (int i = 0; i < r->n_samples * r->n_channels; i++)
            rms += r->sources[s][i] * r->sources[s][i];
        rms = sqrtf(rms / (r->n_samples * r->n_channels));
        fprintf(stderr, "  %s: rms=%.6f\n", r->source_names[s], rms);
    }
    htdemucs_result_free(r);
    htdemucs_free(ctx);
    return 0;
}

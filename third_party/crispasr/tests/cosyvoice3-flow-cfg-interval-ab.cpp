// cosyvoice3 flow interval-CFG A/B harness (manual — needs local GGUF models).
//
// Drives the CFM Euler flow solver (`cosyvoice3_tts_solve_flow_euler`) with FIXED
// mu / spks / cond / x_init, so its output mel depends only on the weights, the
// graph, and CRISPASR_COSYVOICE3_CFG_INTERVAL. This isolates interval-CFG:
//   - K=1 (env unset or =1): EXACT path — must be byte-identical run-to-run and to
//     the legacy build (the new branch is only entered when K>1).
//   - K>1: APPROXIMATE — reuses a stale uncond forward every K steps. Compare its
//     mel to the K=1 mel (cosine, via the dumped raw floats) to characterize the
//     approximation. Content-preservation is judged downstream (mel → hift → ASR).
//
// Usage:
//   cosyvoice3-flow-cfg-interval-ab <llm.gguf> <flow.gguf> [T_mel=64] [out.f32]
//
// If out.f32 is given, the T_mel*80 output mel is written as raw little-endian
// float32 for an external cosine compare between two runs.

#include "cosyvoice3_tts.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static inline uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

// FNV-1a 64 over the raw float bytes — a bit-exact fingerprint.
static uint64_t fnv1a(const float* p, size_t n) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n * sizeof(float); i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <llm.gguf> <flow.gguf> [T_mel=64] [out.f32]\n", argv[0]);
        return 2;
    }
    const char* llm_path = argv[1];
    const char* flow_path = argv[2];
    const int T_mel = argc > 3 ? atoi(argv[3]) : 64;
    const char* out_path = argc > 4 ? argv[4] : nullptr;
    if (T_mel <= 0) {
        fprintf(stderr, "bad T_mel\n");
        return 2;
    }

    cosyvoice3_tts_context_params p = cosyvoice3_tts_context_default_params();
    p.verbosity = 1;
    cosyvoice3_tts_context* ctx = cosyvoice3_tts_init_from_file(llm_path, p);
    if (!ctx) {
        fprintf(stderr, "init_from_file failed: %s\n", llm_path);
        return 1;
    }
    if (cosyvoice3_tts_init_flow_from_file(ctx, flow_path) != 0) {
        fprintf(stderr, "init_flow_from_file failed: %s\n", flow_path);
        cosyvoice3_tts_free(ctx);
        return 1;
    }

    const int mel_dim = 80;
    const int spk_dim = 80; // fh.spk_dim_out
    const size_t mel_n = (size_t)T_mel * mel_dim;

    // Fixed, bounded, deterministic inputs (same across every run).
    std::vector<float> mu(mel_n), cond(mel_n), x_init(mel_n), spks(spk_dim);
    for (int t = 0; t < T_mel; t++)
        for (int c = 0; c < mel_dim; c++) {
            mu[(size_t)t * mel_dim + c] = std::sin(0.03f * t + 0.11f * c) * 0.5f;
            cond[(size_t)t * mel_dim + c] = std::cos(0.017f * t + 0.07f * c) * 0.4f;
        }
    uint32_t s = 0x5EED1234u;
    for (size_t i = 0; i < mel_n; i++)
        x_init[i] = ((lcg(s) >> 8) * (1.0f / 16777216.0f) - 0.5f) * 2.0f; // ~N-ish noise in [-1,1)
    for (int i = 0; i < spk_dim; i++)
        spks[i] = std::sin(0.21f * i) * 0.3f;

    const int n_steps = 10;      // cfm default
    const float cfg_rate = 0.7f; // cfm default

    float* mel = cosyvoice3_tts_solve_flow_euler(ctx, mu.data(), T_mel, spks.data(), cond.data(), x_init.data(),
                                                 n_steps, cfg_rate);
    if (!mel) {
        fprintf(stderr, "solve_flow_euler returned null\n");
        cosyvoice3_tts_free(ctx);
        return 1;
    }

    double sum = 0.0, amax = 0.0;
    bool nan_seen = false;
    for (size_t i = 0; i < mel_n; i++) {
        const float v = mel[i];
        if (std::isnan(v))
            nan_seen = true;
        sum += v;
        if (std::fabs(v) > amax)
            amax = std::fabs(v);
    }
    const char* env = getenv("CRISPASR_COSYVOICE3_CFG_INTERVAL");
    printf("CFG_INTERVAL=%s  T_mel=%d  n=%zu  hash=%016llx  sum=%.6f  max|m|=%.6f  nan=%d\n", env ? env : "(unset=1)",
           T_mel, mel_n, (unsigned long long)fnv1a(mel, mel_n), sum, amax, nan_seen ? 1 : 0);

    if (out_path) {
        FILE* f = fopen(out_path, "wb");
        if (f) {
            fwrite(mel, sizeof(float), mel_n, f);
            fclose(f);
        } else {
            fprintf(stderr, "could not write %s\n", out_path);
        }
    }

    free(mel);
    cosyvoice3_tts_free(ctx);
    return nan_seen ? 1 : 0;
}

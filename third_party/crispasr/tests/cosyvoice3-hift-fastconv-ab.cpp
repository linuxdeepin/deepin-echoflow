// cosyvoice3 HiFT FASTCONV A/B harness (manual — needs local GGUF models).
//
// Drives the deterministic HiFT vocoder (`cosyvoice3_tts_run_hift_inference`)
// with a FIXED mel + FIXED source-noise buffer, so its output depends only on
// the weights and graph — no flow-matching / AR RNG. This isolates the
// CRISPASR_COSYVOICE3_FASTCONV cast-kill and the opt-in SIMDCONV ResBlock
// simdconv path. FASTCONV is expected to preserve the hash. SIMDCONV uses
// a different reduction tree from ggml_conv_1d, so a hash change is allowed;
// use the printed signal stats plus the normal HiFT diff harness for closeness.
//
// Usage:
//   cosyvoice3-hift-fastconv-ab <llm.gguf> <hift.gguf> [T_mel=64]
//
// The LLM GGUF is loaded only to construct a valid context (backend +
// compute_meta); its weights are not exercised by the HiFT path.
//
// Set CRISPASR_COSYVOICE3_FASTCONV_DEBUG=1 and/or
// CRISPASR_COSYVOICE3_SIMDCONV_DEBUG=1 to prove which path engaged. With
// SIMDCONV on, FASTCONV intentionally skips the 72 packed ResBlock kernels.

#include "cosyvoice3_tts.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Deterministic 32-bit LCG so the fixed inputs are reproducible run-to-run
// without pulling in <random> (whose engines could drift across libstdc++).
static inline uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

// FNV-1a 64 over the raw bytes of the float output — a bit-exact fingerprint.
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
        fprintf(stderr, "usage: %s <llm.gguf> <hift.gguf> [T_mel=64]\n", argv[0]);
        return 2;
    }
    const char* llm_path = argv[1];
    const char* hift_path = argv[2];
    const int T_mel = argc > 3 ? atoi(argv[3]) : 64;
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
    if (cosyvoice3_tts_init_hift_from_file(ctx, hift_path) != 0) {
        fprintf(stderr, "init_hift_from_file failed: %s\n", hift_path);
        cosyvoice3_tts_free(ctx);
        return 1;
    }

    // Fixed mel [T_mel, 80] — a smooth, bounded, deterministic pattern.
    const int mel_dim = 80;
    std::vector<float> mel((size_t)T_mel * mel_dim);
    for (int t = 0; t < T_mel; t++)
        for (int c = 0; c < mel_dim; c++)
            mel[(size_t)t * mel_dim + c] = std::sin(0.03f * t + 0.11f * c) * 0.5f;

    // Fixed source-noise [T_mel*480*9] in uniform[0,1) — mirrors the seeded
    // SineGen2 noise buffer the real path supplies.
    const size_t noise_n = (size_t)T_mel * 480 * 9;
    std::vector<float> noise(noise_n);
    uint32_t s = 0xC0FFEEu;
    for (size_t i = 0; i < noise_n; i++)
        noise[i] = (lcg(s) >> 8) * (1.0f / 16777216.0f); // [0,1)

    float* audio = cosyvoice3_tts_run_hift_inference(ctx, mel.data(), T_mel, noise.data());
    if (!audio) {
        fprintf(stderr, "run_hift_inference returned null\n");
        cosyvoice3_tts_free(ctx);
        return 1;
    }
    const size_t n = (size_t)T_mel * 480;

    // Bit-exact fingerprint + a coarse magnitude so a silent/NaN output is
    // obvious (a hash alone can't distinguish "identical" from "identically zero").
    double sum = 0.0, amax = 0.0;
    bool nan_seen = false;
    for (size_t i = 0; i < n; i++) {
        const float v = audio[i];
        if (std::isnan(v))
            nan_seen = true;
        sum += v;
        if (std::fabs(v) > amax)
            amax = std::fabs(v);
    }
    const char* env = getenv("CRISPASR_COSYVOICE3_FASTCONV");
    const char* env2 = getenv("CRISPASR_COSYVOICE3_SIMDCONV");
    printf("FASTCONV=%s  SIMDCONV=%s  T_mel=%d  n=%zu  hash=%016llx  sum=%.6f  max|a|=%.6f  nan=%d\n",
           env ? env : "(default-on)", env2 ? env2 : "(default-off)", T_mel, n,
           (unsigned long long)fnv1a(audio, n), sum, amax, nan_seen ? 1 : 0);

    free(audio);
    cosyvoice3_tts_free(ctx);
    return nan_seen ? 1 : 0;
}

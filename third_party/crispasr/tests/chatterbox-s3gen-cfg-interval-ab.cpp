// chatterbox s3gen interval-CFG A/B harness (manual — needs local GGUF models).
//
// Drives ONLY the S3Gen CFM decoder (chatterbox_synthesize_mel_from_tokens) from a
// FIXED synthetic speech-token sequence — skipping the slow T3 AR stage entirely — so
// the 10-step CFM Euler solver is exercised in isolation. `chatterbox_set_seed` reseeds
// the init-noise mt19937 before each run, so the output mel depends only on the tokens,
// the weights, and CRISPASR_S3GEN_CFG_INTERVAL. This is the fast, deterministic
// verification the full CLI synth (6+ min, T3-AR-bound) can't give on M1.
//
//   K=1 twice → byte-identical mel (determinism + exact single-path at K=1).
//   K=1 vs K=2 → phase-invariant nothing needed here (it's the mel, not audio):
//     mel cosine directly measures how much interval perturbs the CFM output.
//
// Usage: chatterbox-s3gen-cfg-interval-ab <t3.gguf> <s3gen.gguf> [n_tokens=120]

#include "chatterbox.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "portable_env.h"

static uint64_t fnv1a(const float* p, size_t n) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n * sizeof(float); i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = a.size() < b.size() ? a.size() : b.size();
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <t3.gguf> <s3gen.gguf> [n_tokens=120]\n", argv[0]);
        return 2;
    }
    const int n_tokens = argc > 3 ? atoi(argv[3]) : 120;

    chatterbox_context_params p = chatterbox_context_default_params();
    chatterbox_context* ctx = chatterbox_init_from_file(argv[1], p);
    if (!ctx) {
        fprintf(stderr, "chatterbox_init_from_file failed: %s\n", argv[1]);
        return 1;
    }
    if (chatterbox_set_s3gen_path(ctx, argv[2]) != 0) {
        fprintf(stderr, "chatterbox_set_s3gen_path failed: %s\n", argv[2]);
        return 1;
    }

    // Fixed synthetic speech tokens — a deterministic low-id pattern (ids well under
    // the s3 input_embedding vocab). Not real speech, but the interval A/B measures
    // the RELATIVE perturbation of the CFM output, for which any fixed input works.
    std::vector<int32_t> tokens(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        tokens[i] = (i * 37 + 11) % 500;

    auto run = [&](const char* k) -> std::vector<float> {
        setenv("CRISPASR_S3GEN_CFG_INTERVAL", k, 1);
        chatterbox_set_seed(ctx, 42); // reseed init-noise RNG → same noise each run
        int T_mel = 0;
        float* mel = chatterbox_synthesize_mel_from_tokens(ctx, tokens.data(), (int)tokens.size(), &T_mel);
        if (!mel || T_mel <= 0) {
            fprintf(stderr, "synthesize_mel_from_tokens(K=%s) returned null/empty\n", k);
            return {};
        }
        std::vector<float> v(mel, mel + (size_t)T_mel * 80);
        chatterbox_pcm_free(mel);
        return v;
    };

    std::vector<float> k1a = run("1");
    std::vector<float> k1b = run("1");
    std::vector<float> k2 = run("2");
    std::vector<float> k3 = run("3");
    if (k1a.empty() || k1b.empty() || k2.empty()) {
        chatterbox_free(ctx);
        return 1;
    }

    auto stats = [](const std::vector<float>& v, double& sum, double& amax, bool& nan) {
        sum = 0;
        amax = 0;
        nan = false;
        for (float x : v) {
            if (std::isnan(x))
                nan = true;
            sum += x;
            if (std::fabs(x) > amax)
                amax = std::fabs(x);
        }
    };
    double s1, a1;
    bool n1;
    stats(k1a, s1, a1, n1);
    printf("K1a: n=%zu hash=%016llx sum=%.4f max=%.4f nan=%d\n", k1a.size(),
           (unsigned long long)fnv1a(k1a.data(), k1a.size()), s1, a1, n1 ? 1 : 0);
    printf("K1a vs K1b: byte-identical=%s  cos=%.8f  (determinism / exact single-path at K=1)\n",
           (k1a.size() == k1b.size() && fnv1a(k1a.data(), k1a.size()) == fnv1a(k1b.data(), k1b.size())) ? "YES" : "NO",
           cosine(k1a, k1b));
    printf("K1  vs K2 : cos=%.6f  (interval K=2 approximation on the CFM mel)\n", cosine(k1a, k2));
    if (!k3.empty())
        printf("K1  vs K3 : cos=%.6f  (interval K=3)\n", cosine(k1a, k3));

    chatterbox_free(ctx);
    return 0;
}

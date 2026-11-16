// test_beat_this_stages.cpp — per-stage parity for the beat-this graph.
//
//   ./test-beat-this-stages <gguf> <logmel.bin> <out|out_prefix> [stage ...]
//
// Computes one stage per named argument and dumps each for
// tools/cmp_beat_this_stages.py to score against the torch reference.
// Stage-by-stage rather than end-to-end: first divergence is the bug.
//
// Each dump is `int32 ne[4]` followed by the raw f32 payload. ne is the ggml
// shape, which is the reverse of torch's — the comparison script reshapes to
// reversed(ne) and lands on the reference layout with no transpose, so a
// layout slip cannot masquerade as a numerical one.
//
// With no stage arguments this writes only "stem", to <out_prefix> taken
// verbatim, which keeps the original single-file invocation working.
#include "beat_this.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<float> read_f32(const char* p) {
    std::vector<float> v;
    FILE* f = fopen(p, "rb");
    if (!f)
        return v;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t)(n / 4));
    if (fread(v.data(), 4, v.size(), f) != v.size())
        v.clear();
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <gguf> <logmel.bin> <out|out_prefix> [stage ...]\n", argv[0]);
        return 2;
    }
    auto lm = read_f32(argv[2]);
    if (lm.empty()) {
        fprintf(stderr, "cannot read logmel\n");
        return 1;
    }
    const int T = (int)(lm.size() / BEAT_THIS_MEL_BINS);

    beat_this_context* ctx = beat_this_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    std::vector<std::string> stages;
    for (int i = 4; i < argc; i++)
        stages.emplace_back(argv[i]);
    const bool multi = !stages.empty();
    if (!multi)
        stages.emplace_back("stem");

    // The widest stage is the stem's / blk0's, T*32*32; every later block
    // halves frequency as it doubles channels, so this bound holds throughout.
    std::vector<float> buf((size_t)T * 32 * 32);
    int rc = 0;

    for (const std::string& s : stages) {
        int64_t ne[4] = {0, 0, 0, 0};
        const int n = beat_this_debug_stage(ctx, lm.data(), T, s.c_str(), buf.data(), (int)buf.size(), ne);
        if (n <= 0) {
            fprintf(stderr, "stage %s FAILED\n", s.c_str());
            rc = 1;
            continue;
        }
        const std::string path = multi ? std::string(argv[3]) + s + ".bin" : std::string(argv[3]);
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            rc = 1;
            continue;
        }
        const int32_t hdr[4] = {(int32_t)ne[0], (int32_t)ne[1], (int32_t)ne[2], (int32_t)ne[3]};
        fwrite(hdr, sizeof(int32_t), 4, f);
        fwrite(buf.data(), sizeof(float), (size_t)n, f);
        fclose(f);
        printf("%-16s ne=(%lld,%lld,%lld,%lld) %d elems -> %s\n", s.c_str(), (long long)ne[0], (long long)ne[1],
               (long long)ne[2], (long long)ne[3], n, path.c_str());
    }

    beat_this_free(ctx);
    return rc;
}

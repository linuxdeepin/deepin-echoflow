// test_beat_this_frontend.cpp — validate beat_this's log-mel against torchaudio.
//
//   ./test-beat-this-frontend <beat-this.gguf> <input.bin> <ref.bin>
//
// input.bin : float32 mono 22050 Hz PCM
// ref.bin   : float32 (frames, 128) log-mel from the reference pipeline
//             (tools/gen_beat_this_frontend_ref.py)
//
// The front end is the piece most likely to drift silently -- a wrong window,
// a squared magnitude, or the wrong STFT normalization all yield a plausible
// spectrogram and wrong beats. Checking it independently, before the network
// exists, keeps that failure mode out of the graph bring-up.
#include "beat_this.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
        fprintf(stderr, "usage: %s <gguf> <input.bin> <ref.bin>\n", argv[0]);
        return 2;
    }
    auto pcm = read_f32(argv[2]);
    auto ref = read_f32(argv[3]);
    if (pcm.empty() || ref.empty()) {
        fprintf(stderr, "cannot read fixtures\n");
        return 1;
    }

    beat_this_context* ctx = beat_this_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    const int T = beat_this_n_frames((int)pcm.size());
    std::vector<float> got((size_t)T * BEAT_THIS_MEL_BINS);
    const int n = beat_this_logmel(ctx, pcm.data(), (int)pcm.size(), got.data());
    printf("frames: got %d, ref %zu (x%d bins)\n", n, ref.size() / BEAT_THIS_MEL_BINS, BEAT_THIS_MEL_BINS);
    if (n <= 0) {
        beat_this_free(ctx);
        return 1;
    }

    const size_t m = std::min(got.size(), ref.size());
    double num = 0, da = 0, db = 0, mx = 0;
    for (size_t i = 0; i < m; i++) {
        num += (double)got[i] * ref[i];
        da += (double)got[i] * got[i];
        db += (double)ref[i] * ref[i];
        mx = std::max(mx, (double)std::fabs(got[i] - ref[i]));
    }
    const double cos = num / (std::sqrt(da) * std::sqrt(db));
    printf("cos=%.8f max_abs=%.4e  |mine|=%.3f |ref|=%.3f\n", cos, mx, std::sqrt(da), std::sqrt(db));
    const bool ok = cos > 0.9999 && mx < 5e-3;
    printf("%s\n", ok ? "PASS" : "FAIL");
    beat_this_free(ctx);
    return ok ? 0 : 1;
}

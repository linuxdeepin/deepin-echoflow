// test_mel_band_roformer_smoke.cpp — mel-band-roformer source-separation smoke.
//
// Closes the wiring-audit "missing: test" advisory for mel-band-roformer, but
// the real value is that BUILDING this links the backend: mel-band-roformer was
// once dropped from a shipped dylib because nothing referenced its symbols
// (see LEARNINGS "linked in CMake is not evidence the code ships"). A test that
// links + calls the full API is the artifact-level guard that catches that.
//
// Standalone (mirrors test_htdemucs_smoke): takes the GGUF as argv[1] and runs a
// 1 s tone through separate(); no model = usage + skip, so `ctest -L unit` (no
// models) still passes while a live run exercises the real path.

#include "mel_band_roformer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mel-band-roformer.gguf>  (no model → skip)\n", argv[0]);
        return 0; // skip cleanly, like the other model-gated smokes
    }
    mel_band_roformer_params p = mel_band_roformer_default_params();
    mel_band_roformer_context* ctx = mel_band_roformer_init_from_file(argv[1], p);
    if (!ctx) {
        fprintf(stderr, "FAIL: could not load '%s'\n", argv[1]);
        return 1;
    }

    const int sr = mel_band_roformer_sample_rate(ctx);
    if (sr <= 0) {
        fprintf(stderr, "FAIL: sample_rate=%d\n", sr);
        mel_band_roformer_free(ctx);
        return 1;
    }
    const int n = sr; // 1 second, per channel
    std::vector<float> pcm(2 * (size_t)n);
    for (int i = 0; i < n; i++) {
        const float t = (float)i / sr;
        pcm[2 * i] = 0.5f * sinf(2.0f * 3.14159265f * 440.0f * t);
        pcm[2 * i + 1] = 0.3f * sinf(2.0f * 3.14159265f * 880.0f * t);
    }

    mel_band_roformer_result* r = mel_band_roformer_separate(ctx, pcm.data(), n, 2);
    if (!r) {
        fprintf(stderr, "FAIL: separate() returned null\n");
        mel_band_roformer_free(ctx);
        return 1;
    }

    // Contract checks: at least one named stem, sane geometry, finite samples.
    int rc = 0;
    if (r->n_sources < 1 || !r->sources || !r->source_names) {
        fprintf(stderr, "FAIL: n_sources=%d\n", r->n_sources);
        rc = 1;
    }
    for (int s = 0; s < r->n_sources && rc == 0; s++) {
        double rms = 0.0;
        const long total = (long)r->n_samples * r->n_channels;
        for (long i = 0; i < total; i++) {
            const float v = r->sources[s][i];
            if (!std::isfinite(v)) {
                fprintf(stderr, "FAIL: non-finite sample in stem %s\n", r->source_names[s]);
                rc = 1;
                break;
            }
            rms += (double)v * v;
        }
        fprintf(stderr, "  %s: rms=%.6f\n", r->source_names[s], total ? sqrt(rms / total) : 0.0);
    }

    mel_band_roformer_result_free(r);
    mel_band_roformer_free(ctx);
    fprintf(stderr, rc == 0 ? "PASS\n" : "FAIL\n");
    return rc;
}

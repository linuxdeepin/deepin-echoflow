// tools/mbr_diff_probe.cpp — standalone §248 Mel-Band RoFormer dev probe.
//
// Two modes (no dep on the full crispasr-lib, so it builds in isolation):
//   diff:      mbr-diff-probe diff <model.gguf> <ref.gguf> [verbosity]
//   separate:  mbr-diff-probe sep  <model.gguf> <in.wav> <out_prefix>
// (legacy: `mbr-diff-probe <model> <ref> [v]` still runs diff.)

#include "core/crispasr_wav_writer.h"
#include "core/wav_reader.h"
#include "mel_band_roformer.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static int run_sep(const char* model, const char* wav, const char* prefix) {
    std::vector<float> mono;
    int sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(wav, mono, sr)) {
        fprintf(stderr, "sep: cannot read %s\n", wav);
        return 2;
    }
    fprintf(stderr, "sep: read %zu samples @ %d Hz (mono)\n", mono.size(), sr);
    auto* ctx = mel_band_roformer_init_from_file(model, mel_band_roformer_default_params());
    if (!ctx)
        return 2;
    // The model expects its native rate; this probe feeds mono as-is (the real
    // CLI dispatcher resamples). Pass in_channels=1.
    auto* r = mel_band_roformer_separate(ctx, mono.data(), (int)mono.size(), 1);
    if (!r) {
        fprintf(stderr, "sep: separate() returned null\n");
        mel_band_roformer_free(ctx);
        return 1;
    }
    for (int s = 0; s < r->n_sources; s++) {
        const std::string path = std::string(prefix) + "_" + r->source_names[s] + ".wav";
        const std::string blob =
            crispasr_make_wav_int16_interleaved(r->sources[s], r->n_samples, r->n_channels, r->sample_rate);
        std::ofstream(path, std::ios::binary).write(blob.data(), (std::streamsize)blob.size());
        // quick non-silence check
        double e = 0;
        for (int i = 0; i < r->n_samples * r->n_channels; i++)
            e += (double)r->sources[s][i] * r->sources[s][i];
        fprintf(stderr, "sep: wrote %s  (%d ch x %d, rms=%.5f)\n", path.c_str(), r->n_channels, r->n_samples,
                std::sqrt(e / (r->n_samples * r->n_channels)));
    }
    mel_band_roformer_result_free(r);
    mel_band_roformer_free(ctx);
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 5 && std::string(argv[1]) == "sep")
        return run_sep(argv[2], argv[3], argv[4]);
    // diff mode (explicit or legacy positional)
    int base = (argc >= 4 && std::string(argv[1]) == "diff") ? 2 : 1;
    if (argc < base + 2) {
        fprintf(stderr,
                "usage:\n  %s diff <model.gguf> <ref.gguf> [verbosity]\n"
                "  %s sep  <model.gguf> <in.wav> <out_prefix>\n",
                argv[0], argv[0]);
        return 2;
    }
    const int verbosity = (argc > base + 2) ? atoi(argv[base + 2]) : 1;
    return mel_band_roformer_diff(argv[base], argv[base + 1], nullptr, verbosity);
}

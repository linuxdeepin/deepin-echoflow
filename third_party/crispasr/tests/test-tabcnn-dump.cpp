// test-tabcnn-dump.cpp — dump src/tabcnn.cpp stages for numerical validation.
//
// Not a Catch2 test: it is the C++ half of the parity check. Run it, then score
// the output against the torch reference with
// `tools/tabcnn_torch_parity.py --cpp-dump <file>`.
//
//   test-tabcnn-dump <model.gguf> <audio.f32> <stage> <out.bin>
//
// `audio.f32` is raw float32 mono at the model's own sample rate (Python does
// the decode, so this stays dependency-free). `stage` is any name the reference
// dumper emits — cqt_db, conv0_relu, conv1_relu, conv2_relu, pool, dense0_relu,
// logits — or `probs` for the final softmax.
//
// Output format matches tools/cqt_librosa_parity.py: int32 count, then floats.

#include "tabcnn.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <model.gguf> <audio.f32> <stage> <out.bin>\n", argv[0]);
        return 2;
    }
    const char* model = argv[1];
    const char* audio_path = argv[2];
    const char* stage = argv[3];
    const char* out_path = argv[4];

    FILE* af = std::fopen(audio_path, "rb");
    if (!af) {
        std::fprintf(stderr, "cannot open %s\n", audio_path);
        return 1;
    }
    std::fseek(af, 0, SEEK_END);
    const long bytes = std::ftell(af);
    std::fseek(af, 0, SEEK_SET);
    std::vector<float> pcm((size_t)bytes / sizeof(float));
    if (std::fread(pcm.data(), sizeof(float), pcm.size(), af) != pcm.size()) {
        std::fclose(af);
        std::fprintf(stderr, "short read on %s\n", audio_path);
        return 1;
    }
    std::fclose(af);

    tabcnn_context* ctx = tabcnn_init(model, 4);
    if (!ctx) {
        std::fprintf(stderr, "tabcnn_init failed\n");
        return 1;
    }
    // Feed at the model's own rate so the resampler is out of the comparison;
    // the Python side writes the file already resampled.
    const int sr = TABCNN_SAMPLE_RATE;
    const int n_frames = tabcnn_n_frames(ctx, (int)pcm.size(), sr);
    std::fprintf(stderr, "frames=%d  frame_period=%.6f s  silent_class=%d\n", n_frames, tabcnn_frame_period(ctx),
                 tabcnn_silent_class(ctx));

    std::vector<float> out;
    if (std::strcmp(stage, "probs") == 0) {
        out.resize((size_t)n_frames * TABCNN_NUM_STRINGS * TABCNN_NUM_CLASSES);
        const int got = tabcnn_compute(ctx, pcm.data(), (int)pcm.size(), sr, out.data(), n_frames);
        if (got <= 0) {
            std::fprintf(stderr, "tabcnn_compute failed\n");
            tabcnn_free(ctx);
            return 1;
        }
        out.resize((size_t)got * TABCNN_NUM_STRINGS * TABCNN_NUM_CLASSES);
    } else {
        // Generous cap: the widest stage (conv0) is T * 32 * 190 * 7 floats.
        out.resize((size_t)n_frames * 32 * 190 * 7 + 1024);
        const int got = tabcnn_extract_stage(ctx, pcm.data(), (int)pcm.size(), sr, stage, out.data(), (int)out.size());
        if (got <= 0) {
            std::fprintf(stderr, "tabcnn_extract_stage(%s) failed\n", stage);
            tabcnn_free(ctx);
            return 1;
        }
        out.resize((size_t)got);
    }
    tabcnn_free(ctx);

    FILE* f = std::fopen(out_path, "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }
    const int32_t n = (int32_t)out.size();
    std::fwrite(&n, sizeof(int32_t), 1, f);
    std::fwrite(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    std::printf("wrote %s (stage=%s, %d floats)\n", out_path, stage, n);
    return 0;
}

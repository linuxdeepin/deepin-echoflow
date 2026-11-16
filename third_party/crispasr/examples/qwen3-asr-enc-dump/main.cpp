// qwen3-asr-enc-dump — dump the Qwen3-ASR audio-encoder output for a GGUF.
//
// Runs mel + full audio encoder (CPU) on a WAV and writes the encoder
// output as raw float32 with a small header, so two quants can be compared
// off-line (issue #240: does the Q8_0 audio-tower floor repair the
// sub-8-bit encoder drift on the 1.7b as it did on the 0.6b?).
//
// Usage: qwen3-asr-enc-dump model.gguf audio.wav out.bin
//   out.bin layout: int32 N_total, int32 proj_dim, then N_total*proj_dim f32.

#include "core/wav_reader.h"
#include "qwen3_asr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf audio.wav out.bin\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* audio_path = argv[2];
    const char* out_path = argv[3];

    std::vector<float> pcm;
    int sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(audio_path, pcm, sr) || pcm.empty()) {
        fprintf(stderr, "failed to read wav: %s\n", audio_path);
        return 2;
    }
    fprintf(stderr, "audio: %zu samples @ %d Hz\n", pcm.size(), sr);

    qwen3_asr_context_params p = qwen3_asr_context_default_params();
    p.n_threads = 8;
    p.verbosity = 1;
    p.use_gpu = false; // CPU for all quants => pure quantization comparison
    p.flash_attn = false;

    qwen3_asr_context* ctx = qwen3_asr_init_from_file(model_path, p);
    if (!ctx) {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        return 3;
    }

    int n_mels = 0, T_mel = 0;
    float* mel = qwen3_asr_compute_mel(ctx, pcm.data(), (int)pcm.size(), &n_mels, &T_mel);
    if (!mel) {
        fprintf(stderr, "compute_mel failed\n");
        return 4;
    }
    fprintf(stderr, "mel: %d x %d\n", n_mels, T_mel);

    int N_total = 0, proj_dim = 0;
    float* enc = qwen3_asr_run_encoder(ctx, mel, n_mels, T_mel, &N_total, &proj_dim);
    if (!enc) {
        fprintf(stderr, "run_encoder failed\n");
        return 5;
    }
    fprintf(stderr, "encoder: N_total=%d proj_dim=%d\n", N_total, proj_dim);

    FILE* f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open out: %s\n", out_path);
        return 6;
    }
    int32_t hdr[2] = {N_total, proj_dim};
    fwrite(hdr, sizeof(int32_t), 2, f);
    fwrite(enc, sizeof(float), (size_t)N_total * proj_dim, f);
    fclose(f);
    fprintf(stderr, "wrote %s (%d x %d)\n", out_path, N_total, proj_dim);

    free(enc);
    free(mel);
    qwen3_asr_free(ctx);
    return 0;
}

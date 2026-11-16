// bananamind_tts.h -- BananaMind-TTS-V2.1 native ggml runtime.
//
// Tacotron-lite TTS with HiFi-GAN vocoder.
// Architecture:
//   - Text encoder: Embedding -> Conv1d blocks (BN + ReLU) -> BiLSTM
//   - Decoder: Autoregressive GRU-based with location-sensitive attention
//     - Prenet: 2x Linear + ReLU
//     - Attention GRU + location-sensitive attention
//     - Decoder GRU
//     - Mel projection (reduction_factor=4 frames per step)
//     - Stop prediction
//   - Postnet: 5-layer Conv1d + BN + Tanh (mel refinement)
//   - HiFi-GAN vocoder: mel -> waveform
//
// Supports en-us (LJ Speech) and de-de (ThorstenVoice) locales.
// Character-based tokenizer, 22050 Hz output, mono.
//
// Model source: Banaxi-Tech/BananaMind-TTS-V2.1-Preview (Apache 2.0)

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct bananamind_tts_context;

struct bananamind_tts_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
};

struct bananamind_tts_params bananamind_tts_default_params(void);

// Load a BananaMind TTS GGUF (acoustic model + HiFi-GAN vocoder).
// Returns nullptr on failure.
struct bananamind_tts_context* bananamind_tts_init_from_file(const char* path_model,
                                                             struct bananamind_tts_params params);

void bananamind_tts_free(struct bananamind_tts_context* ctx);

// Synthesize text to mono PCM at the model's sample rate (22050 Hz).
// Returns number of samples written, 0 on failure.
// Caller owns the returned buffer (malloc'd; free with free()).
int bananamind_tts_synthesize(struct bananamind_tts_context* ctx, const char* text, float** pcm_out,
                              int* sample_rate_out);

// Query model info.
int bananamind_tts_sample_rate(const struct bananamind_tts_context* ctx);

#ifdef __cplusplus
}
#endif

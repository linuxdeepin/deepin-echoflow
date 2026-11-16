// Sidon speech-restoration runtime (w2v-BERT predictor + continuous DAC).
#pragma once

#include <vector>

struct sidon_context;

struct sidon_context_params {
    int n_threads;
    int verbosity;
    bool use_gpu;
};

sidon_context_params sidon_context_default_params();
sidon_context* sidon_init_from_file(const char* path, sidon_context_params params);
void sidon_free(sidon_context* ctx);

// Input is 16 kHz mono float PCM; output is 48 kHz mono float PCM.
std::vector<float> sidon_restore(sidon_context* ctx, const float* samples, int n_samples);

// Encoder-only feature extraction (plain w2v-BERT forward, no restoration
// recipe): 16 kHz mono PCM -> hidden states (n_frames x hidden_size,
// row-major, ~50 frames/s). Works on any sidon GGUF; the encoder-only
// `sidon.encoder_only=1` GGUFs (e.g. confucius4-tts-w2v) exist solely for
// this. Returns empty on failure.
std::vector<float> sidon_extract_hidden(sidon_context* ctx, const float* pcm_16k, int n_samples, int* n_frames_out);

#pragma once

// miocodec.h — MioCodec v2 audio codec decoder (44.1 kHz, MIT).
//
// Architecture: FSQ quantizer → wave_prenet (Transformer 6L, 768→512)
// → conv_upsample(2×) → interp → ResNet prior → wave_decoder (Transformer
// 8L, 512d, AdaLN-Zero cond=128) → ResNet post → SnakeBeta upsampler(9×)
// → ISTFTHead (n_fft=392, hop=98) → 44.1 kHz waveform.
//
// The encoder path (WavLM → local_encoder → FSQ encode) is handled separately;
// this module implements decode-only (token indices + global embedding → PCM).
//
// Stage names (for crispasr-diff):
//   fsq_decoded          — (T, 768) FSQ codebook lookup output
//   wave_prenet_out      — (T, 512) after wave_prenet transformer
//   wave_prior_net_out   — (512, T') after ResNet prior blocks
//   wave_decoder_out     — (T', 512) after AdaLN-Zero transformer
//   wave_post_net_out    — (512, T') after ResNet post blocks
//   wave_upsampler_out   — (T'', 512) after SnakeBeta upsampler
//   istft_mag_phase      — (394, T'') linear projection (mag+phase)
//   output_waveform      — (samples,) final reconstructed audio

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct miocodec_context;

struct miocodec_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
};

struct miocodec_params miocodec_default_params(void);

struct miocodec_context* miocodec_init_from_file(const char* path, struct miocodec_params params);

void miocodec_free(struct miocodec_context* ctx);

uint32_t miocodec_sample_rate(const struct miocodec_context* ctx);   // 44100
uint32_t miocodec_n_fft(const struct miocodec_context* ctx);         // 392
uint32_t miocodec_hop_length(const struct miocodec_context* ctx);    // 98
uint32_t miocodec_codebook_size(const struct miocodec_context* ctx); // 12800
uint32_t miocodec_token_rate(const struct miocodec_context* ctx);    // 25

// Decode token indices + global embedding to mono float32 PCM at 44.1 kHz.
//
// Args:
//   token_indices: array of n_tokens int32 values in [0, 12799].
//   global_embedding: float32 array of 128 elements (speaker/style vector).
//   target_audio_length: desired output length in samples (0 = auto from n_tokens).
//
// Returns malloc'd float32 PCM buffer. Caller frees with free().
// *out_n_samples is set on success. Returns nullptr on error.
float* miocodec_decode(struct miocodec_context* ctx, const int32_t* token_indices, int n_tokens,
                       const float* global_embedding, int target_audio_length, int* out_n_samples);

// Run decode and extract a named intermediate tensor for diff testing.
// Returns a malloc'd float32 buffer of *out_n elements. Caller frees.
float* miocodec_extract_stage(struct miocodec_context* ctx, const int32_t* token_indices, int n_tokens,
                              const float* global_embedding, int target_audio_length, const char* stage_name,
                              int* out_n);

#ifdef __cplusplus
}
#endif

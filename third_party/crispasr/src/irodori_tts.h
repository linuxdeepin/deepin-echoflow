// irodori_tts.h — Irodori-TTS native ggml runtime.
//
// Aratako/Irodori-TTS: RF-DiT flow matching TTS with zero-shot voice cloning.
// Architecture: TextEncoder (14-layer RoPE transformer) + ReferenceLatentEncoder
// (14-layer) + 24-layer DiT with LowRankAdaLN + JointAttention + half-RoPE +
// SwiGLU + Euler ODE solver (40 steps, CFG) + DAC-VAE decoder (48 kHz).
// ~500M params DiT, 32-dim latent space, Japanese-focused.
//
// Voice cloning via reference audio DAC-VAE latent conditioning.
// Text tokenization via sarashina2.2 (102400 vocab SentencePiece).

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct irodori_tts_context;

struct irodori_tts_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    int seed;                // random seed (0 = non-deterministic)
    int ode_steps;           // number of Euler ODE steps (default 40)
    float cfg_scale_text;    // text CFG guidance scale (default 3.0)
    float cfg_scale_speaker; // speaker CFG guidance scale (default 5.0)
    float cfg_scale_caption; // caption CFG guidance scale (default 3.0, VoiceDesign)
    float speed;             // speech speed factor (>1 = faster, <1 = slower)
    float duration_scale;    // duration multiplier (>1 = longer, <1 = shorter)
    float max_seconds;       // maximum output duration in seconds (default 30)
};

struct irodori_tts_params irodori_tts_default_params(void);

// Load an Irodori-TTS GGUF model. Returns nullptr on failure.
struct irodori_tts_context* irodori_tts_init_from_file(const char* path_model, struct irodori_tts_params params);

void irodori_tts_free(struct irodori_tts_context* ctx);

// Set reference audio for voice cloning. Audio is mono PCM at any sample rate
// (will be resampled internally to 48 kHz for DAC-VAE encoding).
// ref_pcm: pointer to mono float PCM samples
// n_samples: number of samples
// sample_rate: sample rate of the reference audio
// Returns 0 on success, -1 on failure.
int irodori_tts_set_reference(struct irodori_tts_context* ctx, const float* ref_pcm, int n_samples, int sample_rate);

// Set reference from pre-encoded DAC-VAE latents (32-dim, T frames).
// latent: pointer to T*32 float values (row-major, T frames × 32 dims)
// n_frames: number of latent frames
// Returns 0 on success, -1 on failure.
int irodori_tts_set_reference_latent(struct irodori_tts_context* ctx, const float* latent, int n_frames);

// Get the currently-set reference latent (e.g. to cache it to disk after
// encoding). Returns a borrowed pointer to the internal buffer (valid until
// the reference is changed/cleared), the frame count, and the per-frame dim
// (latent_dim * latent_patch_size, = 32). Returns 0 on success, -1 if no
// reference is set. Do not free the returned pointer.
int irodori_tts_get_reference_latent(const struct irodori_tts_context* ctx, const float** out_latent, int* out_frames,
                                     int* out_dim);

// Per-frame dimension of a reference latent (latent_dim * latent_patch_size).
// Useful for validating a cached latent before irodori_tts_set_reference_latent.
int irodori_tts_reference_latent_dim(const struct irodori_tts_context* ctx);

// Clear reference (unconditional generation).
void irodori_tts_clear_reference(struct irodori_tts_context* ctx);

// Set caption text for VoiceDesign style conditioning.
// Pass NULL or "" to clear (unconditional caption).
void irodori_tts_set_caption(struct irodori_tts_context* ctx, const char* caption);

// Synthesize text to mono 48 kHz PCM.
// Returns number of samples written, 0 on failure.
// Caller owns the returned buffer (malloc'd; free with free()).
int irodori_tts_synthesize(struct irodori_tts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out);

// Load the DAC-VAE decoder GGUF (companion model for audio reconstruction).
// Must be called before synthesize for audio output (otherwise outputs silence).
// Returns 0 on success, -1 on failure.
int irodori_tts_set_codec_path(struct irodori_tts_context* ctx, const char* codec_gguf_path);

// Runtime parameter setters.
void irodori_tts_set_seed(struct irodori_tts_context* ctx, int seed);
void irodori_tts_set_ode_steps(struct irodori_tts_context* ctx, int steps);
void irodori_tts_set_cfg_scale_text(struct irodori_tts_context* ctx, float scale);
void irodori_tts_set_cfg_scale_speaker(struct irodori_tts_context* ctx, float scale);
void irodori_tts_set_cfg_scale_caption(struct irodori_tts_context* ctx, float scale);
void irodori_tts_set_speed(struct irodori_tts_context* ctx, float speed);

// Query model info.
int irodori_tts_sample_rate(const struct irodori_tts_context* ctx);
int irodori_tts_vocab_size(const struct irodori_tts_context* ctx);

#ifdef __cplusplus
}
#endif

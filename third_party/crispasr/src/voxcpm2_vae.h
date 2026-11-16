// voxcpm2_vae.h — isolated VoxCPM2 AudioVAE speech upscaler C API.
#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct voxcpm2_vae_context;

struct voxcpm2_vae_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
};

struct voxcpm2_vae_context_params voxcpm2_vae_context_default_params(void);

// Load only the AudioVAE tensors (the `vae.*` namespace) from a VoxCPM2
// GGUF. The GGUF may contain the full TTS model or only the AudioVAE.
struct voxcpm2_vae_context* voxcpm2_vae_init_from_file(const char* path_model,
                                                       struct voxcpm2_vae_context_params params);
void voxcpm2_vae_free(struct voxcpm2_vae_context* ctx);

// Upscale 16 kHz mono float32 speech to 48 kHz mono float32 speech.
// The returned buffer contains exactly 3 * n_samples values and must be
// released with voxcpm2_vae_pcm_free(). Inputs longer than 960000 samples
// are rejected by default; CRISPASR_VOXCPM2_VAE_MAX_SAMPLES overrides the
// cap when the caller has sufficient memory.
float* voxcpm2_vae_upscale(struct voxcpm2_vae_context* ctx, const float* samples, int n_samples, int* out_n_samples);
void voxcpm2_vae_pcm_free(float* pcm);

#ifdef __cplusplus
}
#endif

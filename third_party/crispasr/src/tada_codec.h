// src/tada_codec.h — TADA codec decoder (C ABI).
//
// Converts expanded acoustic feature sequences (512-d vectors at 50 Hz)
// into 24 kHz mono PCM via:
//   1. Linear projection (512 → 1024)
//   2. Local-attention transformer encoder (6 layers, 1024-d, 8 heads, RoPE)
//   3. DAC-style upsampler (WNConv1d + Snake1d, strides [4,4,5,6] → 480×)

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct tada_codec_context;

struct tada_codec_context* tada_codec_init_from_file(const char* path, int n_threads);
struct tada_codec_context* tada_codec_init_from_file_ex(const char* path, int n_threads, bool use_gpu);
struct tada_codec_context* tada_codec_init_from_file_with_backend(const char* path, int n_threads,
                                                                  ggml_backend_t backend, ggml_backend_t backend_cpu);

// Decode expanded features to PCM.
// features: (n_frames, 512) float32 row-major
// token_masks: (n_frames,) int32 — 1 where features are non-zero
// Returns heap-allocated PCM, caller frees with tada_codec_pcm_free().
float* tada_codec_decode(struct tada_codec_context* ctx, const float* features, int n_frames,
                         const int32_t* token_masks, int* out_n_samples);

// Debug/diff helper. Runs the same full codec graph as tada_codec_decode and
// returns the named graph tensor as float32. Stage names are internal graph
// names such as "dump_proj", "dump_attn", and "pcm".
float* tada_codec_extract_stage(struct tada_codec_context* ctx, const float* features, int n_frames,
                                const int32_t* token_masks, const char* stage, int* out_n);

void tada_codec_pcm_free(float* pcm);
void tada_codec_free(struct tada_codec_context* ctx);

#ifdef __cplusplus
}
#endif

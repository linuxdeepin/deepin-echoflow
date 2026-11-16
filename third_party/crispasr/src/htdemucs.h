// src/htdemucs.h — HTDemucs source separation (Meta Demucs v4).
//
// 4-stem music source separation: drums, bass, other, vocals.
// Hybrid model with parallel frequency (STFT) and time (waveform) branches,
// connected by a CrossTransformer at the bottleneck.
//
// Reference: Rouard et al., "Hybrid Transformers for Music Source Separation"
//            https://arxiv.org/abs/2211.08553

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct htdemucs_context;

struct htdemucs_params {
    int n_threads;  // 0 = auto
    bool use_gpu;   // attempt GPU acceleration
    int gpu_device; // GPU device index
};

// Separation result: one waveform per source.
struct htdemucs_result {
    int n_sources;   // always 4 (drums, bass, other, vocals)
    int n_channels;  // 2 (stereo)
    int n_samples;   // per-channel sample count
    int sample_rate; // 44100
    // Interleaved: sources[s] points to n_channels * n_samples floats.
    float** sources;
    const char** source_names; // "drums", "bass", "other", "vocals"
};

htdemucs_params htdemucs_default_params(void);

htdemucs_context* htdemucs_init_from_file(const char* model_path, htdemucs_params params);
void htdemucs_free(htdemucs_context* ctx);

// Separate a stereo audio signal at 44100 Hz.
// pcm: interleaved stereo float32, n_samples per channel.
// Returns null on failure. Caller must call htdemucs_result_free().
htdemucs_result* htdemucs_separate(htdemucs_context* ctx, const float* pcm_stereo, int n_samples);
void htdemucs_result_free(htdemucs_result* r);

// Get the model's expected sample rate (44100).
int htdemucs_sample_rate(const htdemucs_context* ctx);

// Source names.
int htdemucs_n_sources(const htdemucs_context* ctx);
const char* htdemucs_source_name(const htdemucs_context* ctx, int idx);

// Per-stage parity diff against a Python reference dump
// (tools/reference_backends/htdemucs.py). Replays the 44.1 kHz waveform stored
// in the reference's `input_wav` stage, so the comparison is input-aligned and
// independent of any resampler. Returns 0 if every stage passes, 1 on a parity
// failure, 2 on a load error.
int htdemucs_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity);

#ifdef __cplusplus
}
#endif

// zonos_tts.h -- Zonos TTS public C ABI.
//
// Zyphra/Zonos-v0.1-transformer: Apache 2.0 licensed ~500M-param TTS
// model with rich speaker conditioning (pitch, speaking rate, emotion).
// Character-level eSpeak phoneme input, 26-layer GPT-style transformer
// AR backbone generating DAC 44.1 kHz audio codes (9 codebooks x 1024).
//
// Architecture:
//   1. Prefix conditioner: concatenates conditioning embeddings
//      (phonemes, speaker, emotion, fmax, pitch_std, rate, language)
//      through projection + LayerNorm -> prefix hidden states
//   2. AR backbone: 26 transformer layers (d=2048, GQA 16/4, SwiGLU,
//      LayerNorm with bias, RoPE) decode 9 codebook streams with delay
//      pattern (each codebook shifted by 1 step)
//   3. DAC decoder: separate GGUF, 9-codebook RVQ -> conv upsample
//      stack -> 44.1 kHz PCM
//
// The AR backbone uses classifier-free guidance (CFG): during decode,
// both conditioned and unconditioned prefix are prepended, and the
// logits are interpolated: logits = uncond + cfg_scale * (cond - uncond).
//
// Status: skeleton (PLAN #130).

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct zonos_tts_context;

struct zonos_tts_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature;    // 0 = greedy (default 0.0 = greedy, upstream uses min_p=0.1)
    uint64_t seed;        // RNG seed (0 = default)
    int max_audio_tokens; // upper bound on AR decode steps; 0 = default (86*30=2580)
    bool flash_attn;
    float cfg_scale; // classifier-free guidance scale (default 2.0)
};

struct zonos_tts_params zonos_tts_default_params(void);

// Load the Zonos AR transformer GGUF (arch="zonos-tts").
struct zonos_tts_context* zonos_tts_init_from_file(const char* path_model, struct zonos_tts_params params);

// Point at the DAC 44.1 kHz decoder GGUF. Required before synthesize.
int zonos_tts_set_codec_path(struct zonos_tts_context* ctx, const char* path);

// --- Conditioning setters (call before synthesize) ---

// Set pitch standard deviation (0-400). 20-45 = normal, 60-150 = expressive.
void zonos_tts_set_pitch_std(struct zonos_tts_context* ctx, float pitch_std);

// Set speaking rate in phonemes/s (0-40). 10 = slow, 15 = normal, 30 = fast.
void zonos_tts_set_speaking_rate(struct zonos_tts_context* ctx, float rate);

// Set emotion vector (8 floats: happiness, sadness, disgust, fear,
// surprise, anger, other, neutral). Will be normalized to sum=1.
void zonos_tts_set_emotion(struct zonos_tts_context* ctx, const float* emotion, int len);

// Set maximum frequency (0-24000). Use 22050 for voice cloning.
void zonos_tts_set_fmax(struct zonos_tts_context* ctx, float fmax);

// Set language by code string (e.g. "en-us", "de", "fr-fr").
// Returns 0 on success, -1 if language not found.
int zonos_tts_set_language(struct zonos_tts_context* ctx, const char* lang_code);

// Load reference audio for speaker cloning. The speaker embedding will
// be extracted from the audio using the built-in ResNet293 encoder.
// For the C++ runtime, we pre-compute the 128-d LDA embedding.
// Returns 0 on success.
int zonos_tts_set_voice(struct zonos_tts_context* ctx, const char* wav_path);

// Set speaker embedding directly (128 floats, LDA-projected).
void zonos_tts_set_speaker_embedding(struct zonos_tts_context* ctx, const float* emb, int dim);

// --- Synthesis ---

// Synthesize text -> 44.1 kHz mono float32 PCM.
// Uses eSpeak phonemization internally (or the C++ phoneme table).
// Caller frees with zonos_tts_pcm_free. Returns nullptr on failure.
float* zonos_tts_synthesize(struct zonos_tts_context* ctx, const char* text, int* out_n_samples);

// Run the AR backbone only, return raw DAC codes.
// Output shape: n_codebooks * seq_len (interleaved).
int32_t* zonos_tts_synthesize_codes(struct zonos_tts_context* ctx, const char* text, int* out_n_codes,
                                    int* out_n_codebooks);

void zonos_tts_codes_free(int32_t* codes);
void zonos_tts_pcm_free(float* pcm);
void zonos_tts_free(struct zonos_tts_context* ctx);

// --- Diff-harness stage APIs (used by crispasr-diff) ---

// Build the conditioning prefix for diff testing.
// Runs the prefix conditioner on `text` and returns a float buffer of shape
// (2 * prefix_len, d_model): first prefix_len rows = conditioned path,
// next prefix_len rows = unconditioned path.
// Caller frees with free(). Returns nullptr on failure.
// out_prefix_len receives the per-path length; out_d_model receives d_model.
float* zonos_tts_build_conditioning_prefix(struct zonos_tts_context* ctx, const char* text, int* out_prefix_len,
                                           int* out_d_model);

// Run up to n_steps_req AR decode steps and capture CFG-blended logits at each.
// Return layout: float[n_steps_out][n_codebooks][head_vocab_size] where:
//   slot 0      = prefill logits (CFG-blended, no EOS masking)
//   slot k≥1   = AR step k-1 logits (CFG-blended + EOS mask for cb1-8)
// Uses greedy sampling (temperature=0) so inputs are deterministic.
// out_n_steps = 1 + AR steps executed (stopped at n_steps_req or EOS).
// Caller frees with free(). Returns nullptr on failure.
float* zonos_tts_run_ar_steps_dump(struct zonos_tts_context* ctx, const char* text, int n_steps_req, int* out_n_steps,
                                   int* out_n_cb, int* out_vocab);

// Return the last-token backbone hidden state at prefill for diff testing.
// Runs the prefix conditioner + backbone prefill (same path as synthesize_codes)
// and returns a float buffer of shape (2, d_model): row 0 = conditioned path,
// row 1 = unconditioned path (after the final LayerNorm, before head projection).
// out_d_model receives d_model. Caller frees with free(). Returns nullptr on failure.
float* zonos_tts_get_prefill_hidden(struct zonos_tts_context* ctx, const char* text, int* out_d_model);

void zonos_tts_set_n_threads(struct zonos_tts_context* ctx, int n_threads);
void zonos_tts_set_temperature(struct zonos_tts_context* ctx, float temperature);
void zonos_tts_set_seed(struct zonos_tts_context* ctx, uint64_t seed);
void zonos_tts_set_cfg_scale(struct zonos_tts_context* ctx, float cfg_scale);

#ifdef __cplusplus
}
#endif

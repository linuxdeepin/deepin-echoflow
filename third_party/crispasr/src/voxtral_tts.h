// voxtral_tts.h — C API for Voxtral-4B-TTS (mistralai/Voxtral-4B-TTS-2603).
//
// Architecture: Ministral-3B AR backbone (26L GQA)
//             + 3-layer acoustic flow-matching transformer (8-step Euler ODE)
//             + Voxtral codec decoder (4 conv+transformer blocks → 24 kHz PCM)
//
// Text flow: Tekken BPE → embed → [voice prefix] → LLM AR decode
//          → per-frame FM ODE → semantic VQ + acoustic FSQ → codec decode → PCM
//
// 20 preset voices (en/fr/de/es/it/pt/nl/ar/hi, male/female).

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct voxtral_tts_context;

struct voxtral_tts_context_params {
    int n_threads;
    int verbosity;     // 0=silent 1=normal 2=verbose
    bool use_gpu;      // false => force CPU backend
    float temperature; // 0 = greedy argmax (default), >0 = softmax sampling for AR
    int n_ode_steps;   // 0 = default (8) for flow-matching ODE
    float cfg_alpha;   // 0 = default (1.2) classifier-free guidance scale
};

struct voxtral_tts_context_params voxtral_tts_context_default_params(void);

struct voxtral_tts_context* voxtral_tts_init_from_file(const char* path_model,
                                                       struct voxtral_tts_context_params params);

void voxtral_tts_free(struct voxtral_tts_context* ctx);

// Seed the flow-matching noise RNG for reproducible acoustic sampling.
void voxtral_tts_set_seed(struct voxtral_tts_context* ctx, uint64_t seed);

// Synthesize text to 24 kHz mono PCM. `voice` is a preset name
// (e.g. "fr_female", "neutral_male") or NULL for the default voice.
// Returns malloc'd float array; caller frees with voxtral_tts_pcm_free().
float* voxtral_tts_synthesize(struct voxtral_tts_context* ctx, const char* text, const char* voice, int* out_n_samples);

void voxtral_tts_pcm_free(float* pcm);

// Output sample rate (always 24000).
int voxtral_tts_sample_rate(void);

// List available voice presets. Returns a null-terminated array of strings.
// The returned pointer is owned by the context and must not be freed.
const char* const* voxtral_tts_list_voices(struct voxtral_tts_context* ctx, int* out_n_voices);

// crispasr-diff: per-layer frame-0 LLM cos vs a reference GGUF (from
// tools/reference_backends/voxtral_tts.py). Returns 0 if all stages pass (cos>=0.99),
// 1 on divergence, 2 on load error. Text/voice from VOXTRAL_TTS_TEXT/VOICE env.
int voxtral_tts_llm_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

#ifdef __cplusplus
}
#endif

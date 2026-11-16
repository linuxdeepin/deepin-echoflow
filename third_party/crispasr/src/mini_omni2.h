// mini_omni2.h — C API for Mini-Omni2 (gpt-omni/mini-omni2).
//
// Architecture: Whisper-small encoder (80 mel, 12L, 768d)
//             + whisperMLP adapter (SwiGLU 768→4864→896)
//             + Qwen2-0.5B LLM (896d, 24L, GQA 14/2, SwiGLU, RMSNorm)
//
// Capabilities:
//   ASR  — audio → text transcription (A1_T1 mode, _asr token)
//   TTS  — text → audio synthesis (T1_A2 mode, 7-stream SNAC + text)
//   S2S  — audio → audio response (A1_A2 mode, 7-stream SNAC + text)

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mini_omni2_context;

struct mini_omni2_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    float temperature; // 0 = greedy argmax, >0 = softmax sampling
};

struct mini_omni2_context_params mini_omni2_context_default_params(void);

struct mini_omni2_context* mini_omni2_init_from_file(const char* path_model, struct mini_omni2_context_params params);

void mini_omni2_free(struct mini_omni2_context* ctx);

// Override the default instruction. Pass NULL or "" to reset.
// For ASR the model uses _asr token; set_ask overrides the task prompt.
void mini_omni2_set_ask(struct mini_omni2_context* ctx, const char* prompt);
// #292: forward --max-new-tokens; <= 0 keeps the backend's 512 default.
void mini_omni2_set_max_new_tokens(struct mini_omni2_context* ctx, int n);

// ---- ASR: audio → text ----

// Transcribe raw 16 kHz mono PCM audio (pure ASR, uses _asr token).
// Returns malloc'd UTF-8 string, caller frees with free().
char* mini_omni2_transcribe(struct mini_omni2_context* ctx, const float* samples, int n_samples);

// ---- TTS: text → audio ----

// Synthesize speech from text. Returns malloc'd mono PCM at 24 kHz.
// out_n_samples receives the sample count. Caller frees with free().
// Requires a SNAC decoder GGUF loaded via mini_omni2_load_snac().
float* mini_omni2_synthesize(struct mini_omni2_context* ctx, const char* text, int* out_n_samples);

// Load a SNAC 24kHz decoder GGUF for TTS/S2S output.
bool mini_omni2_load_snac(struct mini_omni2_context* ctx, const char* snac_path);

// ---- S2S: audio → audio ----

// Speech-to-speech: audio in → spoken response out.
// out_text: if non-NULL, receives malloc'd response text. Caller frees.
// out_n_samples: receives output PCM sample count.
// Returns output PCM at 24 kHz mono. Caller frees with free().
float* mini_omni2_speech_to_speech(struct mini_omni2_context* ctx, const float* in_samples, int n_in_samples,
                                   char** out_text, int* out_n_samples);

// ---- Pipeline building blocks for differential testing ----

float* mini_omni2_compute_mel(struct mini_omni2_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                              int* out_T_mel);

float* mini_omni2_run_encoder(struct mini_omni2_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                              int* out_dim);

float* mini_omni2_run_adapter(struct mini_omni2_context* ctx, const float* enc, int T_enc, int enc_dim, int* out_T,
                              int* out_dim);

bool mini_omni2_kv_init(struct mini_omni2_context* ctx, int max_ctx);
void mini_omni2_kv_reset(struct mini_omni2_context* ctx);

float* mini_omni2_run_llm_kv(struct mini_omni2_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                             int* out_n_tokens, int* out_vocab_size);

const char* mini_omni2_token_text(struct mini_omni2_context* ctx, int id);

#ifdef __cplusplus
}
#endif

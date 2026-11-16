// canary_qwen.h — public C API for nvidia/canary-qwen-2.5b ggml runtime
//
// SALM (Speech-Augmented Language Model): FastConformer encoder (32L, d=1024)
// → linear projection (1024→2048) → Qwen3-1.7B LLM with merged LoRA.
// English ASR only, up to ~40s audio.
//
// Models are loaded from GGUF files produced by:
//   python models/convert-canary-qwen-to-gguf.py --input <hf_dir> --output X.gguf

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct canary_qwen_context;

struct canary_qwen_context_params {
    int n_threads;
    int verbosity;   // 0=silent 1=normal 2=verbose
    bool use_gpu;    // false => force CPU backend
    bool flash_attn; // flash attention in encoder + LLM
};

struct canary_qwen_context_params canary_qwen_context_default_params(void);

struct canary_qwen_context* canary_qwen_init_from_file(const char* path_model,
                                                       struct canary_qwen_context_params params);

void canary_qwen_free(struct canary_qwen_context* ctx);

// ---- Per-token data returned by canary_qwen_transcribe_ex() ----

struct canary_qwen_token_data {
    int id;        // token id
    char text[64]; // decoded text
    float p;       // softmax probability [0, 1]
};

struct canary_qwen_result {
    char* text;
    struct canary_qwen_token_data* tokens;
    int n_tokens;
};

void canary_qwen_result_free(struct canary_qwen_result* r);

// Transcribe raw 16 kHz mono PCM. Returns malloc'd UTF-8 string (caller owns).
char* canary_qwen_transcribe(struct canary_qwen_context* ctx, const float* samples, int n_samples);

// Like canary_qwen_transcribe but returns per-token data.
struct canary_qwen_result* canary_qwen_transcribe_ex(struct canary_qwen_context* ctx, const float* samples,
                                                     int n_samples);

// Vocabulary helpers
int canary_qwen_n_vocab(struct canary_qwen_context* ctx);
const char* canary_qwen_token_to_str(struct canary_qwen_context* ctx, int token_id);

// Sampling: temperature > 0 switches to sampling. Default 0 = greedy.
void canary_qwen_set_temperature(struct canary_qwen_context* ctx, float temperature, uint64_t seed);

// Beam search width. n > 1 activates beam search; n <= 0 → 1 (greedy).
void canary_qwen_set_beam_size(struct canary_qwen_context* ctx, int n);
// #292: forward --max-new-tokens. <= 0 keeps the backend's 256 default.
void canary_qwen_set_max_new_tokens(struct canary_qwen_context* ctx, int n);

// Hyper-parameters
int canary_qwen_frame_dur_cs(struct canary_qwen_context* ctx);
int canary_qwen_n_mels(struct canary_qwen_context* ctx);
int canary_qwen_sample_rate(struct canary_qwen_context* ctx);

// ---- Stage-level entry points (for crispasr-diff testing) ----

// Log-mel spectrogram of raw 16 kHz mono PCM, row-major (n_mels, T_mel).
float* canary_qwen_compute_mel(struct canary_qwen_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                               int* out_T_mel);

// Run encoder on mel features. Returns malloc'd (T_enc, d_enc=1024) F32.
float* canary_qwen_run_encoder(struct canary_qwen_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                               int* out_d_enc);

// Run projection on encoder output. Returns malloc'd (T_enc, d_llm=2048) F32.
float* canary_qwen_run_projection(struct canary_qwen_context* ctx, const float* enc_out, int T_enc, int d_enc,
                                  int* out_T, int* out_d_llm);

#ifdef __cplusplus
}
#endif

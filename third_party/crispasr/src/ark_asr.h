// src/ark_asr.h — ARK-ASR-3B C runtime API.
//
// ARK-ASR-3B = Whisper-large-v3 encoder (partial RoPE) + MLP adapter +
// Qwen2.5-3B decoder with audio-token injection. See PLAN.md §ARK.
#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct ark_asr_context;

struct ark_asr_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature; // 0 = greedy (only greedy is implemented for now)
    int beam_size;     // 1 = greedy (default); >1 = beam search (core_beam_decode)
};

struct ark_asr_context_params ark_asr_context_default_params(void);

// Initialise from the ARK-ASR GGUF file. Returns nullptr on failure.
struct ark_asr_context* ark_asr_init_from_file(const char* path_model, struct ark_asr_context_params params);

// Transcribe PCM audio (16 kHz mono float32). Caller frees the returned
// malloc'd C string with free(). Returns nullptr on failure.
char* ark_asr_transcribe(struct ark_asr_context* ctx, const float* pcm, int n_samples);

// Optional language hint, e.g. "de" / "en". Stored on the context.
void ark_asr_set_language(struct ark_asr_context* ctx, const char* lang_iso);

// EXPERIMENTAL: set a transcription instruction (e.g. "Transcribe the audio in
// German.") that is BPE-tokenised and prepended to the user turn. Empty/null
// clears it (default promptless behaviour). The CLI/session build this from the
// `-l` language flag. Promptless ARK was not trained on instructions, so this
// is best-effort language steering; the default (no instruction) is unchanged.
void ark_asr_set_ask(struct ark_asr_context* ctx, const char* instruction);

void ark_asr_free(struct ark_asr_context* ctx);

// ---- diff-harness stage extraction (CRISPASR_ARKASR_* debug paths) ----
// Compute the mel spectrogram. Returns malloc'd (n_mels, T_mel) row-major.
float* ark_asr_compute_mel(struct ark_asr_context* ctx, const float* pcm, int n_samples, int* out_n_mels,
                           int* out_T_mel);
// Run encoder+adapter on pre-padded audio; returns malloc'd (hidden, N) audio
// embeddings, N = number of merged frames. Caller frees.
float* ark_asr_run_encoder(struct ark_asr_context* ctx, const float* pcm, int n_samples, int* out_hidden, int* out_n);
// Prefill logits at the last prompt position; returns malloc'd float[vocab].
float* ark_asr_prefill_logits(struct ark_asr_context* ctx, const float* pcm, int n_samples, int* out_vocab);

#ifdef __cplusplus
}
#endif

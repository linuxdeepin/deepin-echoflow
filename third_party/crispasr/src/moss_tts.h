// moss_tts.h — public C API for MOSS-TTS-v1.5 (MossTTSDelay) ggml runtime
//
// Text-to-speech: a Qwen3-8B backbone autoregressively emits 32 RVQ audio
// codebooks under a delay pattern, decoded to 24 kHz audio by a 1.6B
// pure-transformer codec (companion GGUF). Models are produced by:
//   python models/convert-moss-tts-to-gguf.py --input <hf> --output X.gguf \
//          [--codec <hf>]   # writes X.gguf + X-codec.gguf
//
// Architecture: OpenMOSS-Team/MOSS-TTS-v1.5 (Apache-2.0)
//   Backbone : Qwen3-8B (36L, 4096d, 32Q/8KV, head_dim 128, QK-norm, SwiGLU,
//              RoPE NEOX theta 1e6). Input embedding = text_emb + Sum_i audio_emb_i.
//   Heads    : 1 text lm_head + 32 audio codebook heads (n_vq = 32).
//   Delay    : MossTTSDelay staggered-emit state machine (see delay logic).
//   Codec    : transformer RVQ codec (Phase 4; loaded from the companion GGUF).
//
// The backbone graph mirrors qwen3_asr's Qwen3 KV path but additionally exposes
// the per-token last hidden state (fed to the 32 audio heads), as qwen3_tts does.

#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct moss_tts_context;

struct moss_tts_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    bool flash_attn;
};

struct moss_tts_context_params moss_tts_context_default_params(void);

// Load the backbone GGUF. The companion codec GGUF (if present) is loaded
// lazily on the first synthesize() / via moss_tts_set_codec_path().
struct moss_tts_context* moss_tts_init_from_file(const char* path_model, struct moss_tts_context_params params);

void moss_tts_free(struct moss_tts_context* ctx);

// Point the runtime at the companion codec GGUF (arch "moss-tts-codec").
// Returns false if the file can't be opened / parsed.
bool moss_tts_set_codec_path(struct moss_tts_context* ctx, const char* path_codec);

// Voice cloning: encode a reference WAV (mono f32 @ sampling_rate) so subsequent
// synthesize()/generate_codes() clone that voice. Pass samples=NULL or
// n_samples<=0 to clear. Requires the codec (with encoder) loaded. Returns false
// on error (no codec/encoder, or encode failure).
bool moss_tts_set_reference_wav(struct moss_tts_context* ctx, const float* samples, int n_samples);
bool moss_tts_has_reference(const struct moss_tts_context* ctx);

// Convenience: load a reference WAV from `path` (any sample rate; resampled to
// the model rate), encode it, and set it as the voice prompt. path=NULL clears.
bool moss_tts_set_reference_wav_file(struct moss_tts_context* ctx, const char* path);

// ---- Model dims (read from GGUF metadata) ----
int moss_tts_n_vq(const struct moss_tts_context* ctx);
int moss_tts_hidden_size(const struct moss_tts_context* ctx);
int moss_tts_audio_vocab_size(const struct moss_tts_context* ctx); // WITHOUT the +1 pad
int moss_tts_sampling_rate(const struct moss_tts_context* ctx);
bool moss_tts_codec_loaded(const struct moss_tts_context* ctx);

// ---- Tokenizer (Qwen3 gpt2-style BPE, special-token aware) ----
// Returns a malloc'd id array (caller frees); sets *out_n_tokens.
int32_t* moss_tts_tokenize(struct moss_tts_context* ctx, const char* text, int* out_n_tokens);
const char* moss_tts_token_text(struct moss_tts_context* ctx, int token_id);

// Diagnostic (parity): the col-0 prompt token ids the AR loop builds for `text`.
// Caller frees; sets *out_n.
int32_t* moss_tts_debug_prompt_ids(struct moss_tts_context* ctx, const char* text,
                                   const struct moss_tts_synth_params* sp, int* out_n);

// Diagnostic (parity): the step-0 audio-head logits for one `codebook`
// (audio_vocab_size+1 floats). Prefills the prompt like the AR loop's first
// step. Caller frees; sets *out_len.
float* moss_tts_debug_first_audio_logits(struct moss_tts_context* ctx, const char* text,
                                         const struct moss_tts_synth_params* sp, int codebook, int* out_len);

// ---- Backbone (KV-cached) ----
bool moss_tts_kv_init(struct moss_tts_context* ctx, int max_ctx);
void moss_tts_kv_reset(struct moss_tts_context* ctx);

// Run one backbone step with the KV cache. inputs_embeds is (hidden, n_tokens)
// F32 row-major (column-major in ggml: ne[0]=hidden). Returns malloc'd text
// logits (vocab_size,) for the LAST position, and — when out_hidden is non-null
// — writes a malloc'd (hidden,) F32 last-position hidden state to *out_hidden
// (the input to the 32 audio heads). Caller frees both.
float* moss_tts_run_llm_kv(struct moss_tts_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                           int* out_vocab_size, float** out_hidden);

// ---- Audio extension aux graphs ----
// Summed input embedding for a (n_pos, 1+n_vq) int32 prompt grid (col 0 = text
// id, cols 1..n_vq = audio codebook ids or audio_pad_code). Returns malloc'd
// (n_pos, hidden) F32 row-major (ne[0]=hidden per position). Caller frees.
float* moss_tts_compute_input_embeddings(struct moss_tts_context* ctx, const int32_t* prompt_grid, int n_pos);

// Project a (hidden,) F32 last-hidden through all 32 audio heads. Returns
// malloc'd (n_vq, audio_vocab_size+1) F32 row-major. Caller frees.
float* moss_tts_compute_audio_logits(struct moss_tts_context* ctx, const float* hidden);

// ---- End-to-end synthesis ----
struct moss_tts_synth_params {
    int max_new_tokens;     // 0 -> default 4096
    float text_temperature; // <=0 -> greedy
    float text_top_p;
    int text_top_k;
    float audio_temperature;
    float audio_top_p;
    int audio_top_k;
    float audio_repetition_penalty;
    int min_audio_frames;
    int max_audio_frames;
    uint64_t seed;           // 0 -> nondeterministic
    const char* language;    // may be null
    const char* instruction; // may be null
};
struct moss_tts_synth_params moss_tts_synth_default_params(void);

// Synthesize speech for `text`. Returns a malloc'd F32 mono waveform @
// sampling_rate and sets *out_n_samples. Requires a codec (returns null with
// *out_n_samples=0 if the codec is not loaded). Caller frees.
float* moss_tts_synthesize(struct moss_tts_context* ctx, const char* text, const struct moss_tts_synth_params* sp,
                           int* out_n_samples);

// Generate only the RVQ code grid (no codec). Returns malloc'd (n_vq, T_audio)
// int32 row-major and sets *out_n_vq, *out_t_audio. Used for code-parity tests
// and to drive the codec separately. Caller frees.
int32_t* moss_tts_generate_codes(struct moss_tts_context* ctx, const char* text, const struct moss_tts_synth_params* sp,
                                 int* out_n_vq, int* out_t_audio);

void moss_tts_set_seed(struct moss_tts_context* ctx, uint32_t seed);

#ifdef __cplusplus
}
#endif

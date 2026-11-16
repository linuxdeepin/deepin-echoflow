// kyutai_stt.h — C API for Kyutai STT (stt-1b-en_fr, stt-2.6b-en).
//
// Architecture: Mimi audio codec encoder (SEANet CNN + transformer + RVQ)
//             + Causal transformer LM (2048d, 16L, RoPE, SwiGLU, RMSNorm)
//
// Audio flow: 24kHz PCM → SEANet encoder → transformer → downsample →
//             RVQ (32 codebooks) → LM → text tokens → SentencePiece decode

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kyutai_stt_context;

struct kyutai_stt_context_params {
    int n_threads;
    int verbosity;         // 0=silent 1=normal 2=verbose
    bool use_gpu;          // false => force CPU backend
    float temperature;     // 0 = greedy argmax, >0 = softmax sampling
    int beam_size;         // 1 = greedy/sampled (default); >1 = deterministic beam search
    int input_sample_rate; // 0 or 16000 = legacy (resample 16k→24k); 24000 = skip resample
};

struct kyutai_stt_context_params kyutai_stt_context_default_params(void);

struct kyutai_stt_context* kyutai_stt_init_from_file(const char* path_model, struct kyutai_stt_context_params params);

void kyutai_stt_free(struct kyutai_stt_context* ctx);

// High-level: transcribe raw PCM audio (16 kHz default; set
// input_sample_rate=24000 in params to skip the internal resample).
// Returns malloc'd UTF-8 string, caller frees with free().
char* kyutai_stt_transcribe(struct kyutai_stt_context* ctx, const float* samples, int n_samples);

// Variant returning per-emitted-token ids + softmax probs alongside the text.
// All pointers are malloc'd; free with kyutai_stt_result_free.
struct kyutai_stt_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int n_tokens;
};

struct kyutai_stt_result* kyutai_stt_transcribe_with_probs(struct kyutai_stt_context* ctx, const float* samples,
                                                           int n_samples);
void kyutai_stt_result_free(struct kyutai_stt_result* r);

// Token text lookup.
const char* kyutai_stt_token_text(struct kyutai_stt_context* ctx, int id);

// Sticky seed for the multinomial sampler (best-of-N). 0 = leave libc RNG
// state alone; non-zero = `srand(seed)`. Process-global; serialize best-of-N
// at the adapter level.
void kyutai_stt_set_seed(struct kyutai_stt_context* ctx, unsigned int seed);

// Sticky beam size for the per-frame text-token decode. 1 = greedy/sampled
// (default). >1 = deterministic beam search; mutually exclusive with
// temperature sampling. Beam path snapshots the LM KV per beam each frame
// and picks by cumulative log-prob across all T_frames.
void kyutai_stt_set_beam_size(struct kyutai_stt_context* ctx, int beam_size);

// Tell the backend what sample rate the next transcribe's PCM is at.
// 0 or 16000 = legacy (resample 16k→24k); 24000 = skip resample.
void kyutai_stt_set_input_sample_rate(struct kyutai_stt_context* ctx, int rate);

// ---- Per-token + word-level timing (PLAN #61c) ----
//
// Kyutai's "delayed-streams" architecture aligns each emitted text
// token to the audio frame that produced it. Frame rate is 12.5 Hz
// (80 ms per frame); the LM emits one text token per Mimi frame.
// `audio_delay_seconds` (typically 0.5s) is the training-time
// lookahead the LM was given — we subtract it from the LM frame index
// to recover the audio time the token actually corresponds to.

struct kyutai_stt_token_data {
    int id;        // SentencePiece token id
    char text[48]; // decoded text (▁ → space)
    int64_t t0;    // start time, centiseconds (audio-relative, includes t_offset_cs)
    int64_t t1;    // end time,   centiseconds (= t0 + 8, one Mimi frame)
    float p;       // softmax probability of the emitted token [0,1]
};

// Word-level data: sub-tokens grouped at SentencePiece '▁' boundaries.
struct kyutai_stt_word_data {
    char text[64]; // word text (no leading space)
    int64_t t0;    // start time, centiseconds (from first sub-token)
    int64_t t1;    // end time,   centiseconds (from last sub-token)
    float p;       // mean softmax probability across the word's sub-tokens
};

struct kyutai_stt_result_ex {
    char* text;                           // full transcript (malloc'd)
    struct kyutai_stt_token_data* tokens; // per-token timing (malloc'd)
    int n_tokens;
    struct kyutai_stt_word_data* words; // grouped word timings (malloc'd)
    int n_words;
};

void kyutai_stt_result_ex_free(struct kyutai_stt_result_ex* r);

// Returns the model's audio_delay_seconds + audio_silence_prefix_seconds.
// The CLI wrapper uses this to size the silence tail it appends: the tail
// must be at least as long as the lookahead so the LM can flush the final word.
/// Does this checkpoint support `lang` (ISO-639-1, lowercase)? Empty or "auto"
/// always returns 1. stt-1b-en_fr supports en+fr; stt-2.6b-en is English-only,
/// and CrispASR used to warn "English-only model" for both (#366).
int kyutai_stt_supports_language(struct kyutai_stt_context* ctx, const char* lang);

/// Comma-separated languages this checkpoint supports, for diagnostics.
/// Owned by the context; valid until kyutai_stt_free.
const char* kyutai_stt_languages(struct kyutai_stt_context* ctx);

float kyutai_stt_total_lookahead_seconds(struct kyutai_stt_context* ctx);

// Like kyutai_stt_transcribe but returns per-token + word-level timing.
//
// `t_offset_cs`: absolute start of this audio slice in centiseconds.
//   Token t0/t1 = t_offset_cs + (audio_frame * 8). For long audio with
//   VAD slicing, pass (vad_segment_t0_seconds * 100).
//
// Timestamps are accurate to one Mimi frame (80 ms) by construction —
// the kyutai LM's frame-aligned emission means no DTW or auxiliary
// duration head is needed.
struct kyutai_stt_result_ex* kyutai_stt_transcribe_ex(struct kyutai_stt_context* ctx, const float* samples,
                                                      int n_samples, int64_t t_offset_cs);

// ---- Streaming API (PLAN #62c) ----
//
// Mimi's encoder transformer uses non-causal full-sequence attention,
// so true incremental encoding can't bit-match batch. Instead, the
// streaming wrapper buffers 16 kHz PCM into a rolling window and
// re-runs `kyutai_stt_transcribe_ex` over the window every `step_ms`.
// Each decode is bit-exact-batch for its window. Latency >= step_ms;
// for audio longer than `length_ms` only the last `length_ms` is
// transcribed (mirrors whisper's `crispasr_stream_*` semantics).
struct kyutai_stt_stream;

struct kyutai_stt_stream* kyutai_stt_stream_open(struct kyutai_stt_context* ctx, int step_ms, int length_ms);

// Push 16 kHz mono float32 PCM. Returns:
//   0  = still buffering (no new transcript yet)
//   1  = new transcript ready (call kyutai_stt_stream_get_text)
//  <0  = error
int kyutai_stt_stream_feed(struct kyutai_stt_stream* s, const float* pcm_16k, int n_samples);

// Pull the latest decode. `out` receives a NUL-terminated UTF-8 string.
// `*t0_s` / `*t1_s` get the transcribed window's absolute time bounds.
// `*counter` increments per decode; same value = no new text.
// Returns byte count copied (not including NUL), 0 if no output yet, <0 on error.
int kyutai_stt_stream_get_text(struct kyutai_stt_stream* s, char* out, int cap, double* t0_s, double* t1_s,
                               int64_t* counter);

// Force a decode of whatever audio is currently buffered, regardless of
// step boundary. Returns 1 on success, 0 if nothing was buffered, <0 on error.
int kyutai_stt_stream_flush(struct kyutai_stt_stream* s);

void kyutai_stt_stream_close(struct kyutai_stt_stream* s);

// Per-token streaming callback.
typedef void (*kyutai_stt_token_cb)(int tok_id, float prob, void* userdata);

// Like kyutai_stt_transcribe() but fires cb for each emitted (non-pad) text token.
// Falls back to greedy decode (beam_size ignored).
void kyutai_stt_transcribe_cb(struct kyutai_stt_context* ctx, const float* samples, int n_samples,
                              kyutai_stt_token_cb cb, void* userdata);

#ifdef __cplusplus
}
#endif

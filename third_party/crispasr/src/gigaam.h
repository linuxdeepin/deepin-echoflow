// gigaam.h — public C API for the ai-sage/GigaAM-v3 ggml runtime
//
// Russian ASR. A 16-layer rotary Conformer encoder (220 M params) with
// either a CTC head or an RNN-Transducer head, in four shipped flavours:
//
//   gigaam-v3-ctc        charwise vocab (33 Cyrillic chars), lowercase, no
//                        punctuation
//   gigaam-v3-rnnt       same vocab, transducer decode (lower WER)
//   gigaam-v3-e2e-ctc    SentencePiece 256, punctuation + inverse text
//                        normalization baked into the vocabulary
//   gigaam-v3-e2e-rnnt   SentencePiece 1024, punctuation + ITN, best WER
//
// A single GGUF carries the head type (`gigaam.head_type`) and tokenizer
// kind (`gigaam.tokenizer_type`), so one runtime serves all four.
//
// Models come from GGUF files produced by:
//   python models/convert-gigaam-to-gguf.py --model <dir|repo> --revision <rev>
//          --output X.gguf

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct gigaam_context;

struct gigaam_context_params {
    int n_threads;
    bool use_flash; // flash attention in the encoder (default: false)
    int verbosity;  // 0=silent 1=normal 2=verbose
    bool use_gpu;   // false => force CPU backend
};

struct gigaam_context_params gigaam_context_default_params(void);

struct gigaam_context* gigaam_init_from_file(const char* path_model, struct gigaam_context_params params);

void gigaam_free(struct gigaam_context* ctx);

// ---- Per-token / per-word data returned by gigaam_transcribe_ex() ----

struct gigaam_token_data {
    int id;        // vocabulary id
    char text[48]; // decoded text (SentencePiece '▁' converted to ' ')
    int64_t t0;    // start time, centiseconds (absolute, includes t_offset_cs)
    int64_t t1;    // end time, centiseconds
    float p;       // softmax probability of the emitted token [0,1]
};

struct gigaam_word_data {
    char text[64]; // word text (no leading space)
    int64_t t0;
    int64_t t1;
    float p; // mean probability across the word's sub-word tokens
};

struct gigaam_result {
    char* text; // full transcript (malloc'd, caller owns)
    struct gigaam_token_data* tokens;
    int n_tokens;
    struct gigaam_word_data* words;
    int n_words;
};

void gigaam_result_free(struct gigaam_result* r);

// Transcribe raw 16 kHz mono PCM, returning a malloc'd UTF-8 string.
char* gigaam_transcribe(struct gigaam_context* ctx, const float* samples, int n_samples);

// Like gigaam_transcribe but returns per-token timings. Timestamps are one
// encoder frame (40 ms) wide and come from the frame at which the token was
// emitted — for CTC heads that is the argmax frame, for RNN-T the frame the
// transducer emitted on.
//
// t_offset_cs: absolute start of this audio slice in centiseconds (pass
// vad_segment_t0_seconds * 100 when slicing long audio).
struct gigaam_result* gigaam_transcribe_ex(struct gigaam_context* ctx, const float* samples, int n_samples,
                                           int64_t t_offset_cs);

// ---- Model introspection ----

int gigaam_n_vocab(struct gigaam_context* ctx);
int gigaam_blank_id(struct gigaam_context* ctx);
const char* gigaam_token_to_str(struct gigaam_context* ctx, int token_id);
int gigaam_frame_dur_cs(struct gigaam_context* ctx); // centiseconds per encoder frame
int gigaam_n_mels(struct gigaam_context* ctx);
int gigaam_sample_rate(struct gigaam_context* ctx);
int gigaam_d_model(struct gigaam_context* ctx);
int gigaam_n_layers(struct gigaam_context* ctx);
// 1 iff this GGUF carries an RNN-T head (0 = CTC head).
int gigaam_is_rnnt(struct gigaam_context* ctx);
// 1 iff the tokenizer is SentencePiece (0 = charwise). SentencePiece models
// are the `e2e_*` ones and emit punctuation + capitalisation.
int gigaam_is_spm(struct gigaam_context* ctx);
// Approximate encoder frame count for n_samples (for memory policy).
int gigaam_est_enc_frames(struct gigaam_context* ctx, int n_samples);

// Max symbols the RNN-T greedy decoder may emit per encoder frame.
// Matches RNNTGreedyDecoding(max_symbols_per_step=10). <= 0 restores the
// default. No effect on CTC heads.
void gigaam_set_max_symbols(struct gigaam_context* ctx, int max_symbols);

// ---- Stage-level entry points (for crispasr-diff) ----
// Each returns a malloc'd F32 buffer the caller must free(), or nullptr.

// Log-mel spectrogram of raw 16 kHz mono PCM.
// Output layout: row-major (n_mels, T_mel) — each mel band's time series is
// contiguous, which is what the conv1d subsampling consumes.
float* gigaam_compute_mel(struct gigaam_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                          int* out_T_mel);

// Run the encoder on a mel spectrogram (layout as produced above).
// Output layout: row-major (T_enc, d_model).
float* gigaam_run_encoder(struct gigaam_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                          int* out_d_model);

// Run the encoder and capture per-stage intermediates.
//   out[0]    : after pre-encode (conv1d subsampling)
//   out[1..N] : after each conformer layer (N = n_layers)
// Each buffer must hold T_enc * d_model floats; use gigaam_est_enc_frames()
// to size them. Returns 0 on success.
int gigaam_run_encoder_dump(struct gigaam_context* ctx, const float* mel, int n_mels, int T_mel, float** out,
                            int out_count, int* out_T_enc, int* out_d_model);

// CTC heads only: log-softmax'd head output, row-major (T_enc, num_classes).
float* gigaam_ctc_log_probs(struct gigaam_context* ctx, const float* enc_frames, int T_enc, int d_model,
                            int* out_n_classes);

// RNN-T heads only.
// joint.enc projection of every encoder frame — row-major (T_enc, joint_hidden).
float* gigaam_joint_project_encoder(struct gigaam_context* ctx, const float* enc_frames, int T_enc, int d_model,
                                    int* out_joint_hidden);
// Predictor output for the zero embedding + zero state (the t=0 entry point,
// i.e. RNNTDecoder.predict(None, None)). Length pred_hidden.
float* gigaam_predictor_initial(struct gigaam_context* ctx, int* out_pred_hidden);
// One joint step: RAW logits (no log_softmax), length num_classes.
float* gigaam_joint_step(struct gigaam_context* ctx, const float* proj_enc, const float* pred_out,
                         int* out_num_classes);

#ifdef __cplusplus
}
#endif

#pragma once

// confucius4_tts.h — public C ABI for netease-youdao/Confucius4-TTS backend.
//
// Two-stage TTS: T2S (GPT-2, 24L/1280d/20h) generates semantic codes from text
// + speaker conditioning, then S2A (flow-matching DiT+WaveNet) converts them to
// mel spectrograms, which a BigVGAN vocoder renders to 22050 Hz PCM.
//
// The T2S model is a GPT-2 backbone with custom embedding concatenation:
//   [condition_emb(1) | text_emb(T_text) | semantic_emb(T_semantic)]
// where condition_emb comes from a Qwen3TTS-style ECAPA-TDNN speaker encoder
// over Wav2Vec2-BERT layer-17 features, and text_emb is a frozen
// Embedding(32k,4096) → SiLU MLP(4096→1280) projector.
//
// GGUF files (all auto-downloaded by `-m auto`):
//   - T2S (GPT-2 + text projector + ECAPA + baked vocab): confucius4-tts-t2s-q4_k.gguf
//   - S2A (DiT + WaveNet + length regulator + CAMPPlus):  confucius4-tts-s2a-q4_k.gguf
//   - BigVGAN v2 22 kHz vocoder:                          confucius4-tts-bigvgan-22k-f16.gguf
//   - w2v-BERT 2.0 encoder-only (17L, sidon layout):      confucius4-tts-w2v-f16.gguf
//
// Zero-shot voice cloning (`--voice ref.wav --i-have-rights`) is fully native
// when all four files are present. Without --voice conditioning the output is
// unintelligible BY DESIGN — the model is a zero-shot cloner.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct confucius4_tts_context;

struct confucius4_tts_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature;        // T2S sampling temperature (0 = greedy)
    float top_p;              // T2S nucleus sampling
    int top_k;                // T2S top-k sampling
    float repetition_penalty; // T2S repetition penalty
    int max_semantic_tokens;  // max T2S generation length; 0 = default (1520)
    int ode_steps;            // S2A flow-matching ODE steps; 0 = default (25)
    float cfg_rate;           // S2A classifier-free guidance rate; 0 = default (0.7),
                              // negative = disable CFG (single conditioned pass)
    uint64_t seed;            // RNG seed (0 = random)
};

struct confucius4_tts_params confucius4_tts_default_params(void);

// Load the T2S model (GPT-2 + speaker encoder + text projector + tokenizer).
struct confucius4_tts_context* confucius4_tts_init_from_file(const char* path_t2s, struct confucius4_tts_params params);

// Load the S2A model (required before synthesis). Returns 0 on success.
int confucius4_tts_set_s2a_path(struct confucius4_tts_context* ctx, const char* path_s2a);

// Load the BigVGAN vocoder (optional companion GGUF for mel → PCM).
// Without this, synthesis outputs silence at the correct duration.
// Returns 0 on success.
int confucius4_tts_set_vocoder_path(struct confucius4_tts_context* ctx, const char* path_vocoder);

// Load the encoder-only w2v-BERT 2.0 GGUF (confucius4-tts-w2v-f16.gguf,
// sidon layout, 17 layers). With it loaded, confucius4_tts_set_voice_path
// computes the T2S condition_emb fully natively (layer-17 hidden states,
// z-normalised with the baked stats, through the ECAPA encoder).
// Returns 0 on success.
int confucius4_tts_set_w2v_path(struct confucius4_tts_context* ctx, const char* path_w2v);

// Native voice conditioning from a reference WAV (`--voice ref.wav`): computes
// the CAMPPlus style embedding and the 22.05 kHz prompt mel in-process (needs
// the campplus.* bake in the S2A GGUF). The T2S condition_emb still requires
// externally-computed w2v-BERT features (CRISPASR_CONFUCIUS4_COND_DIR or
// confucius4_tts_set_speaker) until w2v-BERT is ported natively.
// Returns 0 on success.
int confucius4_tts_set_voice_path(struct confucius4_tts_context* ctx, const char* wav_path);

// Set reference speaker conditioning from pre-computed Wav2Vec2-BERT features.
// `semantic_features` is (n_frames, 1024) float32 — layer-17 hidden states,
// z-normalised with the baked mean/var. `style_embedding` is (192,) float32
// from CAMPPlus. Both are computed externally until those models are ported.
// Returns 0 on success.
int confucius4_tts_set_speaker(struct confucius4_tts_context* ctx, const float* semantic_features, int n_frames,
                               const float* style_embedding);

// Set pre-computed speaker conditioning directly, bypassing the (unported)
// Wav2Vec2-BERT and ECAPA-TDNN speaker encoders.
//   condition_embedding: (model_dim,) = speaker_encoder(w2v_bert layer 17) in
//                        the reference; prepended to the GPT-2 prefix.
//   style_embedding:     (spk_embed_dim,) from CAMPPlus; the DiT's `spks`.
//   prompt_mel:          (n_prompt_frames, mel_dim) reference mel; prepended to
//                        the flow-matching state and stripped from the output.
// Any pointer may be null to leave that part unset. Returns 0 on success.
int confucius4_tts_set_conditioning(struct confucius4_tts_context* ctx, const float* condition_embedding, int cond_dim,
                                    const float* style_embedding, int style_dim, const float* prompt_mel,
                                    int n_prompt_frames, int mel_dim);

// Synthesize text to 22050 Hz mono float32 PCM.
// `lang` is a language code (e.g. "en", "zh", "ja", "ko").
// Returns malloc'd float[*out_n_samples]. Caller frees with confucius4_tts_pcm_free().
// Returns nullptr on failure.
float* confucius4_tts_synthesize(struct confucius4_tts_context* ctx, const char* text, const char* lang,
                                 int* out_n_samples);

void confucius4_tts_pcm_free(float* pcm);
void confucius4_tts_free(struct confucius4_tts_context* ctx);

int confucius4_tts_sample_rate(const struct confucius4_tts_context* ctx);

#ifdef __cplusplus
}
#endif

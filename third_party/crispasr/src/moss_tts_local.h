// moss_tts_local.h — public C API for MOSS-TTS-Local-Transformer-v1.5 (4B) runtime
//
// Text-to-speech: a Qwen3-4B backbone emits ONE hidden per frame; a 1-layer
// local/depth transformer then autoregressively generates 12 RVQ audio codebooks
// WITHIN that frame (RQ-Transformer style — replaces the 8B's delay pattern),
// decoded to 48 kHz audio by a companion codec (MOSS-Audio-Tokenizer-v2).
// Models are produced by:
//   python models/convert-moss-tts-local-to-gguf.py --input <hf> --output X.gguf
//
// Architecture: OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 (Apache-2.0)
//   Backbone : Qwen3-4B (36L, 2560d, 32Q/8KV, head_dim 128, QK-norm, SwiGLU,
//              RoPE NEOX 1e6, TIED embeddings). Input embed = text + Sum_k audio_k.
//   Local    : 1-layer GPT2-style depth transformer (LayerNorm+bias, fused QKV,
//              RoPE NEOX 1e6, SiLU MLP). Static KV of length n_vq+1 = 13 per frame.
//   Heads    : text_lm_head (tied), 12 audio_lm_heads (tied), binary
//              local_text_lm_head (continue=assistant_slot vs stop=audio_end).
//   Codec    : MOSS-Audio-Tokenizer-v2 (48 kHz, 12 cb; companion GGUF, Phase 3).
//
// See docs/moss-tts/STUDY-4B.md. Public API mirrors moss_tts.h so the CLI/c_api
// integration parallels the 8B moss-tts backend.

#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct moss_tts_local_context;

struct moss_tts_local_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    bool flash_attn;
};

struct moss_tts_local_context_params moss_tts_local_context_default_params(void);

// Load the backbone GGUF (arch "moss-tts-local"). The companion codec GGUF is
// loaded lazily on first synthesize() / via moss_tts_local_set_codec_path().
struct moss_tts_local_context* moss_tts_local_init_from_file(const char* path_model,
                                                             struct moss_tts_local_context_params params);
void moss_tts_local_free(struct moss_tts_local_context* ctx);

// Point the runtime at the companion codec GGUF (arch "moss-tts-local-codec").
bool moss_tts_local_set_codec_path(struct moss_tts_local_context* ctx, const char* path_codec);

// ---- Model dims (read from GGUF metadata) ----
int moss_tts_local_n_vq(const struct moss_tts_local_context* ctx);             // 12
int moss_tts_local_hidden_size(const struct moss_tts_local_context* ctx);      // 2560
int moss_tts_local_audio_vocab_size(const struct moss_tts_local_context* ctx); // 1024 (no +1)
int moss_tts_local_sampling_rate(const struct moss_tts_local_context* ctx);    // 48000
bool moss_tts_local_codec_loaded(const struct moss_tts_local_context* ctx);

// ---- Tokenizer (Qwen3 gpt2-style BPE, special-token aware) ----
int32_t* moss_tts_local_tokenize(struct moss_tts_local_context* ctx, const char* text, int* out_n_tokens);
const char* moss_tts_local_token_text(struct moss_tts_local_context* ctx, int token_id);

// ---- End-to-end synthesis ----
struct moss_tts_local_synth_params {
    int max_new_frames;     // 0 -> default 4096
    float text_temperature; // binary continue/stop head; <=0 -> greedy
    float text_top_p;
    int text_top_k;
    float audio_temperature; // codebook heads; <=0 -> greedy
    float audio_top_p;
    int audio_top_k;
    float audio_repetition_penalty;
    int min_audio_frames;
    int max_audio_frames;
    uint64_t seed;           // 0 -> nondeterministic
    const char* language;    // may be null
    const char* instruction; // may be null

    // Voice cloning. Reference codes as produced by moss_tts_local_codec::encode()
    // — (ref_n_vq, ref_t_audio) int32 row-major, i.e. ref_codes[q*ref_t_audio + t].
    // Null / zero disables cloning and the prompt keeps the literal "None"
    // reference, which is what the reference processor emits when no audio is
    // supplied (processing_moss_tts.py, _build_generation_or_voice_clone_codes).
    const int32_t* ref_codes;
    int ref_n_vq;
    int ref_t_audio;
};
struct moss_tts_local_synth_params moss_tts_local_synth_default_params(void);

// Synthesize speech for `text`. Returns a malloc'd F32 mono waveform @
// sampling_rate; sets *out_n_samples. Requires a codec (returns null with
// *out_n_samples=0 otherwise). Caller frees.
float* moss_tts_local_synthesize(struct moss_tts_local_context* ctx, const char* text,
                                 const struct moss_tts_local_synth_params* sp, int* out_n_samples);

// Generate only the RVQ code grid (no codec). Returns malloc'd (n_vq, T_audio)
// int32 row-major; sets *out_n_vq, *out_t_audio. For code-parity + driving the
// codec separately. Caller frees.
int32_t* moss_tts_local_generate_codes(struct moss_tts_local_context* ctx, const char* text,
                                       const struct moss_tts_local_synth_params* sp, int* out_n_vq, int* out_t_audio);

// True iff the loaded codec GGUF carries encoder tensors, i.e. voice cloning is
// available. Decode-only codec GGUFs (everything published before the encoder
// export) load fine and simply report false here.
bool moss_tts_local_can_clone(const struct moss_tts_local_context* ctx);

// Encode a MONO reference waveform into audio codes for voice cloning. Applies
// the reference preprocessing (resample to the codec rate, loudness-normalise to
// -20 dBFS with a +/-3 dB gain clamp, duplicate to the codec's channel count)
// before analysis, matching processing_moss_tts.py::encode_audios_from_wav.
// Returns malloc'd (n_vq, T_audio) int32 row-major, ready for
// moss_tts_local_synth_params::ref_codes; nullptr if the codec has no encoder.
// Caller frees.
int32_t* moss_tts_local_encode_reference(struct moss_tts_local_context* ctx, const float* pcm_mono, int n_samples,
                                         int sample_rate, int* out_n_vq, int* out_t_audio);

void moss_tts_local_set_seed(struct moss_tts_local_context* ctx, uint32_t seed);

#ifdef __cplusplus
}
#endif

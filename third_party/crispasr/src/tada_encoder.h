/**
 * tada_encoder.h — TADA encoder for voice reference creation (--make-ref).
 *
 * Converts WAV + transcript → aligned acoustic features (token_values +
 * token_positions) that can be saved as a voice reference GGUF.
 *
 * Architecture:
 *   1. Aligner: wav2vec2-large CTC → DP alignment → token positions
 *      (uses wav2vec2-ggml.h directly)
 *   2. WavEncoder: DAC-style strided conv (24kHz → 50Hz, 480× downsample)
 *   3. LocalAttentionEncoder: 6-layer transformer with RoPE + segment attention
 *   4. hidden_linear: Linear(1024 → 512)
 *   5. Post-processing: zero non-token, noise, gather, normalize
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct tada_encoder_context;

struct tada_encoder_params {
    int n_threads = 4;
    int seed = 42;
    int verbosity = 1;
};

tada_encoder_params tada_encoder_default_params();

/**
 * Load the encoder GGUF (WavEncoder + LocalAttentionEncoder + hidden_linear).
 * Returns nullptr on failure.
 */
tada_encoder_context* tada_encoder_init(const char* encoder_gguf_path, tada_encoder_params params);

/**
 * Free the encoder context.
 */
void tada_encoder_free(tada_encoder_context* ctx);

/**
 * Result of the encode pipeline.
 */
struct tada_encoder_result {
    std::vector<float> token_values;      // (n_tokens, 512) flat row-major
    std::vector<float> token_positions;   // (n_tokens,) frame index (~50 fps)
    std::vector<int32_t> token_ids;       // (n_tokens,) BPE token ids (for --align labels)
    std::vector<std::string> token_texts; // (n_tokens,) decoded UTF-8 per token
    int n_tokens = 0;
    int embed_dim = 512;
    int frame_rate = 50; // aligner CTC frames per second (positions / frame_rate = seconds)
};

/**
 * Run the full encoder pipeline:
 *   1. Run aligner (wav2vec2 GGUF) on 16kHz audio → token positions
 *   2. Run WavEncoder + LocalAttentionEncoder on 24kHz audio → features
 *   3. Post-process → token_values
 *
 * @param ctx           Encoder context (WavEncoder + attention)
 * @param aligner_gguf  Path to the aligner GGUF (wav2vec2 format)
 * @param audio_24k     24 kHz mono PCM float32
 * @param n_samples_24k Number of samples
 * @param transcript    Exact text spoken in the audio
 * @param result        Output: token_values and positions
 * @return 0 on success, non-zero on error
 */
int tada_encoder_encode(tada_encoder_context* ctx, const char* aligner_gguf, const float* audio_24k, int n_samples_24k,
                        const char* transcript, tada_encoder_result& result);

/**
 * Run only the WavEncoder + LocalAttentionEncoder stages (no aligner).
 * Requires pre-computed token_positions and token_masks.
 * Used by the diff harness for stage-by-stage comparison.
 *
 * @param ctx           Encoder context
 * @param audio_24k     24 kHz mono PCM float32
 * @param n_samples_24k Number of samples
 * @param token_masks   Binary mask (T_50hz,) — 1 at token positions
 * @param n_frames      Number of 50Hz frames (length of token_masks)
 * @param token_positions Token position indices (n_tokens,)
 * @param n_tokens      Number of tokens
 * @param result        Output
 * @return 0 on success
 */
int tada_encoder_encode_with_positions(tada_encoder_context* ctx, const float* audio_24k, int n_samples_24k,
                                       const int32_t* token_masks, int n_frames, const int32_t* token_positions,
                                       int n_tokens, tada_encoder_result& result);

/**
 * Extract a named intermediate stage for diff testing.
 * Caller must free() the returned pointer.
 *
 * Supported stages:
 *   "encoder_wav_out"    — WavEncoder output (T, 1024)
 *   "encoder_attn_out"   — after LocalAttentionEncoder (T, 1024)
 *   "encoder_hidden"     — after hidden_linear (T, 512)
 *
 * @param n_elem  Output: number of float elements
 * @return pointer to float data (caller frees), or nullptr if stage unknown
 */
float* tada_encoder_extract_stage(tada_encoder_context* ctx, const float* audio_24k, int n_samples_24k,
                                  const int32_t* token_masks, int n_frames, const char* stage, int* n_elem);

/**
 * DP text-audio alignment algorithm.
 * Given softmax probabilities probs[T × V] and token IDs tokens[N],
 * finds the optimal monotonic alignment of tokens to frames.
 *
 * @param probs   Softmax probabilities, (T, V) row-major
 * @param T       Number of frames
 * @param V       Vocabulary size
 * @param tokens  Token IDs to align
 * @param N       Number of tokens
 * @param positions Output: frame index for each token (N,)
 * @return 0 on success
 */
int tada_dp_align(const float* probs, int T, int V, const int32_t* tokens, int N, int32_t* positions);

/**
 * Write an encoder result as a voice reference GGUF file.
 * Compatible with tada_load_prompt() in tada_tts.cpp.
 *
 * @param path       Output .gguf file path
 * @param result     Encoder result with token_values and token_positions
 * @param transcript Text spoken in the reference audio
 * @param language   Language code (nullable, omit for English)
 * @param cloned_from_recording  Stamp `crispasr.voice.cloned_from_recording`
 *                   into the pack. Set when this reference was derived from a
 *                   real person's recording, so the synthesis-time consent and
 *                   Art. 50(4) disclosure gates can recognise the resulting
 *                   .gguf as a clone — they classify by provenance, not by file
 *                   suffix. See examples/cli/crispasr_voice_clone_policy.h.
 * @param consent_attestation  Attestation recorded at bake time (nullable).
 *                   Audit trail only; it does NOT satisfy the synthesis-time
 *                   gate.
 * @return 0 on success
 */
int tada_encoder_write_ref_gguf(const char* path, const tada_encoder_result& result, const char* transcript,
                                const char* language, bool cloned_from_recording = false,
                                const char* consent_attestation = nullptr);

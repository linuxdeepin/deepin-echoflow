// src/beatrice_phone.h — Beatrice v2 PhoneExtractor (Project Beatrice, MIT).
//
// Second of Beatrice's three networks. Takes 16 kHz mono PCM and produces
// 128-dim phone units at 100 Hz — the content representation the
// ConverterNetwork consumes. Deterministic, like the pitch estimator; the RNG
// lives in the vocoder.
//
// Unlike RVC, Beatrice derives its content features itself, so this replaces
// the ContentVec/HuBERT encoder a caller would otherwise have to supply.
//
// NOTE ON FIDELITY. Upstream's PhoneExtractor.merge_weights() folds
// feature_projection's LayerNorm affine into backbone.embed, which is an edge
// approximation (the zero-padded taps do not see the LayerNorm bias) and, with
// this backbone's self-attention, spreads across the whole sequence at
// rel 1.7e-02. Our converter skips that fold and keeps the norm explicit, so
// this path is EXACT where upstream's own inference is not. Numbers from a
// `.beatrice` dump will therefore differ slightly, and ours is the closer one.
//
// See docs/music-transcription/BEATRICE_BLUEPRINT.md.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct beatrice_phone_context;

struct beatrice_phone_params {
    int n_threads; // 0 = auto
    bool use_gpu;
    int gpu_device;
};

beatrice_phone_params beatrice_phone_default_params(void);

beatrice_phone_context* beatrice_phone_init_from_file(const char* model_path, beatrice_phone_params params);
void beatrice_phone_free(beatrice_phone_context* ctx);

int beatrice_phone_channels(const beatrice_phone_context* ctx);    // 128
int beatrice_phone_sample_rate(const beatrice_phone_context* ctx); // 16000
int beatrice_phone_hop_length(const beatrice_phone_context* ctx);  // 160 -> 100 Hz

struct beatrice_phone_result {
    // BIN-MAJOR, TIME FASTEST: element (channel, frame) at channel*n_frames +
    // frame. ggml-native and matches the parity reference, so nothing has to
    // transpose. It is NOT frame-major.
    float* units; // n_channels * n_frames
    int n_frames;
    int n_channels;
};

// `pcm` is mono 16 kHz. n_samples is TRUNCATED to a multiple of 160 (the
// reference warns rather than handling a partial frame), giving
// n_frames = n_samples / 160.
beatrice_phone_result* beatrice_phone_extract(beatrice_phone_context* ctx, const float* pcm, int n_samples);
void beatrice_phone_result_free(beatrice_phone_result* r);

// Per-stage parity diff against tools/beatrice_torch_parity.py --component
// phone_extractor. 0 = every stage passed, 1 = parity failure, 2 = load error.
int beatrice_phone_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

#ifdef __cplusplus
}
#endif

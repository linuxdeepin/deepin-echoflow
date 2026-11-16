// src/rvc_svc.h — RVC voice conversion (SynthesizerTrnMs*NSFsid).
//
// The FULL inference path, not just the vocoder: enc_p (relative-attention
// transformer) + flow (normalizing flow, reversed) + dec (NSF-HiFi-GAN).
// CometBeat confirmed this split — they send ContentVec features + F0 + speaker
// id and we own everything downstream. See docs/music-transcription/
// SVC_RECORD_SHAPES.md for the wire contract and RVC_BLUEPRINT.md for the
// ~15 non-obvious details this reproduces.
//
// INFERENCE IS STOCHASTIC. Two live RNG sites (the z_p latent sample and
// SineGen's additive noise) mean output varies run to run by design. Both are
// INJECTABLE here so the port can be validated at all: waveform correlation
// against a reference run is not a valid test when the reference disagrees with
// itself. tools/rvc_torch_parity.py produces matching buffers.
//
// LICENCE: RVC's code is MIT; CHECKPOINTS vary and some forks are
// non-commercial. The GGUF carries its own tag and the registry gate matches
// on it.

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct rvc_svc_context;

struct rvc_svc_params {
    int n_threads; // 0 = auto
    bool use_gpu;
    int gpu_device;
};

rvc_svc_params rvc_svc_default_params(void);

rvc_svc_context* rvc_svc_init_from_file(const char* model_path, rvc_svc_params params);
void rvc_svc_free(rvc_svc_context* ctx);

// The checkpoint's expected ContentVec dimensionality: 256 (v1, layer 9 +
// final_proj) or 768 (v2, final layer). CometBeat asked for this specifically
// so a v1/v2 mismatch refuses loudly instead of sounding subtly wrong — they
// cannot make that check from the Dart side.
int rvc_svc_content_dim(const rvc_svc_context* ctx);
int rvc_svc_n_speakers(const rvc_svc_context* ctx);
// Output rate is a property of the checkpoint (32k/40k/48k), not a constant.
int rvc_svc_sample_rate(const rvc_svc_context* ctx);

struct rvc_svc_result {
    float* pcm; // mono, rvc_svc_sample_rate()
    int n_samples;
};

// Convert. `content` is n_frames * content_dim, frame-major; `f0_hz` is
// n_frames values in Hz with 0.0 marking unvoiced (we derive the coarse
// mel-quantised pitch ourselves — those constants are model-side and
// replicating them in the caller guarantees drift).
//
// `noise_zp` / `noise_sine` are OPTIONAL. NULL means draw randomly, which is
// what production should do. Supplying them replays a specific draw and makes
// the call deterministic — required for the cross-implementation harness, and
// the only way this port can be validated at all.
//   noise_zp   : inter_channels * n_frames, or NULL
//   noise_sine : n_frames * upsample_product, or NULL
rvc_svc_result* rvc_svc_convert(rvc_svc_context* ctx, const float* content, int n_frames, const float* f0_hz,
                                int speaker_id, const float* noise_zp, const float* noise_sine);
void rvc_svc_result_free(rvc_svc_result* r);

// f0 (Hz) -> coarse 1..255, the exact mel quantisation from pipeline.py:73-137.
// Exposed because it is the one piece a caller might reasonably want to
// cross-check; the conversion path applies it internally.
void rvc_svc_coarse_pitch(const float* f0_hz, int n, int* out_coarse);

// STATUS: the FULL path passes, INCLUDING rvc_svc_convert() end to end —
// 48 comparisons at cos 1.00000000, of which convert_e2e runs this very API
// with the reference'''s noise and reproduces its audio (max_abs 1.4e-05).
// Run it with `crispasr-diff rvc <model.gguf> <ref.gguf> <any.wav>`.
//
// F32 ONLY for now. An f16 GGUF converts fine but aborts at runtime: several
// ops on this path require F32 operands (ggml_scale is F32-only, and an F16
// embedding/bias reaching ggml_add trips
// GGML_ASSERT(src1->type == GGML_TYPE_F32)). Casting the relative-position
// tables fixed the first of those; the rest needs per-op work. Convert with
// --dtype f32 until then — the f16 path is NOT silently wrong, it refuses.
//
// Per-stage parity diff against tools/rvc_torch_parity.py's reference dump.
// The reference carries BOTH noise buffers, which the runtime replays, so the
// comparison is deterministic despite the model being stochastic.
// Returns 0 when every stage passes, 1 on a parity failure, 2 on a load error.
int rvc_svc_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

#ifdef __cplusplus
}
#endif

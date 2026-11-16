// moss_tts_codec.h — MOSS-Audio-Tokenizer transformer RVQ codec (decode path).
//
// Internal C++ interface used by moss_tts.cpp. Loads the companion codec GGUF
// (arch "moss-tts-codec", tensors "moss.codec.*") and decodes a (n_vq, T_audio)
// RVQ code grid into a 24 kHz mono waveform (length T_audio * 1920).
//
// Decode pipeline (ported from pwilkin/openmoss src/codec.cpp, verified against
// MOSS-Audio-Tokenizer config.json):
//   codes -> per-quantizer codebook lookup -> weight-normed oproj conv1x1
//         -> Sum over 32 -> global oproj -> 4 ProjectedTransformer stages
//         (each: optional iproj -> N pre-LN layers with adjacent-pair RoPE
//          (base 10000), sliding-window causal attention (ctx 125/250/500/1000),
//          GELU FFN, per-channel LayerScale -> optional oproj) + patch upsample
//         -> waveform.
// The sliding-window mask is part of the model (not an optimisation).

#pragma once

#include <cstdint>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_backend_sched;
typedef struct ggml_backend_sched* ggml_backend_sched_t;

namespace moss_tts_codec {

struct Codec; // opaque

// Load the codec from a GGUF at `path`, binding tensors on `backend`. `sched`
// is the caller's scheduler, reused for decode graphs. Returns nullptr on error.
Codec* load(const char* path, ggml_backend_t backend, ggml_backend_sched_t sched, int verbosity);

void free(Codec* c);

// Decode (n_vq, t_audio) row-major int32 codes -> mono f32 waveform
// (length t_audio * 1920). Returns empty on error / t_audio <= 0.
std::vector<float> decode(Codec* c, const int32_t* codes, int n_vq, int t_audio);

// True iff the codec GGUF carried the encoder tensors (voice cloning available).
bool encoder_ready(const Codec* c);

// Encode a 24 kHz mono waveform (length n_samples, padded to a multiple of 1920)
// -> (n_vq, t_audio) row-major int32 RVQ codes; sets *n_vq_out, *t_audio_out.
// Returns empty if the encoder isn't available. (Voice cloning.)
std::vector<int32_t> encode(Codec* c, const float* waveform, int64_t n_samples, int& n_vq_out, int& t_audio_out);

} // namespace moss_tts_codec

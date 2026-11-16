// moss_tts_local_codec.h — MOSS-Audio-Tokenizer-v2 codec (decode path).
//
// Companion codec for the 4B moss-tts-local backend. Loads a decode-only codec
// GGUF (arch "moss-tts-local-codec", tensors "codec.*") and decodes a (n_vq, T)
// RVQ code grid (n_vq = 12, 12.5 Hz) into a 48 kHz STEREO waveform.
//
// Decode pipeline (ported from modeling_moss_audio_tokenizer.py, read
// line-by-line — see docs/moss-tts/STUDY-4B.md "P3 codec" + PLAN NOW):
//   codes (12,T)
//     -> per-quantizer: codebook[1024,8][code] -> WNConv1d 8->512 (+bias)
//     -> Sum over 12 -> output_proj WNConv1d 512->768 (+bias)        [(768,T)]
//     -> 6 ProjectedTransformer stages (each: iproj -> N pre-LN layers with
//        adjacent-pair RoPE (base 1e4, head_dim 64), sliding-window causal
//        attention, GELU-erf FFN, per-channel LayerScale 0.01 -> oproj),
//        interleaved with patch upsamplers (pure reshape: x2 x5 then x240)
//     -> (1, 7680*T) mono-INTERLEAVED signal  [L0,R0,L1,R1,...]
//   The runtime returns this interleaved buffer; the caller de-interleaves to
//   stereo / downmixes to mono. The sliding-window mask is part of the model.

#pragma once

#include <cstdint>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_backend_sched;
typedef struct ggml_backend_sched* ggml_backend_sched_t;

namespace moss_tts_local_codec {

struct Codec; // opaque

// Load the codec from a GGUF at `path`, binding tensors on `backend`. `sched` is
// the caller's scheduler, reused for decode graphs. Returns nullptr on error.
Codec* load(const char* path, ggml_backend_t backend, ggml_backend_sched_t sched, int verbosity);

void free(Codec* c);

// Samples of the internal (channel-interleaved) signal per code frame:
//   downsample_rate * num_channels = 3840 * 2 = 7680.
int interleaved_hop(const Codec* c);
int num_channels(const Codec* c);  // 2
int sampling_rate(const Codec* c); // 48000

// True iff the codec GGUF carried encoder tensors, i.e. voice cloning is
// available. Decode-only GGUFs (everything published before the encoder export)
// load fine and simply report false here.
bool encoder_ready(const Codec* c);

// Encode a channel-INTERLEAVED f32 signal (same layout decode() emits) into
// (n_vq, t_audio) row-major int32 codes — the analysis inverse of decode().
// Input is zero-padded up to a whole number of code frames. Returns empty if the
// encoder isn't available or the input is unusable. (Voice cloning.)
std::vector<int32_t> encode(Codec* c, const float* interleaved, int64_t n_samples, int& n_vq_out, int& t_audio_out);

// Decode (n_vq, t_audio) row-major int32 codes -> mono channel-INTERLEAVED f32
// signal of length t_audio * interleaved_hop() (= [L0,R0,L1,R1,...] for stereo).
// Returns empty on error / t_audio <= 0.
std::vector<float> decode(Codec* c, const int32_t* codes, int n_vq, int t_audio);

} // namespace moss_tts_local_codec

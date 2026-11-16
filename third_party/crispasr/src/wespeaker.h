// wespeaker.h — public C API for the WeSpeaker ResNet34-LM ggml runtime.
//
// Speaker-embedding model: 16 kHz mono PCM -> a 256-dim embedding. Used by
// the diarization clustering stack as an alternative to TitaNet.
//
// ⚠ The upstream weights (Wespeaker/wespeaker-voxceleb-resnet34-LM) are
// CC-BY-4.0. Any redistributed GGUF must carry the licence tag + attribution;
// see THIRD_PARTY_NOTICES.txt.
//
// Models come from GGUF files produced by:
//   python models/convert-wespeaker-to-gguf.py \
//       --model Wespeaker/wespeaker-voxceleb-resnet34-LM --output X.gguf

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct wespeaker_context;

struct wespeaker_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;  // false => force CPU backend
};

struct wespeaker_context_params wespeaker_context_default_params(void);

struct wespeaker_context* wespeaker_init_from_file(const char* path_model, struct wespeaker_context_params params);

void wespeaker_free(struct wespeaker_context* ctx);

// ---- Model introspection ----

// A second context sharing `src`'s weights, so several windows can be embedded
// concurrently. Each concurrent caller needs its own; free every one with
// wespeaker_free(), and free the workers BEFORE the context they borrow from.
struct wespeaker_context* wespeaker_init_worker(struct wespeaker_context* src);

int wespeaker_embed_dim(struct wespeaker_context* ctx);   // 256
int wespeaker_sample_rate(struct wespeaker_context* ctx); // 16000
int wespeaker_n_mels(struct wespeaker_context* ctx);      // 80

// Minimum input length in samples. The stem downsamples time by 8, so a clip
// shorter than this produces an empty final feature map and cannot be
// embedded. Callers should skip such windows (FoxNose's MIN_SEGMENT_DURATION
// of 0.4 s is comfortably above it).
int wespeaker_min_samples(struct wespeaker_context* ctx);

// ---- Embedding ----

// Compute the speaker embedding for raw 16 kHz mono PCM in [-1, 1].
// `out_embedding` must have room for wespeaker_embed_dim() floats.
//
// The embedding is returned RAW, exactly as the reference model emits it —
// no L2 normalisation. Callers that need unit vectors (cosine affinity,
// spherical centroids) normalise at their own layer, which is what the
// upstream clustering does.
//
// Returns 0 on success, non-zero on failure (including audio too short).
int wespeaker_embed(struct wespeaker_context* ctx, const float* samples, int n_samples, float* out_embedding);

// Embed `n_win` windows of ONE contiguous span with a single pass of the
// convolutional trunk, instead of one pass per window. win_start/win_end are
// sample offsets into `samples`; out_embeddings receives n_win * embed_dim
// floats. Returns 0 on success.
//
// The sliding window overlaps 50%, so per-window embedding pushes every sample
// through the trunk twice; this does it once. NOT bit-identical to looping
// wespeaker_embed(): CMN is computed over the span, and interior windows see
// real neighbouring audio through the convs instead of zero padding. Validate
// changes here on DER.
int wespeaker_embed_windows(struct wespeaker_context* ctx, const float* samples, int n_samples, const int* win_start,
                            const int* win_end, int n_win, float* out_embeddings);

// Embed `n_win` independent windows of `samples`, batching same-length groups
// through the trunk along the ggml batch dimension (ne[3]). ARITHMETIC-
// IDENTICAL to looping wespeaker_embed(): every window keeps its own fbank and
// its own per-window CMN, and windows never see each other's audio — only the
// forward passes are fused, which turns the seg1 GEMV into a GEMM and lets one
// graph amortise the per-dispatch cost that dominates 1.2 s windows. (Contrast
// wespeaker_embed_windows above, which shares fbank/CMN/conv context across a
// span and therefore moves DER.)
//
// offsets/lengths are sample ranges into `samples`; out_embeddings receives
// n_win * embed_dim floats in caller order. Windows of unequal length are
// grouped by exact fbank frame count — never padded, padding would change CMN
// and conv content — and singleton groups fall back to the per-window graph.
// Batch size per graph is capped by CRISPASR_WESPEAKER_BATCH (default 16,
// max 32). Returns 0 only if EVERY window succeeded; on non-zero the caller
// should fall back to per-window wespeaker_embed() to isolate the failure.
int wespeaker_embed_batch(struct wespeaker_context* ctx, const float* samples, int64_t n_samples,
                          const int64_t* offsets, const int* lengths, int n_win, float* out_embeddings);

// ---- Stage-level entry points (for crispasr-diff) ----

// Kaldi fbank (80 mel, 25/10 ms, hamming) AFTER per-utterance CMN — i.e. the
// exact tensor the network consumes. Returns a malloc'd (T_frames, n_mels)
// row-major buffer the caller must free(), or nullptr.
float* wespeaker_compute_fbank(struct wespeaker_context* ctx, const float* samples, int n_samples, int* out_T,
                               int* out_n_mels);

// Per-stage capture callback. `data` is a row-major buffer of ne2*ne1*ne0
// floats with ne0 the fastest axis, matching the ggml layout:
//   feature maps -> (ne2=channels, ne1=freq, ne0=time) i.e. time fastest
//   stats/embedding -> 1-D, ne1 = ne2 = 1
// Stage names match tools/reference_backends/wespeaker.py: "stem_out",
// "layer1_out".."layer4_out", "stats", "embedding".
typedef void (*wespeaker_stage_cb)(const char* name, const float* data, int ne0, int ne1, int ne2, void* userdata);

// Like wespeaker_embed, but invokes `cb` for every intermediate stage.
// `out_embedding` may be nullptr if only the callback output is wanted.
int wespeaker_embed_staged(struct wespeaker_context* ctx, const float* samples, int n_samples, wespeaker_stage_cb cb,
                           void* userdata, float* out_embedding);

#ifdef __cplusplus
}
#endif

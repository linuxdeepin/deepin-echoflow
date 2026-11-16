// src/core/foxnose_pipeline.h — the FoxNose diarization pipeline (#324).
//
// Ties the pieces together: speech segments -> sliding embedding windows ->
// speaker clustering -> temporal smoothing -> merged speaker turns.
//
// The embedder is INJECTED rather than linked in. That keeps this module free
// of any model dependency, lets the wiring layer choose WeSpeaker or TitaNet,
// and — the reason it matters most — makes the whole pipeline unit-testable
// with a synthetic embedder whose speaker identities are known by
// construction. Weight-free orchestration is exactly the code crispasr-diff
// cannot see (HARD RULE #3b).
//
// Clean-room; see core/spectral_diarize.h for the provenance argument.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace core_foxnose {

// ── Tuned constants (upstream recipe) ─────────────────────────────────────

// Windows shorter than this are not embedded at all — too little speech for a
// stable speaker vector. Such VAD segments inherit the nearest speaker.
constexpr float kMinSegmentSeconds = 0.4f;
// Sliding window over long segments, and its hop.
constexpr float kEmbeddingWindowSeconds = 1.2f;
constexpr float kEmbeddingStepSeconds = 0.6f;
// A segment up to this multiple of the window length is embedded whole
// instead of being slid over.
constexpr float kSingleWindowFactor = 1.5f;
// Adjacent turns of the same speaker closer than this are merged.
constexpr float kMergeGapSeconds = 0.7f;

// ── Types ─────────────────────────────────────────────────────────────────

// A speech region from VAD, in seconds.
struct Speech {
    double start = 0.0;
    double end = 0.0;
};

// One diarized turn.
struct Turn {
    double start = 0.0;
    double end = 0.0;
    int speaker = 0; // dense index; render as SPEAKER_%02d
};

struct Params {
    int min_speakers = 1;
    int max_speakers = 20;
    int num_speakers = 0; // > 0 pins the count and skips estimation
    unsigned seed = 42;
    // How many windows may be embedded concurrently. Each concurrent slot must
    // have its OWN embedder state; see EmbedFn's `worker` argument. 1 keeps the
    // old serial behaviour.
    int n_workers = 1;
};

// Embed `n` samples of 16 kHz mono PCM into `out` (embed_dim floats).
// Return 0 on success; any non-zero result makes the window be skipped, which
// is the correct behaviour for audio the model refuses (too short, silent).
//
// `worker` is in [0, Params::n_workers) and says WHICH embedder state to use.
// diarize() may call this from several threads at once with distinct `worker`
// values, and never twice concurrently with the same one — so an
// implementation only needs one context per worker, not a lock.
using EmbedFn = int (*)(void* userdata, int worker, const float* pcm, int n_samples, float* out);

// Optional fast path: embed SEVERAL windows of one contiguous span at once,
// sharing whatever work they have in common (for a convolutional trunk that is
// most of it, since the windows overlap 50%). `ws`/`we` are sample offsets
// relative to `pcm`. Return 0 on success; non-zero marks the whole span failed
// and those windows are skipped. Null => diarize() embeds window by window.
using EmbedWindowsFn = int (*)(void* userdata, int worker, const float* pcm, int n_samples, const int* ws,
                               const int* we, int n_win, float* out);

// Batched embedding (#324 perf): embed SEVERAL independent windows in one
// call, each an (offset, length) slice of `pcm`, writing n_win * embed_dim
// floats in window order. UNLIKE EmbedWindowsFn the windows share nothing —
// no span CMN, no conv context — so this path is arithmetic-identical to
// per-window EmbedFn and needs no DER gate; the implementation merely fuses
// the forward passes (see wespeaker_embed_batch). Return 0 only when every
// window succeeded; on non-zero diarize() retries the chunk window by window
// through EmbedFn, so a single bad window degrades to the old path instead of
// dropping its neighbours.
using EmbedBatchFn = int (*)(void* userdata, int worker, const float* pcm, int64_t n_samples, const int64_t* offsets,
                             const int* lengths, int n_win, float* out);

// Windows handed to one EmbedBatchFn call. Purely a scheduling unit — chunk
// boundaries are fixed by window index so results never depend on the worker
// count — and the embedder may sub-batch however it likes.
constexpr int kWindowsPerBatch = 32;

// Windows per trunk pass when EmbedWindowsFn is used. Fixed, so the result does
// not depend on the worker count: cepstral mean normalisation is computed over
// the span, so span size is part of the answer, not just of the schedule.
constexpr int kWindowsPerSpan = 32;

// Windows per span, overridable so the accuracy/speed trade can be swept
// without a rebuild. Work per span is (N+1)/2N of the per-window cost — N=32
// is 1.94x, N=8 1.78x, N=4 1.60x — while a smaller N keeps CMN closer to the
// per-window normalisation that the clusterer was tuned on.
int windows_per_span();

struct Result {
    std::vector<Turn> turns;
    int n_speakers = 0;
    std::string reason; // how the speaker count was decided
    int n_windows = 0;  // embedding windows actually produced
    int n_skipped = 0;  // windows the embedder rejected
};

// ── Pipeline ──────────────────────────────────────────────────────────────

// Split one speech region into embedding windows, following the upstream
// rule: skip below kMinSegmentSeconds; embed whole when the region is no
// longer than kSingleWindowFactor * kEmbeddingWindowSeconds; otherwise slide.
std::vector<Speech> window_speech(const Speech& seg);

// Convert overlapping windows into the non-overlapping timeline the labels
// are attached to: boundaries sit at the midpoints between adjacent window
// centres, clamped into the parent region.
std::vector<Speech> window_boundaries(const Speech& seg, const std::vector<Speech>& windows);

// Run the whole pipeline over pre-computed speech regions.
//
// Embedding strategy, in priority order: `embed_windows` (span sharing, the
// DER-trading opt-in) > `embed_batch` (fused independent windows, DER-neutral)
// > plain per-window `embed`. `embed` is always required — it is the fallback
// when a batch call fails.
Result diarize(const float* pcm, int n_samples, int sample_rate, const std::vector<Speech>& speech, EmbedFn embed,
               void* userdata, int embed_dim, const Params& params, EmbedWindowsFn embed_windows = nullptr,
               EmbedBatchFn embed_batch = nullptr);

} // namespace core_foxnose

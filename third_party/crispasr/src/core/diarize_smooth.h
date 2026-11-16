// src/core/diarize_smooth.h — temporal smoothing of speaker labels (#324).
//
// Clustering assigns a speaker to each embedding window independently, so its
// output flickers: a 0.6 s window straddling a pause, or one noisy embedding,
// flips the label mid-utterance. These routines impose temporal continuity on
// that label sequence.
//
// Clean-room, same basis as core/spectral_diarize.h — Viterbi decoding and
// majority filtering are textbook; what comes from the upstream recipe is the
// order of operations and the tuned constants below.
//
// Weight-free, so crispasr-diff cannot see any of it (HARD RULE #3b): the
// guards are the hermetic unit tests in tests/test-diarize-smooth.cpp.

#pragma once

#include <vector>

namespace core_diarize_smooth {

// Cost of changing speaker between adjacent windows in the Viterbi pass.
// Scores are cosine similarities in [-1, 1], so 0.18 is the similarity margin
// a switch has to beat.
constexpr float kSwitchPenalty = 0.18f;

// A run shorter than this is treated as flicker rather than a real turn, both
// when collapsing A-B-A islands and when deciding which original runs are
// sustained enough to survive Viterbi.
constexpr float kMaxFragmentSeconds = 1.2f;

// Small bonus keeping a confident clustering label in place, so the transition
// penalty only removes low-value flicker rather than rewriting the segment.
constexpr float kOriginalLabelAnchor = 0.02f;

// A window's time span, in seconds.
struct Window {
    float start = 0.0f;
    float end = 0.0f;
};

// The unique majority of `labels`, or -1 when there is a tie (or no input).
// Ties must NOT pick a winner — that is what keeps the 3-window filter from
// inventing a label at a genuine speaker boundary.
int majority_label(const std::vector<int>& labels);

// 3-window majority filter (indices i-1 .. i+1). A tie leaves the original
// label untouched. Sequences shorter than 3 pass through unchanged.
std::vector<int> smooth_window_labels(const std::vector<int>& labels);

// L2-normalised centroid per distinct label. `out_label_values` receives the
// label each centroid row corresponds to, in ascending order; labels whose
// centroid degenerates to zero are dropped. Centroids are the mean of the
// NORMALISED member embeddings, then re-normalised.
std::vector<float> speaker_centroids(const float* embeddings, int n, int d, const std::vector<int>& labels,
                                     std::vector<int>* out_label_values);

// Viterbi over `scores` (n_frames, n_labels, row-major): maximise the total
// score with `switch_penalty` subtracted for every label change. Returns the
// best path as label INDICES into the score columns.
std::vector<int> viterbi_smooth(const float* scores, int n_frames, int n_labels, float switch_penalty = kSwitchPenalty);

// Smooth one contiguous speech segment: score each window's embedding against
// every speaker centroid, anchor the original label slightly, and Viterbi-
// decode. `embeddings` holds the segment's windows only (n rows of d).
// Sequences shorter than 3, or with fewer than 2 candidate labels, pass
// through unchanged.
std::vector<int> smooth_segment_temporal(const std::vector<int>& labels, const float* embeddings, int n, int d,
                                         const std::vector<int>& label_values, const float* centroids);

// Collapse short A-B-A islands: an interior run whose neighbours share a label
// and which lasts no longer than `max_seconds` is absorbed into them.
std::vector<int> collapse_short_islands(const std::vector<int>& labels, const std::vector<Window>& windows,
                                        float max_seconds = kMaxFragmentSeconds);

// Restore runs from `original` that lasted longer than `min_seconds`. Viterbi
// is biased toward fewer switches and will happily erase a genuine short turn;
// this puts sustained evidence back.
std::vector<int> restore_sustained_runs(const std::vector<int>& original, const std::vector<int>& smoothed,
                                        const std::vector<Window>& windows, float min_seconds = kMaxFragmentSeconds);

} // namespace core_diarize_smooth

// crispasr_diarize_internal.h — pyannote-seg per-segment scoring,
// extracted from apply_pyannote() so unit tests can drive it with
// synthetic posteriors (no model load, no audio).
//
// NOT part of the public API. Stable between releases is not promised.

#pragma once

#include "crispasr_diarize.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace crispasr_diarize_internal {

// Score the dominant speaker over a half-open frame range derived from
// the buffer-relative cs range [start_cs, end_cs). Returns 0/1/2 for
// the highest-activity speaker, or -1 when the range is empty / pure
// silence / all-classes-below-floor.
//
// `log_probs` is the same shape and semantics as for
// assign_speakers_from_log_posteriors (row-major [T,7] log-softmax).
// `start_cs`/`end_cs` are RELATIVE to log_probs frame 0 (the caller
// must subtract any slice / cache offset). Used by both
// assign_speakers_from_log_posteriors (per ASR segment) and the
// segment-splitting helper (per ASR word).
int score_speaker_for_range(const float* log_probs, int T, double frame_dur_s, int64_t start_cs, int64_t end_cs);

// Assign speaker indices (0/1/2) to each ASR segment given per-frame
// pyannote-seg posteriors.
//
// log_probs: float array of shape [T, 7], row-major. Frame f's 7
//            class log-softmax outputs occupy log_probs[f*7 .. f*7+6].
//            Class layout matches pyannote-seg-3.0:
//              0 = silence,
//              1 = spk0,        2 = spk1,        3 = spk0 + spk1,
//              4 = spk2,        5 = spk0 + spk2, 6 = spk1 + spk2.
//            MUST be log-probabilities — they are exp()'d internally
//            for the per-speaker activity sum.
// T:        number of frames.
// frame_dur_s: seconds per pyannote frame (270 / 16000 = 16.875 ms
//              for pyannote-seg-3.0; sinc stride 10 × 3 maxpools of
//              stride 3).
// slice_t0_cs: absolute centisecond offset at which the buffer starts.
// segs:     in/out. For each segment, segs[i].speaker is set to the
//           speaker with the highest activity probability summed over
//           the segment's frame range, or left at -1 if no frame in
//           range showed any non-silence activity.
//
// Scoring details (intentional, see issue #107):
//   * Per-frame, per-speaker activity is P(spk active in frame) =
//     sum of exp(log_probs[c]) over the classes c that include that
//     speaker. So overlap class 3 (spk0+spk1) contributes to BOTH
//     spk0 and spk1, fixing the previous LUT collapse where it only
//     credited spk0.
//   * Per-segment, we sum those per-frame activity probabilities and
//     argmax over the 3 speakers. This is posterior-weighted (a
//     confident spk0 frame counts more than an uncertain one), unlike
//     the previous one-hot argmax-then-count.
//   * Silence (class 0) and ambiguity are handled naturally — a
//     mostly-silent segment yields tiny activity sums for all
//     speakers, but the argmax still picks the speaker most likely
//     to have been active. Callers wanting "no speaker" semantics on
//     silence should pre-filter with VAD.
void assign_speakers_from_log_posteriors(const float* log_probs, int T, double frame_dur_s, int64_t slice_t0_cs,
                                         std::vector<CrispasrDiarizeSegment>& segs);

// A maximal run of consecutive words assigned to one speaker. Half-open
// word-index range [start, end); `speaker` is the run's speaker id (>=0),
// or -1 when unknown.
struct SpeakerRun {
    std::size_t start; // first word index (inclusive)
    std::size_t end;   // one-past-last word index (exclusive)
    int speaker;       // speaker id, or -1
};

// Group per-word speaker labels into maximal same-speaker runs, then fold
// any run whose word span is shorter than `min_run_cs` into its longer
// neighbour. The fold suppresses single-word track-index flips that
// pyannote-seg (no speaker embeddings) produces mid-phrase — e.g. a lone
// "(speaker 2)" word inside a contiguous "(speaker 1)" run — which would
// otherwise fragment a segment into spurious sub-segments (#107 / #267).
//
// word_spk[i] is word i's speaker (>=0, or -1 for "unknown"). word_t0_cs[i]
// / word_t1_cs[i] are word i's absolute start/end centiseconds; the caller
// substitutes segment bounds for non-positive values so a run's span is
// well-defined. Pass `min_run_cs <= 0` to disable folding (split on every
// speaker change). All three vectors must be the same length.
//
// Pure and model-free so unit tests can drive it directly. Returns an empty
// vector when `word_spk` is empty.
std::vector<SpeakerRun> group_words_into_speaker_runs(const std::vector<int>& word_spk,
                                                      const std::vector<int64_t>& word_t0_cs,
                                                      const std::vector<int64_t>& word_t1_cs, int64_t min_run_cs);

} // namespace crispasr_diarize_internal

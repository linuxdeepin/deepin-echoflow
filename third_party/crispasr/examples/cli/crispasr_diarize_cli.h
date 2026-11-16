// crispasr_diarize_cli.h — CLI-side diarization shim.
//
// The four in-process diarization methods (energy, xcorr, vad-turns,
// pyannote-native) live in `src/crispasr_diarize.h` so every CrispASR
// consumer reaches them through the shared library. This CLI-local
// header keeps the subprocess-based sherpa-onnx method plus the
// `--diarize-method` → method-enum translation that CLI callers rely
// on.
//
// CLI callers: use `crispasr_apply_diarize(..., whisper_params &)`.
// Library callers / wrappers: use
// `crispasr_diarize_segments(..., CrispasrDiarizeOptions &)` from
// `src/crispasr_diarize.h` directly.

#pragma once

#include "crispasr_diarize.h" // from src/ via whisper target's PUBLIC include dir
#include "crispasr_backend.h"
#include "crispasr_vad.h" // crispasr_audio_slice

#include <functional>
#include <vector>

struct whisper_params;         // fwd decl
class CrispasrSpeakerEmbedder; // fwd decl (crispasr_speaker_embedder.h)

/// Cached pyannote-seg posteriors over a full audio buffer. Built once
/// at the start of a run (issue #107 — avoids per-slice pyannote runs
/// that reset local track indices across slices and produce
/// inconsistent speaker labels).
///
/// `log_probs` is row-major [T, 7] log-softmax over the seven
/// pyannote-seg classes (silence, spk0, spk1, spk0+1, spk2, spk0+2,
/// spk1+2). `frame_dur_s` is the audio duration of one frame
/// (270 / 16000 = 16.875 ms for pyannote-seg-3.0). The cache implicitly
/// covers absolute time [0, T*frame_dur_s); pass `slice_t0_cs` when
/// applying it to translate segment cs into frame indices.
struct CrispasrPyannoteCache {
    std::vector<float> log_probs;
    int T = 0;
    double frame_dur_s = 0.0;
    bool valid() const { return T > 0 && (int)log_probs.size() == T * 7 && frame_dur_s > 0.0; }
};

/// Compute pyannote-seg posteriors over `full_audio` (mono, 16 kHz) so
/// the same buffer can be reused for every per-slice diarize call.
/// `params.sherpa_segment_model` selects (or auto-downloads) the GGUF.
/// Returns true on success; `out` is populated with shape [T, 7].
bool crispasr_compute_pyannote_cache(const float* full_audio, int n_samples, const whisper_params& params,
                                     CrispasrPyannoteCache& out);

/// Cached global sherpa-onnx speaker-diarization timeline. Built once
/// over the full audio at the start of a run (issue #110 — avoids
/// per-slice sherpa invocations that reset local speaker IDs across
/// slices and produce inconsistent labels).
///
/// `segments` holds the globally-parsed speaker regions with absolute
/// timestamps. When this cache is passed to `crispasr_apply_diarize`,
/// the sherpa subprocess is NOT re-invoked; instead the per-slice
/// ASR segments are scored against the pre-computed global timeline.
struct CrispasrSherpaCache {
    struct Segment {
        double t0_s;
        double t1_s;
        int speaker;
    };
    std::vector<Segment> segments;
    bool valid() const { return !segments.empty(); }
};

/// Run sherpa-onnx-offline-speaker-diarization once over the full mono
/// 16 kHz audio and parse the global speaker-turn timeline.
/// Returns true on success; `out` is populated with speaker regions.
bool crispasr_compute_sherpa_cache(const float* full_audio, int n_samples, const whisper_params& params,
                                   CrispasrSherpaCache& out);

/// Top-level CLI diarize post-step.
///
/// Routes `params.diarize_method` to either the shared library methods
/// (energy / xcorr / vad-turns / pyannote) or the CLI-local sherpa-ONNX
/// subprocess fallback. Handles auto-download of the pyannote GGUF
/// when `--diarize-method pyannote` was passed without `--sherpa-segment-model`.
/// Mutates each `seg.speaker` in-place, formatting the result as
/// `"(speaker N) "` to match the historical crispasr convention.
///
/// `left` and `right` are per-channel slice buffers when stereo is
/// available. For mono input, both vectors point at the same data and
/// `is_stereo` is false; the dispatcher should call this anyway so the
/// mono-friendly methods (vad-turns, pyannote, sherpa) can still run.
///
/// `pyannote_cache` is optional. When non-null and valid AND the method
/// is `pyannote`, the cached posteriors are used instead of running
/// pyannote-seg again on this slice — required for cross-slice speaker
/// ID consistency (#107).
bool crispasr_apply_diarize(const std::vector<float>& left, const std::vector<float>& right, bool is_stereo,
                            int64_t slice_t0_cs, std::vector<crispasr_segment>& segs, const whisper_params& params,
                            const CrispasrPyannoteCache* pyannote_cache = nullptr,
                            const CrispasrSherpaCache* sherpa_cache = nullptr);

/// Apply a per-slice diarization pass to an ALREADY-MERGED segment list.
///
/// The unified CLI runner diarizes inside its per-slice loop and hands each
/// slice its own segment vector, so it never needs this. The server
/// transcribes every slice first and only then diarizes, so it has to re-walk
/// the merged list and give each slice back its own sub-range.
///
/// That re-walk is the subtle part: `crispasr_apply_diarize` GROWS the vector
/// whenever it splits a segment at a speaker-turn boundary (the pyannote,
/// foxnose and sherpa paths all do word-range splitting), so the sub-range
/// cannot be written back in place — see #324. This rebuilds the list instead,
/// appending whatever each slice returns, so no sub-segment is ever dropped
/// and a shrinking pass can never index past the end.
///
/// A segment belongs to slice `i` when its `t0` is below slice `i+1`'s
/// `t0_cs`; the final slice takes the remainder. `diarize_slice` receives the
/// slice and its segments, and may relabel, split or drop them.
void crispasr_diarize_merged_by_slice(
    std::vector<crispasr_segment>& segs, const std::vector<crispasr_audio_slice>& slices,
    const std::function<void(const crispasr_audio_slice&, std::vector<crispasr_segment>&)>& diarize_slice);

/// #324: run the FoxNose diarizer ONCE over the whole audio, using the final
/// segment list as its speech regions, then relabel and split those segments.
///
/// Per-SLICE diarization cannot give consistent speaker identities: each slice
/// clusters independently and restarts numbering at 0, so `speaker 0` in one
/// slice is a different person from `speaker 0` in the next. This is the same
/// problem the pyannote path solves with a pre-computed posterior cache
/// (#107); FoxNose solves it here instead, after transcription, because it
/// needs the segments as its speech regions and they do not exist beforehand.
///
/// Returns true when it ran (so the caller can skip the per-slice path).
bool crispasr_apply_foxnose_global(std::vector<crispasr_segment>& all_segs, const std::vector<float>& samples,
                                   const whisper_params& params);

/// Re-label each segment's speaker by clustering speaker embeddings
/// extracted from `full_audio`. Operates over the WHOLE finalized
/// segment list (after per-slice diarize + segment splitting), so it
/// gives globally stable speaker IDs across the entire file.
///
/// `embedder` is the pluggable model (TitaNet today; any other adapter
/// later). `full_audio` is the mono 16 kHz buffer; `params` carries
/// the merge threshold and max-speakers settings. Segments shorter
/// than ~250 ms are skipped (embedders generally produce noisy
/// vectors below that), so their existing (pyannote-local) speaker
/// label is left in place.
///
/// No-op when `embedder` is null, `segs` is empty, or `full_audio` is
/// empty / too short — the pyannote-only labels survive unchanged in
/// those cases, which is what makes the system "work sufficiently
/// well without an embedder" (#107 P3).
///
/// When `out_clusters` is non-null it receives the per-segment
/// embeddings and cluster assignment so a later identification stage
/// (issue #266) can reuse them without re-running the embedder.
struct CrispasrClusterEmbeddings {
    std::vector<size_t> seg_idx;   // indices into segs that were embedded
    std::vector<float> embeddings; // seg_idx.size() * dim, row-major
    std::vector<int> labels;       // cluster ID per embedded segment
    int dim = 0;
    int n_clusters = 0;
    bool valid() const { return dim > 0 && n_clusters > 0 && !seg_idx.empty() && labels.size() == seg_idx.size(); }
};

void crispasr_remap_speakers_via_embeddings(std::vector<crispasr_segment>& segs, const float* full_audio, int n_samples,
                                            CrispasrSpeakerEmbedder* embedder, const whisper_params& params,
                                            CrispasrClusterEmbeddings* out_clusters = nullptr);

/// Cluster-level speaker identification (issue #266). For each global
/// speaker cluster in `ce`, match the L2-normalized centroid of the
/// cluster's member embeddings against `db` — which the caller MUST
/// already have narrowed to the claimed roster via speaker_db_retain().
/// A matched cluster's member segments get "(Name) " labels; unmatched
/// clusters keep their anonymous "(speaker N) " labels. One identity
/// per cluster; a mixed slice can never inherit a single name.
/// Returns the number of clusters identified.
int crispasr_identify_speaker_clusters(std::vector<crispasr_segment>& segs, const CrispasrClusterEmbeddings& ce,
                                       const struct speaker_db* db, float threshold, bool no_prints);

/// Standalone identification without diarization (issue #266): the whole
/// recording is treated as ONE speaker cluster. Embeds all eligible
/// segments, centroid-matches against the (already claimed-roster-
/// narrowed) db, and on success labels EVERY segment with the name.
/// Returns true when a name was assigned.
bool crispasr_identify_single_speaker(std::vector<crispasr_segment>& segs, const float* full_audio, int n_samples,
                                      CrispasrSpeakerEmbedder* embedder, const struct speaker_db* db, float threshold,
                                      bool no_prints);

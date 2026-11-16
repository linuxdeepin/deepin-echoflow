#pragma once

// basic_pitch.h — Spotify Basic Pitch backend
// (ICASSP 2022, Apache-2.0, https://github.com/spotify/basic-pitch)
//
// Polyphonic, instrument-agnostic audio → note events. ~20k parameters of
// convolution sitting on top of a large harmonically-stacked CQT; the front end
// is most of the model.
//
//   Input: 22.05 kHz mono, processed in 43844-sample (2 s minus one hop) windows
//   → nnAudio CQT2010v2 (fmin 27.5 Hz, 309 bins, 36 bins/octave, hop 256)
//   → NormalizedLog (power → dB → per-example min/max to [0,1]) → BatchNorm
//   → HarmonicStacking(bins_per_semitone=3,
//                      harmonics=[0.5,1,2,3,4,5,6,7],
//                      n_output_freqs=264)                → (172, 264, 8)
//   → contour head: Conv2D(8, 3x39) + BN + ReLU → Conv2D(1, 5x5) + sigmoid
//                                                          → (172, 264)
//   → note head:    Conv2D(32, 7x7, stride (1,3)) + ReLU
//                   → Conv2D(1, 7x3) + sigmoid             → (172, 88)
//   → onset head:   Conv2D(32, 5x5, stride (1,3)) + BN + ReLU,
//                   concat with the pre-flatten note map,
//                   → Conv2D(1, 3x3) + sigmoid             → (172, 88)
//
// Windows overlap by 30 frames; 15 are dropped from each side before the
// per-window outputs are stitched (inference.py::unwrap_output). Note events
// come from the onset/note posteriorgrams by peak-picking plus the "melodia
// trick" (note_creation.py::output_to_notes_polyphonic).
//
// All BatchNorms are pre-folded into the preceding convolution by the ONNX
// export, so there is no BN to apply at runtime except the single-channel one
// after the CQT, which the GGUF carries as a scale/shift pair.
//
// GGUF produced by models/convert-basic-pitch-to-gguf.py (arch "basic-pitch").

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct basic_pitch_ctx;

struct basic_pitch_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose (also populates raw heads)
    bool use_gpu;

    // Post-processing, defaults from basic_pitch/inference.py.
    float onset_threshold;        // 0.5
    float frame_threshold;        // 0.3
    float minimum_note_length_ms; // 127.7  (→ 11 frames)
    float minimum_frequency;      // <= 0 → no low cut
    float maximum_frequency;      // <= 0 → no high cut
    bool infer_onsets;            // true — add onsets from frame-energy jumps
    bool melodia_trick;           // true — second pass over leftover energy
};

struct basic_pitch_params basic_pitch_default_params(void);

// A single detected note event.
struct basic_pitch_note_event {
    float start_time; // seconds
    float end_time;   // seconds
    int midi_note;    // 21-108 (A0-C8)
    float amplitude;  // mean frame activation over the note, 0-1
    int velocity;     // round(127 * amplitude)
};

struct basic_pitch_result {
    struct basic_pitch_note_event* notes;
    int n_notes;

    // Stitched posteriorgrams, row-major. Only populated when verbosity >= 2.
    float* contour;    // (n_frames, 264)
    float* note_head;  // (n_frames, 88)
    float* onset_head; // (n_frames, 88)
    int n_frames;
    int n_freq_contours; // 264
    int n_freq_notes;    // 88
};

// Initialize from a GGUF file.
struct basic_pitch_ctx* basic_pitch_init_from_file(const char* path, struct basic_pitch_params params);

void basic_pitch_free(struct basic_pitch_ctx* ctx);

// Transcribe float32 mono PCM at 22050 Hz.
// Returns 0 on success, nonzero on error. Free with basic_pitch_result_free().
int basic_pitch_transcribe(struct basic_pitch_ctx* ctx, const float* pcm, int n_samples,
                           struct basic_pitch_result* result);

void basic_pitch_result_free(struct basic_pitch_result* result);

// Expected input sample rate (always 22050).
uint32_t basic_pitch_sample_rate(const struct basic_pitch_ctx* ctx);

// Samples per model window (43844) — the unit the diff harness compares.
uint32_t basic_pitch_window_samples(const struct basic_pitch_ctx* ctx);

// Per-stage parity against tools/reference_backends/basic_pitch.py.
// `pcm_22k` must be the SAME 22050 Hz mono signal the reference ran on; the
// CQT is compared before any head, so a front-end difference cannot be mistaken
// for a model failure.
// Returns 0 when every stage passes, 1 on a parity failure, 2 on a load error.
int basic_pitch_diff(const char* model_gguf, const char* ref_gguf, const float* pcm_22k, int n_samples, int verbosity);

#ifdef __cplusplus
}
#endif

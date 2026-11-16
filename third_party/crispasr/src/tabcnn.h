// tabcnn.h — TabCNN guitar tablature emission scorer.
//
// TabCNN (Wiggins & Kim, ISMIR 2019) predicts, per frame, six INDEPENDENT
// distributions over 21 fret classes — one per guitar string. It contains no
// decoder: no inter-string coupling, no temporal model, no search. That is
// deliberate here. This backend ships the emission scores and nothing else;
// the constrained Viterbi/DP that turns them into a playable fingering (one
// note per string, fret range, capo, hand span) belongs to the caller.
//
//   front end   librosa-compatible CQT, sr 22050, hop 512, 192 bins,
//               24 per octave, fmin C1 (32.70 Hz)
//               -> amplitude_to_db(ref = max of the WHOLE clip) -> [-80, 0]
//               -> /80 + 1                                      -> [0, 1]
//   input       [1, 192, 9]  (192 bins x a 9-frame centred context window)
//   conv        Conv2d(1,32,3) ReLU Conv2d(32,64,3) ReLU Conv2d(64,64,3) ReLU
//               MaxPool2d(2,2)          192x9 -> 190x7 -> 188x5 -> 186x3 -> 93x1
//   dense       flatten 64*93 = 5952 -> Linear(5952,128) ReLU
//   head        Linear(128, 126) -> [6, 21] -> per-string softmax
//
// ⚠️ fmin is C1, NOT the guitar's low E. Assuming E2 is the obvious guess and it
// is wrong — and every wrong value still RUNS, producing plausible tensors that
// pass shape and cosine checks while the model emits garbage. Measured on
// EGSet12 track 01: fmin C1 -> tablature F1 0.771, E1 -> 0.040, E2 -> 0.001.
// The constants live in the GGUF so the runtime cannot drift from the reference
// dumper; they are read back here, never hardcoded.
//
// ⚠️ `ref = max of the whole clip` makes the front end a PER-CLIP normalisation.
// It cannot be computed streaming or chunked without changing the features, so
// this backend is two-pass by construction and `--tab` is not a streaming
// surface. Chunking would reproduce the BTC chunked-CQT bug.
//
// GGUF: tabcnn-f16.gguf. Weights are CC BY 4.0 (EGSet12,
// https://zenodo.org/records/11406378) and REQUIRE attribution.
// ⚠ There is NO Zenodo DOI: `10.5281/zenodo.11406378` looks plausible and 404s.
// The record DOI is the arXiv one, 10.48550/arXiv.2405.14679.
// Blueprint, licence chain and benchmark context:
//   docs/music-transcription/GUITAR_TAB_SPEC.md
// Executable spec for this graph (keep in lockstep):
//   tools/tabcnn_torch_parity.py
// Ground-truth dumper for crispasr-diff:
//   tools/reference_backends/tabcnn.py

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tabcnn_context;

// Fixed geometry. Callers sizing buffers may rely on these; the runtime still
// reads the authoritative values from the GGUF and fails if they disagree.
#define TABCNN_NUM_STRINGS 6
#define TABCNN_NUM_CLASSES 21
#define TABCNN_SAMPLE_RATE 22050

// Initialize from a GGUF model file. Returns NULL on failure.
struct tabcnn_context* tabcnn_init(const char* model_path, int n_threads);

void tabcnn_free(struct tabcnn_context* ctx);

// Frames `tabcnn_compute` will produce for `n_samples` of input at
// `sample_rate`. Lets the caller allocate exactly once. Returns 0 on bad args.
int tabcnn_n_frames(const struct tabcnn_context* ctx, int n_samples, int sample_rate);

// Seconds per output frame (hop / sample_rate), so a caller can place frames on
// its own timeline without assuming the hop.
float tabcnn_frame_period(const struct tabcnn_context* ctx);

// The class index meaning "string not played". Read from the GGUF rather than
// assumed to be the highest index — a decoder that guesses this wrong produces
// confidently wrong tablature with no error anywhere.
int tabcnn_silent_class(const struct tabcnn_context* ctx);

// Open-string MIDI pitch for `string` (0 = lowest, i.e. low E in standard
// tuning), or -1 if unavailable. A capo/transpose-aware decoder needs these
// rather than hardcoding standard tuning.
int tabcnn_string_open_midi(const struct tabcnn_context* ctx, int string);

// THE EMISSION SCORER. Writes n_frames * 6 * 21 LOG-probabilities to `out` in
// frame-major order: out[(f*6 + s)*21 + c] is log P(string s plays fret class c
// at frame f). Log, not probability: a DP sums costs, and handing back raw
// probabilities invites the caller to take log(0).
//
// Rows sum to 1 in probability space per (frame, string). `sample_rate` may be
// anything; audio is resampled internally to the model's rate.
//
// Returns the number of frames written, or 0 on error. Size `out` with
// tabcnn_n_frames() * TABCNN_NUM_STRINGS * TABCNN_NUM_CLASSES.
int tabcnn_compute(struct tabcnn_context* ctx, const float* pcm, int n_samples, int sample_rate, float* out,
                   int max_frames);

// Convenience argmax over the emissions, for the CLI's text output and for
// smoke tests. `out_frets` receives n_frames * 6 int8 values; a value equal to
// tabcnn_silent_class() means the string is not played. This is NOT the
// intended production path — it ignores every playability constraint, which is
// exactly what the caller's decoder exists to apply.
int tabcnn_compute_argmax(struct tabcnn_context* ctx, const float* pcm, int n_samples, int sample_rate,
                          int8_t* out_frets, int max_frames);

// Per-stage extraction for crispasr-diff. `stage` is one of the names emitted
// by tools/reference_backends/tabcnn.py ("cqt_db", "conv0", "pool", "logits",
// ...). Writes at most `max_elems` floats and returns the count, or 0.
// Wired so the reference dumper has a consumer — a dumper with no C++ reader is
// dead code that only looks like coverage.
int tabcnn_extract_stage(struct tabcnn_context* ctx, const float* pcm, int n_samples, int sample_rate,
                         const char* stage, float* out, int max_elems);

// crispasr-diff entry point: score this runtime against a reference archive
// from tools/reference_backends/tabcnn.py. Runs from the `audio` stage in the
// archive, NOT from replayed features, so the CQT front end is covered too.
// Returns 0 when every stage passes.
int tabcnn_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

#ifdef __cplusplus
}
#endif

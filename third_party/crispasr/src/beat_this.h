// beat_this.h — Beat This! beat and downbeat tracking (§251b).
//
// Architecture: CPJKU "Beat This!" (ISMIR 2024), checkpoint `final0`. MIT —
// upstream states code AND published weights are MIT.
//
//   Input:  22050 Hz mono PCM -> log-mel (128 bins, 50 fps)
//   Front:  BN1d -> conv stem (freq 128->32) -> 3x [partial freq/time
//           transformer + conv] (dims 32->256, freq 32->4) -> concat 1024
//           -> Linear 512
//   Body:   6 roformer layers, dim 512, 16 heads, shared RoPE(32)
//   Head:   SumHead -> 2 logits per frame (beat, downbeat)
//   Post:   peak-pick (no DBN) -> beat/downbeat times in seconds
//
// NO DBN POSTPROCESSING. That is the paper's central claim and it matters
// downstream: madmom's DBN is patent-encumbered (Böck) and non-commercially
// licensed, so any beat tracker reaching CometBeat must avoid it. This model's
// dependency list is numpy/torch/torchaudio/einops/rotary-embedding-torch/soxr
// — no madmom anywhere, not even by distillation.
//
// GGUF: beat-this-f16.gguf (~41 MB), converted by
// models/convert-beat-this-to-gguf.py. Blueprint + tensor map + the two
// silent-failure subtleties (RMSNorm form, per-head attention gating) are in
// docs/music-transcription/PLAN.md §251b-1.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct beat_this_context;

#define BEAT_THIS_SAMPLE_RATE 22050
#define BEAT_THIS_N_FFT 1024
#define BEAT_THIS_HOP 441
#define BEAT_THIS_MEL_BINS 128
#define BEAT_THIS_FPS 50
#define BEAT_THIS_DIM 512
#define BEAT_THIS_N_LAYERS 6

// Windowing, from upstream Spect2Frames.spect2frames. `border` frames are
// discarded from each end of every chunk's prediction because the model was
// never trained on its input edges (the training loss max-pools them away), so
// consecutive chunks overlap by 2*border to cover the discarded region.
#define BEAT_THIS_CHUNK_FRAMES 1500
#define BEAT_THIS_BORDER_FRAMES 6

// One detected event. `is_downbeat` marks a beat that also starts a bar; every
// downbeat is also reported as a beat, matching the reference's convention of
// snapping each downbeat to its nearest beat.
struct beat_this_event {
    float time_s;
    int is_downbeat;
};

struct beat_this_context* beat_this_init(const char* model_path, int n_threads);
void beat_this_free(struct beat_this_context* ctx);

// Compute the log-mel front end: 22050 Hz mono -> (n_frames, 128) row-major.
//
// Exposed separately because it is independently testable against torchaudio
// and is the piece most likely to drift silently. Contract, all of it load-
// bearing: STFT n_fft 1024 / hop 441, periodic Hann, center=true with REFLECT
// padding, magnitude (power=1, NOT power=2), divided by sqrt(n_fft) = 32
// (torchaudio's `normalized="frame_length"` divides by the SQUARE ROOT of
// n_fft despite the name — verified empirically, ratio 32.0), projected onto
// the baked [513,128] filterbank, then log1p(1000 * x).
//
// Returns the frame count, or 0 on error. `out` must hold at least
// beat_this_n_frames(n_samples) * 128 floats.
int beat_this_logmel(struct beat_this_context* ctx, const float* pcm_22k, int n_samples, float* out);

// Frames the front end will produce for `n_samples` (center=true: 1 + n/hop).
int beat_this_n_frames(int n_samples);

// Debug/parity: stem forward only. Writes T*32*32 floats as ne (t, f, c)
// — the reverse of the reference's torch (b, c, f, t). Returns the element
// count. Used by tests/test_beat_this_stages.cpp against
// tools/reference_backends/beat_this.py.
int beat_this_debug_stem(struct beat_this_context* ctx, const float* logmel, int T, float* out);

// Debug/parity: run the frontend and return one named intermediate. Stages:
//
//   "stem"                          ne (t, f, c)
//   "blk<N>_attnF" / "blk<N>_ffF"   ne (c, f, t)   N = 0..2
//   "blk<N>_attnT" / "blk<N>_ffT"   ne (c, t, f)
//   "blk<N>_partial"                ne (t, f, c)
//   "blk<N>"                        ne (t, f, c)   after conv+BN+GELU
//
// The attn/ff stages are the residual BRANCH, matching the reference's forward
// hooks — they are `attnF(x)`, NOT `x + attnF(x)`. Comparing a branch against a
// post-residual activation is the trap this port was designed around.
//
// Writes the tensor's ggml ne into `ne_out[4]` (may be NULL) and returns the
// element count, or 0 on error or an unknown stage. Because ggml ne is the
// reverse of torch's shape, reshaping the dump to reversed(ne) in numpy lands
// exactly on the reference's layout with no transpose.
int beat_this_debug_stage(struct beat_this_context* ctx, const float* logmel, int T, const char* stage, float* out,
                          int max_out, int64_t* ne_out);

// Framewise beat/downbeat LOGITS for a whole piece, with upstream's 1500-frame
// chunking (border 6, keep_first overlap) applied. `beat` and `downbeat` must
// each hold T floats. Returns T, or 0 on error.
//
// Exposed separately from beat_this_track() so the windowing can be tested
// against the reference independently of the peak-picking: a seam bug and a
// threshold bug both show up as "wrong beats" at the event level.
int beat_this_logits(struct beat_this_context* ctx, const float* logmel, int T, float* beat, float* downbeat);

// Peak-pick framewise logits into events. Needs no model, so postprocessing can
// be scored against the reference's OWN logits — which is the only way to tell
// a peak-picking bug apart from an upstream numerical difference.
//
// Reproduces upstream Postprocessor(type="minimal"): maxima over a +/-3 frame
// window, logit > 0, runs of peaks <=1 frame apart collapsed to their MEAN
// frame index (fractional, deliberately), /50 for seconds, then each downbeat
// snapped to its nearest beat. NO DBN — see the header note above.
int beat_this_events_from_logits(const float* beat, const float* downbeat, int T, struct beat_this_event* out,
                                 int max_events);

// Full pipeline: audio -> events. Returns the event count, or 0/-1 on error.
int beat_this_track(struct beat_this_context* ctx, const float* pcm_22k, int n_samples, struct beat_this_event* out,
                    int max_events);

// Estimated tempo in BPM from the detected beats (median inter-beat interval),
// or 0 when there are too few beats.
float beat_this_tempo_bpm(const struct beat_this_event* events, int n_events);

int beat_this_sample_rate(const struct beat_this_context* ctx);

#ifdef __cplusplus
}
#endif

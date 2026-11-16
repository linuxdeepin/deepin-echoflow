// src/beatrice_pitch.h — Beatrice v2 PitchEstimator (Project Beatrice, MIT).
//
// The first of Beatrice's three networks. Takes 16 kHz mono PCM and produces,
// at a 100 Hz frame rate, 448 pitch-bin logits plus a per-frame energy — and
// the banded-argmax quantised pitch derived from them.
//
// Beatrice's LICENCE IS MIT, source and trained models alike
// (fierce-cats/beatrice-trainer). Unlike RVC this needs no acceptance gate. The
// GGUF still carries its own tag because a checkpoint from some other training
// run need not share the base's terms.
//
// UNLIKE the rest of the model, this component is DETERMINISTIC — the RNG in
// Beatrice lives in the vocoder's overlap_add, not here. So this one can be
// validated by direct comparison, with no noise injection.
//
// See docs/music-transcription/BEATRICE_BLUEPRINT.md for the ~10 non-obvious
// details this reproduces, and for what the parity harness cannot see.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct beatrice_pitch_context;

struct beatrice_pitch_params {
    int n_threads; // 0 = auto
    bool use_gpu;
    int gpu_device;
};

beatrice_pitch_params beatrice_pitch_default_params(void);

beatrice_pitch_context* beatrice_pitch_init_from_file(const char* model_path, beatrice_pitch_params params);
void beatrice_pitch_free(beatrice_pitch_context* ctx);

// Geometry, read from the GGUF rather than assumed.
int beatrice_pitch_n_bins(const beatrice_pitch_context* ctx);          // 448
int beatrice_pitch_bins_per_octave(const beatrice_pitch_context* ctx); // 96
int beatrice_pitch_sample_rate(const beatrice_pitch_context* ctx);     // 16000
int beatrice_pitch_hop_length(const beatrice_pitch_context* ctx);      // 160 -> 100 Hz

struct beatrice_pitch_result {
    // BIN-MAJOR, TIME FASTEST: element (bin, frame) at bin*n_frames + frame.
    // This is the ggml-native layout of the graph output and matches the parity
    // reference, so nothing has to transpose. It is NOT frame-major.
    float* logits; // n_bins * n_frames
    // n_frames, banded argmax. NOTE: this NEVER returns 0. sample_pitch forces
    // bin 0 (the unvoiced class) to -100 before the argmax, so it is excluded by
    // construction -- measured: 0 zeros over 1100 frames of jfk.wav INCLUDING
    // its silent stretches, minimum bin 8. There is therefore NO voicing
    // information here; an earlier version of this comment said 0 meant
    // unvoiced and was simply wrong.
    //
    // For a usable voicing decision use `energy` below, NOT this and NOT
    // pitch_features[0] -- see beatrice_pitch_to_f0_hz.
    //
    // Frequency: f = 55 * 2**(bin / 96) Hz -- 96 bins/octave anchored at A1,
    // 12.5 cents per bin. Recovered empirically from octave pairs; the trainer
    // source never states it.
    int* quantized;
    float* energy; // n_frames
    // The 3 pitch features, CHANNEL-MAJOR / TIME-FASTEST, 3 * n_frames:
    //   [0] unvoiced_proba     -- the model's unvoiced-class probability. It is
    //                             EFFECTIVELY INERT on this checkpoint (max
    //                             8.9e-07 measured); exposed because
    //                             ConverterNetwork consumes it, NOT because it
    //                             is a usable voicing gate.
    //   [1] half_pitch_proba
    //   [2] double_pitch_proba   -- these two DO have real range (up to ~0.6/0.7)
    // ConverterNetwork prepends `energy` to these to form the 4 channels that
    // embed_pitch_features consumes.
    float* pitch_features;
    int n_frames;
    int n_bins;
};

// bin -> Hz. f = 55 * 2**(bin / 96). Accuracy is at that quantisation floor
// (median 2.4 cents on off-grid tones) roughly between 62 Hz and 780 Hz; above
// ~800 Hz the MODEL fails, not the mapping. Measure with crispasr-f0-eval.
float beatrice_pitch_bin_to_hz(int bin);

// Convert the quantised bins to Hz, writing 0.0 for frames judged unvoiced --
// exactly the convention rvc_svc_convert() expects for its `f0_hz` input, and at
// the same 100 Hz rate.
//
// GATE ON ENERGY, NOT ON unvoiced_proba. Measured on this checkpoint,
// pitch_features[0] never exceeds 8.9e-07 on any frame of jfk.wav (median
// 1.7e-08) -- the unvoiced class is effectively never activated, so any
// probability-style threshold is inert. It does carry signal (correlation
// +0.508 with -energy; the quietest 15% of frames average ~40x the loudest),
// but not on a scale a threshold can use.
//
// `energy` is the practical voicing signal and this API already returns it:
// cosine-windowed log10 energy, floored at -1.5 (its clamp) for silence, up to
// ~+0.8 for loud voiced speech. MEASURED on jfk.wav against CREPE: thresholds of
// -1.4/-1.2/-1.0 change nothing (real speech rarely sits that quiet), while
// -0.5 cuts the median disagreement from 56.5 to 17.4 cents and lifts
// within-50-cents from 49.0% to 85.5%. Start around -0.5 and tune per corpus;
// it is a loudness gate, so it will also drop genuinely quiet voiced frames.
//
// Passing energy_threshold <= -1.5 disables gating so every frame gets a pitch,
// which is WRONG for anything driving a source signal: the model reports a pitch
// for silence exactly as confidently as for speech.
void beatrice_pitch_to_f0_hz(const beatrice_pitch_result* r, float energy_threshold, float* out_f0_hz);

// `pcm` is mono 16 kHz. Frame count is
// floor((n_samples + (win-hop) - win) / hop) + 1 with win=560, hop=160.
beatrice_pitch_result* beatrice_pitch_estimate(beatrice_pitch_context* ctx, const float* pcm, int n_samples);
void beatrice_pitch_result_free(beatrice_pitch_result* r);

// Per-stage parity diff against tools/beatrice_torch_parity.py's dump.
// Returns 0 when every stage passes, 1 on a parity failure, 2 on a load error.
int beatrice_pitch_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

#ifdef __cplusplus
}
#endif

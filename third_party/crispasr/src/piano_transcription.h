#pragma once

// piano_transcription.h — Piano transcription backend
// (ByteDance/Kong, Apache-2.0, https://github.com/qiuqiangkong/piano_transcription_inference)
//
// Transcribes piano audio to MIDI note events using a CRNN architecture:
//   Input: 16 kHz mono audio
//   → STFT (n_fft=2048, hop=160) → LogMel (229 bins, 30-8000 Hz) → BN
//   → 4× AcousticModelCRnn8Dropout (frame/onset/offset/velocity):
//       4× ConvBlock(Conv2d 3×3 + BN2d + ReLU, AvgPool2d (1,2))
//       FC(1792→768) + BN1d + ReLU
//       BiGRU(768→256, 2 layers) → FC(512→88) → sigmoid
//   → Onset refinement: cat(onset, sqrt(onset)*velocity) → BiGRU → FC → sigmoid
//   → Frame refinement: cat(frame, onset, offset) → BiGRU → FC → sigmoid
//   → Post-processing: regression binarization → note detection → MIDI events
//
// Output: 4 heads × (T, 88) for 88 piano keys at 100fps,
//         plus detected note events with onset/offset times and velocities.
//
// GGUF produced by models/convert-piano-transcription-to-gguf.py

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct piano_transcription_ctx;

struct piano_transcription_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;

    // Post-processing thresholds
    float onset_threshold;        // default 0.3
    float offset_threshold;       // default 0.3
    float frame_threshold;        // default 0.1
    float pedal_offset_threshold; // default 0.2
};

struct piano_transcription_params piano_transcription_default_params(void);

// A single detected note event.
struct piano_note_event {
    float onset_time;  // seconds
    float offset_time; // seconds
    int midi_note;     // 21-108 (A0-C8)
    int velocity;      // 0-127
};

// A single detected pedal event.
struct piano_pedal_event {
    float onset_time;
    float offset_time;
};

// Transcription result.
struct piano_transcription_result {
    piano_note_event* note_events;
    int n_notes;
    piano_pedal_event* pedal_events;
    int n_pedals;

    // Raw neural network outputs (T × 88), row-major.
    // Only populated if verbosity >= 2.
    float* frame_output;
    float* onset_output;
    float* offset_output;
    float* velocity_output;
    int n_frames;
    int n_classes; // 88
};

// Initialize from a GGUF file.
struct piano_transcription_ctx* piano_transcription_init_from_file(const char* path,
                                                                   struct piano_transcription_params params);

void piano_transcription_free(struct piano_transcription_ctx* ctx);

// Transcribe float32 mono PCM at 16 kHz.
// Returns 0 on success, nonzero on error.
// Result is written to *result. Caller must free with piano_transcription_result_free().
int piano_transcription_transcribe(struct piano_transcription_ctx* ctx, const float* pcm, int n_samples,
                                   struct piano_transcription_result* result);

// Free a result's internal allocations.
void piano_transcription_result_free(struct piano_transcription_result* result);

// Per-stage parity diff against tools/reference_backends/piano_transcription.py.
// The reference dump carries no input_audio, so the caller passes the SAME
// 16 kHz mono PCM the reference ran on; mel_spectrogram is compared first so a
// front-end difference cannot masquerade as a model failure.
// Returns 0 when every stage passes, 1 on a parity failure, 2 on a load error.
int piano_transcription_diff(const char* model_gguf, const char* ref_gguf, const float* pcm_16k, int n_samples,
                             int verbosity);

// Return expected sample rate (always 16000).
uint32_t piano_transcription_sample_rate(const struct piano_transcription_ctx* ctx);

// Diff harness helpers — run individual stages and return intermediate tensors.
// Each returns a malloc'd float buffer; caller frees with free().
// *out_n receives the number of elements.

// Compute log-mel spectrogram with BN0.
// Output shape: (T, 229), row-major.
float* piano_transcription_mel_spectrogram(struct piano_transcription_ctx* ctx, const float* pcm, int n_samples,
                                           int* out_n);

// Run a single acoustic model on mel input.
// model_name: "frame", "onset", "offset", "velocity"
// Output shape: (T, 88), row-major.
float* piano_transcription_acoustic_model(struct piano_transcription_ctx* ctx, const float* mel, int n_mel_elements,
                                          const char* model_name, int* out_n);

#ifdef __cplusplus
}
#endif

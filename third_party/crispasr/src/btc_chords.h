// src/btc_chords.h — BTC chord recognition (Bi-directional Transformer,
// Park et al., ISMIR 2019).
//
// Frame-level chord labels over a constant-Q spectrogram. Upstream code
// (jayg996/BTC-ISMIR19) is MIT; the SHIPPED WEIGHTS are CC-BY-NC-SA because
// they were trained on Isophonics / Robbie Williams / UsPop2002 annotations,
// so the GGUF carries that licence and the registry gate refuses to download
// it without --accept-license cc-by-nc-sa-4.0.
//
// See docs/music-transcription/BTC_BLUEPRINT.md for the ten details that are
// not the obvious default, and tools/btc_torch_parity.py for the executable
// spec this runtime must match.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct btc_chords_context;

struct btc_chords_params {
    int n_threads; // 0 = auto
    bool use_gpu;
    int gpu_device;
};

// One recognised chord region.
struct btc_chord_span {
    double start_ms;
    double end_ms;
    int label;        // index into the model's chord vocabulary
    float confidence; // softmax probability of `label`
};

struct btc_chords_result {
    int n_spans;
    btc_chord_span* spans;
    int n_chords;             // vocabulary size (25 or 170)
    const char** chord_names; // n_chords entries
};

btc_chords_params btc_chords_default_params(void);

btc_chords_context* btc_chords_init_from_file(const char* model_path, btc_chords_params params);
void btc_chords_free(btc_chords_context* ctx);

// Recognise chords in mono PCM. The model expects 22050 Hz; other rates are
// resampled internally. Caller frees with btc_chords_result_free().
btc_chords_result* btc_chords_recognize(btc_chords_context* ctx, const float* pcm, int n_samples, int sample_rate);
void btc_chords_result_free(btc_chords_result* r);

int btc_chords_vocab_size(const btc_chords_context* ctx);
const char* btc_chords_label_name(const btc_chords_context* ctx, int label);

// Per-stage parity diff against tools/btc_torch_parity.py's reference dump.
// Returns 0 when every stage passes, 1 on a parity failure, 2 on a load error.
int btc_chords_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

#ifdef __cplusplus
}
#endif

// webrtc_vad.h — WebRTC VAD (GMM-based voice activity detection).
//
// Architecture: 6-band Gaussian Mixture Model — zero weights, zero dependencies.
//   Input: 16kHz int16 PCM → 30ms frames → per-frame speech/non-speech decision
//   Pure algorithmic (no neural network, no GGUF model file needed).
//
// Usage: init, feed 16kHz PCM audio, get speech segments.
// The "model path" is a sentinel — pass "webrtc" or any path containing "webrtc".
// Aggressiveness mode (0-3) is controlled via CRISPASR_WEBRTC_VAD_MODE env var.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct webrtc_vad_segment {
    float start_sec;
    float end_sec;
};

// Detect speech segments in 16kHz mono float32 PCM audio.
// Returns array of segments (caller must free with free()).
// n_segments is set to the number of segments found.
// mode: 0 (least aggressive) to 3 (most aggressive). -1 = use env var or default (1).
// Returns 0 on success, -1 on error.
int webrtc_vad_detect(const float* samples, int n_samples, struct webrtc_vad_segment** segments, int* n_segments,
                      float threshold, float min_speech_sec, float min_silence_sec, int mode);

#ifdef __cplusplus
}
#endif

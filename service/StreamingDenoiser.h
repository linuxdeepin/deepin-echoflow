// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_STREAMING_DENOISER_H
#define ECHOFLOW_STREAMING_DENOISER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace echoflow {

// One state per recording, 16 kHz mono in/out. Resampling and RNNoise state
// persist across append calls; finish drains the lookahead and preserves length.
class StreamingDenoiser {
public:
    StreamingDenoiser();
    ~StreamingDenoiser();
    StreamingDenoiser(const StreamingDenoiser&) = delete;
    StreamingDenoiser& operator=(const StreamingDenoiser&) = delete;
    std::vector<float> append(const float* samples, size_t count);
    std::vector<int16_t> append(const int16_t* samples, size_t count);
    std::vector<float> finish();
    std::vector<int16_t> finishS16();
    double processingMs() const;
    size_t inputSamples() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Lightweight speech probability for metering. This only decides whether to
// animate the microphone; it must never gate or discard recognition audio.
class StreamingSpeechDetector {
public:
    explicit StreamingSpeechDetector(int sampleRate);
    ~StreamingSpeechDetector();
    StreamingSpeechDetector(const StreamingSpeechDetector&) = delete;
    StreamingSpeechDetector& operator=(const StreamingSpeechDetector&) = delete;
    double probability(const float* mono, size_t count);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace echoflow

#endif

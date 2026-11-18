// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_AUDIO_LEVEL_METER_H
#define ECHOFLOW_AUDIO_LEVEL_METER_H

#include "AudioLevelFrame.h"
#include "AudioSpectrum.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace echoflow {

class StreamingSpeechDetector;

// Fixed 50 ms windows, independent of recorder read sizes and ASR progress.
// Recording paths enable background suppression; amplitude tests can bypass it.
class AudioLevelMeter {
public:
    using Update = std::function<void(const AudioLevelFrame&)>;
    AudioLevelMeter(int sampleRate, int channels, Update update, bool speechOnly = false);
    ~AudioLevelMeter();
    void append(const int16_t* samples, size_t count);
    void appendSample(double sample);

private:
    Update update_;
    size_t windowSamples_;
    size_t channels_;
    std::vector<double> samples_;
    std::unique_ptr<StreamingSpeechDetector> speechDetector_;
    bool speechActive_ = false;
    std::unique_ptr<AudioSpectrum> spectrum_;
    double backgroundPower_ = 1e-5;
    bool backgroundKnown_ = false;
};

// Reads newly written samples from an active WAV, whose size fields may not
// be finalized yet. It never opens another microphone stream.
class WavAudioLevelReader {
public:
    explicit WavAudioLevelReader(std::filesystem::path path, bool speechOnly = false);
    std::optional<AudioLevelFrame> poll();

private:
    bool readAt(std::uint64_t offset, void* buffer, size_t count);
    bool readHeader(std::uint64_t size);
    std::filesystem::path path_;
    std::ifstream input_;
    std::uint64_t chunkOffset_ = 12;
    std::uint64_t sampleOffset_ = 0;
    unsigned format_ = 0;
    unsigned bytesPerSample_ = 0;
    std::uint64_t windowBytes_ = 0;
    bool invalid_ = false;
    bool speechOnly_;
    std::unique_ptr<AudioLevelMeter> meter_;
    std::optional<AudioLevelFrame> latest_;
};

} // namespace echoflow

#endif

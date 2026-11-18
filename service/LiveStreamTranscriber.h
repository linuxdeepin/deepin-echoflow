// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_LIVE_STREAM_TRANSCRIBER_H
#define ECHOFLOW_LIVE_STREAM_TRANSCRIBER_H

#include "LiveStreamEndpoint.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace echoflow {

struct LiveStreamConfig {
    int maxSegmentMs = 8000;
    int endSilenceMs = 500;
    int prePaddingMs = 200;
    int minSpeechMs = 100;
    int shortPauseMs = 200;
};

struct Config;

// Keep enough room to validate speech even with a long minimum duration.
LiveStreamConfig makeLiveStreamConfig(const Config& config);

struct LiveStreamDecoder {
    std::function<void()> open;
    // nullopt means PCM was buffered without a new decode; empty text retracts a hypothesis.
    std::function<std::optional<std::string>(const int16_t*, size_t)> feed;
    std::function<std::string()> flush;
    std::function<void()> close;
    // Skip an interim decode immediately before a known segment-final decode.
    std::function<void()> deferPreview;
    // Quiet gating is reversible; a checkpoint revises text without closing.
    std::function<void(bool)> setSilence;
    std::function<std::optional<std::string>()> checkpoint;
};

// Runs on the ASR worker. Each native stream owns only one bounded utterance;
// confirmed text survives reopening the stream, model weights stay loaded.
class LiveStreamTranscriber {
public:
    using Publish = std::function<void(const std::string&)>;

    LiveStreamTranscriber(LiveStreamConfig config, LiveStreamDecoder decoder,
                          Publish publish = {}, Publish confirmed = {},
                          std::unique_ptr<ILiveStreamEndpoint> endpoint = {});
    void append(const int16_t* samples, size_t count);
    std::string finish();
    void cancel();
    size_t confirmedSegments() const { return confirmedSegments_; }

private:
    void openStream();
    void setDecoderSilence(bool quiet);
    void processFrame(const int16_t* samples, size_t count);
    void processAnalyzedAudio(const LiveEndpointState& state, bool finishing = false);
    void processAudio(const int16_t* samples, size_t count, const LiveEndpointState& state,
                      bool allowCheckpoint, bool resetEndpoint);
    void bufferPausedAudio(const int16_t* samples, size_t count);
    void replayPausedAudio(size_t count, bool validSpeech);
    void clearPause();
    void feed(const int16_t* samples, size_t count, bool validSpeech);
    void finalizeSegment();
    void confirmSegment();
    void discardSegment();
    void updateActivePreview(std::string text, bool force = false);
    void publishText();
    std::string combinedText() const;

    LiveStreamDecoder decoder_;
    Publish publish_;
    Publish confirmed_;
    std::unique_ptr<ILiveStreamEndpoint> endpoint_;
    LiveEndpointState endpointState_;
    size_t endSilenceSamples_;
    size_t shortPauseSamples_;
    size_t continuationSilence_ = 0;
    size_t maxSegmentSamples_;
    size_t prePaddingSamples_;
    size_t segmentSamples_ = 0;
    size_t confirmedSegments_ = 0;
    std::vector<int16_t> frame_;
    std::vector<int16_t> pendingAudio_;
    std::vector<int16_t> leadIn_;
    std::vector<int16_t> pausedAudio_;
    size_t pausedTailSamples_ = 0;
    std::optional<double> pauseNoisePower_;
    std::string confirmedText_;
    std::string activeText_;
    std::string publishedText_;
    bool streamOpen_ = false;
    bool deferredPreviewRewrite_ = false;
    bool pauseCheckpointed_ = false;
    bool decoderSilence_ = false;
    bool continuation_ = false;
    bool finished_ = false;
};

} // namespace echoflow

#endif

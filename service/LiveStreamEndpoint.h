// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_LIVE_STREAM_ENDPOINT_H
#define ECHOFLOW_LIVE_STREAM_ENDPOINT_H

#include "Config.h"
#include "VadBackend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace echoflow {

struct LiveEndpointState {
    bool active = false;
    bool enoughSpeech = false;
    bool ended = false;
    std::optional<size_t> trailingSilenceSamples;
    size_t nextSpeechSamples = 0;
    bool awaitingAnalysis = false;
};

class ILiveStreamEndpoint {
public:
    virtual ~ILiveStreamEndpoint() = default;
    virtual LiveEndpointState append(const int16_t* samples, size_t count) = 0;
    virtual LiveEndpointState finish() = 0;
    // Preserve the supplied suffix of the audio already passed to append().
    virtual void reset(const int16_t* retainedAudio = nullptr, size_t retainSamples = 0) = 0;
    virtual std::string name() const = 0;
};

std::unique_ptr<ILiveStreamEndpoint> makeEnergyStreamEndpoint(int minSpeechMs, int endSilenceMs);
// The backend owns thresholds and minimum durations; the wrapper preserves a
// bounded history for Silero and analyzes at most every 100 ms (sooner for
// a shorter configured silence duration).
std::unique_ptr<ILiveStreamEndpoint> makeVadStreamEndpoint(
    std::unique_ptr<IVadBackend> backend, int minSpeechMs, int maxSegmentMs, int endSilenceMs);
std::unique_ptr<ILiveStreamEndpoint> makeLiveStreamEndpoint(const Config& config,
                                                         int maxSegmentMs = 8000);

} // namespace echoflow

#endif

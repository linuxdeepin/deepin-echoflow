// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "LiveStreamEndpoint.h"

#include "AudioSegmenter.h"
#include "SileroVadBackend.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace echoflow {
namespace {

constexpr size_t kEnergyFrameSamples = 320;

AudioSegmenterConfig energyConfig(int minSpeechMs, int endSilenceMs)
{
    AudioSegmenterConfig config;
    config.minSegmentMs = minSpeechMs;
    config.endSilenceMs = endSilenceMs;
    config.prePaddingMs = 0;
    config.maxSegmentMs = 0;
    config.forceOverlapMs = 0;
    return config;
}

class EnergyEndpoint final : public ILiveStreamEndpoint {
public:
    EnergyEndpoint(int minSpeechMs, int endSilenceMs)
        : detector_(energyConfig(minSpeechMs, endSilenceMs)) {}

    LiveEndpointState append(const int16_t* samples, size_t count) override
    {
        const auto ended = detector_.append(samples, count);
        partialFrameSamples_ = (partialFrameSamples_ + count) % kEnergyFrameSamples;
        state_ = {detector_.hasActiveSpeech(), detector_.hasEnoughSpeech(), !ended.empty(),
                  detector_.trailingSilenceSamples()};
        return state_;
    }
    LiveEndpointState finish() override
    {
        if (partialFrameSamples_) {
            const int16_t silence[kEnergyFrameSamples] = {};
            return append(silence, kEnergyFrameSamples - partialFrameSamples_);
        }
        return state_;
    }
    void reset(size_t) override { detector_.reset(); state_ = {}; partialFrameSamples_ = 0; }
    std::string name() const override { return "energy"; }
private:
    AudioSegmenter detector_;
    LiveEndpointState state_;
    size_t partialFrameSamples_ = 0;
};

class VadEndpoint final : public ILiveStreamEndpoint {
public:
    VadEndpoint(std::unique_ptr<IVadBackend> backend, int minSpeechMs,
                int maxSegmentMs, int endSilenceMs)
        : backend_(std::move(backend)), minSpeechSamples_(static_cast<size_t>(minSpeechMs) * 16),
          maxHistorySamples_((static_cast<size_t>(maxSegmentMs) + endSilenceMs + 1000) * 16)
    {
        if (!backend_) throw std::invalid_argument("missing stream VAD backend");
        analysisCadenceSamples_ = static_cast<size_t>(std::min(100, endSilenceMs)) * 16;
    }
    LiveEndpointState append(const int16_t* samples, size_t count) override
    {
        buffer_.insert(buffer_.end(), samples, samples + count);
        totalSamples_ += count;
        sinceAnalysis_ += count;
        if (buffer_.size() > maxHistorySamples_) {
            const size_t removed = buffer_.size() - maxHistorySamples_;
            buffer_.erase(buffer_.begin(), buffer_.begin() + removed);
            bufferStart_ += removed;
        }
        state_.ended = false;
        state_.nextSpeechSamples = 0;
        state_.awaitingAnalysis = sinceAnalysis_ < analysisCadenceSamples_;
        if (sinceAnalysis_ >= analysisCadenceSamples_) analyze();
        return state_;
    }
    LiveEndpointState finish() override { analyze(); return state_; }
    void reset(size_t retainSamples) override
    {
        // Keep a second of already-captured lead-in: resetting at an onset
        // otherwise deprives Silero's front-end of its required context.
        retainSamples = std::min(retainSamples, buffer_.size());
        consumed_ = totalSamples_ - retainSamples;
        const size_t kept = retainSamples + std::min(buffer_.size() - retainSamples, size_t(16000));
        buffer_.erase(buffer_.begin(), buffer_.end() - kept);
        bufferStart_ = totalSamples_ - kept;
        sinceAnalysis_ = 0;
        state_ = {};
    }
    std::string name() const override { return backend_->name(); }
private:
    void analyze()
    {
        sinceAnalysis_ = 0;
        state_ = {};
        if (buffer_.empty()) return;
        const auto intervals = backend_->detect(buffer_.data(), buffer_.size());
        for (const auto& interval : intervals) {
            const size_t begin = std::max(consumed_, bufferStart_ + std::min(interval.beginSample, buffer_.size()));
            const size_t end = bufferStart_ + std::min(interval.endSample, buffer_.size());
            if (end <= begin) continue;
            if (end == totalSamples_) {
                const size_t quiet = std::min(end - begin, interval.trailingSilenceSamples.value_or(0));
                // Context retained across a reset may contain only old speech and new silence.
                if (quiet == end - begin) continue;
                state_.active = true;
                if (state_.ended) state_.nextSpeechSamples = end - begin;
                state_.trailingSilenceSamples = interval.trailingSilenceSamples;
                state_.enoughSpeech = end - begin - quiet >= minSpeechSamples_;
            } else if (end - begin >= minSpeechSamples_) {
                state_.ended = true;
            }
        }
    }
    std::unique_ptr<IVadBackend> backend_;
    size_t minSpeechSamples_;
    size_t maxHistorySamples_;
    size_t totalSamples_ = 0;
    size_t bufferStart_ = 0;
    size_t consumed_ = 0;
    size_t sinceAnalysis_ = 0;
    size_t analysisCadenceSamples_ = 1600;
    std::vector<int16_t> buffer_;
    LiveEndpointState state_;
};

} // namespace

std::unique_ptr<ILiveStreamEndpoint> makeEnergyStreamEndpoint(int minSpeechMs, int endSilenceMs)
{
    if (minSpeechMs <= 0 || endSilenceMs <= 0) {
        throw std::invalid_argument("stream VAD durations must be positive");
    }
    return std::make_unique<EnergyEndpoint>(minSpeechMs, endSilenceMs);
}

std::unique_ptr<ILiveStreamEndpoint> makeVadStreamEndpoint(
    std::unique_ptr<IVadBackend> backend, int minSpeechMs, int maxSegmentMs, int endSilenceMs)
{
    if (minSpeechMs <= 0 || maxSegmentMs <= 0 || endSilenceMs <= 0) {
        throw std::invalid_argument("stream VAD durations must be positive");
    }
    return std::make_unique<VadEndpoint>(std::move(backend), minSpeechMs, maxSegmentMs, endSilenceMs);
}

std::unique_ptr<ILiveStreamEndpoint> makeLiveStreamEndpoint(const Config& config, int maxSegmentMs)
{
    if (config.vadBackend == "energy") {
        return makeEnergyStreamEndpoint(config.vadMinSpeechMs, config.vadMinSilenceMs);
    }
    if (config.vadBackend == "silero") {
        return makeVadStreamEndpoint(std::make_unique<SileroVadBackend>(
            config.vadModelPath, config.vadThreshold, config.vadEndThreshold,
            config.vadMinSpeechMs, config.vadMinSilenceMs),
            config.vadMinSpeechMs, maxSegmentMs, config.vadMinSilenceMs);
    }
    throw std::invalid_argument("unsupported VAD backend: " + config.vadBackend);
}

} // namespace echoflow

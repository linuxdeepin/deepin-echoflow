// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "LiveStreamTranscriber.h"

#include "Config.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace echoflow {
namespace {

constexpr size_t kFrameSamples = 320; // 20 ms, 16 kHz mono.
constexpr size_t kMinTailFrameSamples = 80; // Include partial STOP frames of at least 5 ms.
constexpr double kMinTailPower = 40.0 * 40.0;
constexpr size_t kPreferredPauseWindowSamples = 800 * 16;
constexpr size_t kMutablePreviewTailCharacters = 4;

bool rewritesPreviewPrefix(const std::string& displayed, const std::string& next)
{
    const auto isContinuation = [](unsigned char byte) { return (byte & 0xc0) == 0x80; };
    auto first = std::mismatch(displayed.begin(), displayed.end(), next.begin(), next.end()).first;
    if (first == displayed.end()) return false;
    // The first differing byte may lie inside a Chinese character or another UTF-8 code point.
    while (first != displayed.begin() && isContinuation(*first)) --first;
    size_t characters = 0;
    for (; first != displayed.end(); ++first) {
        if (!isContinuation(*first) && ++characters > kMutablePreviewTailCharacters) return true;
    }
    return false;
}

double acPower(const int16_t* samples, size_t count)
{
    double sum = 0;
    double squareSum = 0;
    for (size_t i = 0; i < count; ++i) {
        const double sample = samples[i];
        sum += sample;
        squareSum += sample * sample;
    }
    const double mean = sum / count;
    return squareSum / count - mean * mean;
}

std::string joinDisjointText(const std::string& left, const std::string& right)
{
    if (left.empty()) return right;
    if (right.empty()) return left;
    // No audio overlaps across streams: identical words may be intentional.
    const auto characterAt = [](const std::string& text, size_t offset) {
        const auto lead = static_cast<unsigned char>(text[offset]);
        if (lead < 0x80) return static_cast<uint32_t>(lead);
        const size_t width = lead < 0xe0 ? 2 : (lead < 0xf0 ? 3 : 4);
        uint32_t character = lead & (0x7f >> width);
        for (size_t i = 1; i < width && offset + i < text.size(); ++i) {
            character = (character << 6) | (static_cast<unsigned char>(text[offset + i]) & 0x3f);
        }
        return character;
    };
    size_t lastOffset = left.size() - 1;
    while (lastOffset && (static_cast<unsigned char>(left[lastOffset]) & 0xc0) == 0x80) {
        --lastOffset;
    }
    const auto isCjk = [](uint32_t c) {
        return (c >= 0x2e80 && c <= 0x9fff) || (c >= 0xac00 && c <= 0xd7af)
            || (c >= 0xf900 && c <= 0xfaff) || (c >= 0xff00 && c <= 0xffef)
            || (c >= 0x20000 && c <= 0x3134f);
    };
    const auto isSpace = [](uint32_t c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0xa0 || c == 0x3000;
    };
    const uint32_t last = characterAt(left, lastOffset);
    const uint32_t first = characterAt(right, 0);
    const bool punctuation = first < 128 && std::string(".,!?;:)]}").find(
        static_cast<char>(first)) != std::string::npos;
    const bool space = !isCjk(last) && !isCjk(first) && !isSpace(last) && !isSpace(first)
        && !punctuation;
    return left + (space ? " " : "") + right;
}

} // namespace

LiveStreamConfig makeLiveStreamConfig(const Config& config)
{
    LiveStreamConfig stream;
    stream.minSpeechMs = config.vadMinSpeechMs;
    stream.endSilenceMs = config.vadMinSilenceMs;
    if (config.vadBackend == "silero") stream.prePaddingMs = 1000;
    if (stream.minSpeechMs <= 0 || stream.endSilenceMs <= 0
        || stream.minSpeechMs > std::numeric_limits<int>::max() - stream.prePaddingMs
        || config.streamStepMs < 100 || config.streamStepMs > 4000) {
        throw std::invalid_argument("invalid live stream durations");
    }
    stream.maxSegmentMs = std::max({stream.maxSegmentMs, config.streamStepMs + 200,
                                   stream.minSpeechMs + stream.prePaddingMs});
    return stream;
}

LiveStreamTranscriber::LiveStreamTranscriber(LiveStreamConfig config,
                                             LiveStreamDecoder decoder,
                                             Publish publish, Publish confirmed,
                                             std::unique_ptr<ILiveStreamEndpoint> endpoint)
    : decoder_(std::move(decoder)), publish_(std::move(publish)),
      confirmed_(std::move(confirmed)),
      endpoint_(endpoint ? std::move(endpoint)
                         : makeEnergyStreamEndpoint(config.minSpeechMs, config.endSilenceMs)),
      endSilenceSamples_(static_cast<size_t>(config.endSilenceMs) * 16),
      shortPauseSamples_(static_cast<size_t>(std::min(config.shortPauseMs, config.endSilenceMs)) * 16),
      maxSegmentSamples_(static_cast<size_t>(config.maxSegmentMs) * 16),
      prePaddingSamples_(static_cast<size_t>(config.prePaddingMs) * 16)
{
    if (config.maxSegmentMs < 20 || config.endSilenceMs <= 0
        || config.minSpeechMs <= 0 || config.shortPauseMs <= 0 || config.prePaddingMs < 0
        || config.prePaddingMs >= config.maxSegmentMs
        || config.minSpeechMs > config.maxSegmentMs - config.prePaddingMs
        || !decoder_.open || !decoder_.feed || !decoder_.flush || !decoder_.close) {
        throw std::invalid_argument("invalid live stream configuration");
    }
    frame_.reserve(kFrameSamples);
}

void LiveStreamTranscriber::append(const int16_t* samples, size_t count)
{
    if (finished_ || !samples || count == 0) return;
    while (count) {
        const size_t take = std::min(count, kFrameSamples - frame_.size());
        frame_.insert(frame_.end(), samples, samples + take);
        samples += take;
        count -= take;
        if (frame_.size() == kFrameSamples) {
            processFrame(frame_.data(), frame_.size());
            frame_.clear();
        }
    }
}

void LiveStreamTranscriber::openStream()
{
    decoder_.open();
    streamOpen_ = true;
    deferredPreviewRewrite_ = false;
    pauseCheckpointed_ = false;
    if (decoder_.setSilence) decoder_.setSilence(decoderSilence_);
}

void LiveStreamTranscriber::setDecoderSilence(bool quiet)
{
    decoderSilence_ = quiet;
    if (decoder_.setSilence) decoder_.setSilence(quiet);
}

void LiveStreamTranscriber::processFrame(const int16_t* samples, size_t count)
{
    pendingAudio_.insert(pendingAudio_.end(), samples, samples + count);
    const auto state = endpoint_->append(samples, count);
    if (!state.awaitingAnalysis) processAnalyzedAudio(state);
}

void LiveStreamTranscriber::processAnalyzedAudio(const LiveEndpointState& state, bool finishing)
{
    const size_t nextSpeech = std::min(state.nextSpeechSamples, pendingAudio_.size());
    if (nextSpeech) {
        const size_t previousSamples = pendingAudio_.size() - nextSpeech;
        processAudio(pendingAudio_.data(), previousSamples, {false, false, true, {}}, false, false);
        endpoint_->reset(nextSpeech);
        auto nextState = state;
        nextState.ended = false;
        nextState.nextSpeechSamples = 0;
        processAudio(pendingAudio_.data() + previousSamples, nextSpeech, nextState, !finishing, true);
    } else {
        processAudio(pendingAudio_.data(), pendingAudio_.size(), state, !finishing, true);
    }
    pendingAudio_.clear();
}

void LiveStreamTranscriber::processAudio(const int16_t* samples, size_t count,
                                         const LiveEndpointState& endpointState,
                                         bool allowCheckpoint, bool resetEndpoint)
{
    endpointState_ = endpointState;
    const bool speechActive = endpointState.active;
    const size_t trailingSilence = endpointState.trailingSilenceSamples.value_or(0);
    const bool quiet = !speechActive || endpointState.ended || trailingSilence > 0;
    setDecoderSilence(quiet);
    const bool validSpeech = continuation_ || endpointState.ended || endpointState.enoughSpeech;
    if (!streamOpen_) {
        if (!speechActive && !endpointState.ended) {
            if (continuation_) {
                continuationSilence_ += count;
                if (continuationSilence_ >= endSilenceSamples_) continuation_ = false;
            }
            leadIn_.insert(leadIn_.end(), samples, samples + count);
            if (leadIn_.size() > prePaddingSamples_) {
                leadIn_.erase(leadIn_.begin(), leadIn_.end() - prePaddingSamples_);
            }
            return;
        }
        continuationSilence_ = 0;
        openStream();
        if (endpointState.ended && decoder_.deferPreview) decoder_.deferPreview();
        if (!leadIn_.empty()) feed(leadIn_.data(), leadIn_.size(), validSpeech);
        leadIn_.clear();
    }
    // Batched VAD can first observe resumed speech after another short gap.
    // A quiet tail shorter than withheld audio means that buffer includes voice.
    const bool withheldVoice = pauseCheckpointed_ && speechActive && !endpointState.ended
        && endpointState.trailingSilenceSamples
        && trailingSilence < pausedAudio_.size() + count;
    if (!quiet || withheldVoice) {
        if (pauseCheckpointed_) replayPausedAudio(pausedAudio_.size(), validSpeech);
        else pauseNoisePower_.reset();
    }

    if (pauseCheckpointed_) {
        bufferPausedAudio(samples, count);
    } else if (quiet && trailingSilence > 0 && count > 0) {
        const size_t noiseSamples = std::min({count, trailingSilence, kFrameSamples});
        const double power = acPower(samples + count - noiseSamples, noiseSamples);
        pauseNoisePower_ = std::min(pauseNoisePower_.value_or(power), power);
    }

    const size_t verifiedSilence = pausedTailSamples_ > 0
        ? std::min(trailingSilence, pausedAudio_.size() - pausedTailSamples_) : trailingSilence;
    const size_t pauseWindow = std::min(kPreferredPauseWindowSamples, maxSegmentSamples_ / 10);
    // Only cut ahead of the limit when the current batch ends entirely in the pause.
    // An analysis that straddles the limit can then drop only excess quiet samples.
    const bool preferPause = validSpeech && endpointState_.active && !endpointState.ended
        && verifiedSilence >= std::max(shortPauseSamples_, count)
        && segmentSamples_ + (pauseCheckpointed_ ? pausedAudio_.size() : count)
            >= maxSegmentSamples_ - pauseWindow;
    const bool closing = endpointState.ended || !speechActive || preferPause;
    if (closing && decoder_.deferPreview) decoder_.deferPreview();

    if (pauseCheckpointed_) {
        // Keep suspicious audio until confirmation, unless waiting would exceed the cap.
        const size_t remaining = maxSegmentSamples_ - segmentSamples_;
        if (!closing && pausedTailSamples_ > 0
            && pausedAudio_.size() >= remaining) {
            replayPausedAudio(std::max(remaining, pausedTailSamples_), validSpeech);
        }
    } else {
        // A gap can have hit the hard segment boundary before new speech.
        if (!streamOpen_) {
            openStream();
        }
        feed(samples, preferPause ? std::min(count, maxSegmentSamples_ - segmentSamples_) : count,
             validSpeech);
    }
    if (closing) {
        if (endpointState.ended || continuation_ || preferPause) {
            finalizeSegment();
        } else {
            // The detector discarded a burst shorter than minSpeechMs.
            discardSegment();
        }
        continuation_ = false;
        if (resetEndpoint) endpoint_->reset();
        endpointState_ = {};
    } else if (allowCheckpoint && streamOpen_ && quiet && validSpeech && endpointState_.active
               && !pauseCheckpointed_ && decoder_.checkpoint && trailingSilence >= shortPauseSamples_) {
        if (auto text = decoder_.checkpoint()) {
            pauseCheckpointed_ = true;
            updateActivePreview(std::move(*text), true);
        }
    }
}

void LiveStreamTranscriber::bufferPausedAudio(const int16_t* samples, size_t count)
{
    // This is only a second look at audio after a checkpoint, not a new VAD onset.
    const double threshold = std::max(kMinTailPower, 4 * pauseNoisePower_.value_or(0)); // Twice the noise RMS.
    for (size_t offset = 0; offset < count; offset += kFrameSamples) {
        const size_t length = std::min(kFrameSamples, count - offset);
        if (length >= kMinTailFrameSamples && acPower(samples + offset, length) >= threshold) {
            pausedTailSamples_ = pausedAudio_.size() + offset + length;
        }
    }
    pausedAudio_.insert(pausedAudio_.end(), samples, samples + count);
    const size_t lookback = std::max(prePaddingSamples_, kFrameSamples);
    if (pausedTailSamples_ == 0 && pausedAudio_.size() > lookback) {
        pausedAudio_.erase(pausedAudio_.begin(), pausedAudio_.end() - lookback);
    }
}

void LiveStreamTranscriber::replayPausedAudio(size_t count, bool validSpeech)
{
    // feed() can close and reopen the stream, so it cannot use the member buffer.
    auto audio = std::move(pausedAudio_);
    clearPause();
    if (count == 0) return;
    const bool quiet = decoderSilence_;
    setDecoderSilence(true);
    feed(audio.data(), count, validSpeech);
    setDecoderSilence(quiet);
}

void LiveStreamTranscriber::clearPause()
{
    pauseCheckpointed_ = false;
    pausedAudio_.clear();
    pausedTailSamples_ = 0;
    pauseNoisePower_.reset();
}

void LiveStreamTranscriber::feed(const int16_t* samples, size_t count, bool validSpeech)
{
    while (count) {
        const size_t take = std::min(count, maxSegmentSamples_ - segmentSamples_);
        if (segmentSamples_ + take == maxSegmentSamples_ && decoder_.deferPreview) decoder_.deferPreview();
        auto preview = decoder_.feed(samples, take);
        segmentSamples_ += take;
        samples += take;
        count -= take;
        if (preview) updateActivePreview(std::move(*preview));
        if (segmentSamples_ == maxSegmentSamples_) {
            const bool valid = continuation_ || validSpeech;
            if (valid) confirmSegment();
            else discardSegment();
            continuation_ = valid;
            continuationSilence_ = 0;
            endpoint_->reset();
            endpointState_ = {};
            if (count) {
                openStream();
            }
        }
    }
}

void LiveStreamTranscriber::finalizeSegment()
{
    if (pausedTailSamples_ > 0) replayPausedAudio(pausedTailSamples_, true);
    confirmSegment();
}

void LiveStreamTranscriber::confirmSegment()
{
    if (!streamOpen_) return;
    if (!pauseCheckpointed_) activeText_ = decoder_.flush();
    confirmedText_ = combinedText();
    activeText_.clear();
    decoder_.close();
    streamOpen_ = false;
    segmentSamples_ = 0;
    clearPause();
    ++confirmedSegments_;
    if (confirmed_) confirmed_(confirmedText_);
    publishText();
}

void LiveStreamTranscriber::discardSegment()
{
    if (!streamOpen_) return;
    decoder_.close();
    streamOpen_ = false;
    segmentSamples_ = 0;
    clearPause();
    activeText_.clear();
    publishText();
}

void LiveStreamTranscriber::updateActivePreview(std::string text, bool force)
{
    // Give disruptive live revisions one more decode to settle. Never constrain
    // the model with displayed text or delay checkpoint/final results.
    if (!force && !deferredPreviewRewrite_ && rewritesPreviewPrefix(activeText_, text)) {
        deferredPreviewRewrite_ = true;
        return;
    }
    deferredPreviewRewrite_ = false;
    activeText_ = std::move(text);
    publishText();
}

std::string LiveStreamTranscriber::combinedText() const
{
    return joinDisjointText(confirmedText_, activeText_);
}

void LiveStreamTranscriber::publishText()
{
    const std::string text = combinedText();
    if (text == publishedText_) return;
    publishedText_ = text;
    if (publish_) publish_(text);
}

std::string LiveStreamTranscriber::finish()
{
    if (finished_) return confirmedText_;
    if (!frame_.empty()) {
        endpoint_->append(frame_.data(), frame_.size());
        pendingAudio_.insert(pendingAudio_.end(), frame_.begin(), frame_.end());
    }
    frame_.clear();
    // A continuation may be shorter than the minimum at a forced boundary;
    // it still belongs to speech validated in the preceding stream.
    endpointState_ = endpoint_->finish();
    if (!pendingAudio_.empty()) {
        const auto finalState = endpointState_;
        processAnalyzedAudio(finalState, true);
    }
    if (!streamOpen_ && (endpointState_.active || endpointState_.ended) && !leadIn_.empty()) {
        // A batched VAD may first report a short utterance at final analysis.
        openStream();
        feed(leadIn_.data(), leadIn_.size(), endpointState_.enoughSpeech || endpointState_.ended);
    }
    if (continuation_ || endpointState_.enoughSpeech || endpointState_.ended) finalizeSegment();
    else discardSegment();
    continuation_ = false;
    continuationSilence_ = 0;
    endpoint_->reset();
    endpointState_ = {};
    leadIn_.clear();
    finished_ = true;
    return confirmedText_;
}

void LiveStreamTranscriber::cancel()
{
    finished_ = true;
    if (streamOpen_) decoder_.close();
    streamOpen_ = false;
    continuation_ = false;
    continuationSilence_ = 0;
    segmentSamples_ = 0;
    clearPause();
    confirmedSegments_ = 0;
    frame_.clear();
    pendingAudio_.clear();
    leadIn_.clear();
    activeText_.clear();
    confirmedText_.clear();
    publishedText_.clear();
    endpoint_->reset();
    endpointState_ = {};
}

} // namespace echoflow

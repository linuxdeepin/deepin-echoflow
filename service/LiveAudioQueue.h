// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_LIVE_AUDIO_QUEUE_H
#define ECHOFLOW_LIVE_AUDIO_QUEUE_H

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <vector>

namespace echoflow {

// Caller serializes access. Backlog beyond the RAM budget spills to an
// anonymous temporary file, preserving every sample and its original order.
class LiveAudioQueue {
public:
    using Clock = std::chrono::steady_clock;
    explicit LiveAudioQueue(size_t memorySamples = 16000 * 5);
    void append(std::vector<int16_t> samples, Clock::time_point enqueued = Clock::now());
    // Age of the oldest sample in the returned packet, including time in the
    // disk spool. An empty queue reports zero. Caller serializes access.
    std::vector<int16_t> take(std::chrono::milliseconds* waited = nullptr,
                              Clock::time_point now = Clock::now());
    void clear();
    size_t size() const { return queuedSamples_; }
    bool empty() const { return queuedSamples_ == 0; }
    bool spilling() const { return static_cast<bool>(spill_); }

private:
    const size_t memoryLimit_;
    size_t memorySamples_ = 0;
    size_t queuedSamples_ = 0;
    struct Packet {
        std::vector<int16_t> samples;
        Clock::time_point enqueued;
    };
    std::deque<Packet> memory_;
    struct CloseFile {
        void operator()(std::FILE* file) const { std::fclose(file); }
    };
    std::unique_ptr<std::FILE, CloseFile> spill_;
    std::uint64_t spillRead_ = 0;
    std::uint64_t spillWrite_ = 0;
    size_t spillRecordRemaining_ = 0;
    Clock::time_point spillRecordTime_{};
};

} // namespace echoflow

#endif

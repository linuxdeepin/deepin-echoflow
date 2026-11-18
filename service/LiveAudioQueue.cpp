// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "LiveAudioQueue.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace echoflow {
namespace {
void transfer(int fd, void* buffer, size_t bytes, std::uint64_t offset, bool write)
{
    auto* next = static_cast<char*>(buffer);
    while (bytes) {
        const auto count = write ? pwrite(fd, next, bytes, static_cast<off_t>(offset))
                                 : pread(fd, next, bytes, static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) throw std::system_error(errno, std::generic_category(), "live audio spool");
        if (count == 0) throw std::runtime_error("incomplete live audio spool transfer");
        bytes -= static_cast<size_t>(count);
        offset += static_cast<size_t>(count);
        next += count;
    }
}
}

LiveAudioQueue::LiveAudioQueue(size_t memorySamples) : memoryLimit_(memorySamples) {}

void LiveAudioQueue::append(std::vector<int16_t> samples, Clock::time_point enqueued)
{
    if (samples.empty()) return;
    const size_t count = samples.size();
    if (!spill_ && count <= memoryLimit_ - memorySamples_) {
        memory_.push_back({std::move(samples), enqueued});
        memorySamples_ += count;
    } else {
        if (!spill_) {
            spill_.reset(std::tmpfile());
            if (!spill_) throw std::system_error(errno, std::generic_category(), "live audio spool open");
        }
        // Metadata spills with PCM so diagnostics do not create an unbounded
        // in-memory timestamp queue when recognition falls behind.
        std::array<std::int64_t, 2> header{
            static_cast<std::int64_t>(count),
            std::chrono::duration_cast<std::chrono::nanoseconds>(enqueued.time_since_epoch()).count()};
        transfer(fileno(spill_.get()), header.data(), sizeof(header), spillWrite_, true);
        spillWrite_ += sizeof(header);
        const auto bytes = count * sizeof(int16_t);
        transfer(fileno(spill_.get()), samples.data(), bytes, spillWrite_, true);
        spillWrite_ += bytes;
    }
    queuedSamples_ += count;
}

std::vector<int16_t> LiveAudioQueue::take(std::chrono::milliseconds* waited, Clock::time_point now)
{
    const auto reportAge = [&](Clock::time_point enqueued) {
        if (waited) *waited = std::max(std::chrono::milliseconds::zero(),
            std::chrono::duration_cast<std::chrono::milliseconds>(now - enqueued));
    };
    if (waited) *waited = std::chrono::milliseconds::zero();
    if (!memory_.empty()) {
        auto packet = std::move(memory_.front());
        memory_.pop_front();
        memorySamples_ -= packet.samples.size();
        queuedSamples_ -= packet.samples.size();
        reportAge(packet.enqueued);
        return std::move(packet.samples);
    }
    if (empty()) return {};
    const size_t count = std::min<size_t>(3200, queuedSamples_);
    std::vector<int16_t> result(count);
    size_t offset = 0;
    while (offset < count) {
        if (!spillRecordRemaining_) {
            std::array<std::int64_t, 2> header{};
            transfer(fileno(spill_.get()), header.data(), sizeof(header), spillRead_, false);
            spillRead_ += sizeof(header);
            if (header[0] <= 0 || static_cast<std::uint64_t>(header[0]) > queuedSamples_) {
                throw std::runtime_error("invalid live audio spool record");
            }
            spillRecordRemaining_ = static_cast<size_t>(header[0]);
            spillRecordTime_ = Clock::time_point(std::chrono::duration_cast<Clock::duration>(
                std::chrono::nanoseconds(header[1])));
        }
        if (offset == 0) reportAge(spillRecordTime_);
        const size_t take = std::min(count - offset, spillRecordRemaining_);
        const size_t bytes = take * sizeof(int16_t);
        transfer(fileno(spill_.get()), result.data() + offset, bytes, spillRead_, false);
        spillRead_ += bytes;
        spillRecordRemaining_ -= take;
        offset += take;
        queuedSamples_ -= take;
    }
    if (spillRead_ == spillWrite_) {
        spill_.reset();
        spillRead_ = spillWrite_ = 0;
        spillRecordRemaining_ = 0;
    }
    return result;
}

void LiveAudioQueue::clear()
{
    memory_.clear();
    spill_.reset();
    memorySamples_ = queuedSamples_ = 0;
    spillRead_ = spillWrite_ = 0;
    spillRecordRemaining_ = 0;
    spillRecordTime_ = {};
}

} // namespace echoflow

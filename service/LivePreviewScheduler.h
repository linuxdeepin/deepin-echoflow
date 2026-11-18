// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_LIVE_PREVIEW_SCHEDULER_H
#define ECHOFLOW_LIVE_PREVIEW_SCHEDULER_H

#include <algorithm>
#include <chrono>
#include <cstddef>

namespace echoflow {

// Admission controls interim inference only; every sample still reaches the
// decoder. Its live-decode switch coalesces pending audio into one update.
class LivePreviewScheduler {
public:
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::milliseconds;
    explicit LivePreviewScheduler(int stepMs, Clock::time_point now = Clock::now())
        : stepMs_(std::clamp(stepMs, 100, 4000)), completed_(now) {}

    void beginSegment() { pendingSamples_ = 0; }

    bool admit(size_t samples, bool backlog, bool finishing, Clock::time_point now)
    {
        pendingSamples_ += samples;
        if (finishing) return false;
        const auto interval = intervalMs();
        if (pendingSamples_ < static_cast<size_t>(interval) * 16) return false;
        // Leave CPU time between expensive previews. Under backlog, periodically
        // admit one update rather than blacking out until the segment boundary.
        const auto rest = std::max(0LL, interval - costMs_);
        const auto wait = backlog ? std::max(1500LL, rest) : rest;
        if (now - completed_ < Milliseconds(wait)) return false;
        // Keep the budget until a decode really ran: R2T2 may still be
        // collecting its first 160 ms of lookahead after admission.
        return true;
    }

    void decoded(Milliseconds duration, Clock::time_point completed)
    {
        // React immediately to rising cost and recover gradually after it falls.
        costMs_ = std::max(static_cast<long long>(duration.count()), costMs_ * 3 / 4);
        completed_ = completed;
        pendingSamples_ = 0;
    }

    long long intervalMs() const
    {
        return std::max(static_cast<long long>(stepMs_), std::min(2000LL, costMs_ * 5 / 4));
    }

private:
    int stepMs_;
    long long costMs_ = 0;
    size_t pendingSamples_ = 0;
    Clock::time_point completed_;
};

} // namespace echoflow

#endif

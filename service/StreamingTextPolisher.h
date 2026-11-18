// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_STREAMING_TEXT_POLISHER_H
#define ECHOFLOW_STREAMING_TEXT_POLISHER_H

#include "Interfaces.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace echoflow {

// The polisher must outlive this worker. Updates carry a session token and
// raw-prefix snapshot; consumers must reject stale updates. finish() freezes
// the final result and invalidates pending model work.
class StreamingTextPolisher {
public:
    using Update = std::function<void(std::uint64_t, const std::string&, const std::string&)>;
    explicit StreamingTextPolisher(ITextPolisher& polisher, Update update = {});
    ~StreamingTextPolisher();
    std::uint64_t start();
    void appendConfirmed(const std::string& cumulativeText);
    std::string finish(const std::string& finalText,
                       std::chrono::milliseconds wait = std::chrono::milliseconds(1500));
    void cancel();
    void setRecognitionBusy(bool busy);

private:
    struct Piece {
        std::string raw;
        std::string result;
        bool done = false;
    };
    struct Session {
        std::uint64_t id = 0;
        std::atomic<bool> abort{false};
        std::string raw;
        std::vector<Piece> pieces;
        bool sealed = false;
    };
    struct Job {
        std::shared_ptr<Session> session;
        size_t index;
    };

    void appendLocked(const std::string& cumulativeText);
    void run();
    static std::string render(const Session& session);

    ITextPolisher& polisher_;
    Update update_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::shared_ptr<Session> session_;
    std::deque<Job> jobs_;
    bool stopping_ = false;
    bool recognitionBusy_ = false;
    // Epoch invalidates an in-flight result even if recognition became idle
    // before the model next polled its cancellation callback.
    std::atomic<std::uint64_t> recognitionEpoch_{0};
    std::uint64_t nextSession_ = 0;
    std::thread worker_;
};

} // namespace echoflow

#endif

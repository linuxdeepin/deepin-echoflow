// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "StreamingTextPolisher.h"

#include "PolishPrompt.h"
#include "log.h"

#include <algorithm>
#include <exception>

namespace echoflow {
namespace {
constexpr size_t kMaxPendingPieces = 8;
constexpr auto kPieceTimeout = std::chrono::seconds(5);
}

StreamingTextPolisher::StreamingTextPolisher(ITextPolisher& polisher, Update update)
    : polisher_(polisher), update_(std::move(update)),
      worker_(&StreamingTextPolisher::run, this)
{
}

StreamingTextPolisher::~StreamingTextPolisher()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        if (session_) session_->abort = true;
        jobs_.clear();
    }
    ready_.notify_all();
    worker_.join();
}

std::uint64_t StreamingTextPolisher::start()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (session_) session_->abort = true;
    jobs_.clear();
    session_ = std::make_shared<Session>();
    session_->id = ++nextSession_;
    ready_.notify_all();
    return session_->id;
}

void StreamingTextPolisher::appendLocked(const std::string& text)
{
    if (!session_ || session_->sealed || session_->abort || text == session_->raw) return;
    if (text.rfind(session_->raw, 0) != 0) {
        // A changed confirmed prefix cannot safely be stitched to cached edits.
        session_->abort = true;
        jobs_.clear();
        return;
    }
    const auto pieces = splitTranscriptForPolish(text.substr(session_->raw.size()), kPolishChunkMaxChars);
    session_->raw = text;
    for (const auto& raw : pieces) {
        const size_t index = session_->pieces.size();
        const bool queued = jobs_.size() < kMaxPendingPieces;
        session_->pieces.push_back({raw, raw, !queued});
        if (queued) jobs_.push_back({session_, index});
    }
    ready_.notify_all();
}

void StreamingTextPolisher::appendConfirmed(const std::string& text)
{
    std::lock_guard<std::mutex> lock(mutex_);
    appendLocked(text);
}

std::string StreamingTextPolisher::render(const Session& session)
{
    std::string result;
    for (const auto& piece : session.pieces) result += piece.done ? piece.result : piece.raw;
    return result;
}

std::string StreamingTextPolisher::finish(const std::string& text, std::chrono::milliseconds wait)
{
    const auto started = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    if (!session_ || session_->abort || session_->sealed) return text;
    appendLocked(text);
    const auto session = session_;
    if (session->abort) return text;
    session->sealed = true;
    ready_.wait_until(lock, started + wait, [&] {
        return session->abort || std::all_of(session->pieces.begin(), session->pieces.end(),
                                             [](const Piece& piece) { return piece.done; });
    });
    if (session->abort) return text;
    const std::string result = render(*session);
    const size_t unfinished = static_cast<size_t>(std::count_if(
        session->pieces.begin(), session->pieces.end(), [](const Piece& piece) { return !piece.done; }));
    session->abort = true;
    jobs_.clear();
    lock.unlock();
    ready_.notify_all();
    log("stream polish finish: wait_ms=" + std::to_string(std::chrono::duration_cast<
        std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count())
        + ", unfinished_pieces=" + std::to_string(unfinished));
    return result;
}

void StreamingTextPolisher::cancel()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (session_) session_->abort = true;
        jobs_.clear();
    }
    ready_.notify_all();
}

void StreamingTextPolisher::setRecognitionBusy(bool busy)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (recognitionBusy_ == busy) return;
        recognitionBusy_ = busy;
        if (busy) ++recognitionEpoch_;
    }
    ready_.notify_all();
}

void StreamingTextPolisher::run()
{
    while (true) {
        Job job;
        std::string raw;
        std::uint64_t epoch;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stopping_ || (!recognitionBusy_ && !jobs_.empty()); });
            if (stopping_) return;
            job = std::move(jobs_.front());
            jobs_.pop_front();
            raw = job.session->pieces[job.index].raw;
            epoch = recognitionEpoch_.load();
        }
        if (job.session->abort) continue;
        std::string result = raw;
        const auto deadline = std::chrono::steady_clock::now() + kPieceTimeout;
        const auto aborted = [this, session = job.session, deadline, epoch] {
            return session->abort || recognitionEpoch_.load() != epoch
                || std::chrono::steady_clock::now() >= deadline;
        };
        const size_t first = raw.find_first_not_of(" \t\r\n");
        const size_t last = raw.find_last_not_of(" \t\r\n");
        try {
            if (first != std::string::npos && !aborted()) {
                const auto polished = polisher_.polish(raw.substr(first, last - first + 1), aborted);
                if (!polished.empty() && !aborted()) {
                    result = raw.substr(0, first) + polished + raw.substr(last + 1);
                }
            }
        } catch (const std::exception& error) {
            log(std::string("stream polish fallback: ") + error.what());
        } catch (...) {
            log("stream polish fallback: unknown error");
        }
        std::string rawPrefix;
        std::string polishedPrefix;
        bool changed = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!job.session->abort && recognitionEpoch_.load() != epoch) {
                // A temporary ASR interruption leaves this piece pending.
                if (jobs_.size() == kMaxPendingPieces) {
                    const auto overflow = jobs_.back();
                    overflow.session->pieces[overflow.index].done = true;
                    jobs_.pop_back();
                }
                jobs_.push_front(job);
            } else if (!job.session->abort) {
                auto& piece = job.session->pieces[job.index];
                changed = result != piece.raw;
                piece.result = std::move(result);
                piece.done = true;
                if (changed && update_) {
                    rawPrefix = job.session->raw;
                    polishedPrefix = render(*job.session);
                }
            }
        }
        ready_.notify_all();
        if (changed && update_ && !job.session->abort) {
            try {
                update_(job.session->id, rawPrefix, polishedPrefix);
            } catch (const std::exception& error) {
                log(std::string("stream polish preview failed: ") + error.what());
            } catch (...) {
                log("stream polish preview failed: unknown error");
            }
        }
    }
}

} // namespace echoflow

#pragma once

#include <cstddef>
#include <algorithm>
#include <vector>

namespace core_realtime {

// /v1/realtime is an explicit-commit protocol. Keep one bounded turn and run
// ASR exactly once when the client commits (or when the safety cap is hit).
// Re-decoding the full growing prefix on every append is quadratic work for
// backends whose streaming cache only lives inside a single inference call.
class TurnBuffer {
public:
    struct AppendResult {
        std::size_t consumed = 0;
        bool full = false;
    };

    explicit TurnBuffer(std::size_t max_samples) : max_samples_(max_samples) {}

    AppendResult append(const float* samples, std::size_t count) {
        if (!samples || count == 0)
            return {};
        if (max_samples_ > 0) {
            const std::size_t room = audio_.size() < max_samples_ ? max_samples_ - audio_.size() : 0;
            count = std::min(count, room);
        }
        audio_.insert(audio_.end(), samples, samples + count);
        return {count, max_samples_ > 0 && audio_.size() >= max_samples_};
    }

    const std::vector<float>& audio() const { return audio_; }
    bool empty() const { return audio_.empty(); }
    std::size_t size() const { return audio_.size(); }
    void clear() { audio_.clear(); }

private:
    std::size_t max_samples_;
    std::vector<float> audio_;
};

} // namespace core_realtime

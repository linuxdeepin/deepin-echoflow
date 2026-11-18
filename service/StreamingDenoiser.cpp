// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "StreamingDenoiser.h"

#include <rnnoise.h>
#include <samplerate.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>

namespace echoflow {
namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t kFrame48 = 480;

void checkSrc(int error)
{
    if (error) throw std::runtime_error(std::string("denoiser resampling: ") + src_strerror(error));
}

std::vector<float> resample(SRC_STATE* state, const float* samples, size_t count,
                            double ratio, bool final)
{
    std::vector<float> result;
    std::array<float, 4096> buffer;
    float dummy = 0;
    do {
        SRC_DATA data {};
        data.data_in = count ? samples : &dummy;
        data.input_frames = static_cast<long>(count);
        data.data_out = buffer.data();
        data.output_frames = static_cast<long>(buffer.size());
        data.src_ratio = ratio;
        data.end_of_input = final ? 1 : 0;
        checkSrc(src_process(state, &data));
        samples = count ? samples + data.input_frames_used : nullptr;
        count -= static_cast<size_t>(data.input_frames_used);
        result.insert(result.end(), buffer.begin(), buffer.begin() + data.output_frames_gen);
        if (!data.input_frames_used && !data.output_frames_gen) break;
        if (!count && data.output_frames_gen < data.output_frames) break;
    } while (true);
    if (count) throw std::runtime_error("denoiser resampler did not consume its input");
    return result;
}

void initializeRnnoise()
{
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        std::unique_ptr<DenoiseState, decltype(&rnnoise_destroy)> state(rnnoise_create(nullptr), rnnoise_destroy);
        if (!state) throw std::runtime_error("cannot initialize RNNoise");
        std::array<float, kFrame48> zero{};
        rnnoise_process_frame(state.get(), zero.data(), zero.data());
    });
}

std::vector<int16_t> toS16(const std::vector<float>& samples)
{
    std::vector<int16_t> result;
    result.reserve(samples.size());
    for (const auto sample : samples) {
        result.push_back(static_cast<int16_t>(std::lround(
            std::clamp(sample * 32768.0f, -32768.0f, 32767.0f))));
    }
    return result;
}
}

struct StreamingDenoiser::Impl {
    std::unique_ptr<DenoiseState, decltype(&rnnoise_destroy)> noise{nullptr, rnnoise_destroy};
    std::unique_ptr<SRC_STATE, decltype(&src_delete)> up{nullptr, src_delete};
    std::unique_ptr<SRC_STATE, decltype(&src_delete)> down{nullptr, src_delete};
    std::vector<float> pending;
    size_t input = 0;
    size_t emitted = 0;
    size_t skip = kFrame48; // RNNoise's overlap/add returns the preceding 10 ms.
    double elapsedMs = 0;
    bool ended = false;

    Impl()
    {
        initializeRnnoise();
        if (rnnoise_get_frame_size() != kFrame48) throw std::runtime_error("unsupported RNNoise frame size");
        noise.reset(rnnoise_create(nullptr));
        if (!noise) throw std::runtime_error("cannot create RNNoise recording state");
        int error = 0;
        up.reset(src_new(SRC_SINC_FASTEST, 1, &error));
        checkSrc(error);
        down.reset(src_new(SRC_SINC_FASTEST, 1, &error));
        checkSrc(error);
        pending.reserve(kFrame48 * 2);
    }

    std::vector<float> process48(const std::vector<float>& samples, bool final)
    {
        pending.insert(pending.end(), samples.begin(), samples.end());
        if (final && !pending.empty()) {
            pending.resize((pending.size() + kFrame48 - 1) / kFrame48 * kFrame48, 0);
        }
        if (final) pending.resize(pending.size() + kFrame48, 0); // Release RNNoise's last frame.
        std::vector<float> cleaned;
        size_t consumed = 0;
        while (pending.size() - consumed >= kFrame48) {
            std::array<float, kFrame48> frame;
            for (size_t i = 0; i < kFrame48; ++i) frame[i] = pending[consumed + i] * 32768.0f;
            rnnoise_process_frame(noise.get(), frame.data(), frame.data());
            const size_t offset = std::min(skip, kFrame48);
            skip -= offset;
            for (size_t i = offset; i < kFrame48; ++i) cleaned.push_back(frame[i] / 32768.0f);
            consumed += kFrame48;
        }
        pending.erase(pending.begin(), pending.begin() + consumed);
        auto result = resample(down.get(), cleaned.data(), cleaned.size(), 1.0 / 3, final);
        if (result.size() > input - emitted) result.resize(input - emitted);
        // Only rounding at resampler boundaries may need padding; audio itself
        // is drained by the final zero frame, rather than discarded at STOP.
        if (final && result.size() < input - emitted) {
            if (input - emitted - result.size() > 1) {
                throw std::runtime_error("denoiser resampler failed to drain the final audio");
            }
            result.resize(input - emitted, 0);
        }
        for (auto& value : result) value = std::isfinite(value) ? std::clamp(value, -1.0f, 1.0f) : 0;
        emitted += result.size();
        return result;
    }
};

StreamingDenoiser::StreamingDenoiser() : impl_(std::make_unique<Impl>()) {}
StreamingDenoiser::~StreamingDenoiser() = default;

std::vector<float> StreamingDenoiser::append(const float* samples, size_t count)
{
    if (impl_->ended) throw std::logic_error("denoiser already finished");
    if (!samples && count) throw std::invalid_argument("missing denoiser samples");
    const auto started = Clock::now();
    impl_->input += count;
    std::vector<float> sanitized(count);
    for (size_t i = 0; i < count; ++i) {
        sanitized[i] = std::isfinite(samples[i]) ? std::clamp(samples[i], -1.0f, 1.0f) : 0;
    }
    const auto up = resample(impl_->up.get(), sanitized.data(), count, 3, false);
    auto result = impl_->process48(up, false);
    impl_->elapsedMs += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    return result;
}

std::vector<int16_t> StreamingDenoiser::append(const int16_t* samples, size_t count)
{
    if (!samples && count) throw std::invalid_argument("missing denoiser samples");
    std::vector<float> pcm(count);
    for (size_t i = 0; i < count; ++i) pcm[i] = samples[i] / 32768.0f;
    return toS16(append(pcm.data(), pcm.size()));
}

std::vector<float> StreamingDenoiser::finish()
{
    if (impl_->ended) return {};
    impl_->ended = true;
    if (impl_->input == 0) return {};
    const auto started = Clock::now();
    const auto up = resample(impl_->up.get(), nullptr, 0, 3, true);
    auto result = impl_->process48(up, true);
    impl_->elapsedMs += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    return result;
}

std::vector<int16_t> StreamingDenoiser::finishS16() { return toS16(finish()); }
double StreamingDenoiser::processingMs() const { return impl_->elapsedMs; }
size_t StreamingDenoiser::inputSamples() const { return impl_->input; }

struct StreamingSpeechDetector::Impl {
    std::unique_ptr<DenoiseState, decltype(&rnnoise_destroy)> state{nullptr, rnnoise_destroy};
    std::unique_ptr<SRC_STATE, decltype(&src_delete)> up{nullptr, src_delete};
    std::vector<float> pending;
    double ratio;

    explicit Impl(int sampleRate) : ratio(48000.0 / sampleRate)
    {
        if (sampleRate < 1000 || sampleRate > 384000) throw std::invalid_argument("unsupported speech detector rate");
        initializeRnnoise();
        state.reset(rnnoise_create(nullptr));
        if (!state) throw std::runtime_error("cannot create waveform speech detector");
        int error = 0;
        up.reset(src_new(SRC_SINC_FASTEST, 1, &error));
        checkSrc(error);
    }
};

StreamingSpeechDetector::StreamingSpeechDetector(int sampleRate)
    : impl_(std::make_unique<Impl>(sampleRate)) {}
StreamingSpeechDetector::~StreamingSpeechDetector() = default;

double StreamingSpeechDetector::probability(const float* mono, size_t count)
{
    if (!mono && count) throw std::invalid_argument("missing speech detector samples");
    const auto audio = resample(impl_->up.get(), mono, count, impl_->ratio, false);
    impl_->pending.insert(impl_->pending.end(), audio.begin(), audio.end());
    double sum = 0;
    size_t frames = 0;
    size_t consumed = 0;
    while (impl_->pending.size() - consumed >= kFrame48) {
        std::array<float, kFrame48> frame;
        for (size_t i = 0; i < frame.size(); ++i) {
            const float sample = impl_->pending[consumed + i];
            frame[i] = std::isfinite(sample) ? std::clamp(sample, -1.0f, 1.0f) * 32768.0f : 0;
        }
        sum += rnnoise_process_frame(impl_->state.get(), frame.data(), frame.data());
        ++frames;
        consumed += kFrame48;
    }
    impl_->pending.erase(impl_->pending.begin(), impl_->pending.begin() + consumed);
    return frames ? sum / frames : 0;
}

} // namespace echoflow

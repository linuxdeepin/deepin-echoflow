// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "AudioLevelMeter.h"
#include "StreamingDenoiser.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <utility>

namespace echoflow {
namespace {
std::uint64_t littleEndian(const unsigned char* bytes, size_t count)
{
    std::uint64_t value = 0;
    for (size_t i = 0; i < count; ++i) value |= std::uint64_t(bytes[i]) << (8 * i);
    return value;
}
}

AudioLevelMeter::AudioLevelMeter(int sampleRate, int channels, Update update, bool speechOnly)
    : update_(std::move(update)), windowSamples_(0), channels_(static_cast<size_t>(channels))
{
    if (sampleRate <= 0 || channels <= 0) throw std::invalid_argument("invalid audio level format");
    windowSamples_ = static_cast<size_t>(std::max(1, sampleRate / 20)) * channels;
    samples_.reserve(windowSamples_);
    spectrum_ = std::make_unique<AudioSpectrum>(sampleRate, windowSamples_ / channels_, channels_);
    if (speechOnly) speechDetector_ = std::make_unique<StreamingSpeechDetector>(sampleRate);
}

AudioLevelMeter::~AudioLevelMeter() = default;

void AudioLevelMeter::append(const int16_t* samples, size_t count)
{
    if (!samples) return;
    for (size_t i = 0; i < count; ++i) appendSample(samples[i] / 32768.0);
}

void AudioLevelMeter::appendSample(double sample)
{
    samples_.push_back(std::isfinite(sample) ? std::clamp(sample, -1.0, 1.0) : 0);
    if (samples_.size() < windowSamples_) return;
    const auto powerFor = [this](size_t begin, size_t end) {
        if (end <= begin) return 0.0;
        std::vector<double> sums(channels_, 0);
        double squares = 0;
        for (size_t i = begin; i < end; ++i) {
            sums[i % channels_] += samples_[i];
            squares += samples_[i] * samples_[i];
        }
        // Remove DC independently in each channel, including each bar's slice.
        const double frames = static_cast<double>(end - begin) / channels_;
        double dcPower = 0;
        for (const double sum : sums) dcPower += (sum / frames) * (sum / frames);
        dcPower /= channels_;
        return std::max(0.0, squares / (end - begin) - dcPower);
    };
    AudioLevelFrame frame;
    constexpr double kNoisePower = 1e-5;
    const double power = powerFor(0, samples_.size());
    const size_t frames = samples_.size() / channels_;
    const auto spectrum = spectrum_->analyze(samples_);
    double displayPower = power;
    if (speechDetector_) {
        std::vector<float> mono(frames, 0);
        for (size_t i = 0; i < samples_.size(); ++i) {
            mono[i / channels_] += static_cast<float>(samples_[i] / channels_);
        }
        const double probability = speechDetector_->probability(mono.data(), mono.size());
        // Combine spectral shape with VAD confidence. Quiet, structured speech
        // can open the display below a strict VAD threshold; stationary broad
        // noise instead teaches the ambient floor, even when its level is high.
        const bool broadbandNoise = spectrum.flatness > 0.35 && probability < 0.95;
        const bool speechEvidence = !broadbandNoise &&
            (probability >= 0.8 || (probability >= 0.35 && spectrum.flatness < 0.2));
        if (!speechEvidence) {
            if (!backgroundKnown_) {
                backgroundPower_ = std::max(power, kNoisePower);
                backgroundKnown_ = true;
            } else {
                const double rate = power < backgroundPower_ ? 0.2 : 0.05;
                backgroundPower_ += rate * (power - backgroundPower_);
                backgroundPower_ = std::max(backgroundPower_, kNoisePower);
            }
        }
        const double threshold = speechEvidence ? kNoisePower
            : std::max(kNoisePower, backgroundPower_ * (speechActive_ ? 2 : 4));
        speechActive_ = !broadbandNoise && power > threshold;
        if (!speechActive_) displayPower = 0;
        else displayPower = speechEvidence ? power : std::max(0.0, power - backgroundPower_);
    }
    frame.level = displayPower <= kNoisePower ? 0.0
        : std::clamp((10.0 * std::log10(displayPower) + 50.0) / 42.0, 0.0, 1.0);
    // Keep one absolute loudness scale for the frame. Band energy distributes
    // this level without stretching each spectrum to full height.
    const double total = std::accumulate(spectrum.power.begin(), spectrum.power.end(), 0.0);
    const double reference = std::max(power, total);
    for (size_t band = 0; band < frame.bars.size(); ++band) {
        const double bandPower = spectrum.power[band];
        frame.bars[band] = reference > 0 && bandPower > 1e-10
            ? frame.level * std::sqrt(bandPower / reference) : 0;
    }
    samples_.clear();
    if (update_) update_(frame);
}

WavAudioLevelReader::WavAudioLevelReader(std::filesystem::path path, bool speechOnly)
    : path_(std::move(path)), speechOnly_(speechOnly)
{
}

bool WavAudioLevelReader::readAt(std::uint64_t offset, void* buffer, size_t count)
{
    input_.clear();
    input_.seekg(static_cast<std::streamoff>(offset));
    input_.read(static_cast<char*>(buffer), static_cast<std::streamsize>(count));
    return static_cast<size_t>(input_.gcount()) == count;
}

bool WavAudioLevelReader::readHeader(std::uint64_t size)
{
    unsigned char header[12];
    if (size < sizeof(header) || !readAt(0, header, sizeof(header))) return false;
    if (std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4)) {
        invalid_ = true;
        return false;
    }
    while (chunkOffset_ + 8 <= size) {
        unsigned char chunk[8];
        if (!readAt(chunkOffset_, chunk, sizeof(chunk))) return false;
        const auto length = littleEndian(chunk + 4, 4);
        if (std::memcmp(chunk, "data", 4) == 0) {
            if (!meter_) { invalid_ = true; return false; }
            sampleOffset_ = chunkOffset_ + 8;
            return true;
        }
        if (chunkOffset_ + 8 + length > size) return false;
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            unsigned char fmt[40] = {};
            const auto count = static_cast<size_t>(std::min<std::uint64_t>(length, sizeof(fmt)));
            if (count < 16 || !readAt(chunkOffset_ + 8, fmt, count)) {
                invalid_ = true;
                return false;
            }
            format_ = static_cast<unsigned>(littleEndian(fmt, 2));
            const auto channels = static_cast<int>(littleEndian(fmt + 2, 2));
            const auto rate = littleEndian(fmt + 4, 4);
            const auto bits = littleEndian(fmt + 14, 2);
            if (format_ == 0xfffe && count >= 40) {
                format_ = static_cast<unsigned>(littleEndian(fmt + 24, 2));
            }
            if (channels <= 0 || channels > 64 || rate == 0 || rate > 768000
                || !((format_ == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32))
                     || (format_ == 3 && (bits == 32 || bits == 64)))
                || littleEndian(fmt + 12, 2) != bits / 8 * channels) {
                invalid_ = true;
                return false;
            }
            bytesPerSample_ = static_cast<unsigned>(bits / 8);
            windowBytes_ = static_cast<std::uint64_t>(std::max<std::uint64_t>(1, rate / 20))
                * channels * bytesPerSample_;
            meter_ = std::make_unique<AudioLevelMeter>(static_cast<int>(rate), channels,
                [this](const AudioLevelFrame& frame) { latest_ = frame; }, speechOnly_);
        }
        chunkOffset_ += 8 + length + (length & 1);
    }
    return false;
}

std::optional<AudioLevelFrame> WavAudioLevelReader::poll()
{
    latest_.reset();
    if (invalid_) return {};
    if (!input_.is_open()) {
        input_.clear();
        input_.open(path_, std::ios::binary);
        if (!input_) return {};
    }
    std::error_code error;
    const auto size = std::filesystem::file_size(path_, error);
    if (error || (!sampleOffset_ && !readHeader(size))) return {};
    std::array<unsigned char, 8192> buffer;
    // Retain at most 250 ms after a delayed poll, publishing only its latest
    // completed window. Skips preserve sample/channel alignment.
    if (size > sampleOffset_ && (size - sampleOffset_) / windowBytes_ > 4) {
        sampleOffset_ += ((size - sampleOffset_) / windowBytes_ - 4) * windowBytes_;
    }
    while (sampleOffset_ + bytesPerSample_ <= size) {
        size_t count = static_cast<size_t>(std::min<std::uint64_t>(buffer.size(), size - sampleOffset_));
        count -= count % bytesPerSample_;
        if (!readAt(sampleOffset_, buffer.data(), count)) break;
        for (size_t i = 0; i < count; i += bytesPerSample_) {
            const auto raw = littleEndian(buffer.data() + i, bytesPerSample_);
            double sample;
            if (format_ == 3 && bytesPerSample_ == 4) {
                const auto word = static_cast<std::uint32_t>(raw);
                float value;
                std::memcpy(&value, &word, sizeof(value));
                sample = value;
            } else if (format_ == 3) {
                std::memcpy(&sample, &raw, sizeof(sample));
            } else if (bytesPerSample_ == 1) {
                sample = (static_cast<int>(raw) - 128) / 128.0;
            } else {
                const auto scale = std::int64_t(1) << (8 * bytesPerSample_ - 1);
                const auto value = raw >= static_cast<std::uint64_t>(scale)
                    ? static_cast<std::int64_t>(raw) - 2 * scale : static_cast<std::int64_t>(raw);
                sample = static_cast<double>(value) / scale;
            }
            meter_->appendSample(sample);
        }
        sampleOffset_ += count;
    }
    return latest_;
}

} // namespace echoflow

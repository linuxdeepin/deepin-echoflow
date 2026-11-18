// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "AudioSpectrum.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace echoflow {
namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr std::array<double, 8> kBandEdges{80, 200, 400, 800, 1600, 2600, 4000, 7600};
}

AudioSpectrum::AudioSpectrum(int sampleRate, size_t frames, size_t channels)
    : sampleRate_(sampleRate), frames_(frames), channels_(channels)
{
    if (sampleRate <= 0 || frames == 0 || channels == 0) {
        throw std::invalid_argument("invalid audio spectrum format");
    }
    size_t count = 2;
    while (count < frames) count *= 2;
    fft_.resize(count);
    power_.resize(count / 2 + 1);
    window_.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
        window_[i] = frames == 1 ? 1 : 0.5 * (1 - std::cos(2 * kPi * i / (frames - 1)));
        windowPower_ += window_[i] * window_[i];
    }
    if (windowPower_ == 0) {
        std::fill(window_.begin(), window_.end(), 1);
        windowPower_ = static_cast<double>(frames);
    }
}

void AudioSpectrum::transform()
{
    const size_t count = fft_.size();
    for (size_t i = 1, j = 0; i < count; ++i) {
        size_t bit = count >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(fft_[i], fft_[j]);
    }
    for (size_t length = 2; length <= count; length *= 2) {
        const auto step = std::polar(1.0, -2 * kPi / length);
        for (size_t offset = 0; offset < count; offset += length) {
            std::complex<double> phase{1, 0};
            for (size_t j = 0; j < length / 2; ++j) {
                const auto left = fft_[offset + j];
                const auto right = phase * fft_[offset + j + length / 2];
                fft_[offset + j] = left + right;
                fft_[offset + j + length / 2] = left - right;
                phase *= step;
            }
        }
    }
}

AudioSpectrumFrame AudioSpectrum::analyze(const std::vector<double>& interleaved)
{
    if (interleaved.size() != frames_ * channels_) {
        throw std::invalid_argument("incomplete audio spectrum frame");
    }
    std::fill(power_.begin(), power_.end(), 0);
    const double normalization = fft_.size() * windowPower_ * channels_;
    for (size_t channel = 0; channel < channels_; ++channel) {
        double mean = 0;
        for (size_t i = 0; i < frames_; ++i) mean += interleaved[i * channels_ + channel];
        mean /= frames_;
        std::fill(fft_.begin(), fft_.end(), std::complex<double>{0, 0});
        for (size_t i = 0; i < frames_; ++i) {
            fft_[i] = (interleaved[i * channels_ + channel] - mean) * window_[i];
        }
        transform();
        for (size_t bin = 1; bin < power_.size(); ++bin) {
            const double multiplicity = bin == fft_.size() / 2 ? 1 : 2;
            power_[bin] += multiplicity * std::norm(fft_[bin]) / normalization;
        }
    }
    AudioSpectrumFrame result;
    double sum = 0, logarithms = 0;
    size_t bins = 0;
    for (size_t bin = 1; bin < power_.size(); ++bin) {
        const double frequency = static_cast<double>(bin) * sampleRate_ / fft_.size();
        if (frequency < kBandEdges.front() || frequency >= kBandEdges.back()) continue;
        for (size_t band = 0; band < result.power.size(); ++band) {
            if (frequency >= kBandEdges[band] && frequency < kBandEdges[band + 1]) {
                result.power[band] += power_[bin];
                break;
            }
        }
        sum += power_[bin];
        logarithms += std::log(std::max(power_[bin], 1e-20));
        ++bins;
    }
    if (bins && sum > 1e-15) {
        result.flatness = std::clamp(std::exp(logarithms / bins) / (sum / bins), 0.0, 1.0);
    }
    return result;
}

} // namespace echoflow

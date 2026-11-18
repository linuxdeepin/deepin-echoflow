// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_AUDIO_SPECTRUM_H
#define ECHOFLOW_AUDIO_SPECTRUM_H

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

namespace echoflow {

struct AudioSpectrumFrame {
    std::array<double, 7> power{};
    double flatness = 0;
};

// Hann-windowed FFT of one current audio frame. Power is averaged across
// channels, avoiding phase cancellation when stereo microphone channels differ.
class AudioSpectrum {
public:
    AudioSpectrum(int sampleRate, size_t frames, size_t channels);
    AudioSpectrumFrame analyze(const std::vector<double>& interleaved);

private:
    void transform();
    int sampleRate_;
    size_t frames_;
    size_t channels_;
    double windowPower_ = 0;
    std::vector<double> window_;
    std::vector<std::complex<double>> fft_;
    std::vector<double> power_;
};

} // namespace echoflow

#endif

// glint - Psychoacoustic masking model implementation
// MIT License - Clean-room implementation
//
// Bark-band masking model operating directly on MDCT coefficients.
// No FFT needed — MDCT coefficients are already frequency-ordered.

#include "psycho.hpp"
#include "tables.hpp"
#include <cmath>
#include <algorithm>

namespace glint {

// Bark band boundaries (MDCT bin indices) for different sample rates.
// Computed from: bark(f) = 13*atan(0.00076*f) + 3.5*atan((f/7500)^2)
// with f = bin_index * (sample_rate / 1152) for 576 MDCT bins.

// 44100 Hz (MPEG-1 sr_index 0)
static const int bark_band_44100[26] = {
    0, 2, 4, 6, 8, 10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 74,
    92, 114, 142, 176, 220, 276, 348, 438, 512, 576
};

// 48000 Hz (MPEG-1 sr_index 1)
static const int bark_band_48000[26] = {
    0, 2, 4, 5, 7, 9, 11, 14, 17, 20, 25, 30, 37, 45, 55, 68,
    84, 105, 130, 162, 202, 254, 320, 402, 470, 576
};

// 32000 Hz (MPEG-1 sr_index 2)
static const int bark_band_32000[26] = {
    0, 3, 6, 9, 12, 15, 18, 22, 27, 33, 40, 49, 60, 74, 90, 111,
    138, 170, 213, 264, 330, 414, 500, 560, 572, 576
};

// For MPEG-2 rates (22050, 24000, 16000) — sr_index 3,4,5
// These have the same 576 bins but cover half the frequency range,
// so bark bands are wider in bin space.
// 22050 Hz
static const int bark_band_22050[26] = {
    0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 80, 98, 120, 148,
    184, 228, 284, 352, 440, 520, 560, 572, 575, 576
};

// 24000 Hz
static const int bark_band_24000[26] = {
    0, 4, 7, 11, 14, 18, 22, 28, 34, 40, 50, 60, 74, 90, 110, 136,
    168, 210, 260, 324, 404, 488, 552, 570, 574, 576
};

// 16000 Hz
static const int bark_band_16000[26] = {
    0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 82, 100, 120, 148, 180, 222,
    276, 342, 420, 496, 552, 570, 574, 575, 575, 576
};

const int* PsychoModel::get_bark_bands(int sr_index) {
    switch (sr_index) {
    case 0: return bark_band_44100;
    case 1: return bark_band_48000;
    case 2: return bark_band_32000;
    case 3: return bark_band_22050;
    case 4: return bark_band_24000;
    case 5: return bark_band_16000;
    default: return bark_band_44100;
    }
}

PsychoModel::PsychoModel() : spread_init_(false) {}

void PsychoModel::init_spread_matrix() {
    if (spread_init_) return;
    for (int b = 0; b < 25; b++) {
        for (int j = 0; j < 25; j++) {
            if (j >= b) {
                // Upward masking: 3 dB/bark decay
                spread_matrix_[b][j] = std::pow(10.0, -0.3 * (j - b));
            } else {
                // Downward masking: 1.5 dB/bark decay
                spread_matrix_[b][j] = std::pow(10.0, -0.15 * (b - j));
            }
        }
    }
    spread_init_ = true;
}

void PsychoModel::compute_masking(const double* mdct, double* threshold, int sr_index) {
    init_spread_matrix();

    const int* bark_band = get_bark_bands(sr_index);

    // Step 1: Compute per-bark-band energy from MDCT coefficients
    double bark_energy[25];
    for (int b = 0; b < 25; b++) {
        double e = 0.0;
        for (int i = bark_band[b]; i < bark_band[b + 1]; i++)
            e += mdct[i] * mdct[i];
        bark_energy[b] = e;
    }

    // Step 2: Apply spreading function (simultaneous masking)
    // Each band's masking threshold is based on energy spread from
    // NEIGHBORING bands only (exclude self to avoid masking own signal).
    double masking[25];
    for (int b = 0; b < 25; b++) {
        double sum = 0.0;
        for (int j = 0; j < 25; j++) {
            if (j == b) continue;  // exclude self-masking
            sum += bark_energy[j] * spread_matrix_[b][j];
        }
        masking[b] = sum;
    }

    // Step 3: Apply absolute threshold of hearing as floor
    // Use a conservative noise-to-mask ratio: -20 dB (0.01)
    // This ensures we only zero coefficients that are strongly masked
    // by neighboring bands' energy.
    for (int b = 0; b < 25; b++) {
        double ath_linear = std::pow(10.0, (tables::ath_cb[b] - 96.0) / 10.0);
        masking[b] = std::max(masking[b] * 0.01, ath_linear);
    }

    // Step 4: Map bark-band thresholds to individual coefficients
    for (int i = 0; i < 576; i++) {
        int b = 0;
        while (b < 24 && i >= bark_band[b + 1]) b++;

        int band_width = bark_band[b + 1] - bark_band[b];
        if (band_width < 1) band_width = 1;

        threshold[i] = masking[b] / band_width;
    }
}

} // namespace glint

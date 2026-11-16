// glint - Psychoacoustic masking model
// MIT License - Clean-room implementation

#ifndef GLINT_PSYCHO_HPP
#define GLINT_PSYCHO_HPP

namespace glint {

class PsychoModel {
public:
    PsychoModel();

    // Compute masking thresholds for 576 MDCT coefficients.
    // Input: 576 MDCT coefficients (double)
    // Output: 576 masking thresholds (double) — zero coefficient if |xr|^2 < threshold
    void compute_masking(const double* mdct, double* threshold, int sr_index);

private:
    // Precomputed bark band boundaries for each sample rate group
    // Index 0: 44100 Hz, 1: 48000 Hz, 2: 32000 Hz (sr_index 0,1,2 for MPEG-1)
    // For MPEG-2 rates (sr_index >= 3), we use a scaled version
    static const int kNumBarkBands = 25;

    // Get bark band boundaries for a given sr_index
    static const int* get_bark_bands(int sr_index);

    // Precomputed spreading function matrix (25x25)
    double spread_matrix_[25][25];
    bool spread_init_;
    void init_spread_matrix();
};

} // namespace glint

#endif // GLINT_PSYCHO_HPP

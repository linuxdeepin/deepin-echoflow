// glint - Subband analysis filter (polyphase filter bank)
// MIT License - Clean-room implementation

#ifndef GLINT_SUBBAND_HPP
#define GLINT_SUBBAND_HPP

#include <cstdint>

namespace glint {

static constexpr int kNumSubbands = 32;
static constexpr int kTimeSlots = 36;

#if !defined(GLINT_FIXED_POINT) || defined(GLINT_BOTH_PATHS)
// Double-precision subband analysis
class SubbandAnalysis {
public:
    SubbandAnalysis();
    void analyze(const int16_t* pcm, double out[kNumSubbands][kTimeSlots], int num_slots = kTimeSlots);
    void analyze_float(const float* pcm, double out[kNumSubbands][kTimeSlots], int num_slots = kTimeSlots);
    void reset();

private:
    double window_buf_[512];
    int window_offset_;
    void process_slot(const double* samples, double subband_out[kNumSubbands]);
};
#endif

#ifdef GLINT_FIXED_POINT
// Fixed-point (Q24) subband analysis
class SubbandAnalysisFP {
public:
    SubbandAnalysisFP();
    void analyze(const int16_t* pcm, int32_t out[kNumSubbands][kTimeSlots], int num_slots = kTimeSlots);
    void reset();

private:
    // Window buffer stores raw int16 values as double for SSE2 compatibility.
    // This eliminates int->double conversion in the windowing inner loop.
    double window_buf_d_[512];
    int window_offset_;
    void process_slot(const int16_t* samples, int32_t subband_out[kNumSubbands]);
};
#endif

} // namespace glint

#endif

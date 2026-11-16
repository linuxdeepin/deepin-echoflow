// glint - Opus encoder tonality analysis (PLAN § O4, tonality item)
// MIT License - Clean-room implementation
//
// Reduced port of the reference tonality_analysis (src/analysis.c, float
// path): the FFT-phase-predictability tonality estimate, its spectral
// slope, the activity estimate and the per-band leakage boosts. The
// music/speech MLP, bandwidth detection and BFCC features are NOT ported
// — nothing the CELT-only encoder consumes needs them. All outputs are
// encoder POLICY (they steer VBR targets / trim / dynalloc); validity of
// the stream never depends on them.
//
// Feed 48 kHz float PCM (±1.0, interleaved) frame by frame; the analyzer
// downmixes + halves to 24 kHz and updates once 480 new samples (20 ms)
// have accumulated. info().valid stays 0 until the first full update.

#pragma once

#include "opus_mdct.hpp"

namespace glint {
namespace opus {

constexpr int kLeakBands = 19;

struct AnalysisInfo {
    int valid = 0;
    float tonality = 0;
    float tonality_slope = 0;
    float activity = 0;
    int leak_boost[kLeakBands] = {};
};

class TonalityAnalyzer {
public:
    TonalityAnalyzer();
    void init();

    // One 48 kHz frame of C-channel interleaved float PCM.
    void feed(const float* pcm, int frame_size, int C);

    const AnalysisInfo& info() const { return info_; }

private:
    static constexpr int kBufSize = 720;  // 30 ms at 24 kHz
    static constexpr int kNbTBands = 18;
    static constexpr int kNbFrames = 8;

    void process();

    CeltImdct::MixedFft fft_;
    float window_[240];
    float inmem_[kBufSize];
    int mem_fill_ = 240;
    float resamp_state_[3] = {};
    float angle_[240] = {}, d_angle_[240] = {}, d2_angle_[240] = {};
    float e_[kNbFrames][kNbTBands] = {};
    float log_e_[kNbFrames][kNbTBands] = {};
    float low_e_[kNbTBands] = {}, high_e_[kNbTBands] = {};
    float prev_band_tonality_[kNbTBands] = {};
    float prev_tonality_ = 0;
    int count_ = 0;
    int e_count_ = 0;
    AnalysisInfo info_;
};

}  // namespace opus
}  // namespace glint

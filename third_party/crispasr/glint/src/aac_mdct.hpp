// glint - AAC forward MDCT (all four window sequences)
// MIT License - Clean-room implementation
//
// 2048-point (long/start/stop) and 8x256-point (eight-short) MDCTs per
// ISO/IEC 13818-7 / 14496-3:
//   X[k] = 2 * sum_{n=0}^{N-1} z[n] cos(2pi/N (n + n0)(k + 1/2)),  n0 = (N/2+1)/2
// computed as a +/- fold followed by a DCT-IV via an N/8-point complex FFT.
// Sine windows. Short windows occupy samples [448, 1600) of the 2048 frame;
// LONG_START/LONG_STOP bridge shapes keep TDAC across the transition.

#ifndef GLINT_AAC_MDCT_HPP
#define GLINT_AAC_MDCT_HPP

#include "aac_coder_types_fwd.hpp"

namespace glint {
namespace aac {

constexpr int kAacFrameLen = 1024;  // coefficients / new samples per frame

enum AacWindowSequence {
    kSeqLong  = 0,  // ONLY_LONG_SEQUENCE
    kSeqStart = 1,  // LONG_START_SEQUENCE
    kSeqShort = 2,  // EIGHT_SHORT_SEQUENCE
    kSeqStop  = 3,  // LONG_STOP_SEQUENCE
};

// prev/cur: previous and current 1024-sample blocks (unwindowed, PcmT
// storage). spec: 1024 coefficients out. For kSeqShort the layout is
// window-major: spec[128*w + k] = coefficient k of short window w.
void aac_mdct_frame(int window_sequence, const PcmT* prev, const PcmT* cur,
                    SpecT* spec);

}  // namespace aac
}  // namespace glint

#endif  // GLINT_AAC_MDCT_HPP

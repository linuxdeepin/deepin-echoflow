// SILK stereo prediction decode + MS->LR unmixing — RFC 6716 section 4.2.7.1
// MIT License - Clean-room implementation
//
// Stereo SILK codes mid/side with two Q13 predictors (joint MSBs + per-
// predictor fine indices). The unmix interpolates the predictors over the
// first 8 ms, applies a 3-tap smoothed mid prediction into the side, then
// forms L/R. One-sample lookahead per channel is buffered in the state
// (x1/x2 carry frame_length + 2 samples: [prev2 | frame]).

#pragma once

#include <cstdint>

#include "opus_ec.hpp"

namespace glint {
namespace opus {
namespace silk {

struct StereoDecState {
    int16_t pred_prev_q13[2] = {};
    int16_t s_mid[2] = {};
    int16_t s_side[2] = {};
};

void stereo_decode_pred(RangeDecoder& dec, int32_t* pred_q13);
int stereo_decode_mid_only(RangeDecoder& dec);

// x1/x2 hold frame_length + 2 samples; on entry [2..] is the new frame,
// on exit [1 .. frame_length+1) is L/R (reference indexing convention).
void stereo_ms_to_lr(StereoDecState* st, int16_t* x1, int16_t* x2,
                     const int32_t* pred_q13, int fs_khz, int frame_length);

}  // namespace silk
}  // namespace opus
}  // namespace glint

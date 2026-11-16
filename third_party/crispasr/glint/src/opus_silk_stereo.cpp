// SILK stereo prediction decode + MS->LR unmixing — RFC 6716 section 4.2.7.1
// MIT License - Clean-room implementation

#include "opus_silk_stereo.hpp"

#include <cstring>

#include "opus_silk_math.hpp"
#include "opus_silk_tables.hpp"

namespace glint {
namespace opus {
namespace silk {

namespace {
constexpr int kStereoInterpLenMs = 8;
// 0.5 / STEREO_QUANT_SUB_STEPS(5) in Q16.
constexpr int32_t kHalfSubStepQ16 = 6554;
}  // namespace

void stereo_decode_pred(RangeDecoder& dec, int32_t* pred_q13) {
    int ix[2][3];
    // Joint MSB index (5x5), then per-predictor fine indices.
    int n = dec.dec_icdf(kStereoPredJointIcdf, 8);
    ix[0][2] = n / 5;
    ix[1][2] = n - 5 * ix[0][2];
    for (n = 0; n < 2; n++) {
        ix[n][0] = dec.dec_icdf(kUniform3Icdf, 8);
        ix[n][1] = dec.dec_icdf(kUniform5Icdf, 8);
    }

    for (n = 0; n < 2; n++) {
        ix[n][0] += 3 * ix[n][2];
        int32_t low_q13 = kStereoPredQuantQ13[ix[n][0]];
        int32_t step_q13 = smulwb(
            kStereoPredQuantQ13[ix[n][0] + 1] - low_q13, kHalfSubStepQ16);
        pred_q13[n] = smlabb(low_q13, step_q13, 2 * ix[n][1] + 1);
    }

    // Predictors are applied cascaded; store the difference.
    pred_q13[0] -= pred_q13[1];
}

int stereo_decode_mid_only(RangeDecoder& dec) {
    return dec.dec_icdf(kStereoOnlyCodeMidIcdf, 8);
}

void stereo_ms_to_lr(StereoDecState* st, int16_t* x1, int16_t* x2,
                     const int32_t* pred_q13, int fs_khz,
                     int frame_length) {
    // Two-sample carry between frames (3-tap filter lookahead).
    std::memcpy(x1, st->s_mid, 2 * sizeof(int16_t));
    std::memcpy(x2, st->s_side, 2 * sizeof(int16_t));
    std::memcpy(st->s_mid, &x1[frame_length], 2 * sizeof(int16_t));
    std::memcpy(st->s_side, &x2[frame_length], 2 * sizeof(int16_t));

    // Interpolate the predictors over the first 8 ms.
    int32_t pred0_q13 = st->pred_prev_q13[0];
    int32_t pred1_q13 = st->pred_prev_q13[1];
    int interp_len = kStereoInterpLenMs * fs_khz;
    int32_t denom_q16 = (1 << 16) / interp_len;
    int32_t delta0_q13 = rshift_round(
        smulbb(pred_q13[0] - st->pred_prev_q13[0], denom_q16), 16);
    int32_t delta1_q13 = rshift_round(
        smulbb(pred_q13[1] - st->pred_prev_q13[1], denom_q16), 16);

    int n = 0;
    for (int phase = 0; phase < 2; phase++) {
        int end = phase == 0 ? interp_len : frame_length;
        for (; n < end; n++) {
            if (phase == 0) {
                pred0_q13 += delta0_q13;
                pred1_q13 += delta1_q13;
            }
            // Smoothed mid (3-tap) into the side, plus direct mid term.
            int32_t sum = static_cast<int32_t>(
                static_cast<uint32_t>(add32_ovflw(
                    x1[n] + static_cast<int32_t>(x1[n + 2]),
                    static_cast<int32_t>(
                        static_cast<uint32_t>(x1[n + 1]) << 1)))
                << 9);
            sum = smlawb(static_cast<int32_t>(x2[n + 1]) << 8, sum,
                         pred0_q13);
            sum = smlawb(sum, static_cast<int32_t>(x1[n + 1]) << 11,
                         pred1_q13);
            x2[n + 1] =
                static_cast<int16_t>(sat16(rshift_round(sum, 8)));
        }
        pred0_q13 = pred_q13[0];
        pred1_q13 = pred_q13[1];
    }
    st->pred_prev_q13[0] = static_cast<int16_t>(pred_q13[0]);
    st->pred_prev_q13[1] = static_cast<int16_t>(pred_q13[1]);

    // Mid/side to left/right.
    for (n = 0; n < frame_length; n++) {
        int32_t sum = x1[n + 1] + static_cast<int32_t>(x2[n + 1]);
        int32_t diff = x1[n + 1] - static_cast<int32_t>(x2[n + 1]);
        x1[n + 1] = static_cast<int16_t>(sat16(sum));
        x2[n + 1] = static_cast<int16_t>(sat16(diff));
    }
}

}  // namespace silk
}  // namespace opus
}  // namespace glint

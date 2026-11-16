// SILK frame decoding: parameters + core synthesis — RFC 6716 section 4.2.7
// MIT License - Clean-room implementation

#include "opus_silk_frame.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "opus_silk_excitation.hpp"
#include "opus_silk_math.hpp"
#include "opus_silk_nlsf.hpp"
#include "opus_silk_plc.hpp"
#include "opus_silk_tables.hpp"

namespace glint {
namespace opus {
namespace silk {

namespace {
constexpr int32_t kQuantLevelAdjustQ10 = 80;
constexpr int32_t kBweAfterLossQ16 = 63570;
constexpr int kMaxFrameLength = 320;
constexpr int kLtpMemMax = 320;  // 20 ms of history at 16 kHz
}  // namespace

void decode_parameters(DecoderState* st, DecoderControl* ctrl,
                       int cond_coding) {
    int16_t nlsf_q15[kMaxLpcOrder], nlsf0_q15[kMaxLpcOrder];

    gains_dequant(ctrl->gains_q16, st->indices.gains_indices,
                  &st->prev_gain_index,
                  cond_coding == kCodeConditionally ? 1 : 0, st->nb_subfr);

    nlsf_decode(nlsf_q15, st->indices.nlsf_indices, *st->nlsf_cb);
    nlsf2a(ctrl->pred_coef_q12[1], nlsf_q15, st->lpc_order);

    // No interpolation right after a reset (protects a post-switch loss).
    if (st->first_frame_after_reset == 1)
        st->indices.nlsf_interp_coef_q2 = 4;

    if (st->indices.nlsf_interp_coef_q2 < 4) {
        // First-half coefficients from NLSFs interpolated with the
        // previous frame's.
        for (int i = 0; i < st->lpc_order; i++) {
            nlsf0_q15[i] = static_cast<int16_t>(
                st->prev_nlsf_q15[i] +
                ((st->indices.nlsf_interp_coef_q2 *
                  (nlsf_q15[i] - st->prev_nlsf_q15[i])) >> 2));
        }
        nlsf2a(ctrl->pred_coef_q12[0], nlsf0_q15, st->lpc_order);
    } else {
        std::memcpy(ctrl->pred_coef_q12[0], ctrl->pred_coef_q12[1],
                    st->lpc_order * sizeof(int16_t));
    }

    std::memcpy(st->prev_nlsf_q15, nlsf_q15,
                st->lpc_order * sizeof(int16_t));

    if (st->loss_cnt) {
        bwexpander(ctrl->pred_coef_q12[0], st->lpc_order, kBweAfterLossQ16);
        bwexpander(ctrl->pred_coef_q12[1], st->lpc_order, kBweAfterLossQ16);
    }

    if (st->indices.signal_type == kTypeVoiced) {
        decode_pitch(st->indices.lag_index, st->indices.contour_index,
                     ctrl->pitch_lags, st->fs_khz, st->nb_subfr);

        const int8_t* cbk = kLtpVqPtrsQ7[st->indices.per_index];
        for (int k = 0; k < st->nb_subfr; k++) {
            int ix = st->indices.ltp_index[k];
            for (int i = 0; i < kLtpOrder; i++)
                ctrl->ltp_coef_q14[k * kLtpOrder + i] =
                    static_cast<int16_t>(
                        static_cast<int32_t>(cbk[ix * kLtpOrder + i]) << 7);
        }

        ctrl->ltp_scale_q14 = kLtpScalesTableQ14[st->indices.ltp_scale_index];
    } else {
        std::memset(ctrl->pitch_lags, 0, st->nb_subfr * sizeof(int));
        std::memset(ctrl->ltp_coef_q14, 0,
                    kLtpOrder * st->nb_subfr * sizeof(int16_t));
        st->indices.per_index = 0;
        ctrl->ltp_scale_q14 = 0;
    }
}

void bwexpander(int16_t* ar, int order, int32_t chirp_q16) {
    // Rounded multiplies here on purpose: SMULWB's truncation bias can
    // destabilize the filter (reference note).
    int32_t chirp_minus_one_q16 = chirp_q16 - 65536;
    for (int i = 0; i < order - 1; i++) {
        ar[i] = static_cast<int16_t>(
            rshift_round64(static_cast<int64_t>(chirp_q16) * ar[i], 16));
        chirp_q16 += static_cast<int32_t>(rshift_round64(
            static_cast<int64_t>(chirp_q16) * chirp_minus_one_q16, 16));
    }
    ar[order - 1] = static_cast<int16_t>(
        rshift_round64(static_cast<int64_t>(chirp_q16) * ar[order - 1], 16));
}

void lpc_analysis_filter(int16_t* out, const int16_t* in,
                         const int16_t* b_q12, int len, int order) {
    for (int ix = order; ix < len; ix++) {
        const int16_t* p = &in[ix - 1];
        // Wrapping accumulation on purpose (invalid streams only).
        int32_t pred_q12 = smulbb(p[0], b_q12[0]);
        for (int j = 1; j < order; j++)
            pred_q12 = smlabb(pred_q12, p[-j], b_q12[j]);
        int32_t out_q12 = static_cast<int32_t>(
            (static_cast<uint32_t>(static_cast<int32_t>(p[1]) << 12)) -
            static_cast<uint32_t>(pred_q12));
        out[ix] = static_cast<int16_t>(sat16(rshift_round(out_q12, 12)));
    }
    std::memset(out, 0, order * sizeof(int16_t));
}

void decode_core(DecoderState* st, DecoderControl& ctrl, int16_t* xq,
                 const int16_t* pulses) {
    int16_t slt[kLtpMemMax];
    int32_t slt_q15[kLtpMemMax + kMaxFrameLength];
    int32_t res_q14[80];  // one subframe
    int32_t slpc_q14[80 + kMaxLpcOrder];
    int16_t a_q12[kMaxLpcOrder];

    int32_t offset_q10 =
        kQuantizationOffsetsQ10[st->indices.signal_type >> 1]
                               [st->indices.quant_offset_type];
    int nlsf_interp_flag = st->indices.nlsf_interp_coef_q2 < 4 ? 1 : 0;

    // Excitation: pulses to Q14 with level adjust, offset, and LCG sign
    // dithering; the seed also absorbs each pulse value (wrapping).
    int32_t rand_seed = st->indices.seed;
    for (int i = 0; i < st->frame_length; i++) {
        rand_seed = silk_rand(rand_seed);
        int32_t e = static_cast<int32_t>(pulses[i]) << 14;
        if (e > 0)
            e -= kQuantLevelAdjustQ10 << 4;
        else if (e < 0)
            e += kQuantLevelAdjustQ10 << 4;
        e += offset_q10 << 4;
        if (rand_seed < 0) e = -e;
        st->exc_q14[i] = e;
        rand_seed = add32_ovflw(rand_seed, pulses[i]);
    }

    std::memcpy(slpc_q14, st->slpc_q14_buf,
                kMaxLpcOrder * sizeof(int32_t));

    const int32_t* pexc_q14 = st->exc_q14;
    int16_t* pxq = xq;
    int slt_buf_idx = st->ltp_mem_length;
    int lag = 0;

    for (int k = 0; k < st->nb_subfr; k++) {
        const int32_t* pres_q14 = res_q14;
        std::memcpy(a_q12, ctrl.pred_coef_q12[k >> 1],
                    st->lpc_order * sizeof(int16_t));
        int16_t* b_q14 = &ctrl.ltp_coef_q14[k * kLtpOrder];
        int signal_type = st->indices.signal_type;

        int32_t gain_q10 = ctrl.gains_q16[k] >> 6;
        int32_t inv_gain_q31 = inverse32_varq(ctrl.gains_q16[k], 47);

        // Rescale the filter state when the gain steps.
        int32_t gain_adj_q16 = 1 << 16;
        if (ctrl.gains_q16[k] != st->prev_gain_q16) {
            gain_adj_q16 =
                div32_varq(st->prev_gain_q16, ctrl.gains_q16[k], 16);
            for (int i = 0; i < kMaxLpcOrder; i++)
                slpc_q14[i] = smulww(gain_adj_q16, slpc_q14[i]);
        }
        st->prev_gain_q16 = ctrl.gains_q16[k];

        // Avoid an abrupt transition from voiced concealment to unvoiced
        // decoding: the first half of the first good frame after a voiced
        // loss keeps a weak (0.25) centered pitch tap at the concealment
        // lag. Written into ctrl so the PLC update below sees it.
        if (st->loss_cnt && st->prev_signal_type == kTypeVoiced &&
            st->indices.signal_type != kTypeVoiced && k < kMaxNbSubfr / 2) {
            std::memset(b_q14, 0, kLtpOrder * sizeof(int16_t));
            b_q14[kLtpOrder / 2] = 4096;  // 0.25 in Q14
            signal_type = kTypeVoiced;
            ctrl.pitch_lags[k] = st->lag_prev;
        }

        if (signal_type == kTypeVoiced) {
            lag = ctrl.pitch_lags[k];

            if (k == 0 || (k == 2 && nlsf_interp_flag)) {
                // Rewhiten past output into an unscaled LTP state.
                int start_idx = st->ltp_mem_length - lag - st->lpc_order -
                                kLtpOrder / 2;
                if (k == 2) {
                    std::memcpy(&st->out_buf[st->ltp_mem_length], xq,
                                2 * st->subfr_length * sizeof(int16_t));
                }
                lpc_analysis_filter(
                    &slt[start_idx],
                    &st->out_buf[start_idx + k * st->subfr_length], a_q12,
                    st->ltp_mem_length - start_idx, st->lpc_order);

                if (k == 0) {
                    // Downscale the LTP state to cut inter-frame
                    // dependency on independently coded frames.
                    inv_gain_q31 = static_cast<int32_t>(
                        static_cast<uint32_t>(
                            smulwb(inv_gain_q31, ctrl.ltp_scale_q14))
                        << 2);
                }
                for (int i = 0; i < lag + kLtpOrder / 2; i++)
                    slt_q15[slt_buf_idx - i - 1] = smulwb(
                        inv_gain_q31, slt[st->ltp_mem_length - i - 1]);
            } else if (gain_adj_q16 != 1 << 16) {
                for (int i = 0; i < lag + kLtpOrder / 2; i++)
                    slt_q15[slt_buf_idx - i - 1] = smulww(
                        gain_adj_q16, slt_q15[slt_buf_idx - i - 1]);
            }
        }

        if (signal_type == kTypeVoiced) {
            // 5-tap long-term prediction, then feed the result back as
            // future LTP state.
            const int32_t* pred_lag =
                &slt_q15[slt_buf_idx - lag + kLtpOrder / 2];
            for (int i = 0; i < st->subfr_length; i++) {
                // +2 counters SMLAWB's round-to-minus-infinity bias.
                int32_t ltp_pred_q13 = 2;
                ltp_pred_q13 = smlawb(ltp_pred_q13, pred_lag[0], b_q14[0]);
                ltp_pred_q13 = smlawb(ltp_pred_q13, pred_lag[-1], b_q14[1]);
                ltp_pred_q13 = smlawb(ltp_pred_q13, pred_lag[-2], b_q14[2]);
                ltp_pred_q13 = smlawb(ltp_pred_q13, pred_lag[-3], b_q14[3]);
                ltp_pred_q13 = smlawb(ltp_pred_q13, pred_lag[-4], b_q14[4]);
                pred_lag++;

                res_q14[i] = add32_ovflw(
                    pexc_q14[i],
                    static_cast<int32_t>(
                        static_cast<uint32_t>(ltp_pred_q13) << 1));
                slt_q15[slt_buf_idx] = static_cast<int32_t>(
                    static_cast<uint32_t>(res_q14[i]) << 1);
                slt_buf_idx++;
            }
        } else {
            pres_q14 = pexc_q14;
        }

        for (int i = 0; i < st->subfr_length; i++) {
            // Short-term LPC synthesis (order/2 counters the SMLAWB bias).
            int32_t lpc_pred_q10 = st->lpc_order >> 1;
            for (int j = 0; j < st->lpc_order; j++)
                lpc_pred_q10 = smlawb(
                    lpc_pred_q10, slpc_q14[kMaxLpcOrder + i - 1 - j],
                    a_q12[j]);

            slpc_q14[kMaxLpcOrder + i] = add_sat32(
                pres_q14[i], lshift_sat32(lpc_pred_q10, 4));
            pxq[i] = static_cast<int16_t>(sat16(rshift_round(
                smulww(slpc_q14[kMaxLpcOrder + i], gain_q10), 8)));
        }

        std::memcpy(slpc_q14, &slpc_q14[st->subfr_length],
                    kMaxLpcOrder * sizeof(int32_t));
        pexc_q14 += st->subfr_length;
        pxq += st->subfr_length;
    }

    std::memcpy(st->slpc_q14_buf, slpc_q14,
                kMaxLpcOrder * sizeof(int32_t));
}

int decode_frame(DecoderState* st, RangeDecoder* dec, int16_t* xq,
                 int cond_coding, bool lost, bool lbrr) {
    DecoderControl ctrl;
    ctrl.ltp_scale_q14 = 0;
    int mv_len = st->ltp_mem_length - st->frame_length;

    // LBRR decode of a frame that carries no LBRR copy => concealment
    // (reference: the flag check lives inside silk_decode_frame).
    if (lbrr && !st->lbrr_flags[st->n_frames_decoded]) lost = true;


    if (!lost) {
        int16_t pulses[(kMaxFrameLength + kShellCodecFrameLength - 1) &
                       ~(kShellCodecFrameLength - 1)];

        decode_indices(st, *dec, st->n_frames_decoded, lbrr, cond_coding);
        decode_pulses(*dec, pulses, st->indices.signal_type,
                      st->indices.quant_offset_type, st->frame_length);
        decode_parameters(st, &ctrl, cond_coding);
        decode_core(st, ctrl, xq, pulses);

        // Output history: keep the last ltp_mem_length samples.
        std::memmove(st->out_buf, &st->out_buf[st->frame_length],
                     mv_len * sizeof(int16_t));
        std::memcpy(&st->out_buf[mv_len], xq,
                    st->frame_length * sizeof(int16_t));

        plc_update(st, &ctrl);
        st->loss_cnt = 0;
        st->prev_signal_type = st->indices.signal_type;
        st->first_frame_after_reset = 0;
    } else {
        // Packet lost: extrapolate from state (increments loss_cnt); the
        // range decoder is never touched.
        plc_conceal(st, &ctrl, xq);

        std::memmove(st->out_buf, &st->out_buf[st->frame_length],
                     mv_len * sizeof(int16_t));
        std::memcpy(&st->out_buf[mv_len], xq,
                    st->frame_length * sizeof(int16_t));
    }

    // Comfort-noise estimation (good frames) / generation (lost frames),
    // then the loss->good energy glue fade.
    cng(st, ctrl, xq, st->frame_length);
    plc_glue_frames(st, xq, st->frame_length);

    st->lag_prev = ctrl.pitch_lags[st->nb_subfr - 1];
    return st->frame_length;
}

}  // namespace silk
}  // namespace opus
}  // namespace glint

// SILK packet-loss concealment + comfort noise — RFC 6716 section 4.4
// MIT License - Clean-room implementation

#include "opus_silk_plc.hpp"

#include <algorithm>
#include <cstring>

#include "opus_silk_frame.hpp"
#include "opus_silk_indices.hpp"
#include "opus_silk_math.hpp"
#include "opus_silk_nlsf.hpp"

namespace glint {
namespace opus {
namespace silk {

namespace {

// Attenuation per consecutive loss (index min(loss_cnt, 1)).
constexpr int kNbAtt = 2;
constexpr int16_t kHarmAttQ15[kNbAtt] = { 32440, 31130 };  // 0.99, 0.95
constexpr int16_t kRandAttVoicedQ15[kNbAtt] = { 31130, 26214 };    // 0.95, 0.8
constexpr int16_t kRandAttUnvoicedQ15[kNbAtt] = { 32440, 29491 };  // 0.99, 0.9

constexpr int32_t kBweCoefQ16 = 64881;             // 0.99 in Q16
constexpr int32_t kVPitchGainStartMinQ14 = 11469;  // 0.7 in Q14
constexpr int32_t kVPitchGainStartMaxQ14 = 15565;  // 0.95 in Q14
constexpr int kMaxPitchLagMs = 18;
constexpr int kRandBufSize = 128;                  // excitation-reuse window
constexpr int kLog2InvLpcGainHighThres = 3;        // 8 dB LPC gain
constexpr int kLog2InvLpcGainLowThres = 8;         // 24 dB LPC gain
constexpr int32_t kPitchDriftFacQ16 = 655;         // 0.01 in Q16

constexpr int kCngBufMaskMax = 255;   // 2^floor(log2(MAX_FRAME_LENGTH)) - 1
constexpr int32_t kCngGainSmthQ16 = 4634;                // 0.25^(1/4)
constexpr int32_t kCngGainSmthThresholdQ16 = 46396;      // -3 dB
constexpr int32_t kCngNlsfSmthQ16 = 16348;               // 0.25

constexpr int kMaxFrameLength = 320;
constexpr int kLtpMemMax = 320;  // 20 ms of history at 16 kHz

// a - (b << shift), two's-complement wrap (reference silk_SUB_LSHIFT32).
inline int32_t sub_lshift32(int32_t a, int32_t b, int shift) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) -
                                (static_cast<uint32_t>(b) << shift));
}

// Approximate sqrt of a non-negative int32 (reference silk_SQRT_APPROX):
// 2^(lz/2) seed picked by leading-zero parity, one fractional correction.
int32_t sqrt_approx(int32_t x) {
    if (x <= 0) return 0;
    int32_t lz, frac_q7;
    clz_frac(x, &lz, &frac_q7);
    int32_t y = (lz & 1) ? 32768 : 46214;  // 46214 = sqrt(2) * 32768
    y >>= (lz >> 1);
    return smlawb(y, y, smulbb(213, frac_q7));
}

// Energy of an int16 vector plus the right-shift that made it fit an int32
// with headroom (reference silk_sum_sqr_shift): a first pass at the maximal
// possible shift estimates the magnitude, a second pass recomputes at the
// final shift. Pairwise uint32 accumulation, wrap intentional.
void sum_sqr_shift(int32_t* energy, int* shift, const int16_t* x, int len) {
    int shft = 31 - clz32(static_cast<uint32_t>(len));
    int32_t nrg = len;  // conservative rounding headstart
    int i;
    for (i = 0; i < len - 1; i += 2) {
        uint32_t t = static_cast<uint32_t>(
            smlabb(smulbb(x[i], x[i]), x[i + 1], x[i + 1]));
        nrg = static_cast<int32_t>(static_cast<uint32_t>(nrg) + (t >> shft));
    }
    if (i < len) {
        uint32_t t = static_cast<uint32_t>(smulbb(x[i], x[i]));
        nrg = static_cast<int32_t>(static_cast<uint32_t>(nrg) + (t >> shft));
    }
    shft = std::max(0, shft + 3 - clz32(static_cast<uint32_t>(nrg)));
    nrg = 0;
    for (i = 0; i < len - 1; i += 2) {
        uint32_t t = static_cast<uint32_t>(
            smlabb(smulbb(x[i], x[i]), x[i + 1], x[i + 1]));
        nrg = static_cast<int32_t>(static_cast<uint32_t>(nrg) + (t >> shft));
    }
    if (i < len) {
        uint32_t t = static_cast<uint32_t>(smulbb(x[i], x[i]));
        nrg = static_cast<int32_t>(static_cast<uint32_t>(nrg) + (t >> shft));
    }
    *shift = shft;
    *energy = nrg;
}

// Energies of the last two subframes of the previous frame's excitation,
// scaled back to the signal domain with the last two gains (reference
// silk_PLC_energy). The quieter one seeds the random component.
void plc_energy(int32_t* energy1, int* shift1, int32_t* energy2, int* shift2,
                const int32_t* exc_q14, const int32_t* prev_gain_q10,
                int subfr_length, int nb_subfr) {
    int16_t exc_buf[2 * 80];  // two subframes, 5 ms at 16 kHz max
    int16_t* p = exc_buf;
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < subfr_length; i++)
            p[i] = static_cast<int16_t>(sat16(
                smulww(exc_q14[i + (k + nb_subfr - 2) * subfr_length],
                       prev_gain_q10[k]) >> 8));
        p += subfr_length;
    }
    sum_sqr_shift(energy1, shift1, exc_buf, subfr_length);
    sum_sqr_shift(energy2, shift2, &exc_buf[subfr_length], subfr_length);
}

// The reference resets PLC state lazily on an internal-rate mismatch at the
// top of silk_PLC (decoder_set_fs does not touch it); fs_khz == 0 at init
// makes this fire on first use too.
void plc_ensure_fs(DecoderState* st) {
    if (st->fs_khz != st->plc.fs_khz) {
        plc_reset(st);
        st->plc.fs_khz = st->fs_khz;
    }
}

// Fill length samples by picking randomly (LCG-indexed) from the smoothed
// CNG excitation history (reference silk_CNG_exc).
void cng_exc(int32_t* exc_q14, const int32_t* exc_buf_q14, int length,
             int32_t* rand_seed) {
    int exc_mask = kCngBufMaskMax;
    while (exc_mask > length) exc_mask >>= 1;
    int32_t seed = *rand_seed;
    for (int i = 0; i < length; i++) {
        seed = silk_rand(seed);
        int idx = (seed >> 24) & exc_mask;
        exc_q14[i] = exc_buf_q14[idx];
    }
    *rand_seed = seed;
}

}  // namespace

void plc_reset(DecoderState* st) {
    PlcState& plc = st->plc;
    plc.pitch_l_q8 = static_cast<int32_t>(st->frame_length)
                     << 7;  // half a frame in Q8
    plc.prev_gain_q16[0] = 1 << 16;
    plc.prev_gain_q16[1] = 1 << 16;
    plc.subfr_length = 20;
    plc.nb_subfr = 2;
}

void plc_update(DecoderState* st, DecoderControl* ctrl) {
    plc_ensure_fs(st);
    PlcState& plc = st->plc;

    st->prev_signal_type = st->indices.signal_type;
    int32_t ltp_gain_q14 = 0;
    if (st->indices.signal_type == kTypeVoiced) {
        // Walk back from the last subframe while a full pitch period still
        // fits; keep the subframe with the strongest LTP tap sum.
        for (int j = 0;
             j * st->subfr_length < ctrl->pitch_lags[st->nb_subfr - 1];
             j++) {
            if (j == st->nb_subfr) break;
            int32_t temp_gain_q14 = 0;
            for (int i = 0; i < kLtpOrder; i++)
                temp_gain_q14 +=
                    ctrl->ltp_coef_q14[(st->nb_subfr - 1 - j) * kLtpOrder + i];
            if (temp_gain_q14 > ltp_gain_q14) {
                ltp_gain_q14 = temp_gain_q14;
                std::memcpy(
                    plc.ltp_coef_q14,
                    &ctrl->ltp_coef_q14[(st->nb_subfr - 1 - j) * kLtpOrder],
                    kLtpOrder * sizeof(int16_t));
                plc.pitch_l_q8 = ctrl->pitch_lags[st->nb_subfr - 1 - j] << 8;
            }
        }

        // Collapse to a single centered tap, clamped into [0.7, 0.95].
        std::memset(plc.ltp_coef_q14, 0, kLtpOrder * sizeof(int16_t));
        plc.ltp_coef_q14[kLtpOrder / 2] = static_cast<int16_t>(ltp_gain_q14);

        if (ltp_gain_q14 < kVPitchGainStartMinQ14) {
            int32_t scale_q10 = (kVPitchGainStartMinQ14 << 10) /
                                std::max(ltp_gain_q14, int32_t{1});
            for (int i = 0; i < kLtpOrder; i++)
                plc.ltp_coef_q14[i] = static_cast<int16_t>(
                    smulbb(plc.ltp_coef_q14[i], scale_q10) >> 10);
        } else if (ltp_gain_q14 > kVPitchGainStartMaxQ14) {
            int32_t scale_q14 = (kVPitchGainStartMaxQ14 << 14) /
                                std::max(ltp_gain_q14, int32_t{1});
            for (int i = 0; i < kLtpOrder; i++)
                plc.ltp_coef_q14[i] = static_cast<int16_t>(
                    smulbb(plc.ltp_coef_q14[i], scale_q14) >> 14);
        }
    } else {
        plc.pitch_l_q8 = smulbb(st->fs_khz, 18) << 8;  // 18 ms fallback lag
        std::memset(plc.ltp_coef_q14, 0, kLtpOrder * sizeof(int16_t));
    }

    std::memcpy(plc.prev_lpc_q12, ctrl->pred_coef_q12[1],
                st->lpc_order * sizeof(int16_t));
    plc.prev_ltp_scale_q14 = static_cast<int16_t>(ctrl->ltp_scale_q14);
    std::memcpy(plc.prev_gain_q16, &ctrl->gains_q16[st->nb_subfr - 2],
                2 * sizeof(int32_t));
    plc.subfr_length = st->subfr_length;
    plc.nb_subfr = st->nb_subfr;
}

void plc_conceal(DecoderState* st, DecoderControl* ctrl, int16_t* frame) {
    plc_ensure_fs(st);
    PlcState& plc = st->plc;
    int16_t slt[kLtpMemMax];
    int32_t slt_q14[kLtpMemMax + kMaxFrameLength];
    int16_t a_q12[kMaxLpcOrder];

    int32_t prev_gain_q10[2] = { plc.prev_gain_q16[0] >> 6,
                                 plc.prev_gain_q16[1] >> 6 };

    if (st->first_frame_after_reset)
        std::memset(plc.prev_lpc_q12, 0, sizeof(plc.prev_lpc_q12));

    // Random component seed: reuse the quieter of the last two excitation
    // subframes (geometry of the frame the state was SAVED from).
    int32_t energy1, energy2;
    int shift1, shift2;
    plc_energy(&energy1, &shift1, &energy2, &shift2, st->exc_q14,
               prev_gain_q10, st->subfr_length, st->nb_subfr);
    const int32_t* rand_ptr;
    if ((energy1 >> shift2) < (energy2 >> shift1)) {
        rand_ptr = &st->exc_q14[std::max(
            0, (plc.nb_subfr - 1) * plc.subfr_length - kRandBufSize)];
    } else {
        rand_ptr = &st->exc_q14[std::max(
            0, plc.nb_subfr * plc.subfr_length - kRandBufSize)];
    }

    int16_t* b_q14 = plc.ltp_coef_q14;
    int16_t rand_scale_q14 = plc.rand_scale_q14;

    int att = std::min(kNbAtt - 1, st->loss_cnt);
    int32_t harm_gain_q15 = kHarmAttQ15[att];
    int32_t rand_gain_q15 = st->prev_signal_type == kTypeVoiced
                                ? kRandAttVoicedQ15[att]
                                : kRandAttUnvoicedQ15[att];

    // LPC concealment: bandwidth-expand the previous frame's filter.
    bwexpander(plc.prev_lpc_q12, st->lpc_order, kBweCoefQ16);
    std::memcpy(a_q12, plc.prev_lpc_q12, st->lpc_order * sizeof(int16_t));

    if (st->loss_cnt == 0) {
        // First lost frame: derive the random-component gain.
        rand_scale_q14 = 1 << 14;
        if (st->prev_signal_type == kTypeVoiced) {
            // The stronger the pitch prediction, the less noise.
            for (int i = 0; i < kLtpOrder; i++)
                rand_scale_q14 =
                    static_cast<int16_t>(rand_scale_q14 - b_q14[i]);
            rand_scale_q14 = std::max<int16_t>(3277, rand_scale_q14);  // 0.2
            rand_scale_q14 = static_cast<int16_t>(
                smulbb(rand_scale_q14, plc.prev_ltp_scale_q14) >> 14);
        } else {
            // Reduce noise under a high-gain (strongly shaped) LPC filter.
            int32_t inv_gain_q30 =
                lpc_inverse_pred_gain(plc.prev_lpc_q12, st->lpc_order);
            int32_t down_scale_q30 =
                std::min((int32_t{1} << 30) >> kLog2InvLpcGainHighThres,
                         inv_gain_q30);
            down_scale_q30 =
                std::max((int32_t{1} << 30) >> kLog2InvLpcGainLowThres,
                         down_scale_q30);
            down_scale_q30 <<= kLog2InvLpcGainHighThres;
            rand_gain_q15 = smulwb(down_scale_q30, rand_gain_q15) >> 14;
        }
    }

    int32_t rand_seed = plc.rand_seed;
    int lag = rshift_round(plc.pitch_l_q8, 8);
    int slt_buf_idx = st->ltp_mem_length;

    // Rewhiten past output into LTP state (as decode_core does on
    // independent frames), scaled by the inverse of the last gain.
    int idx = st->ltp_mem_length - lag - st->lpc_order - kLtpOrder / 2;
    lpc_analysis_filter(&slt[idx], &st->out_buf[idx], a_q12,
                        st->ltp_mem_length - idx, st->lpc_order);
    int32_t inv_gain_q30 = inverse32_varq(plc.prev_gain_q16[1], 46);
    inv_gain_q30 = std::min(inv_gain_q30, INT32_MAX >> 1);
    for (int i = idx + st->lpc_order; i < st->ltp_mem_length; i++)
        slt_q14[i] = smulwb(inv_gain_q30, slt[i]);

    // LTP synthesis: attenuated pitch prediction plus scaled random reuse
    // of the previous excitation.
    for (int k = 0; k < st->nb_subfr; k++) {
        const int32_t* pred_lag = &slt_q14[slt_buf_idx - lag + kLtpOrder / 2];
        for (int i = 0; i < st->subfr_length; i++) {
            // +2 counters SMLAWB's round-to-minus-infinity bias.
            int32_t ltp_pred_q12 = 2;
            ltp_pred_q12 = smlawb(ltp_pred_q12, pred_lag[0], b_q14[0]);
            ltp_pred_q12 = smlawb(ltp_pred_q12, pred_lag[-1], b_q14[1]);
            ltp_pred_q12 = smlawb(ltp_pred_q12, pred_lag[-2], b_q14[2]);
            ltp_pred_q12 = smlawb(ltp_pred_q12, pred_lag[-3], b_q14[3]);
            ltp_pred_q12 = smlawb(ltp_pred_q12, pred_lag[-4], b_q14[4]);
            pred_lag++;

            rand_seed = silk_rand(rand_seed);
            int j = (rand_seed >> 25) & (kRandBufSize - 1);
            slt_q14[slt_buf_idx] = static_cast<int32_t>(
                static_cast<uint32_t>(
                    smlawb(ltp_pred_q12, rand_ptr[j], rand_scale_q14))
                << 2);
            slt_buf_idx++;
        }

        // Decay the pitch gain and the noise gain per subframe.
        for (int j = 0; j < kLtpOrder; j++)
            b_q14[j] =
                static_cast<int16_t>(smulbb(harm_gain_q15, b_q14[j]) >> 15);
        rand_scale_q14 = static_cast<int16_t>(
            smulbb(rand_scale_q14, rand_gain_q15) >> 15);

        // Drift the pitch lag up slowly (capped at 18 ms).
        plc.pitch_l_q8 =
            smlawb(plc.pitch_l_q8, plc.pitch_l_q8, kPitchDriftFacQ16);
        plc.pitch_l_q8 = std::min(
            plc.pitch_l_q8, smulbb(kMaxPitchLagMs, st->fs_khz) << 8);
        lag = rshift_round(plc.pitch_l_q8, 8);
    }

    // LPC synthesis over the concealed residual (in place at the tail of
    // slt_q14, exactly like the reference's aliased buffers).
    int32_t* slpc = &slt_q14[st->ltp_mem_length - kMaxLpcOrder];
    std::memcpy(slpc, st->slpc_q14_buf, kMaxLpcOrder * sizeof(int32_t));

    for (int i = 0; i < st->frame_length; i++) {
        // order/2 counters the SMLAWB bias.
        int32_t lpc_pred_q10 = st->lpc_order >> 1;
        for (int j = 0; j < st->lpc_order; j++)
            lpc_pred_q10 =
                smlawb(lpc_pred_q10, slpc[kMaxLpcOrder + i - 1 - j], a_q12[j]);

        slpc[kMaxLpcOrder + i] = add_sat32(slpc[kMaxLpcOrder + i],
                                           lshift_sat32(lpc_pred_q10, 4));
        frame[i] = static_cast<int16_t>(sat16(rshift_round(
            smulww(slpc[kMaxLpcOrder + i], prev_gain_q10[1]), 8)));
    }

    std::memcpy(st->slpc_q14_buf, &slpc[st->frame_length],
                kMaxLpcOrder * sizeof(int32_t));

    plc.rand_seed = rand_seed;
    plc.rand_scale_q14 = rand_scale_q14;
    for (int i = 0; i < kMaxNbSubfr; i++) ctrl->pitch_lags[i] = lag;

    st->loss_cnt++;  // the reference silk_PLC increments after concealing
}

void plc_glue_frames(DecoderState* st, int16_t* frame, int length) {
    PlcState& plc = st->plc;
    if (st->loss_cnt) {
        sum_sqr_shift(&plc.conc_energy, &plc.conc_energy_shift, frame,
                      length);
        plc.last_frame_lost = 1;
        return;
    }
    if (plc.last_frame_lost) {
        // Energy of the first good frame, normalized against the concealed
        // one; fade in when the good frame is louder.
        int32_t energy;
        int energy_shift;
        sum_sqr_shift(&energy, &energy_shift, frame, length);
        if (energy_shift > plc.conc_energy_shift) {
            plc.conc_energy >>= energy_shift - plc.conc_energy_shift;
        } else if (energy_shift < plc.conc_energy_shift) {
            energy >>= plc.conc_energy_shift - energy_shift;
        }
        if (energy > plc.conc_energy) {
            int lz = clz32(static_cast<uint32_t>(plc.conc_energy)) - 1;
            plc.conc_energy = static_cast<int32_t>(
                static_cast<uint32_t>(plc.conc_energy) << lz);
            energy >>= std::max(24 - lz, 0);
            int32_t frac_q24 =
                plc.conc_energy / std::max(energy, int32_t{1});
            int32_t gain_q16 = static_cast<int32_t>(
                static_cast<uint32_t>(sqrt_approx(frac_q24)) << 4);
            int32_t slope_q16 = ((int32_t{1} << 16) - gain_q16) / length;
            // 4x steeper to avoid missing onsets after DTX.
            slope_q16 <<= 2;
            for (int i = 0; i < length; i++) {
                frame[i] = static_cast<int16_t>(smulwb(gain_q16, frame[i]));
                gain_q16 += slope_q16;
                if (gain_q16 > (1 << 16)) break;
            }
        }
    }
    plc.last_frame_lost = 0;
}

void cng_reset(DecoderState* st) {
    CngState& c = st->cng;
    int32_t nlsf_step_q15 = 32767 / (st->lpc_order + 1);
    int32_t nlsf_acc_q15 = 0;
    for (int i = 0; i < st->lpc_order; i++) {
        nlsf_acc_q15 += nlsf_step_q15;
        c.smth_nlsf_q15[i] = static_cast<int16_t>(nlsf_acc_q15);
    }
    c.smth_gain_q16 = 0;
    c.rand_seed = 3176576;
}

void cng(DecoderState* st, const DecoderControl& ctrl, int16_t* frame,
         int length) {
    CngState& c = st->cng;

    // Lazy reset on rate change / first use (same pattern as PLC).
    if (st->fs_khz != c.fs_khz) {
        cng_reset(st);
        c.fs_khz = st->fs_khz;
    }

    if (st->loss_cnt == 0 &&
        st->prev_signal_type == kTypeNoVoiceActivity) {
        // Inactive good frame: update the comfort-noise estimates.
        for (int i = 0; i < st->lpc_order; i++)
            c.smth_nlsf_q15[i] = static_cast<int16_t>(
                c.smth_nlsf_q15[i] +
                smulwb(static_cast<int32_t>(st->prev_nlsf_q15[i]) -
                           c.smth_nlsf_q15[i],
                       kCngNlsfSmthQ16));

        // Excitation history from the loudest subframe.
        int32_t max_gain_q16 = 0;
        int subfr = 0;
        for (int i = 0; i < st->nb_subfr; i++) {
            if (ctrl.gains_q16[i] > max_gain_q16) {
                max_gain_q16 = ctrl.gains_q16[i];
                subfr = i;
            }
        }
        std::memmove(&c.exc_buf_q14[st->subfr_length], c.exc_buf_q14,
                     (st->nb_subfr - 1) * st->subfr_length * sizeof(int32_t));
        std::memcpy(c.exc_buf_q14, &st->exc_q14[subfr * st->subfr_length],
                    st->subfr_length * sizeof(int32_t));

        // Smooth the gain; adapt down fast when 3 dB above a subframe.
        for (int i = 0; i < st->nb_subfr; i++) {
            c.smth_gain_q16 += smulwb(ctrl.gains_q16[i] - c.smth_gain_q16,
                                      kCngGainSmthQ16);
            if (smulww(c.smth_gain_q16, kCngGainSmthThresholdQ16) >
                ctrl.gains_q16[i])
                c.smth_gain_q16 = ctrl.gains_q16[i];
        }
    }

    if (st->loss_cnt) {
        // Add comfort noise on top of the concealed frame. Target gain =
        // sqrt(smoothed^2 - 32*residual^2): the PLC's own (decaying)
        // output already covers part of the noise floor.
        int32_t sig_q14[kMaxFrameLength + kMaxLpcOrder];
        int16_t a_q12[kMaxLpcOrder];

        int32_t gain_q16 =
            smulww(st->plc.rand_scale_q14, st->plc.prev_gain_q16[1]);
        if (gain_q16 >= (1 << 21) || c.smth_gain_q16 > (1 << 23)) {
            // Large gains: square in Q16-top halves to avoid overflow.
            gain_q16 = smultt(gain_q16, gain_q16);
            gain_q16 = sub_lshift32(smultt(c.smth_gain_q16, c.smth_gain_q16),
                                    gain_q16, 5);
            gain_q16 = static_cast<int32_t>(
                static_cast<uint32_t>(sqrt_approx(gain_q16)) << 16);
        } else {
            gain_q16 = smulww(gain_q16, gain_q16);
            gain_q16 = sub_lshift32(smulww(c.smth_gain_q16, c.smth_gain_q16),
                                    gain_q16, 5);
            gain_q16 = static_cast<int32_t>(
                static_cast<uint32_t>(sqrt_approx(gain_q16)) << 8);
        }
        int32_t gain_q10 = gain_q16 >> 6;

        cng_exc(&sig_q14[kMaxLpcOrder], c.exc_buf_q14, length, &c.rand_seed);

        // Shape with the smoothed NLSFs.
        nlsf2a(a_q12, c.smth_nlsf_q15, st->lpc_order);

        std::memcpy(sig_q14, c.synth_state, kMaxLpcOrder * sizeof(int32_t));
        for (int i = 0; i < length; i++) {
            // order/2 counters the SMLAWB bias.
            int32_t lpc_pred_q10 = st->lpc_order >> 1;
            for (int j = 0; j < st->lpc_order; j++)
                lpc_pred_q10 = smlawb(
                    lpc_pred_q10, sig_q14[kMaxLpcOrder + i - 1 - j], a_q12[j]);

            sig_q14[kMaxLpcOrder + i] = add_sat32(
                sig_q14[kMaxLpcOrder + i], lshift_sat32(lpc_pred_q10, 4));

            frame[i] = add_sat16(
                frame[i], sat16(rshift_round(
                              smulww(sig_q14[kMaxLpcOrder + i], gain_q10),
                              8)));
        }
        std::memcpy(c.synth_state, &sig_q14[length],
                    kMaxLpcOrder * sizeof(int32_t));
    } else {
        std::memset(c.synth_state, 0, st->lpc_order * sizeof(int32_t));
    }
}

}  // namespace silk
}  // namespace opus
}  // namespace glint

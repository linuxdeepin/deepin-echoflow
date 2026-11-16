// SILK frame side-info decoding — RFC 6716 sections 4.2.7.3-4.2.7.6
// MIT License - Clean-room implementation

#include "opus_silk_indices.hpp"

#include <cstring>

#include "opus_silk_math.hpp"

namespace glint {
namespace opus {
namespace silk {

namespace {
constexpr int kMinLagMs = 2;
constexpr int kMaxLagMs = 18;
// Gain quantizer levels cover 2..88 dB (log2 domain, Q7).
constexpr int kMaxDeltaGainQuant = 36;
constexpr int kMinDeltaGainQuant = -4;
constexpr int32_t kGainOffset = (2 * 128) / 6 + 16 * 128;
constexpr int32_t kInvScaleQ16 =
    (65536 * (((88 - 2) * 128) / 6)) / (kNLevelsQGain - 1);
}  // namespace

bool DecoderState::set_fs(int fs_khz_new, int nb_subfr_new,
                          int32_t fs_api_hz_new) {
    nb_subfr = nb_subfr_new;
    subfr_length = 5 * fs_khz_new;  // 5 ms subframes
    int frame_length_new = subfr_length * nb_subfr;

    bool resampler_changed =
        fs_khz != fs_khz_new || fs_api_hz != fs_api_hz_new;
    fs_api_hz = fs_api_hz_new;

    if (fs_khz != fs_khz_new || frame_length_new != frame_length) {
        if (fs_khz_new == 8) {
            pitch_contour_icdf = nb_subfr == kMaxNbSubfr
                                     ? kPitchContourNbIcdf
                                     : kPitchContour10MsNbIcdf;
        } else {
            pitch_contour_icdf = nb_subfr == kMaxNbSubfr
                                     ? kPitchContourIcdf
                                     : kPitchContour10MsIcdf;
        }
        if (fs_khz != fs_khz_new) {
            ltp_mem_length = 20 * fs_khz_new;
            if (fs_khz_new == 16) {
                nlsf_cb = &kNlsfCbWb;
                lpc_order = kMaxLpcOrder;
                pitch_lag_low_bits_icdf = kUniform8Icdf;
            } else {
                nlsf_cb = &kNlsfCbNbMb;
                lpc_order = 10;
                pitch_lag_low_bits_icdf =
                    fs_khz_new == 12 ? kUniform6Icdf : kUniform4Icdf;
            }
            // Internal-rate switch: history is meaningless, reset it.
            first_frame_after_reset = 1;
            lag_prev = 100;
            prev_gain_index = 10;
            prev_signal_type = kTypeNoVoiceActivity;
            std::memset(out_buf, 0, sizeof(out_buf));
            std::memset(slpc_q14_buf, 0, sizeof(slpc_q14_buf));
        }
        fs_khz = fs_khz_new;
        frame_length = frame_length_new;
    }
    return resampler_changed;
}

void nlsf_unpack(int16_t* ec_ix, uint8_t* pred_q8, const NlsfCodebook& cb,
                 int cb1_index) {
    const uint8_t* sel = &cb.ecSel[cb1_index * cb.order / 2];
    for (int i = 0; i < cb.order; i += 2) {
        // Each byte selects (table, predictor) for two coefficients.
        uint8_t entry = *sel++;
        ec_ix[i] = static_cast<int16_t>(((entry >> 1) & 7) *
                                        (2 * kNlsfQuantMaxAmplitude + 1));
        pred_q8[i] = cb.predQ8[i + (entry & 1) * (cb.order - 1)];
        ec_ix[i + 1] = static_cast<int16_t>(
            ((entry >> 5) & 7) * (2 * kNlsfQuantMaxAmplitude + 1));
        pred_q8[i + 1] =
            cb.predQ8[i + ((entry >> 4) & 1) * (cb.order - 1) + 1];
    }
}

void decode_indices(DecoderState* st, RangeDecoder& dec, int frame_index,
                    bool decode_lbrr, int cond_coding) {
    int16_t ec_ix[kMaxLpcOrder];
    uint8_t pred_q8[kMaxLpcOrder];

    // Signal type and quantizer offset, VAD-conditioned.
    int ix;
    if (decode_lbrr || st->vad_flags[frame_index])
        ix = dec.dec_icdf(kTypeOffsetVadIcdf, 8) + 2;
    else
        ix = dec.dec_icdf(kTypeOffsetNoVadIcdf, 8);
    st->indices.signal_type = static_cast<int8_t>(ix >> 1);
    st->indices.quant_offset_type = static_cast<int8_t>(ix & 1);

    // Gains: first subframe delta-coded only across dependent frames.
    if (cond_coding == kCodeConditionally) {
        st->indices.gains_indices[0] =
            static_cast<int8_t>(dec.dec_icdf(kDeltaGainIcdf, 8));
    } else {
        st->indices.gains_indices[0] = static_cast<int8_t>(
            dec.dec_icdf(kGainIcdf[st->indices.signal_type], 8) << 3);
        st->indices.gains_indices[0] = static_cast<int8_t>(
            st->indices.gains_indices[0] + dec.dec_icdf(kUniform8Icdf, 8));
    }
    for (int i = 1; i < st->nb_subfr; i++)
        st->indices.gains_indices[i] =
            static_cast<int8_t>(dec.dec_icdf(kDeltaGainIcdf, 8));

    // NLSF: stage-1 vector (type-conditioned half of the iCDF), then
    // per-coefficient stage-2 residuals with rail-extension escapes.
    st->indices.nlsf_indices[0] = static_cast<int8_t>(dec.dec_icdf(
        &st->nlsf_cb->cb1Icdf[(st->indices.signal_type >> 1) *
                               st->nlsf_cb->nVectors],
        8));
    nlsf_unpack(ec_ix, pred_q8, *st->nlsf_cb,
                st->indices.nlsf_indices[0]);
    for (int i = 0; i < st->nlsf_cb->order; i++) {
        ix = dec.dec_icdf(&st->nlsf_cb->ecIcdf[ec_ix[i]], 8);
        if (ix == 0)
            ix -= dec.dec_icdf(kNlsfExtIcdf, 8);
        else if (ix == 2 * kNlsfQuantMaxAmplitude)
            ix += dec.dec_icdf(kNlsfExtIcdf, 8);
        st->indices.nlsf_indices[i + 1] =
            static_cast<int8_t>(ix - kNlsfQuantMaxAmplitude);
    }

    // Interpolation factor exists only for 20 ms frames.
    if (st->nb_subfr == kMaxNbSubfr)
        st->indices.nlsf_interp_coef_q2 = static_cast<int8_t>(
            dec.dec_icdf(kNlsfInterpolationFactorIcdf, 8));
    else
        st->indices.nlsf_interp_coef_q2 = 4;

    if (st->indices.signal_type == kTypeVoiced) {
        // Pitch lag: delta from the previous frame when permitted and the
        // delta index is nonzero; otherwise absolute (coarse * fs/2 + fine).
        int decode_absolute = 1;
        if (cond_coding == kCodeConditionally &&
            st->ec_prev_signal_type == kTypeVoiced) {
            int delta = dec.dec_icdf(kPitchDeltaIcdf, 8);
            if (delta > 0) {
                st->indices.lag_index = static_cast<int16_t>(
                    st->ec_prev_lag_index + (delta - 9));
                decode_absolute = 0;
            }
        }
        if (decode_absolute) {
            st->indices.lag_index = static_cast<int16_t>(
                dec.dec_icdf(kPitchLagIcdf, 8) * (st->fs_khz >> 1));
            st->indices.lag_index = static_cast<int16_t>(
                st->indices.lag_index +
                dec.dec_icdf(st->pitch_lag_low_bits_icdf, 8));
        }
        st->ec_prev_lag_index = st->indices.lag_index;

        st->indices.contour_index =
            static_cast<int8_t>(dec.dec_icdf(st->pitch_contour_icdf, 8));

        // LTP: periodicity class, then a filter index per subframe.
        st->indices.per_index =
            static_cast<int8_t>(dec.dec_icdf(kLtpPerIndexIcdf, 8));
        for (int k = 0; k < st->nb_subfr; k++)
            st->indices.ltp_index[k] = static_cast<int8_t>(dec.dec_icdf(
                kLtpGainIcdfPtrs[st->indices.per_index], 8));

        if (cond_coding == kCodeIndependently)
            st->indices.ltp_scale_index =
                static_cast<int8_t>(dec.dec_icdf(kLtpScaleIcdf, 8));
        else
            st->indices.ltp_scale_index = 0;
    }
    st->ec_prev_signal_type = st->indices.signal_type;

    st->indices.seed = static_cast<int8_t>(dec.dec_icdf(kUniform4Icdf, 8));
}

void gains_dequant(int32_t* gain_q16, const int8_t* ind, int8_t* prev_ind,
                   int conditional, int nb_subfr) {
    for (int k = 0; k < nb_subfr; k++) {
        if (k == 0 && conditional == 0) {
            // Absolute index, but never more than 16 steps below the chain.
            int v = *prev_ind - 16;
            *prev_ind = static_cast<int8_t>(ind[k] > v ? ind[k] : v);
        } else {
            int ind_tmp = ind[k] + kMinDeltaGainQuant;
            // Deltas above the threshold count double (wide-range escape).
            int thr = 2 * kMaxDeltaGainQuant - kNLevelsQGain + *prev_ind;
            if (ind_tmp > thr)
                *prev_ind = static_cast<int8_t>(*prev_ind + (ind_tmp << 1) -
                                                thr);
            else
                *prev_ind = static_cast<int8_t>(*prev_ind + ind_tmp);
        }
        if (*prev_ind < 0) *prev_ind = 0;
        if (*prev_ind > kNLevelsQGain - 1)
            *prev_ind = kNLevelsQGain - 1;

        int32_t lg = smulwb(kInvScaleQ16, *prev_ind) + kGainOffset;
        gain_q16[k] = log2lin(lg < 3967 ? lg : 3967);
    }
}

void decode_pitch(int16_t lag_index, int8_t contour_index, int* pitch_lags,
                  int fs_khz, int nb_subfr) {
    const int8_t* cb;
    int cbk_size;
    if (fs_khz == 8) {
        if (nb_subfr == kMaxNbSubfr) {
            cb = &kCbLagsStage2[0][0];
            cbk_size = 11;
        } else {
            cb = &kCbLagsStage2_10ms[0][0];
            cbk_size = 3;
        }
    } else {
        if (nb_subfr == kMaxNbSubfr) {
            cb = &kCbLagsStage3[0][0];
            cbk_size = 34;
        } else {
            cb = &kCbLagsStage3_10ms[0][0];
            cbk_size = 12;
        }
    }
    int min_lag = kMinLagMs * fs_khz;
    int max_lag = kMaxLagMs * fs_khz;
    int lag = min_lag + lag_index;
    for (int k = 0; k < nb_subfr; k++) {
        int v = lag + cb[k * cbk_size + contour_index];
        pitch_lags[k] = v < min_lag ? min_lag : (v > max_lag ? max_lag : v);
    }
}

}  // namespace silk
}  // namespace opus
}  // namespace glint

// SILK top-level decoder — RFC 6716 section 4.2
// MIT License - Clean-room implementation

#include "opus_silk_decoder.hpp"

#include <cstring>

#include "opus_silk_excitation.hpp"
#include "opus_silk_frame.hpp"
#include "opus_silk_tables.hpp"

namespace glint {
namespace opus {
namespace silk {

int SilkDecoder::decode(RangeDecoder& dec, int16_t* samples_out,
                        int channels_api, int channels_internal,
                        int internal_khz, int32_t api_hz, int payload_ms,
                        bool new_packet, bool fec) {
    DecoderState* ch = channel_state;
    int32_t ms_pred_q13[2] = { 0, 0 };
    int decode_only_middle = 0;

    if (new_packet) {
        for (int n = 0; n < channels_internal; n++)
            ch[n].n_frames_decoded = 0;
    }

    // Mono -> stereo transition: fresh side-channel state.
    if (channels_internal > n_channels_internal)
        ch[1] = DecoderState();

    // Stereo -> mono collapse at an unchanged rate: the right output gets
    // one frame through the old side resampler to stay continuous.
    int stereo_to_mono = channels_internal == 1 &&
                         n_channels_internal == 2 &&
                         internal_khz == ch[0].fs_khz;

    if (ch[0].n_frames_decoded == 0) {
        int n_frames_per_packet, nb_subfr;
        switch (payload_ms) {
        case 10: n_frames_per_packet = 1; nb_subfr = 2; break;
        case 20: n_frames_per_packet = 1; nb_subfr = 4; break;
        case 40: n_frames_per_packet = 2; nb_subfr = 4; break;
        case 60: n_frames_per_packet = 3; nb_subfr = 4; break;
        default: return -11;
        }
        if (internal_khz != 8 && internal_khz != 12 && internal_khz != 16)
            return -12;
        for (int n = 0; n < channels_internal; n++) {
            ch[n].n_frames_per_packet = n_frames_per_packet;
            if (ch[n].set_fs(internal_khz, nb_subfr, api_hz))
                resampler[n].init(internal_khz,
                                  static_cast<int>(api_hz / 1000));
        }
    }

    // API/internal stereo transitions reset the unmix + side resampler.
    if (channels_api == 2 && channels_internal == 2 &&
        (n_channels_api == 1 || n_channels_internal == 1)) {
        std::memset(stereo.pred_prev_q13, 0, sizeof(stereo.pred_prev_q13));
        std::memset(stereo.s_side, 0, sizeof(stereo.s_side));
        resampler[1] = resampler[0];
    }
    n_channels_api = channels_api;
    n_channels_internal = channels_internal;

    if (ch[0].n_frames_decoded == 0) {
        // Header: per-frame VAD bits + LBRR flag, per channel; then the
        // LBRR distribution when flagged.
        for (int n = 0; n < channels_internal; n++) {
            for (int i = 0; i < ch[n].n_frames_per_packet; i++)
                ch[n].vad_flags[i] = dec.dec_bit_logp(1);
            ch[n].lbrr_flag = dec.dec_bit_logp(1);
        }
        for (int n = 0; n < channels_internal; n++) {
            std::memset(ch[n].lbrr_flags, 0, sizeof(ch[n].lbrr_flags));
            if (ch[n].lbrr_flag) {
                if (ch[n].n_frames_per_packet == 1) {
                    ch[n].lbrr_flags[0] = 1;
                } else {
                    int32_t sym =
                        dec.dec_icdf(
                            kLbrrFlagsIcdfPtr[ch[n].n_frames_per_packet -
                                              2],
                            8) +
                        1;
                    for (int i = 0; i < ch[n].n_frames_per_packet; i++)
                        ch[n].lbrr_flags[i] = (sym >> i) & 1;
                }
            }
        }

        // Normal decode: LBRR (redundancy) data is parsed and discarded.
        // FEC decode consumes it in the frame loop below instead.
        if (!fec)
        for (int i = 0; i < ch[0].n_frames_per_packet; i++) {
            for (int n = 0; n < channels_internal; n++) {
                if (!ch[n].lbrr_flags[i]) continue;
                int16_t pulses[320];
                if (channels_internal == 2 && n == 0) {
                    stereo_decode_pred(dec, ms_pred_q13);
                    if (ch[1].lbrr_flags[i] == 0)
                        decode_only_middle = stereo_decode_mid_only(dec);
                }
                int cond = i > 0 && ch[n].lbrr_flags[i - 1]
                               ? kCodeConditionally
                               : kCodeIndependently;
                decode_indices(&ch[n], dec, i, true, cond);
                decode_pulses(dec, pulses, ch[n].indices.signal_type,
                              ch[n].indices.quant_offset_type,
                              ch[n].frame_length);
            }
        }
    }

    // Stereo predictors for THIS frame (+ mid-only if the side is
    // silent). FEC: only when this frame has an LBRR copy; else reuse
    // the previous predictors.
    if (channels_internal == 2) {
        if (!fec || ch[0].lbrr_flags[ch[0].n_frames_decoded] == 1) {
            stereo_decode_pred(dec, ms_pred_q13);
            if ((!fec &&
                 ch[1].vad_flags[ch[0].n_frames_decoded] == 0) ||
                (fec && ch[1].lbrr_flags[ch[0].n_frames_decoded] == 0))
                decode_only_middle = stereo_decode_mid_only(dec);
            else
                decode_only_middle = 0;
        } else {
            ms_pred_q13[0] = stereo.pred_prev_q13[0];
            ms_pred_q13[1] = stereo.pred_prev_q13[1];
        }
    }

    // First side frame after mid-only coding starts from silence.
    if (channels_internal == 2 && decode_only_middle == 0 &&
        prev_decode_only_middle == 1) {
        std::memset(ch[1].out_buf, 0, sizeof(ch[1].out_buf));
        std::memset(ch[1].slpc_q14_buf, 0, sizeof(ch[1].slpc_q14_buf));
        ch[1].lag_prev = 100;
        ch[1].prev_gain_index = 10;
        ch[1].prev_signal_type = kTypeNoVoiceActivity;
        ch[1].first_frame_after_reset = 1;
    }

    int has_side =
        !fec ? !decode_only_middle
             : (!prev_decode_only_middle ||
                (channels_internal == 2 &&
                 ch[1].lbrr_flags[ch[1].n_frames_decoded] == 1));
    int16_t ch_out[2][322];  // frame + two-sample carry per channel
    int n_dec = ch[0].frame_length;

    for (int n = 0; n < channels_internal; n++) {
        if (n == 0 || has_side) {
            int frame_index = ch[0].n_frames_decoded - n;
            int cond;
            if (frame_index <= 0)
                cond = kCodeIndependently;
            else if (fec)
                cond = ch[n].lbrr_flags[frame_index - 1]
                           ? kCodeConditionally
                           : kCodeIndependently;
            else if (n > 0 && prev_decode_only_middle)
                cond = kCodeIndependentlyNoLtpScaling;
            else
                cond = kCodeConditionally;
            decode_frame(&ch[n], dec, &ch_out[n][2], cond, fec);
        } else {
            std::memset(&ch_out[n][2], 0, n_dec * sizeof(int16_t));
        }
        ch[n].n_frames_decoded++;
    }

    if (channels_api == 2 && channels_internal == 2) {
        stereo_ms_to_lr(&stereo, ch_out[0], ch_out[1], ms_pred_q13,
                        ch[0].fs_khz, n_dec);
    } else {
        // Mono: same two-sample carry, no unmix.
        std::memcpy(ch_out[0], stereo.s_mid, 2 * sizeof(int16_t));
        std::memcpy(stereo.s_mid, &ch_out[0][n_dec], 2 * sizeof(int16_t));
    }

    int n_out = static_cast<int>(
        static_cast<int64_t>(n_dec) * api_hz / (ch[0].fs_khz * 1000));

    int16_t resample_buf[960];
    int16_t* resample_out =
        channels_api == 2 ? resample_buf : samples_out;
    int n_mixed = channels_api < channels_internal ? channels_api
                                                   : channels_internal;
    for (int n = 0; n < n_mixed; n++) {
        // The unmix consumed one sample of delay; resample from offset 1.
        resampler[n].process(resample_out, &ch_out[n][1], n_dec);
        if (channels_api == 2) {
            for (int i = 0; i < n_out; i++)
                samples_out[n + 2 * i] = resample_out[i];
        }
    }

    // Mono stream on a stereo API: duplicate (or, right after a stereo
    // collapse, run the right channel through its own resampler once).
    if (channels_api == 2 && channels_internal == 1) {
        if (stereo_to_mono) {
            resampler[1].process(resample_out, &ch_out[0][1], n_dec);
            for (int i = 0; i < n_out; i++)
                samples_out[1 + 2 * i] = resample_out[i];
        } else {
            for (int i = 0; i < n_out; i++)
                samples_out[1 + 2 * i] = samples_out[2 * i];
        }
    }

    prev_decode_only_middle = decode_only_middle;
    return n_out;
}

int SilkDecoder::decode_lost(int16_t* samples_out, int channels_api,
                             int payload_ms, bool new_packet) {
    DecoderState* ch = channel_state;
    const int channels_internal =
        n_channels_internal > 0 ? n_channels_internal : 1;
    const int internal_khz = ch[0].fs_khz > 0 ? ch[0].fs_khz : 16;
    const int32_t api_hz = ch[0].fs_api_hz > 0 ? ch[0].fs_api_hz : 48000;
    int32_t ms_pred_q13[2];

    if (new_packet) {
        for (int n = 0; n < channels_internal; n++)
            ch[n].n_frames_decoded = 0;
    }
    if (ch[0].n_frames_decoded == 0) {
        int n_frames_per_packet, nb_subfr;
        switch (payload_ms) {
        case 10: n_frames_per_packet = 1; nb_subfr = 2; break;
        case 20: n_frames_per_packet = 1; nb_subfr = 4; break;
        case 40: n_frames_per_packet = 2; nb_subfr = 4; break;
        case 60: n_frames_per_packet = 3; nb_subfr = 4; break;
        default: return -11;
        }
        for (int n = 0; n < channels_internal; n++) {
            ch[n].n_frames_per_packet = n_frames_per_packet;
            if (ch[n].set_fs(internal_khz, nb_subfr, api_hz))
                resampler[n].init(internal_khz,
                                  static_cast<int>(api_hz / 1000));
        }
    }

    // Predictors carry over; no bitstream to read.
    ms_pred_q13[0] = stereo.pred_prev_q13[0];
    ms_pred_q13[1] = stereo.pred_prev_q13[1];

    int has_side = !prev_decode_only_middle;
    int16_t ch_out[2][322];
    int n_dec = ch[0].frame_length;

    for (int n = 0; n < channels_internal; n++) {
        if (n == 0 || has_side) {
            int frame_index = ch[0].n_frames_decoded - n;
            int cond = frame_index <= 0
                           ? kCodeIndependently
                           : (n > 0 && prev_decode_only_middle
                                  ? kCodeIndependentlyNoLtpScaling
                                  : kCodeConditionally);
            decode_frame(&ch[n], nullptr, &ch_out[n][2], cond,
                         /*lost=*/true);
        } else {
            std::memset(&ch_out[n][2], 0, n_dec * sizeof(int16_t));
        }
        ch[n].n_frames_decoded++;
    }

    if (channels_api == 2 && channels_internal == 2) {
        stereo_ms_to_lr(&stereo, ch_out[0], ch_out[1], ms_pred_q13,
                        ch[0].fs_khz, n_dec);
    } else {
        std::memcpy(ch_out[0], stereo.s_mid, 2 * sizeof(int16_t));
        std::memcpy(stereo.s_mid, &ch_out[0][n_dec], 2 * sizeof(int16_t));
    }

    int n_out = static_cast<int>(
        static_cast<int64_t>(n_dec) * api_hz / (ch[0].fs_khz * 1000));

    int16_t resample_buf[960];
    int16_t* resample_out =
        channels_api == 2 ? resample_buf : samples_out;
    int n_mixed = channels_api < channels_internal ? channels_api
                                                   : channels_internal;
    for (int n = 0; n < n_mixed; n++) {
        resampler[n].process(resample_out, &ch_out[n][1], n_dec);
        if (channels_api == 2) {
            for (int i = 0; i < n_out; i++)
                samples_out[n + 2 * i] = resample_out[i];
        }
    }
    if (channels_api == 2 && channels_internal == 1) {
        for (int i = 0; i < n_out; i++)
            samples_out[1 + 2 * i] = samples_out[2 * i];
    }

    // Post-loss: drop the gain-chain clamp so energy can come back down.
    for (int n = 0; n < channels_internal; n++)
        ch[n].prev_gain_index = 10;
    // prev_decode_only_middle intentionally NOT updated on loss.
    return n_out;
}

}  // namespace silk
}  // namespace opus
}  // namespace glint

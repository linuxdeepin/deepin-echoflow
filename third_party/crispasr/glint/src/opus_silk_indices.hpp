// SILK frame side-info decoding — RFC 6716 sections 4.2.7.3-4.2.7.6
// MIT License - Clean-room implementation
//
// The wire layer of one SILK frame: signal type / quantizer offset, gain
// indices (independent MSB+LSB or delta-coded), the two-stage NLSF vector
// quantizer indices (with extension escapes at both rails), the optional
// NLSF interpolation factor, and — for voiced frames — pitch lag
// (absolute or delta), pitch contour, LTP filter indices per subframe,
// LTP scaling, plus the excitation LCG seed.

#pragma once

#include <cstdint>

#include "opus_ec.hpp"
#include "opus_silk_plc.hpp"
#include "opus_silk_tables.hpp"

namespace glint {
namespace opus {
namespace silk {

constexpr int kMaxNbSubfr = 4;
constexpr int kTypeNoVoiceActivity = 0;
constexpr int kTypeUnvoiced = 1;
constexpr int kTypeVoiced = 2;
// Conditional-coding modes (which inter-frame predictions are allowed).
constexpr int kCodeIndependently = 0;
constexpr int kCodeIndependentlyNoLtpScaling = 1;
constexpr int kCodeConditionally = 2;

// All quantization indices of one SILK frame. Zero-initialized like the
// reference decoder state: fields not decoded for a frame type (e.g. pitch
// for unvoiced) keep their previous/initial values.
struct SideInfoIndices {
    int8_t gains_indices[kMaxNbSubfr] = {};
    int8_t ltp_index[kMaxNbSubfr] = {};
    int8_t nlsf_indices[kMaxLpcOrder + 1] = {};  // stage-1 + residuals
    int16_t lag_index = 0;
    int8_t contour_index = 0;
    int8_t signal_type = 0;
    int8_t quant_offset_type = 0;
    int8_t nlsf_interp_coef_q2 = 0;
    int8_t per_index = 0;
    int8_t ltp_scale_index = 0;
    int8_t seed = 0;
};

// Decoder state (grows as more of the decode chain lands). set_fs() derives
// everything sampling-rate-dependent, mirroring the reference.
struct DecoderState {
    int fs_khz = 0;              // internal rate: 8, 12 or 16
    int nb_subfr = 0;            // 2 (10 ms) or 4 (20 ms)
    int frame_length = 0;
    int subfr_length = 0;
    int ltp_mem_length = 0;
    int lpc_order = 0;           // 10 (NB/MB) or 16 (WB)
    const NlsfCodebook* nlsf_cb = nullptr;
    const uint8_t* pitch_lag_low_bits_icdf = nullptr;
    const uint8_t* pitch_contour_icdf = nullptr;
    int vad_flags[3] = { 0, 0, 0 };
    int lbrr_flag = 0;
    int lbrr_flags[3] = { 0, 0, 0 };
    int n_frames_per_packet = 0;
    int32_t fs_api_hz = 0;
    int8_t ec_prev_signal_type = 0;
    int16_t ec_prev_lag_index = 0;
    int8_t prev_gain_index = 0;  // reference LastGainIndex (gain chain)
    SideInfoIndices indices;

    // Synthesis state (decode_core / decode_frame).
    int n_frames_decoded = 0;
    int first_frame_after_reset = 1;
    int loss_cnt = 0;
    int8_t prev_signal_type = 0;
    int lag_prev = 0;   // set_fs puts it at 100 on any rate change
    int32_t prev_gain_q16 = 65536;
    int16_t prev_nlsf_q15[kMaxLpcOrder] = {};
    int16_t out_buf[320 + 2 * 80] = {};   // ltp_mem + 2 subframes
    int32_t slpc_q14_buf[kMaxLpcOrder] = {};
    int32_t exc_q14[320] = {};

    // Packet-loss concealment + comfort noise (opus_silk_plc.hpp). Their
    // defaults mirror silk_init_decoder; both reset themselves lazily when
    // fs_khz changes (the reference does NOT reset them in decoder_set_fs).
    PlcState plc;
    CngState cng;

    // Mirrors the reference decoder_set_fs: rate-dependent tables plus the
    // guarded state resets on an internal-rate change. Returns true when
    // (fs_khz, fs_api_hz) changed, i.e. the output resampler must be
    // re-initialized (held by the top-level decoder here, not this state).
    bool set_fs(int fs_khz_new, int nb_subfr_new, int32_t fs_api_hz_new);
    // Convenience for fixed-rate harnesses.
    void set_fs(int fs_khz_new, int nb_subfr_new) {
        set_fs(fs_khz_new, nb_subfr_new, 48000);
    }
};

// Select the stage-2 entropy tables and predictors for a stage-1 vector.
void nlsf_unpack(int16_t* ec_ix, uint8_t* pred_q8, const NlsfCodebook& cb,
                 int cb1_index);

// Decode all side info of one frame into st->indices.
void decode_indices(DecoderState* st, RangeDecoder& dec, int frame_index,
                    bool decode_lbrr, int cond_coding);

// Gain indices -> linear Q16 gains, chained through *prev_ind.
void gains_dequant(int32_t* gain_q16, const int8_t* ind, int8_t* prev_ind,
                   int conditional, int nb_subfr);

// Lag + contour indices -> per-subframe pitch lags (samples).
void decode_pitch(int16_t lag_index, int8_t contour_index, int* pitch_lags,
                  int fs_khz, int nb_subfr);

}  // namespace silk
}  // namespace opus
}  // namespace glint

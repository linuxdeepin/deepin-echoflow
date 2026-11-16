// SILK packet-loss concealment + comfort noise — RFC 6716 section 4.4
// MIT License - Clean-room implementation
//
// PLC extrapolates lost frames from the last good frame's state: the LPC
// residual is rebuilt from an attenuated single-tap pitch predictor plus
// scaled random re-use of the previous excitation, then run through the
// bandwidth-expanded previous LPC filter. CNG keeps a smoothed NLSF/gain/
// excitation estimate updated on inactive good frames and adds shaped
// comfort noise to concealed frames. Glue fades the first good frame after
// a loss toward the concealed energy. All exact fixed point — byte-identical
// with the reference decoder (gate: tools/crosscheck_opus_silk_frame.py,
// which mixes good and lost frames).
//
// Reset semantics (mirrors the reference exactly): silk_init_decoder zeroes
// the structs, then resets them while frame_length/LPC_order are still 0 —
// the default member values below reproduce that. On an internal-rate
// change the reference does NOT reset in decoder_set_fs; both silk_PLC and
// silk_CNG lazily reset when their own fs_khz snapshot mismatches the
// decoder's, which the fs_khz == 0 default also triggers on first use.

#pragma once

#include <cstdint>

#include "opus_silk_tables.hpp"

namespace glint {
namespace opus {
namespace silk {

struct DecoderState;    // opus_silk_indices.hpp
struct DecoderControl;  // opus_silk_frame.hpp

// Reference silk_PLC_struct (minus the deep-PLC field, not compiled here).
struct PlcState {
    int32_t pitch_l_q8 = 0;                  // concealment pitch lag, Q8
    int16_t ltp_coef_q14[kLtpOrder] = {};    // concealment LTP filter
    int16_t prev_lpc_q12[kMaxLpcOrder] = {};
    int last_frame_lost = 0;
    int32_t rand_seed = 0;                   // excitation-reuse index LCG
    int16_t rand_scale_q14 = 0;              // random-component gain
    int32_t conc_energy = 0;                 // concealed-frame energy (glue)
    int conc_energy_shift = 0;
    int16_t prev_ltp_scale_q14 = 0;
    int32_t prev_gain_q16[2] = { 1 << 16, 1 << 16 };  // last two gains
    int fs_khz = 0;                          // 0 = needs reset (lazy)
    int nb_subfr = 2;                        // geometry of the saved state
    int subfr_length = 20;
};

// Reference silk_CNG_struct. rand_seed default per silk_CNG_Reset; the NLSF
// smoother starts all-zero because LPC_order is 0 at init time (the lazy fs
// reset rebuilds it with the real order before first use).
struct CngState {
    int32_t exc_buf_q14[320] = {};           // MAX_FRAME_LENGTH history
    int16_t smth_nlsf_q15[kMaxLpcOrder] = {};
    int32_t synth_state[kMaxLpcOrder] = {};
    int32_t smth_gain_q16 = 0;
    int32_t rand_seed = 3176576;
    int fs_khz = 0;                          // 0 = needs reset (lazy)
};

// Reference silk_PLC_Reset: re-center the concealment pitch (half a frame)
// and unity gains; the random/LPC memories deliberately survive.
void plc_reset(DecoderState* st);

// Reference silk_PLC(..., lost == 0): snapshot the last good frame's pitch,
// LTP gain (centered single tap, clamped to [0.7, 0.95]), LPC, LTP scale
// and last two gains for later concealment. Call on every clean frame,
// after decode_core and the out_buf update.
void plc_update(DecoderState* st, DecoderControl* ctrl);

// Reference silk_PLC(..., lost == 1): synthesize frame_length concealed
// samples into `frame`, decay the concealment state, write the drifted lag
// into all ctrl->pitch_lags entries, and increment st->loss_cnt.
void plc_conceal(DecoderState* st, DecoderControl* ctrl, int16_t* frame);

// Reference silk_PLC_glue_frames: on lost frames record the concealed
// energy; on the first good frame after a loss, fade the output in from the
// concealed energy level (4x-steepened slope) to hide the level jump.
void plc_glue_frames(DecoderState* st, int16_t* frame, int length);

// Reference silk_CNG_Reset: mid-spaced NLSF ladder for st->lpc_order.
void cng_reset(DecoderState* st);

// Reference silk_CNG: on inactive good frames update the smoothed NLSF /
// gain / excitation estimates; on lost frames (st->loss_cnt != 0) add
// comfort noise shaped by the smoothed LPC on top of the PLC output.
void cng(DecoderState* st, const DecoderControl& ctrl, int16_t* frame,
         int length);

}  // namespace silk
}  // namespace opus
}  // namespace glint

// SILK sample-rate converter (decode side) — RFC 6716 section 4.2.9
// MIT License - Clean-room implementation
//
// Converts the SILK decoder's internal rate (8/12/16 kHz) to the Opus API
// rate. Exact fixed point: output is byte-identical with the reference
// resampler (gate: tools/crosscheck_opus_silk_resampler.py).
//
// Decoder reachability analysis (reference silk_resampler_init, forEnc=0):
// the only decode-side init is silk/decoder_set_fs.c with
// fs_in = fs_kHz*1000, fs_kHz in {8,12,16}, and fs_out = the API rate,
// which opus_decoder_init restricts to {8,12,16,24,48} kHz. Processing is
// silk/dec_API.c: one silk_resampler() call per channel per SILK frame,
// in_len = 10 or 20 ms at the internal rate. That makes 15 reachable
// (in,out) pairs across four kernels:
//
//   copy    : 8->8, 12->12, 16->16              (delay 4, 9, 12)
//   up2_HQ  : 8->16, 12->24        (exact 2x)   (delay 2, 7)
//   IIR_FIR : 8->12, 8->24, 8->48, 12->16,      (2x allpass upsample +
//             12->48, 16->24, 16->48             12-phase FIR interpolation)
//   down_FIR: 12->8 (2:3, 18-tap/2-frac),       (AR2 + polyphase FIR)
//             16->8 (1:2, 24-tap),
//             16->12 (3:4, 18-tap/3-frac)
//
// NOT decoder-reachable (and therefore not implemented): the 36-tap
// down-FIR branches (ratios 1:3 / 1:4 / 1:6 need fs_in >= 24 kHz, which
// only the encoder passes), silk_resampler_down2 / down2_3 (encoder VAD
// and pitch analysis only), the encoder delay matrix, and the LQ 2/3
// coefficients.
//
// Call contract (mirrors the reference asserts): in_len >= fs_in_khz
// (1 ms), out must hold in_len * fs_out / fs_in samples. The first
// input_delay samples of each call are consumed from the previous call's
// tail (delay equalization across modes), so the stream is continuous
// across calls of any admissible length.

#pragma once

#include <cstdint>

namespace glint {
namespace opus {
namespace silk {

struct Resampler {
    // State (reference silk_resampler_state_struct).
    int32_t s_iir[6];                 // allpass / AR2 states
    union {
        int32_t i32[24];              // down_FIR history (max order 24)
        int16_t i16[8];               // IIR_FIR history (RESAMPLER_ORDER_FIR_12)
    } s_fir;
    int16_t delay_buf[16];            // <= 1 ms at 16 kHz
    int fn;                           // kernel selector
    int batch_size;                   // 10 ms of input samples
    int32_t inv_ratio_q16;            // in/out ratio, rounded up
    int fir_order;
    int fir_fracs;
    int fs_in_khz;
    int fs_out_khz;
    int input_delay;
    const int16_t* coefs;

    // fs_in_khz in {8,12,16}, fs_out_khz in {8,12,16,24,48}.
    // Returns 0 on success, -1 on an unreachable rate pair.
    int init(int fs_in_khz, int fs_out_khz);

    // Convert in_len input samples (in_len >= fs_in_khz) into
    // in_len * fs_out / fs_in output samples. State carries across calls.
    void process(int16_t* out, const int16_t* in, int in_len);
};

}  // namespace silk
}  // namespace opus
}  // namespace glint

// SILK frame decoding: parameters + core synthesis — RFC 6716 section 4.2.7
// MIT License - Clean-room implementation
//
// decode_parameters assembles per-subframe controls from the decoded
// indices (linear gains, LPC coefficients for both interpolation halves via
// the NLSF chain, LTP filter taps, pitch lags); decode_core runs the
// excitation reconstruction (LCG dithering, quantization offsets) through
// the LTP (5-tap, voiced only, with rewhitened history) and short-term LPC
// synthesis filters, all in exact fixed point; decode_frame glues them to
// the range decoder and maintains the output history buffer.
//
// PLC and CNG (opus_silk_plc.hpp) hook in at frame level: clean frames run
// the state upkeep (silk_PLC update + silk_CNG estimation + glue fade-in
// after a loss); lost frames are synthesized entirely from state — no range
// decoder reads. Both paths are byte-identical with the reference
// silk_decode_frame (fuzz gate mixes good and lost frames).

#pragma once

#include <cstdint>

#include "opus_ec.hpp"
#include "opus_silk_indices.hpp"

namespace glint {
namespace opus {
namespace silk {

// Per-frame derived controls (reference silk_decoder_control).
struct DecoderControl {
    int pitch_lags[kMaxNbSubfr];
    int32_t gains_q16[kMaxNbSubfr];
    int16_t pred_coef_q12[2][kMaxLpcOrder];  // [half][coef]
    int16_t ltp_coef_q14[kLtpOrder * kMaxNbSubfr];
    int32_t ltp_scale_q14;
};

void decode_parameters(DecoderState* st, DecoderControl* ctrl,
                       int cond_coding);

// ctrl is I/O: on the first good frame after voiced concealment the
// voiced-PLC transition patch rewrites the LTP taps and pitch lags of the
// first half's subframes (the PLC update afterwards must see that).
void decode_core(DecoderState* st, DecoderControl& ctrl, int16_t* xq,
                 const int16_t* pulses);

// Whitening filter used to rebuild the LTP history from past output.
void lpc_analysis_filter(int16_t* out, const int16_t* in,
                         const int16_t* b_q12, int len, int order);

// int16 bandwidth expansion (post-loss LPC smoothing).
void bwexpander(int16_t* ar, int order, int32_t chirp_q16);

// Decode one frame into xq (returns frame_length samples). lost == true
// conceals from decoder state alone — dec may be null and is never read
// (reference silk_decode_frame with FLAG_PACKET_LOST). lbrr == true
// decodes the frame's LBRR (redundancy) copy; if this frame carries no
// LBRR data, it falls back to concealment, exactly like the reference.
int decode_frame(DecoderState* st, RangeDecoder* dec, int16_t* xq,
                 int cond_coding, bool lost, bool lbrr = false);
// Clean-frame convenience (the pre-PLC signature; other decode layers use
// this form).
inline int decode_frame(DecoderState* st, RangeDecoder& dec, int16_t* xq,
                        int cond_coding, bool lbrr = false) {
    return decode_frame(st, &dec, xq, cond_coding, false, lbrr);
}

}  // namespace silk
}  // namespace opus
}  // namespace glint

// SILK excitation (pulse) decoding — RFC 6716 section 4.2.7.8
// MIT License - Clean-room implementation
//
// The excitation is coded per 16-sample shell blocks: a frame-wide rate
// level selects the pulse-count PDF; counts above 16 chain LSB extensions;
// each block's pulse positions come from a fixed binary-split ("shell")
// tree; extension LSBs then refine every sample; signs are coded only for
// nonzero samples with a PDF conditioned on (signal type, quantization
// offset type, pulse count).

#pragma once

#include <cstdint>

#include "opus_ec.hpp"

namespace glint {
namespace opus {
namespace silk {

// frame_length is in samples at the internal rate (max 320 = 20 ms @ 16 kHz;
// the 10 ms @ 12 kHz case rounds up to a partial last block).
void decode_pulses(RangeDecoder& dec, int16_t* pulses, int signal_type,
                   int quant_offset_type, int frame_length);

}  // namespace silk
}  // namespace opus
}  // namespace glint

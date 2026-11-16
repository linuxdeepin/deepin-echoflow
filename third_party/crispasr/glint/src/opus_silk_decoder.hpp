// SILK top-level decoder — RFC 6716 section 4.2 (dec_API.c equivalent)
// MIT License - Clean-room implementation
//
// Orchestrates one SILK "Opus frame": header flags (per-frame VAD + LBRR
// flag + LBRR distribution), LBRR data skipping on normal decode, stereo
// predictor/mid-only decode, per-channel frame decode with the conditional-
// coding rules, MS->LR unmixing (or the mono two-sample carry), and
// resampling from the internal rate (8/12/16 kHz) to the API rate.
//
// Scope: clean streams (no PLC yet — PLAN § O2). Multi-frame packets
// (40/60 ms) are handled by calling decode() once per 20 ms frame with
// new_packet set only on the first, exactly like opus_decode_native.

#pragma once

#include <cstdint>

#include "opus_ec.hpp"
#include "opus_silk_indices.hpp"
#include "opus_silk_resampler.hpp"
#include "opus_silk_stereo.hpp"

namespace glint {
namespace opus {
namespace silk {

struct SilkDecoder {
    DecoderState channel_state[2];
    StereoDecState stereo;
    Resampler resampler[2];
    int n_channels_api = 0;
    int n_channels_internal = 0;
    int prev_decode_only_middle = 0;

    // Decode one 10/20 ms SILK frame set from dec into interleaved int16
    // PCM at api_hz. payload_ms is the TOC audio size (10/20/40/60);
    // frames beyond the first are decoded by repeated calls with
    // new_packet = false. Returns samples per channel, or negative error.
    // fec == true decodes the packet's LBRR (redundancy) copies instead
    // of the normal frames (reference silk_Decode FLAG_DECODE_LBRR).
    int decode(RangeDecoder& dec, int16_t* samples_out, int channels_api,
               int channels_internal, int internal_khz, int32_t api_hz,
               int payload_ms, bool new_packet, bool fec = false);

    // Conceal one frame set (lost packet / DTX / transition source) using
    // the previous configuration; no bitstream reads
    // (reference silk_Decode with FLAG_PACKET_LOST).
    int decode_lost(int16_t* samples_out, int channels_api, int payload_ms,
                    bool new_packet);
};

}  // namespace silk
}  // namespace opus
}  // namespace glint

// glint - Opus multistream (surround) decoder — RFC 7845 mapping family 1
// MIT License - Clean-room implementation
//
// A multistream payload is the streams' packets back to back: streams
// 0..N-2 use self-delimited framing (opus_packet_parse_ext), the last
// uses regular framing. Stream s < coupled_streams decodes stereo and
// occupies mapping indices {2s, 2s+1} (left/right); mono stream s
// occupies index coupled_streams + s; mapping[c] == 255 mutes output
// channel c. Reference: opus_multistream_decoder.c.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "opus_decoder.hpp"

namespace glint {
namespace opus {

class OpusMsDecoder {
public:
    // channels: output channel count (1..8 for family 1); streams: total
    // elementary streams; coupled: how many of them are stereo;
    // mapping[channels]: stream-channel index per output channel.
    // fs: output rate like OpusDecoder::init. Returns 0 or -1 on an
    // invalid layout.
    int init(int channels, int streams, int coupled,
             const uint8_t* mapping, int32_t fs = 48000);

    // Decode one multistream packet into interleaved float PCM
    // (channels() wide). Returns samples per channel or a negative
    // error. max_samples guards pcm (per channel).
    int decode(const uint8_t* data, int32_t len, float* pcm,
               int max_samples);

    int channels() const { return channels_; }
    // XOR of the streams' final ranges (the reference multistream
    // OPUS_GET_FINAL_RANGE convention).
    uint32_t final_range() const;

private:
    int channels_ = 0;
    int streams_ = 0;
    int coupled_ = 0;
    uint8_t mapping_[8] = {};
    std::vector<std::unique_ptr<OpusDecoder>> dec_;
    std::vector<float> buf_;
};

}  // namespace opus
}  // namespace glint

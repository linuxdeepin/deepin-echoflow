// glint - Opus multistream (surround) decoder — RFC 7845 mapping family 1
// MIT License - Clean-room implementation

#include "opus_ms_decoder.hpp"

#include <cstring>

namespace glint {
namespace opus {

int OpusMsDecoder::init(int channels, int streams, int coupled,
                        const uint8_t* mapping, int32_t fs) {
    if (channels < 1 || channels > 8 || streams < 1 ||
        coupled < 0 || coupled > streams || streams + coupled > 255)
        return -1;
    for (int c = 0; c < channels; c++) {
        if (mapping[c] != 255 && mapping[c] >= streams + coupled)
            return -1;
    }
    channels_ = channels;
    streams_ = streams;
    coupled_ = coupled;
    std::memcpy(mapping_, mapping, static_cast<size_t>(channels));
    dec_.clear();
    for (int s = 0; s < streams; s++) {
        dec_.push_back(std::make_unique<OpusDecoder>());
        dec_.back()->init(s < coupled ? 2 : 1, fs);
    }
    buf_.assign(2 * 5760, 0.0f);
    return 0;
}

int OpusMsDecoder::decode(const uint8_t* data, int32_t len, float* pcm,
                          int max_samples) {
    if (data == nullptr || len <= 0) return -1;
    if (len < 2 * streams_ - 1) return -4;

    int frame_out = -1;
    for (int s = 0; s < streams_; s++) {
        const bool self_delimited = s != streams_ - 1;
        OpusPacket pkt;
        int32_t packet_offset = 0;
        int err = opus_packet_parse_ext(data, len, self_delimited, &pkt,
                                        &packet_offset);
        if (err) return err;

        int stream_ch = s < coupled_ ? 2 : 1;
        int total = dec_[s]->decode_parsed(pkt, buf_.data(), 5760);
        if (total < 0) return total;
        if (frame_out < 0) {
            frame_out = total;
            if (frame_out > max_samples) return -2;
        } else if (total != frame_out) {
            return -4;
        }

        // Distribute this stream's channels per the mapping.
        for (int c = 0; c < channels_; c++) {
            int src = -1;
            if (s < coupled_) {
                if (mapping_[c] == 2 * s) src = 0;
                if (mapping_[c] == 2 * s + 1) src = 1;
            } else {
                if (mapping_[c] == coupled_ + s) src = 0;
            }
            if (src < 0) continue;
            for (int i = 0; i < frame_out; i++)
                pcm[i * channels_ + c] = buf_[i * stream_ch + src];
        }

        data += packet_offset;
        len -= packet_offset;
    }

    // Muted channels.
    for (int c = 0; c < channels_; c++) {
        if (mapping_[c] != 255) continue;
        for (int i = 0; i < frame_out; i++) pcm[i * channels_ + c] = 0.0f;
    }
    return frame_out;
}

uint32_t OpusMsDecoder::final_range() const {
    uint32_t r = 0;
    for (const auto& d : dec_) r ^= d->final_range();
    return r;
}

}  // namespace opus
}  // namespace glint

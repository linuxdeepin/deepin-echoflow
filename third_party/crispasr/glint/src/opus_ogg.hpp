// Ogg Opus container reading — RFC 3533 (Ogg) + RFC 7845 (Opus mapping)
// MIT License - Clean-room implementation
//
// Minimal demuxer for decoding .opus files: page parsing with CRC checks,
// packet reassembly (lacing + cross-page continuation), the OpusHead /
// OpusTags header packets, and the edit-list semantics (pre-skip trimming
// at the front, granule-position trimming at the end, Q7.8 dB output gain).
//
// Scope: single logical stream (the first Opus stream found), channel
// mapping families 0 and 1 with up to 2 channels; multistream/surround is
// deferred (PLAN § O3).

#pragma once

#include <cstddef> // size_t — GCC needs it explicitly (MSVC/clang pull it in transitively)
#include <cstdint>
#include <vector>

namespace glint {
namespace opus {

struct OpusHead {
    int version = 0;
    int channels = 0;
    int pre_skip = 0;
    uint32_t input_sample_rate = 0;
    int output_gain_q8 = 0;  // Q7.8 dB, applied by the decoder
    int mapping_family = 0;
    // Family 1 (surround): stream layout + per-channel mapping table.
    // Family 0 fills the equivalent trivial layout.
    int stream_count = 0;
    int coupled_count = 0;
    uint8_t mapping[8] = {};
};

class OggOpusReader {
public:
    // Parse from a whole file in memory. Returns 0 or a negative error
    // (-1 malformed, -2 not an Opus stream, -3 unsupported mapping).
    int parse(const uint8_t* data, size_t len);

    const OpusHead& head() const { return head_; }
    // Audio packets in order (views into the caller's buffer copied out).
    int packet_count() const { return static_cast<int>(packets_.size()); }
    const std::vector<uint8_t>& packet(int i) const { return packets_[i]; }
    // Total PCM samples per channel at 48 kHz after the edit list
    // (granule of the last page minus pre-skip); -1 if unknown.
    int64_t total_samples() const { return total_samples_; }
    // Linear gain factor from output_gain_q8.
    double output_gain() const;

private:
    OpusHead head_;
    std::vector<std::vector<uint8_t>> packets_;
    int64_t total_samples_ = -1;
};

// Ogg page CRC (poly 0x04C11DB7, no reflection, zero init/xor), exposed
// for tests.
uint32_t ogg_crc(const uint8_t* data, size_t len);

// Ogg Opus writer (RFC 7845 muxing): OpusHead/OpusTags header pages, then
// audio packets with 48 kHz granule positions (pre-skip included per the
// spec). Packets up to one page (255 segments) each; one packet per page
// for simplicity (valid, just slightly less compact).
class OggOpusWriter {
public:
    // pre_skip: decoder samples to discard (the encoder's delay).
    void begin(int channels, int pre_skip, uint32_t input_sample_rate);
    // samples48: duration of this packet per channel at 48 kHz.
    void add_packet(const uint8_t* data, size_t len, int samples48);
    // Marks the final page (EOS) and returns the file bytes.
    const std::vector<uint8_t>& finish();

private:
    void write_page(const uint8_t* body, size_t len, int htype,
                    uint64_t granule);

    std::vector<uint8_t> out_;
    std::vector<uint8_t> pending_;
    int pending_samples_ = 0;
    uint32_t serial_ = 0x676c6e74;  // 'glnt'
    uint32_t pageno_ = 0;
    uint64_t granule_ = 0;
    int pre_skip_ = 0;
    bool open_ = false;
};

}  // namespace opus
}  // namespace glint

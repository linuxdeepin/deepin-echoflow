// glint - FLAC decoder
// MIT License - Clean-room implementation
//
// Dependency-free decoder for native FLAC streams. Implemented from the FLAC
// format description only; no third-party codec source consulted.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace glint {
namespace flac {

struct StreamInfo {
    int min_block_size = 0;
    int max_block_size = 0;
    int sample_rate = 0;
    int channels = 0;
    int bits_per_sample = 0;
    uint64_t total_samples = 0;
    bool valid = false;
};

// Decode a complete native FLAC stream ("fLaC" marker + metadata + frames) to
// interleaved float PCM (+-1.0). Returns 0 on success, negative on malformed or
// unsupported input.
int decode(const uint8_t* data, size_t len, std::vector<float>& pcm,
           int& sample_rate, int& channels);

// Internals exposed for focused tests.
int parse_streaminfo(const uint8_t* data, size_t len, StreamInfo& si);
uint8_t crc8(const uint8_t* data, size_t len);
uint16_t crc16(const uint8_t* data, size_t len);

}  // namespace flac
}  // namespace glint

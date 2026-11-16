// glint - Vorbis I bit reader (LSB-first packing, spec §2)
// MIT License - Clean-room implementation
//
// Vorbis packs values into the bitstream low bit first (the opposite of the
// MPEG/AAC MSB-first convention). An n-bit unsigned integer is read so that
// the FIRST bit taken from the stream is the LEAST significant bit of the
// result (Vorbis I spec §2.1.2 "bit(s) are read/written low bit to high
// bit"). Bytes fill low bit first: bit 0 of byte 0 is the first bit.

#pragma once

#include <cstddef>
#include <cstdint>

namespace glint {
namespace vorbis {

class BitReader {
public:
    BitReader(const uint8_t* data, size_t len)
        : data_(data), len_(len) {}

    // True once a read has run past the end of the buffer. All reads after
    // that point return 0; callers check overrun() to reject the stream.
    bool overrun() const { return overrun_; }

    // Total bits consumed so far (for packet-length accounting).
    size_t bits_read() const { return bit_pos_; }
    size_t bit_length() const { return len_ * 8; }

    // Read an unsigned little-endian value of `n` bits (0 <= n <= 32).
    // n == 0 returns 0. On overrun sets the flag and returns 0-padded bits.
    uint32_t read(int n) {
        if (n <= 0) return 0;
        uint32_t v = 0;
        for (int i = 0; i < n; i++) {
            size_t byte = bit_pos_ >> 3;
            if (byte >= len_) {
                overrun_ = true;
                return v;  // remaining bits are 0
            }
            int bit = static_cast<int>(bit_pos_ & 7);
            uint32_t b = (data_[byte] >> bit) & 1u;
            v |= b << i;
            bit_pos_++;
        }
        return v;
    }

    // Read one bit as a bool/flag.
    int read_bit() { return static_cast<int>(read(1)); }

    // Read `n` bits and sign-extend as a two's-complement value.
    int32_t read_signed(int n) {
        uint32_t u = read(n);
        if (n > 0 && n < 32 && (u & (1u << (n - 1))))
            u |= ~((1u << n) - 1);  // sign extend
        return static_cast<int32_t>(u);
    }

private:
    const uint8_t* data_ = nullptr;
    size_t len_ = 0;
    size_t bit_pos_ = 0;
    bool overrun_ = false;
};

// ilog(x): position of the highest set bit (Vorbis spec §9.2.1). ilog(0)=0,
// ilog(1)=1, ilog(2)=2, ilog(4)=3, ilog(7)=3, ilog(8)=4.
inline int ilog(uint32_t x) {
    int n = 0;
    while (x) { n++; x >>= 1; }
    return n;
}

}  // namespace vorbis
}  // namespace glint

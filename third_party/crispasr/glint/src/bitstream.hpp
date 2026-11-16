// glint - Bitstream writer
// MIT License - Clean-room implementation

#ifndef GLINT_BITSTREAM_HPP
#define GLINT_BITSTREAM_HPP

#include <cstdint>
#include <cstring>

namespace glint {

// Maximum frame size: 320kbps at 32kHz = 320000/8/32000*1152 + padding + overhead
// Conservative max: ~1728 bytes (for 320kbps/32kHz) + some margin
#ifdef GLINT_SMALL_BUFFERS
// 1536 covers the largest legal frame (320 kbps @ 32 kHz = 1440 B + pad);
// 1024 could not hold 320k at any MPEG-1 rate and silently overflowed.
static constexpr int kMaxFrameSize = 1536;
static constexpr int kMainDataBufSize = 2048;
#else
static constexpr int kMaxFrameSize = 2048;
// Main data buffer size: larger to handle bit reservoir spanning
static constexpr int kMainDataBufSize = 8192;
#endif

class BitstreamWriter {
public:
    BitstreamWriter() { reset(); }

    void reset() {
        byte_pos_ = 0;
        cache_ = 0;
        bits_in_cache_ = 0;
        std::memset(buf_, 0, sizeof(buf_));
    }

    // Write `num_bits` bits from `value` (MSB-first, big-endian)
    // Uses a 32-bit accumulator cache for speed
    void write_bits(uint32_t value, int num_bits) {
        // Handle large writes (>25 bits) by splitting to avoid
        // overflow of the 32-bit cache
        if (num_bits > 25) {
            int top = num_bits - 25;
            write_bits(value >> top, 25);
            write_bits(value & ((1u << top) - 1), top);
            return;
        }
        cache_ = (cache_ << num_bits) | (value & ((1u << num_bits) - 1));
        bits_in_cache_ += num_bits;
        while (bits_in_cache_ >= 8) {
            bits_in_cache_ -= 8;
            buf_[byte_pos_++] = static_cast<uint8_t>((cache_ >> bits_in_cache_) & 0xFF);
        }
    }

    // Flush remaining bits in cache to the buffer (zero-padded on right)
    void flush() {
        if (bits_in_cache_ > 0) {
            buf_[byte_pos_++] = static_cast<uint8_t>((cache_ << (8 - bits_in_cache_)) & 0xFF);
            bits_in_cache_ = 0;
            cache_ = 0;
        }
    }

    // Get current bit position (total bits written)
    int bit_count() const { return byte_pos_ * 8 + bits_in_cache_; }

    // Get current byte position (rounds up if partial byte in cache)
    int byte_count() const { return byte_pos_ + (bits_in_cache_ > 0 ? 1 : 0); }

    // Pad to byte boundary
    void byte_align() {
        if (bits_in_cache_ > 0) {
            flush();
        }
    }

    // Get pointer to buffer
    const uint8_t* data() const { return buf_; }
    uint8_t* data() { return buf_; }

    // Set write position
    void set_position(int byte_offset, int bit_offset = 0) {
        flush();
        cache_ = 0;
        bits_in_cache_ = 0;
        byte_pos_ = byte_offset;
    }

    // Get buffer capacity
    static constexpr int capacity() { return kMainDataBufSize; }

private:
    uint8_t buf_[kMainDataBufSize];
    uint32_t cache_;
    int byte_pos_;
    int bits_in_cache_;
};

// Frame assembler: builds complete MP3 frames
class FrameAssembler {
public:
    FrameAssembler() : out_size_(0) {}

    void reset() {
        out_size_ = 0;
        std::memset(out_buf_, 0, sizeof(out_buf_));
    }

    // Write frame header (32 bits)
    // mpeg_version: 1 = MPEG-1, 0 = MPEG-2, -1 = MPEG-2.5
    void write_header(int bitrate_index, int samplerate_index,
                      int padding, int channel_mode, int mode_ext,
                      int mpeg_version = 1) {
        header_.reset();
        // Header format: sync(11 bits) + MPEG_ID(2 bits) + layer(2 bits) + protection(1 bit)
        // MPEG_ID: 11 = MPEG-1, 10 = MPEG-2, 00 = MPEG-2.5
        // Sync word (11 bits): all 1s
        header_.write_bits(0x7FF, 11);
        // MPEG ID (2 bits)
        if (mpeg_version == 1)
            header_.write_bits(0x3, 2);  // 11 = MPEG-1
        else if (mpeg_version == 0)
            header_.write_bits(0x2, 2);  // 10 = MPEG-2
        else
            header_.write_bits(0x0, 2);  // 00 = MPEG-2.5
        // Layer (2 bits): 01 = Layer III
        header_.write_bits(0x01, 2);
        // Protection (1 bit): 1 = no CRC
        header_.write_bits(1, 1);
        // Bitrate index (4 bits)
        header_.write_bits(bitrate_index, 4);
        // Sample rate index (2 bits)
        header_.write_bits(samplerate_index, 2);
        // Padding (1 bit)
        header_.write_bits(padding, 1);
        // Private (1 bit): 0
        header_.write_bits(0, 1);
        // Channel mode (2 bits)
        header_.write_bits(channel_mode, 2);
        // Mode extension (2 bits)
        header_.write_bits(mode_ext, 2);
        // Copyright (1 bit): 0
        header_.write_bits(0, 1);
        // Original (1 bit): 1
        header_.write_bits(1, 1);
        // Emphasis (2 bits): 00 = none
        header_.write_bits(0, 2);
    }

    // Access header writer (for reading back)
    const BitstreamWriter& header() const { return header_; }

    // Get side info writer (call after write_header)
    BitstreamWriter& side_info() { return side_info_; }

    // Get main data writer
    BitstreamWriter& main_data() { return main_data_; }

    // Assemble final frame into output buffer
    // Returns pointer to output and sets size
    const uint8_t* assemble(int frame_size, int* out_size) {
        std::memset(out_buf_, 0, sizeof(out_buf_));
        int pos = 0;

        // Copy header (4 bytes)
        std::memcpy(out_buf_ + pos, header_.data(), 4);
        pos += 4;

        // Copy side info
        int si_bytes = side_info_.byte_count();
        std::memcpy(out_buf_ + pos, side_info_.data(), si_bytes);
        pos += si_bytes;

        // Copy main data
        int md_bytes = main_data_.byte_count();
        if (pos + md_bytes > frame_size) {
            md_bytes = frame_size - pos;
        }
        std::memcpy(out_buf_ + pos, main_data_.data(), md_bytes);
        pos = frame_size;  // Pad to exact frame size

        out_size_ = frame_size;
        *out_size = frame_size;
        return out_buf_;
    }

    const uint8_t* output() const { return out_buf_; }
    int output_size() const { return out_size_; }

private:
    BitstreamWriter header_;
    BitstreamWriter side_info_;
    BitstreamWriter main_data_;
    uint8_t out_buf_[kMaxFrameSize];
    int out_size_;
};

} // namespace glint

#endif // GLINT_BITSTREAM_HPP

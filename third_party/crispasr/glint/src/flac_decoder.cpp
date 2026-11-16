// glint - FLAC decoder
// MIT License - Clean-room implementation

#include "flac_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace glint {
namespace flac {
namespace {

class BitReader {
public:
    BitReader(const uint8_t* d, size_t n) : data_(d), len_(n) {}

    size_t bit_pos() const { return bit_; }
    size_t byte_pos() const { return (bit_ + 7) >> 3; }
    bool ok() const { return ok_; }

    uint32_t read(int n) {
        if (n < 0 || n > 32 || bit_ + (size_t)n > len_ * 8) {
            ok_ = false;
            return 0;
        }
        uint32_t v = 0;
        for (int i = 0; i < n; i++) {
            v = (v << 1) | ((data_[bit_ >> 3] >> (7 - (bit_ & 7))) & 1);
            bit_++;
        }
        return v;
    }

    int32_t read_signed(int n) {
        if (n <= 0) return 0;
        uint32_t u = read(n);
        if (!ok_) return 0;
        if (n == 32) return static_cast<int32_t>(u);
        uint32_t sign = 1u << (n - 1);
        return (u & sign) ? static_cast<int32_t>(u | (~0u << n))
                          : static_cast<int32_t>(u);
    }

    int read_unary() {
        int z = 0;
        while (ok_ && bit_ < len_ * 8) {
            if (read(1)) return z;
            z++;
            if (z > 31) {
                ok_ = false;
                return -1;
            }
        }
        ok_ = false;
        return -1;
    }

    void byte_align() {
        size_t r = bit_ & 7;
        if (r) read((int)(8 - r));
    }

private:
    const uint8_t* data_;
    size_t len_;
    size_t bit_ = 0;
    bool ok_ = true;
};

static uint16_t be16(const uint8_t* p) {
    return (uint16_t)((p[0] << 8) | p[1]);
}

static uint32_t be24(const uint8_t* p) {
    return (uint32_t)((p[0] << 16) | (p[1] << 8) | p[2]);
}

static bool read_utf8_uint(const uint8_t* d, size_t n, size_t& pos,
                           uint64_t& out) {
    if (pos >= n) return false;
    uint8_t b0 = d[pos++];
    int extra = 0;
    uint64_t v = 0;
    if ((b0 & 0x80) == 0) {
        out = b0;
        return true;
    } else if ((b0 & 0xE0) == 0xC0) {
        extra = 1; v = b0 & 0x1F;
    } else if ((b0 & 0xF0) == 0xE0) {
        extra = 2; v = b0 & 0x0F;
    } else if ((b0 & 0xF8) == 0xF0) {
        extra = 3; v = b0 & 0x07;
    } else if ((b0 & 0xFC) == 0xF8) {
        extra = 4; v = b0 & 0x03;
    } else if ((b0 & 0xFE) == 0xFC) {
        extra = 5; v = b0 & 0x01;
    } else {
        return false;
    }
    if (pos + (size_t)extra > n) return false;
    for (int i = 0; i < extra; i++) {
        uint8_t b = d[pos++];
        if ((b & 0xC0) != 0x80) return false;
        v = (v << 6) | (b & 0x3F);
    }
    out = v;
    return true;
}

static int block_size_from_code(int code, const uint8_t* d, size_t n,
                                size_t& pos) {
    switch (code) {
    case 1: return 192;
    case 2: case 3: case 4: case 5: return 576 << (code - 2);
    case 6:
        if (pos >= n) return -1;
        return d[pos++] + 1;
    case 7:
        if (pos + 2 > n) return -1;
        { int v = (int)be16(d + pos) + 1; pos += 2; return v; }
    case 8: case 9: case 10: case 11:
    case 12: case 13: case 14: case 15:
        return 256 << (code - 8);
    default:
        return -1;
    }
}

static int sample_rate_from_code(int code, int stream_rate, const uint8_t* d,
                                 size_t n, size_t& pos) {
    static const int rates[12] = {
        0, 88200, 176400, 192000, 8000, 16000,
        22050, 24000, 32000, 44100, 48000, 96000
    };
    if (code == 0) return stream_rate;
    if (code >= 1 && code <= 11) return rates[code];
    if (code == 12) {
        if (pos >= n) return -1;
        return (int)d[pos++] * 1000;
    }
    if (code == 13) {
        if (pos + 2 > n) return -1;
        int v = (int)be16(d + pos); pos += 2; return v;
    }
    if (code == 14) {
        if (pos + 2 > n) return -1;
        int v = (int)be16(d + pos) * 10; pos += 2; return v;
    }
    return -1;
}

static int bits_from_code(int code, int stream_bps) {
    switch (code) {
    case 0: return stream_bps;
    case 1: return 8;
    case 2: return 12;
    case 4: return 16;
    case 5: return 20;
    case 6: return 24;
    default: return -1;
    }
}

static int64_t rice_to_signed(uint64_t v) {
    return (v & 1) ? -static_cast<int64_t>((v + 1) >> 1)
                   : static_cast<int64_t>(v >> 1);
}

static bool decode_residual(BitReader& br, int order, int block_size,
                            std::vector<int64_t>& out) {
    int method = (int)br.read(2);
    if (method != 0 && method != 1) return false;
    int partitions = 1 << (int)br.read(4);
    int param_bits = method == 0 ? 4 : 5;
    int escape = method == 0 ? 15 : 31;
    if (!br.ok() || partitions <= 0) return false;
    if (block_size % partitions != 0) return false;
    int part_len = block_size / partitions;
    int pos = order;
    for (int p = 0; p < partitions; p++) {
        int n = part_len - (p == 0 ? order : 0);
        if (n < 0 || pos + n > block_size) return false;
        int param = (int)br.read(param_bits);
        if (param == escape) {
            int raw_bits = (int)br.read(5);
            for (int i = 0; i < n; i++) out[pos++] = br.read_signed(raw_bits);
        } else {
            for (int i = 0; i < n; i++) {
                int q = br.read_unary();
                uint32_t r = param ? br.read(param) : 0;
                if (!br.ok() || q < 0) return false;
                out[pos++] = rice_to_signed(((uint64_t)q << param) | r);
            }
        }
    }
    return br.ok() && pos == block_size;
}

static int64_t fixed_predict(const std::vector<int64_t>& s, int i, int order) {
    switch (order) {
    case 0: return 0;
    case 1: return s[i - 1];
    case 2: return 2 * s[i - 1] - s[i - 2];
    case 3: return 3 * s[i - 1] - 3 * s[i - 2] + s[i - 3];
    case 4: return 4 * s[i - 1] - 6 * s[i - 2] + 4 * s[i - 3] - s[i - 4];
    default: return 0;
    }
}

static bool decode_subframe(BitReader& br, int bps, int block_size,
                            std::vector<int64_t>& out) {
    if (bps <= 0 || bps > 32 || block_size <= 0) return false;
    if (br.read(1) != 0) return false;
    int type = (int)br.read(6);
    int wasted = 0;
    if (br.read(1)) {
        int z = br.read_unary();
        if (z < 0) return false;
        wasted = z + 1;
        bps -= wasted;
        if (bps <= 0) return false;
    }
    out.assign(block_size, 0);
    if (type == 0) {
        int64_t v = br.read_signed(bps);
        std::fill(out.begin(), out.end(), v);
    } else if (type == 1) {
        for (int i = 0; i < block_size; i++) out[i] = br.read_signed(bps);
    } else if (type >= 8 && type <= 12) {
        int order = type - 8;
        for (int i = 0; i < order; i++) out[i] = br.read_signed(bps);
        if (!decode_residual(br, order, block_size, out)) return false;
        for (int i = order; i < block_size; i++)
            out[i] += fixed_predict(out, i, order);
    } else if (type >= 32 && type <= 63) {
        int order = type - 31;
        if (order > block_size) return false;
        for (int i = 0; i < order; i++) out[i] = br.read_signed(bps);
        int prec = (int)br.read(4) + 1;
        if (prec == 16) return false;
        int shift = (int)br.read_signed(5);
        if (shift < 0) return false;
        int32_t coeff[32] = {};
        for (int i = 0; i < order; i++) coeff[i] = br.read_signed(prec);
        if (!decode_residual(br, order, block_size, out)) return false;
        for (int i = order; i < block_size; i++) {
            int64_t sum = 0;
            for (int j = 0; j < order; j++) sum += coeff[j] * out[i - j - 1];
            out[i] += sum >> shift;
        }
    } else {
        return false;
    }
    if (!br.ok()) return false;
    if (wasted) {
        for (auto& v : out) v <<= wasted;
    }
    return true;
}

struct FrameHeader {
    int block_size = 0;
    int sample_rate = 0;
    int channels = 0;
    int chan_assign = 0;
    int bits_per_sample = 0;
    size_t data_offset = 0;
};

static bool parse_frame_header(const uint8_t* d, size_t n, const StreamInfo& si,
                               FrameHeader& h) {
    if (n < 6 || d[0] != 0xFF || (d[1] & 0xFC) != 0xF8) return false;
    if (d[1] & 0x02) return false;
    int bs_code = d[2] >> 4;
    int sr_code = d[2] & 0x0F;
    int ca = d[3] >> 4;
    int bps_code = (d[3] >> 1) & 7;
    if (d[3] & 1) return false;
    size_t pos = 4;
    uint64_t num = 0;
    if (!read_utf8_uint(d, n, pos, num)) return false;
    int block = block_size_from_code(bs_code, d, n, pos);
    int rate = sample_rate_from_code(sr_code, si.sample_rate, d, n, pos);
    int bps = bits_from_code(bps_code, si.bits_per_sample);
    if (block <= 0 || rate <= 0 || bps <= 0) return false;
    if (pos >= n) return false;
    if (crc8(d, pos) != d[pos]) return false;
    pos++;
    h.block_size = block;
    h.sample_rate = rate;
    h.chan_assign = ca;
    h.bits_per_sample = bps;
    if (ca <= 7) h.channels = ca + 1;
    else if (ca <= 10) h.channels = 2;
    else return false;
    h.data_offset = pos;
    return true;
}

static int decode_frame(const uint8_t* d, size_t n, const StreamInfo& si,
                        std::vector<int64_t>& frame, FrameHeader& h,
                        size_t& consumed) {
    if (!parse_frame_header(d, n, si, h)) return -1;
    if (h.channels != si.channels || h.sample_rate != si.sample_rate)
        return -1;
    BitReader br(d + h.data_offset, n - h.data_offset);
    std::vector<std::vector<int64_t>> ch(h.channels);
    for (int c = 0; c < h.channels; c++) {
        int bps = h.bits_per_sample;
        if ((h.chan_assign == 8 && c == 1) ||
            (h.chan_assign == 9 && c == 0) ||
            (h.chan_assign == 10 && c == 1))
            bps++;
        if (!decode_subframe(br, bps, h.block_size, ch[c])) return -1;
    }
    br.byte_align();
    size_t end = h.data_offset + br.byte_pos();
    if (!br.ok() || end + 2 > n) return -1;
    uint16_t got_crc = be16(d + end);
    uint16_t want_crc = crc16(d, end);
    if (got_crc != want_crc) return -1;
    consumed = end + 2;

    frame.resize((size_t)h.block_size * h.channels);
    if (h.chan_assign <= 7) {
        for (int i = 0; i < h.block_size; i++)
            for (int c = 0; c < h.channels; c++)
                frame[(size_t)i * h.channels + c] = ch[c][i];
    } else if (h.chan_assign == 8) {
        for (int i = 0; i < h.block_size; i++) {
            int64_t left = ch[0][i];
            int64_t right = left - ch[1][i];
            frame[(size_t)i * 2] = left;
            frame[(size_t)i * 2 + 1] = right;
        }
    } else if (h.chan_assign == 9) {
        for (int i = 0; i < h.block_size; i++) {
            int64_t right = ch[1][i];
            int64_t left = ch[0][i] + right;
            frame[(size_t)i * 2] = left;
            frame[(size_t)i * 2 + 1] = right;
        }
    } else {
        for (int i = 0; i < h.block_size; i++) {
            int64_t mid = ch[0][i];
            int64_t side = ch[1][i];
            int64_t left = mid + ((side + (side & 1)) >> 1);
            int64_t right = left - side;
            frame[(size_t)i * 2] = left;
            frame[(size_t)i * 2 + 1] = right;
        }
    }
    return 0;
}

}  // namespace

uint8_t crc8(const uint8_t* data, size_t len) {
    uint8_t crc = 0;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07)
                               : (uint8_t)(crc << 1);
    }
    return crc;
}

uint16_t crc16(const uint8_t* data, size_t len) {
    uint16_t crc = 0;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x8005)
                                 : (uint16_t)(crc << 1);
    }
    return crc;
}

int parse_streaminfo(const uint8_t* data, size_t len, StreamInfo& si) {
    if (!data || len != 34) return -1;
    si.min_block_size = be16(data);
    si.max_block_size = be16(data + 2);
    uint64_t x = ((uint64_t)data[10] << 56) | ((uint64_t)data[11] << 48) |
                 ((uint64_t)data[12] << 40) | ((uint64_t)data[13] << 32) |
                 ((uint64_t)data[14] << 24) | ((uint64_t)data[15] << 16) |
                 ((uint64_t)data[16] << 8) | data[17];
    si.sample_rate = (int)((x >> 44) & 0xFFFFF);
    si.channels = (int)((x >> 41) & 0x7) + 1;
    si.bits_per_sample = (int)((x >> 36) & 0x1F) + 1;
    si.total_samples = x & 0xFFFFFFFFFULL;
    si.valid = si.min_block_size > 0 && si.max_block_size >= si.min_block_size &&
               si.sample_rate > 0 && si.channels >= 1 && si.channels <= 8 &&
               si.bits_per_sample >= 4 && si.bits_per_sample <= 32;
    return si.valid ? 0 : -1;
}

int decode(const uint8_t* data, size_t len, std::vector<float>& pcm,
           int& sample_rate, int& channels) {
    pcm.clear();
    sample_rate = channels = 0;
    if (!data || len < 42 || std::memcmp(data, "fLaC", 4) != 0) return -1;

    size_t pos = 4;
    StreamInfo si;
    bool saw_streaminfo = false;
    bool last = false;
    while (!last) {
        if (pos + 4 > len) return -1;
        uint8_t hdr = data[pos++];
        last = (hdr & 0x80) != 0;
        int type = hdr & 0x7F;
        uint32_t sz = be24(data + pos);
        pos += 3;
        if (pos + sz > len) return -1;
        if (type == 0) {
            if (saw_streaminfo || sz != 34) return -1;
            if (parse_streaminfo(data + pos, sz, si) != 0) return -1;
            saw_streaminfo = true;
        }
        pos += sz;
    }
    if (!saw_streaminfo) return -1;

    std::vector<int64_t> fr;
    FrameHeader fh;
    sample_rate = si.sample_rate;
    channels = si.channels;
    const double scale = std::ldexp(1.0, si.bits_per_sample - 1);
    uint64_t samples_done = 0;
    while (pos + 2 < len) {
        size_t consumed = 0;
        if (decode_frame(data + pos, len - pos, si, fr, fh, consumed) != 0)
            return -1;
        size_t frames = (size_t)fh.block_size;
        if (si.total_samples && samples_done + frames > si.total_samples)
            frames = (size_t)(si.total_samples - samples_done);
        size_t old = pcm.size();
        pcm.resize(old + frames * si.channels);
        for (size_t i = 0; i < frames * (size_t)si.channels; i++)
            pcm[old + i] = static_cast<float>(fr[i] / scale);
        samples_done += frames;
        pos += consumed;
        if (si.total_samples && samples_done >= si.total_samples) break;
    }
    return pcm.empty() ? -1 : 0;
}

}  // namespace flac
}  // namespace glint

// Ogg Opus container reading — RFC 3533 + RFC 7845
// MIT License - Clean-room implementation

#include "opus_ogg.hpp"

#include <cmath>
#include <cstring>

namespace glint {
namespace opus {

namespace {

// CRC table for the Ogg polynomial (0x04C11DB7, MSB-first, no reflection).
const uint32_t* crc_table() {
    static uint32_t table[256];
    static bool once = [] {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t r = i << 24;
            for (int j = 0; j < 8; j++)
                r = (r << 1) ^ ((r & 0x80000000u) ? 0x04C11DB7u : 0);
            table[i] = r;
        }
        return true;
    }();
    (void)once;
    return table;
}

uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
           static_cast<uint32_t>(p[2]) << 16 |
           static_cast<uint32_t>(p[3]) << 24;
}

uint64_t le64(const uint8_t* p) {
    return static_cast<uint64_t>(le32(p)) |
           static_cast<uint64_t>(le32(p + 4)) << 32;
}

}  // namespace

uint32_t ogg_crc(const uint8_t* data, size_t len) {
    const uint32_t* t = crc_table();
    uint32_t crc = 0;
    for (size_t i = 0; i < len; i++)
        crc = (crc << 8) ^ t[(crc >> 24) ^ data[i]];
    return crc;
}

double OggOpusReader::output_gain() const {
    // Q7.8 dB: gain = 10^(g/(20*256)).
    return std::pow(10.0, head_.output_gain_q8 / (20.0 * 256.0));
}

int OggOpusReader::parse(const uint8_t* data, size_t len) {
    size_t off = 0;
    bool have_stream = false;
    uint32_t serial = 0;
    int header_packets = 0;  // OpusHead + OpusTags
    std::vector<uint8_t> pending;  // packet continued across pages
    bool continued_ok = false;
    int64_t last_granule = -1;

    while (off + 27 <= len) {
        if (std::memcmp(data + off, "OggS", 4) != 0) return -1;
        const uint8_t* page = data + off;
        if (page[4] != 0) return -1;  // stream_structure_version
        int htype = page[5];
        uint64_t granule = le64(page + 6);
        uint32_t pserial = le32(page + 14);
        int nsegs = page[26];
        if (off + 27 + nsegs > len) return -1;
        const uint8_t* segtab = page + 27;
        size_t body = 0;
        for (int i = 0; i < nsegs; i++) body += segtab[i];
        size_t page_len = 27 + nsegs + body;
        if (off + page_len > len) return -1;

        // CRC check with the CRC field zeroed.
        {
            uint8_t hdr[27 + 255];
            std::memcpy(hdr, page, 27 + nsegs);
            hdr[22] = hdr[23] = hdr[24] = hdr[25] = 0;
            uint32_t crc = ogg_crc(hdr, 27 + nsegs);
            const uint8_t* b = page + 27 + nsegs;
            const uint32_t* t = crc_table();
            for (size_t i = 0; i < body; i++)
                crc = (crc << 8) ^ t[(crc >> 24) ^ b[i]];
            if (crc != le32(page + 22)) return -1;
        }

        // First BOS page with an OpusHead wins; other streams are skipped.
        if (!have_stream) {
            bool bos = (htype & 0x02) != 0;
            bool is_opus = bos && nsegs >= 1 && body >= 8 &&
                           std::memcmp(page + 27 + nsegs, "OpusHead", 8) == 0;
            if (!is_opus) {
                off += page_len;
                continue;
            }
            have_stream = true;
            serial = pserial;
        } else if (pserial != serial) {
            off += page_len;
            continue;
        }

        // Reassemble packets from the lacing values.
        const uint8_t* p = page + 27 + nsegs;
        if (!(htype & 0x01)) {
            // Not a continuation: any dangling partial packet is dropped.
            pending.clear();
            continued_ok = true;
        }
        for (int i = 0; i < nsegs; i++) {
            pending.insert(pending.end(), p, p + segtab[i]);
            p += segtab[i];
            if (segtab[i] < 255) {
                // Packet complete.
                if (continued_ok) {
                    if (header_packets == 0) {
                        // OpusHead (RFC 7845 section 5.1).
                        if (pending.size() < 19 ||
                            std::memcmp(pending.data(), "OpusHead", 8) != 0)
                            return -2;
                        head_.version = pending[8];
                        if ((head_.version >> 4) != 0) return -2;
                        head_.channels = pending[9];
                        head_.pre_skip =
                            pending[10] | (pending[11] << 8);
                        head_.input_sample_rate = le32(&pending[12]);
                        head_.output_gain_q8 = static_cast<int16_t>(
                            pending[16] | (pending[17] << 8));
                        head_.mapping_family = pending[18];
                        if (head_.channels < 1 || head_.mapping_family > 1)
                            return -3;
                        if (head_.mapping_family == 0) {
                            if (head_.channels > 2) return -2;  // RFC 7845
                            head_.stream_count = 1;
                            head_.coupled_count = head_.channels - 1;
                            for (int c = 0; c < head_.channels; c++)
                                head_.mapping[c] =
                                    static_cast<uint8_t>(c);
                        } else {
                            // Family 1: stream/coupled counts + mapping
                            // table (up to 8 channels).
                            if (head_.channels > 8) return -3;
                            if (pending.size() < 21u + head_.channels)
                                return -2;
                            head_.stream_count = pending[19];
                            head_.coupled_count = pending[20];
                            if (head_.stream_count < 1 ||
                                head_.coupled_count > head_.stream_count)
                                return -2;
                            for (int c = 0; c < head_.channels; c++)
                                head_.mapping[c] = pending[21 + c];
                        }
                        header_packets = 1;
                    } else if (header_packets == 1) {
                        if (pending.size() < 8 ||
                            std::memcmp(pending.data(), "OpusTags", 8) != 0)
                            return -2;
                        header_packets = 2;
                    } else {
                        packets_.push_back(pending);
                    }
                }
                pending.clear();
                continued_ok = true;
            }
        }
        if (header_packets == 2 && granule != ~UINT64_C(0))
            last_granule = static_cast<int64_t>(granule);
        off += page_len;
    }

    if (!have_stream || header_packets < 2) return -2;
    if (last_granule >= 0)
        total_samples_ = last_granule - head_.pre_skip;
    return 0;
}

}  // namespace opus
}  // namespace glint

namespace glint {
namespace opus {

// (definitions appended below the reader; see header for the contract)

void OggOpusWriter::write_page(const uint8_t* body, size_t len, int htype,
                               uint64_t granule) {
    // Segment table: 255-lacing chain; a multiple-of-255 packet gets a
    // terminating 0 lacing value.
    uint8_t segs[255];
    int nsegs = 0;
    size_t left = len;
    for (;;) {
        uint8_t v = left >= 255 ? 255 : static_cast<uint8_t>(left);
        segs[nsegs++] = v;
        left -= v;
        if (v < 255) break;
        if (nsegs == 255) break;  // callers keep packets under one page
    }
    uint8_t hdr[27];
    std::memcpy(hdr, "OggS", 4);
    hdr[4] = 0;
    hdr[5] = static_cast<uint8_t>(htype);
    for (int i = 0; i < 8; i++)
        hdr[6 + i] = static_cast<uint8_t>(granule >> (8 * i));
    for (int i = 0; i < 4; i++)
        hdr[14 + i] = static_cast<uint8_t>(serial_ >> (8 * i));
    for (int i = 0; i < 4; i++)
        hdr[18 + i] = static_cast<uint8_t>(pageno_ >> (8 * i));
    hdr[22] = hdr[23] = hdr[24] = hdr[25] = 0;  // CRC placeholder
    hdr[26] = static_cast<uint8_t>(nsegs);
    pageno_++;

    size_t page_off = out_.size();
    out_.insert(out_.end(), hdr, hdr + 27);
    out_.insert(out_.end(), segs, segs + nsegs);
    out_.insert(out_.end(), body, body + len);
    uint32_t crc = ogg_crc(out_.data() + page_off,
                           27 + nsegs + len);
    for (int i = 0; i < 4; i++)
        out_[page_off + 22 + i] = static_cast<uint8_t>(crc >> (8 * i));
}

void OggOpusWriter::begin(int channels, int pre_skip,
                          uint32_t input_sample_rate) {
    out_.clear();
    pageno_ = 0;
    granule_ = 0;
    pre_skip_ = pre_skip;
    open_ = true;

    uint8_t head[19];
    std::memcpy(head, "OpusHead", 8);
    head[8] = 1;  // version
    head[9] = static_cast<uint8_t>(channels);
    head[10] = static_cast<uint8_t>(pre_skip);
    head[11] = static_cast<uint8_t>(pre_skip >> 8);
    for (int i = 0; i < 4; i++)
        head[12 + i] = static_cast<uint8_t>(input_sample_rate >> (8 * i));
    head[16] = head[17] = 0;  // output gain 0 dB
    head[18] = 0;             // mapping family 0
    write_page(head, sizeof(head), 0x02 /*BOS*/, 0);

    static const char kVendor[] = "glint";
    uint8_t tags[8 + 4 + sizeof(kVendor) - 1 + 4];
    std::memcpy(tags, "OpusTags", 8);
    uint32_t vlen = sizeof(kVendor) - 1;
    for (int i = 0; i < 4; i++)
        tags[8 + i] = static_cast<uint8_t>(vlen >> (8 * i));
    std::memcpy(tags + 12, kVendor, vlen);
    std::memset(tags + 12 + vlen, 0, 4);  // zero user comments
    write_page(tags, sizeof(tags), 0, 0);
}

void OggOpusWriter::add_packet(const uint8_t* data, size_t len,
                               int samples48) {
    // One packet per page (trivial lacing, exact per-page granules); each
    // page is written when the NEXT packet arrives so the final one can
    // carry the EOS flag. Granule = decoded samples INCLUDING pre-skip
    // (RFC 7845 section 4).
    if (!pending_.empty()) {
        write_page(pending_.data(), pending_.size(), 0,
                   granule_ + pre_skip_);
    }
    pending_.assign(data, data + len);
    granule_ += static_cast<uint64_t>(samples48);
    pending_samples_ = samples48;
}

const std::vector<uint8_t>& OggOpusWriter::finish() {
    if (open_) {
        if (!pending_.empty())
            write_page(pending_.data(), pending_.size(), 0x04 /*EOS*/,
                       granule_ + pre_skip_);
        pending_.clear();
        open_ = false;
    }
    return out_;
}

}  // namespace opus
}  // namespace glint

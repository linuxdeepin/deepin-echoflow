// glint - Ogg demux for Vorbis (RFC 3533 framing)
// MIT License - Clean-room implementation
//
// Extracts the packets of the FIRST logical bitstream from an in-memory Ogg
// file: page walk with CRC validation and lacing-based packet reassembly
// (including cross-page continuation). Shares the Ogg CRC with the Opus
// reader (glint::opus::ogg_crc). Scope matches what .sf3 / general .ogg
// Vorbis needs: one logical Vorbis stream per buffer.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "opus_ogg.hpp"  // glint::opus::ogg_crc

namespace glint {
namespace vorbis {

// Reassemble every packet of the first logical stream into `packets`.
// Returns 0 on success, negative on a malformed container. `last_granule`
// receives the granule position of the final non-(-1) page, or -1.
inline int ogg_demux_first_stream(const uint8_t* data, size_t len,
                                  std::vector<std::vector<uint8_t>>& packets,
                                  int64_t* last_granule) {
    packets.clear();
    if (last_granule) *last_granule = -1;
    size_t off = 0;
    bool have_stream = false;
    uint32_t serial = 0;
    std::vector<uint8_t> pending;
    bool continued_ok = false;
    int64_t lg = -1;

    while (off + 27 <= len) {
        const uint8_t* page = data + off;
        if (std::memcmp(page, "OggS", 4) != 0) return -1;
        if (page[4] != 0) return -1;  // stream_structure_version
        int htype = page[5];
        uint64_t granule = 0;
        for (int i = 0; i < 8; i++)
            granule |= static_cast<uint64_t>(page[6 + i]) << (8 * i);
        uint32_t pserial = static_cast<uint32_t>(page[14]) |
                           static_cast<uint32_t>(page[15]) << 8 |
                           static_cast<uint32_t>(page[16]) << 16 |
                           static_cast<uint32_t>(page[17]) << 24;
        int nsegs = page[26];
        if (off + 27 + static_cast<size_t>(nsegs) > len) return -1;
        const uint8_t* segtab = page + 27;
        size_t body = 0;
        for (int i = 0; i < nsegs; i++) body += segtab[i];
        size_t page_len = 27 + static_cast<size_t>(nsegs) + body;
        if (off + page_len > len) return -1;

        // CRC check (CRC field zeroed for the computation).
        {
            uint8_t hdr[27 + 255];
            std::memcpy(hdr, page, 27 + nsegs);
            hdr[22] = hdr[23] = hdr[24] = hdr[25] = 0;
            uint32_t crc = glint::opus::ogg_crc(hdr, 27 + nsegs);
            // Continue the running CRC over the page body.
            const uint8_t* b = page + 27 + nsegs;
            std::vector<uint8_t> full(hdr, hdr + 27 + nsegs);
            full.insert(full.end(), b, b + body);
            crc = glint::opus::ogg_crc(full.data(), full.size());
            uint32_t stored = static_cast<uint32_t>(page[22]) |
                              static_cast<uint32_t>(page[23]) << 8 |
                              static_cast<uint32_t>(page[24]) << 16 |
                              static_cast<uint32_t>(page[25]) << 24;
            if (crc != stored) return -1;
        }

        // Lock onto the first logical stream (its BOS page).
        if (!have_stream) {
            if (!(htype & 0x02)) { off += page_len; continue; }  // want BOS
            have_stream = true;
            serial = pserial;
        } else if (pserial != serial) {
            off += page_len;
            continue;  // ignore other multiplexed streams
        }

        const uint8_t* p = page + 27 + nsegs;
        if (!(htype & 0x01)) {  // fresh page, not a continuation
            pending.clear();
            continued_ok = true;
        }
        for (int i = 0; i < nsegs; i++) {
            pending.insert(pending.end(), p, p + segtab[i]);
            p += segtab[i];
            if (segtab[i] < 255) {  // packet boundary
                if (continued_ok) packets.push_back(pending);
                pending.clear();
                continued_ok = true;
            }
        }
        if (granule != ~UINT64_C(0)) lg = static_cast<int64_t>(granule);
        if (htype & 0x04) break;  // EOS
        off += page_len;
    }

    if (!have_stream) return -1;
    if (last_granule) *last_granule = lg;
    return 0;
}

}  // namespace vorbis
}  // namespace glint

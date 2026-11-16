// crispasr_opus_writer.h — Ogg Opus serializer for TTS float32 PCM, built on
// the in-tree glint Opus encoder (glint_opus_encode_file). Header-only like
// crispasr_mp3_writer.h / crispasr_aac_writer.h; consumers link the `glint`
// static library.
//
// Unlike the MP3/AAC writers this does NOT prepend an ID3v2 AI-provenance tag —
// an Ogg stream must begin with the "OggS" capture pattern. Instead the
// AI-provenance is injected into the stream's OpusTags comment header (the
// RFC 7845 place for it), rewriting that Ogg page in-tree (glint's C ABI writes
// a fixed OpusTags), mirroring the mp3/aac TXXX/LIST provenance.
//
// glint's Opus encoder is CELT-only, fullband, 48 kHz — input is resampled to
// 48 kHz here (linear, good enough for speech; mirrors the server's Opus path).
// The result is a standard, playable .opus file (verified decodable by ffmpeg
// and libopus).

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <glint/glint.h>

// Ogg page CRC (RFC 3533: poly 0x04C11DB7, MSB-first, no reflection, zero
// init/xor). Table built once, thread-safe via the C++11 local-static guard.
inline uint32_t crispasr_ogg_crc32(const uint8_t* data, size_t len) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t r = i << 24;
            for (int j = 0; j < 8; j++)
                r = (r & 0x80000000u) ? ((r << 1) ^ 0x04c11db7u) : (r << 1);
            t[i] = r;
        }
        return t;
    }();
    uint32_t crc = 0;
    for (size_t i = 0; i < len; i++)
        crc = (crc << 8) ^ table[((crc >> 24) & 0xff) ^ data[i]];
    return crc;
}

// Inject AI-provenance comments into the OpusTags header of an in-memory Ogg
// Opus stream (glint output). Finds the OpusTags page (its body begins with
// "OpusTags"), appends the comments, and rewrites that one page with a fresh
// segment table + CRC. Other pages (their CRCs/seqnos are self-contained) are
// untouched. Returns true on success; leaves `ogg` unchanged and returns false
// if the layout is unexpected (caller still ships the valid, untagged stream).
inline bool crispasr_opus_inject_provenance(std::string& ogg) {
    const uint8_t* d = reinterpret_cast<const uint8_t*>(ogg.data());
    const size_t n = ogg.size();
    size_t pos = 0;
    while (pos + 27 <= n) {
        if (std::memcmp(d + pos, "OggS", 4) != 0)
            return false;
        const int n_seg = d[pos + 26];
        if (pos + 27 + (size_t)n_seg > n)
            return false;
        const uint8_t* seg = d + pos + 27;
        size_t body_len = 0;
        for (int i = 0; i < n_seg; i++)
            body_len += seg[i];
        const size_t body_off = pos + 27 + (size_t)n_seg;
        if (body_off + body_len > n)
            return false;

        if (body_len >= 8 && std::memcmp(d + body_off, "OpusTags", 8) == 0) {
            const uint8_t* b = d + body_off;
            auto rd32 = [&](size_t o) -> uint32_t {
                return (uint32_t)b[o] | ((uint32_t)b[o + 1] << 8) | ((uint32_t)b[o + 2] << 16) |
                       ((uint32_t)b[o + 3] << 24);
            };
            size_t o = 8;
            if (o + 4 > body_len)
                return false;
            uint32_t vlen = rd32(o);
            o += 4;
            if (o + vlen + 4 > body_len)
                return false;
            std::string vendor(reinterpret_cast<const char*>(b + o), vlen);
            o += vlen;
            uint32_t ccount = rd32(o);
            o += 4;
            std::vector<std::string> comments;
            for (uint32_t i = 0; i < ccount; i++) {
                if (o + 4 > body_len)
                    return false;
                uint32_t clen = rd32(o);
                o += 4;
                if (o + clen > body_len)
                    return false;
                comments.emplace_back(reinterpret_cast<const char*>(b + o), clen);
                o += clen;
            }
            // AI-provenance (mirrors the mp3/aac TXXX frames).
            comments.emplace_back("AI_GENERATED=true");
            comments.emplace_back("GENERATOR=CrispASR");
            comments.emplace_back("AI_CONTENT_NOTICE=This audio was synthesized by an AI text-to-speech model. "
                                  "It is not a recording of a human speaker.");

            // Rebuild the OpusTags packet.
            std::string pkt;
            pkt.append("OpusTags", 8);
            auto put32 = [&](uint32_t v) {
                pkt.push_back((char)(v & 0xff));
                pkt.push_back((char)((v >> 8) & 0xff));
                pkt.push_back((char)((v >> 16) & 0xff));
                pkt.push_back((char)((v >> 24) & 0xff));
            };
            put32((uint32_t)vendor.size());
            pkt += vendor;
            put32((uint32_t)comments.size());
            for (const auto& c : comments) {
                put32((uint32_t)c.size());
                pkt += c;
            }

            // Ogg lacing for the single packet: floor(L/255) × 255, then L%255.
            std::vector<uint8_t> lacing;
            size_t rem = pkt.size();
            while (rem >= 255) {
                lacing.push_back(255);
                rem -= 255;
            }
            lacing.push_back((uint8_t)rem);
            if (lacing.size() > 255)
                return false; // won't happen for a tags packet, but stay safe

            // Rebuild the page: header[0..21] (OggS..page_seq) verbatim, CRC
            // zeroed, new segment count/table, new body. Then set CRC.
            std::string page;
            page.append(reinterpret_cast<const char*>(d + pos), 22);
            page.append(4, '\0'); // CRC placeholder
            page.push_back((char)lacing.size());
            page.append(reinterpret_cast<const char*>(lacing.data()), lacing.size());
            page += pkt;
            uint32_t crc = crispasr_ogg_crc32(reinterpret_cast<const uint8_t*>(page.data()), page.size());
            page[22] = (char)(crc & 0xff);
            page[23] = (char)((crc >> 8) & 0xff);
            page[24] = (char)((crc >> 16) & 0xff);
            page[25] = (char)((crc >> 24) & 0xff);

            std::string result;
            result.reserve(pos + page.size() + (n - (body_off + body_len)));
            result.append(ogg.data(), pos);
            result += page;
            result.append(ogg.data() + body_off + body_len, n - (body_off + body_len));
            ogg.swap(result);
            return true;
        }
        pos = body_off + body_len;
    }
    return false;
}

// Encode float32 mono PCM in [-1, 1] to a complete Ogg Opus stream via glint.
// Samples outside [-1, 1] are clamped. Non-48 kHz input is linearly resampled
// to 48 kHz first (glint Opus is 48 kHz only). Returns empty on failure.
inline std::string crispasr_make_opus_glint(const float* pcm, int n_samples, int sample_rate, int bitrate_bps = 64000) {
    if (!pcm || n_samples <= 0 || sample_rate <= 0)
        return {};

    std::vector<float> resampled;
    if (sample_rate != 48000) {
        const int out_n = (int)((int64_t)n_samples * 48000 / sample_rate);
        if (out_n <= 0)
            return {};
        resampled.resize(out_n);
        for (int i = 0; i < out_n; i++) {
            float pos = (float)i * (float)sample_rate / 48000.0f;
            int s0 = (int)pos;
            int s1 = std::min(s0 + 1, n_samples - 1);
            float frac = pos - (float)s0;
            resampled[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
        }
        pcm = resampled.data();
        n_samples = out_n;
    }

    // Clamp to [-1, 1] (same semantics as the MP3/AAC writers).
    std::vector<float> clamped(pcm, pcm + n_samples);
    for (float& s : clamped) {
        if (s > 1.0f)
            s = 1.0f;
        if (s < -1.0f)
            s = -1.0f;
    }

    int size = 0;
    uint8_t* data = glint_opus_encode_file(clamped.data(), n_samples, 1, bitrate_bps, /*vbr=*/0, &size);
    if (!data || size <= 0) {
        if (data)
            glint_free(data);
        return {};
    }
    std::string out((const char*)data, (size_t)size);
    glint_free(data);
    // Inject AI-provenance into OpusTags (best-effort; a failure leaves the
    // valid, untagged stream intact).
    crispasr_opus_inject_provenance(out);
    return out;
}

// Public entry: glint Ogg Opus. Kept as a thin wrapper so a future libopus A/B
// path can slot in behind CRISPASR_OPUS_ENCODER without touching call sites.
inline std::string crispasr_make_opus(const float* pcm, int n_samples, int sample_rate, int bitrate_bps = 64000) {
    return crispasr_make_opus_glint(pcm, n_samples, sample_rate, bitrate_bps);
}

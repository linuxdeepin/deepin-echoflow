// crispasr_mp4_writer.h — minimal ISO-BMFF (MP4/M4A) muxer for a single audio
// track, so AAC and Opus TTS output can carry a native C2PA manifest (raw ADTS
// AAC and Ogg Opus have no C2PA embedding path; MP4 does — see docs/tts.md).
//
// Consumes glint's encoder output: AAC-LC access units (from the ADTS blob) or
// Opus packets. Produces `ftyp | mdat | moov` (non-faststart, so the chunk
// offset is known before moov is built). Header-only; links the glint lib.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "crispasr_aac_writer.h" // crispasr_make_aac (ADTS) + target-rate

namespace crispasr_mp4 {

using Bytes = std::string;

inline void u8(Bytes& b, uint8_t v) {
    b.push_back((char)v);
}
inline void u16(Bytes& b, uint16_t v) {
    u8(b, v >> 8);
    u8(b, v);
}
inline void u32(Bytes& b, uint32_t v) {
    u8(b, v >> 24);
    u8(b, v >> 16);
    u8(b, v >> 8);
    u8(b, v);
}
inline void tag(Bytes& b, const char* t) {
    b.append(t, 4);
}
// box = size(4) + type(4) + payload
inline Bytes box(const char* type, const Bytes& payload) {
    Bytes b;
    u32(b, (uint32_t)(8 + payload.size()));
    tag(b, type);
    b += payload;
    return b;
}

// AAC-LC AudioSpecificConfig (2 bytes) from sample-rate index + channels.
inline int aac_sr_index(int sr) {
    static const int r[] = {96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350};
    for (int i = 0; i < 13; i++)
        if (r[i] == sr)
            return i;
    return 6; // 24000 default
}

// ---- MPEG-4 descriptor length (1-byte form is enough for our small ASCs) ----
inline void desc(Bytes& b, uint8_t t, const Bytes& body) {
    u8(b, t);
    u8(b, (uint8_t)body.size()); // sizes < 128 for our data
    b += body;
}

// Build the `esds` box for an AAC-LC track from its AudioSpecificConfig.
inline Bytes esds_aac(const Bytes& asc, uint32_t avg_bitrate) {
    Bytes dsi = asc; // DecoderSpecificInfo payload
    Bytes dcd;       // DecoderConfigDescriptor payload
    u8(dcd, 0x40);   // objectTypeIndication = MPEG-4 Audio
    u8(dcd, 0x15);   // streamType=5(audio)<<2 | upStream=0 | reserved=1
    u8(dcd, 0);
    u8(dcd, 0);
    u8(dcd, 0);            // bufferSizeDB (24-bit) = 0
    u32(dcd, avg_bitrate); // maxBitrate
    u32(dcd, avg_bitrate); // avgBitrate
    desc(dcd, 0x05, dsi);  // DecoderSpecificInfo
    Bytes esd;             // ES_Descriptor payload
    u16(esd, 0);           // ES_ID
    u8(esd, 0);            // flags
    desc(esd, 0x04, dcd);  // DecoderConfigDescriptor
    Bytes sl;
    u8(sl, 0x02);
    desc(esd, 0x06, sl); // SLConfigDescriptor (0x02 = MP4)
    Bytes payload;
    u32(payload, 0);          // version + flags (full box)
    desc(payload, 0x03, esd); // ES_Descriptor
    return box("esds", payload);
}

// stsd for AAC ("mp4a" audio sample entry + esds).
inline Bytes stsd_mp4a(int sample_rate, int channels, const Bytes& esds) {
    Bytes se; // mp4a sample entry
    for (int i = 0; i < 6; i++)
        u8(se, 0); // reserved
    u16(se, 1);    // data_reference_index
    u32(se, 0);
    u32(se, 0); // reserved (version/rev/vendor)
    u16(se, (uint16_t)channels);
    u16(se, 16); // sample size
    u16(se, 0);
    u16(se, 0);                           // predefined + reserved
    u32(se, (uint32_t)sample_rate << 16); // 16.16 sample rate
    se += esds;
    Bytes mp4a = box("mp4a", se);
    Bytes payload;
    u32(payload, 0);
    u32(payload, 1); // version/flags + entry_count
    payload += mp4a;
    return box("stsd", payload);
}
// stsd for Opus ("Opus" sample entry + dOps).
inline Bytes stsd_opus(int sample_rate, int channels, int preskip) {
    Bytes dops;                  // dOps payload
    u8(dops, 0);                 // version
    u8(dops, (uint8_t)channels); // OutputChannelCount
    u16(dops, (uint16_t)preskip);
    u32(dops, 48000); // InputSampleRate (Opus is always 48k internally)
    u16(dops, 0);     // OutputGain
    u8(dops, 0);      // ChannelMappingFamily = 0
    Bytes dOps = box("dOps", dops);
    Bytes se;
    for (int i = 0; i < 6; i++)
        u8(se, 0);
    u16(se, 1);
    u32(se, 0);
    u32(se, 0);
    u16(se, (uint16_t)channels);
    u16(se, 16);
    u16(se, 0);
    u16(se, 0);
    u32(se, (uint32_t)sample_rate << 16);
    se += dOps;
    Bytes opus = box("Opus", se);
    Bytes payload;
    u32(payload, 0);
    u32(payload, 1);
    payload += opus;
    return box("stsd", payload);
}

// Assemble the full MP4 given the stsd, per-sample sizes, sample duration and
// media timescale/duration. Layout: ftyp | mdat | moov (mdat data offset known).
inline Bytes assemble(const Bytes& ftyp, const Bytes& stsd, const std::vector<uint32_t>& sizes, uint32_t sample_delta,
                      uint32_t timescale, uint64_t media_duration, const Bytes& mdat_data) {
    const uint32_t n = (uint32_t)sizes.size();
    // mdat data begins at ftyp.size() + 8 (mdat header)
    const uint32_t mdat_data_off = (uint32_t)ftyp.size() + 8;

    Bytes stts;
    u32(stts, 0);
    u32(stts, 1);
    u32(stts, n);
    u32(stts, sample_delta);
    Bytes stsc;
    u32(stsc, 0);
    u32(stsc, 1);
    u32(stsc, 1);
    u32(stsc, n);
    u32(stsc, 1);
    Bytes stsz;
    u32(stsz, 0);
    u32(stsz, 0);
    u32(stsz, n);
    for (uint32_t s : sizes)
        u32(stsz, s);
    Bytes stco;
    u32(stco, 0);
    u32(stco, 1);
    u32(stco, mdat_data_off);
    Bytes stbl = box("stbl", stsd + box("stts", stts) + box("stsc", stsc) + box("stsz", stsz) + box("stco", stco));

    Bytes smhd;
    u32(smhd, 0);
    u16(smhd, 0);
    u16(smhd, 0);
    Bytes dref;
    u32(dref, 0);
    u32(dref, 1);
    {
        Bytes u;
        u32(u, 1);
        dref += box("url ", u);
    }
    Bytes dinf = box("dinf", box("dref", dref));
    Bytes minf = box("minf", box("smhd", smhd) + dinf + stbl);

    Bytes hdlr;
    u32(hdlr, 0);
    u32(hdlr, 0);
    tag(hdlr, "soun");
    u32(hdlr, 0);
    u32(hdlr, 0);
    u32(hdlr, 0);
    hdlr.append("CrispASR", 8);
    u8(hdlr, 0);
    Bytes mdhd;
    u32(mdhd, 0);
    u32(mdhd, 0);
    u32(mdhd, 0);
    u32(mdhd, timescale);
    u32(mdhd, (uint32_t)media_duration);
    u16(mdhd, 0x55c4);
    u16(mdhd, 0); // language 'und' + predefined
    Bytes mdia = box("mdia", box("mdhd", mdhd) + box("hdlr", hdlr) + minf);

    Bytes tkhd;
    u32(tkhd, 0x00000007);
    u32(tkhd, 0);
    u32(tkhd, 0);
    u32(tkhd, 1);
    u32(tkhd, 0);
    u32(tkhd, (uint32_t)(media_duration * 1000 / (timescale ? timescale : 1))); // duration in movie timescale
    u32(tkhd, 0);
    u32(tkhd, 0);
    u16(tkhd, 0);
    u16(tkhd, 0);
    u16(tkhd, 0x0100);
    u16(tkhd, 0); // layer/altgrp/vol/res
    const uint32_t mat[9] = {0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000};
    for (uint32_t m : mat)
        u32(tkhd, m);
    u32(tkhd, 0);
    u32(tkhd, 0); // width/height = 0 (audio)
    Bytes trak = box("trak", box("tkhd", tkhd) + mdia);

    Bytes mvhd;
    u32(mvhd, 0);
    u32(mvhd, 0);
    u32(mvhd, 0);
    u32(mvhd, 1000);
    u32(mvhd, (uint32_t)(media_duration * 1000 / (timescale ? timescale : 1)));
    u32(mvhd, 0x00010000);
    u16(mvhd, 0x0100);
    u16(mvhd, 0);
    u32(mvhd, 0);
    u32(mvhd, 0);
    for (uint32_t m : mat)
        u32(mvhd, m);
    for (int i = 0; i < 6; i++)
        u32(mvhd, 0); // predefined
    u32(mvhd, 2);     // next_track_ID
    Bytes moov = box("moov", box("mvhd", mvhd) + trak);

    Bytes mdat = box("mdat", mdat_data);
    return ftyp + mdat + moov;
}

inline Bytes ftyp_box(const char* major, const std::vector<const char*>& compat) {
    Bytes p;
    tag(p, major);
    u32(p, 0);
    for (auto c : compat)
        tag(p, c);
    return box("ftyp", p);
}

// ---- ADTS -> raw AAC AUs ----
// Parse an ADTS blob (7-byte headers, no CRC) into access units.
inline bool parse_adts(const std::string& adts, size_t start, std::vector<std::pair<size_t, size_t>>& aus,
                       int& sr_index, int& channels) {
    size_t o = start;
    sr_index = -1;
    channels = 1;
    while (o + 7 <= adts.size()) {
        const uint8_t* d = (const uint8_t*)adts.data() + o;
        if (d[0] != 0xFF || (d[1] & 0xF0) != 0xF0)
            return !aus.empty();
        int frame_len = ((d[3] & 3) << 11) | (d[4] << 3) | (d[5] >> 5);
        if (frame_len < 7 || o + frame_len > adts.size())
            return !aus.empty();
        if (sr_index < 0) {
            sr_index = (d[2] >> 2) & 0xF;
            channels = ((d[2] & 1) << 2) | (d[3] >> 6);
        }
        int hdr = (d[1] & 1) ? 7 : 9; // protection_absent -> 7-byte header
        aus.emplace_back(o + hdr, (size_t)(frame_len - hdr));
        o += frame_len;
    }
    return !aus.empty();
}

// Encode float32 mono PCM to AAC-LC in an MP4 (M4A) container.
inline std::string make_aac_mp4(const float* pcm, int n_samples, int sample_rate, int bitrate_kbps = 96) {
    std::string adts = crispasr_make_aac(pcm, n_samples, sample_rate, bitrate_kbps);
    if (adts.empty())
        return {};
    // skip the leading ID3v2 tag the AAC writer prepends
    size_t start = 0;
    if (adts.size() >= 10 && adts.compare(0, 3, "ID3") == 0) {
        auto ss = [&](int i) { return (uint32_t)((uint8_t)adts[i] & 0x7f); };
        start = 10 + ((ss(6) << 21) | (ss(7) << 14) | (ss(8) << 7) | ss(9));
    }
    std::vector<std::pair<size_t, size_t>> aus;
    int sr_index = 6, channels = 1;
    if (!parse_adts(adts, start, aus, sr_index, channels))
        return {};
    const int enc_rate = crispasr_aac_target_rate(sample_rate);

    Bytes mdat_data;
    std::vector<uint32_t> sizes;
    for (auto& au : aus) {
        mdat_data.append(adts.data() + au.first, au.second);
        sizes.push_back((uint32_t)au.second);
    }

    Bytes asc;
    u8(asc, (uint8_t)((2 << 3) | (sr_index >> 1)));
    u8(asc, (uint8_t)(((sr_index & 1) << 7) | (channels << 3)));
    Bytes stsd = stsd_mp4a(enc_rate, channels, esds_aac(asc, (uint32_t)bitrate_kbps * 1000));
    uint64_t media_duration = (uint64_t)aus.size() * 1024;
    Bytes ftyp = ftyp_box("M4A ", {"M4A ", "mp42", "isom"});
    return assemble(ftyp, stsd, sizes, 1024, (uint32_t)enc_rate, media_duration, mdat_data);
}

// Encode float32 mono PCM to Opus in an MP4 container. glint's Opus encoder is
// CELT-only 48 kHz, 20 ms frames (960 samples), pre-skip 120 — input is
// linearly resampled to 48 kHz first.
inline std::string make_opus_mp4(const float* pcm, int n_samples, int sample_rate, int bitrate_kbps = 64) {
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
            int s0 = (int)pos, s1 = s0 + 1 < n_samples ? s0 + 1 : n_samples - 1;
            float frac = pos - (float)s0;
            resampled[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
        }
        pcm = resampled.data();
        n_samples = out_n;
    }
    glint_opus_enc_t enc = glint_opus_enc_create(1, bitrate_kbps * 1000, 1);
    if (!enc)
        return {};
    const int fs = 960; // 20 ms @ 48 kHz
    Bytes mdat_data;
    std::vector<uint32_t> sizes;
    std::vector<float> frame((size_t)fs);
    uint8_t pkt[1276];
    for (int off = 0; off < n_samples; off += fs) {
        int got = n_samples - off < fs ? n_samples - off : fs;
        for (int i = 0; i < got; i++)
            frame[(size_t)i] = pcm[off + i];
        for (int i = got; i < fs; i++)
            frame[(size_t)i] = 0.0f;
        int sz = glint_opus_encode(enc, frame.data(), fs, pkt, (int)sizeof(pkt));
        if (sz > 0) {
            mdat_data.append((const char*)pkt, (size_t)sz);
            sizes.push_back((uint32_t)sz);
        }
    }
    glint_opus_enc_destroy(enc);
    if (sizes.empty())
        return {};
    Bytes stsd = stsd_opus(48000, 1, 120);
    uint64_t media_duration = (uint64_t)sizes.size() * fs;
    Bytes ftyp = ftyp_box("isom", {"isom", "iso2", "mp41"});
    return assemble(ftyp, stsd, sizes, (uint32_t)fs, 48000, media_duration, mdat_data);
}

} // namespace crispasr_mp4

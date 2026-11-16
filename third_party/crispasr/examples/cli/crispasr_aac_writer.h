// crispasr_aac_writer.h — AAC-LC (ADTS) serializer for TTS float32 PCM,
// built on our in-tree glint encoder (glint/, experimental phase-1 AAC:
// long blocks, CBR-average). Header-only like crispasr_wav_writer.h;
// consumers link the `glint` static library.
//
// AI-generated provenance: the ADTS stream is prefixed with the ID3v2
// TXXX tag from crispasr_make_id3v2_ai_tag() (crispasr_wav_writer.h).
// A leading ID3v2 tag on raw ADTS is common practice and skipped by
// decoders (ffmpeg's ADTS demuxer, Apple's parsers).

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <glint/glint.h>

#include "core/crispasr_wav_writer.h"

// AAC supports the 12 standard sample rates. Return `sr` when directly
// encodable, otherwise the nearest supported rate at or above it
// (keeps full bandwidth), capped at 96 kHz.
inline int crispasr_aac_target_rate(int sr) {
    static const int rates[] = {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000};
    for (int r : rates)
        if (r == sr)
            return r;
    for (int r : rates)
        if (r > sr)
            return r;
    return 96000;
}

// Encode float32 mono PCM in [-1, 1] to an AAC-LC ADTS blob via glint,
// with the ID3v2 AI-provenance tag prepended. Samples outside [-1, 1]
// are clamped (same semantics as crispasr_make_wav_int16). Non-AAC
// sample rates are linearly resampled first (good enough for speech).
// Returns empty on failure.
inline std::string crispasr_make_aac(const float* pcm, int n_samples, int sample_rate, int bitrate_kbps = 96) {
    if (!pcm || n_samples <= 0 || sample_rate <= 0)
        return {};

    const int enc_rate = crispasr_aac_target_rate(sample_rate);
    std::vector<float> resampled;
    if (enc_rate != sample_rate) {
        const int out_n = (int)((int64_t)n_samples * enc_rate / sample_rate);
        if (out_n <= 0)
            return {};
        resampled.resize(out_n);
        for (int i = 0; i < out_n; i++) {
            float pos = (float)i * (float)sample_rate / (float)enc_rate;
            int s0 = (int)pos;
            int s1 = std::min(s0 + 1, n_samples - 1);
            float frac = pos - (float)s0;
            resampled[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
        }
        pcm = resampled.data();
        n_samples = out_n;
    }

    glint_aac_config cfg = {};
    cfg.sample_rate = enc_rate;
    cfg.num_channels = 1;
    cfg.bitrate = bitrate_kbps;
    glint_aac_t enc = glint_aac_create(&cfg);
    if (!enc)
        return {};

    const int spf = glint_aac_samples_per_frame(enc); // 1024
    std::string out = crispasr_make_id3v2_ai_tag();
    std::vector<float> frame((size_t)spf);
    const float* ch[1] = {frame.data()};
    for (int off = 0; off < n_samples; off += spf) {
        const int got = std::min(spf, n_samples - off);
        for (int i = 0; i < got; i++) {
            float s = pcm[off + i];
            if (s > 1.0f)
                s = 1.0f;
            if (s < -1.0f)
                s = -1.0f;
            frame[(size_t)i] = s;
        }
        std::fill(frame.begin() + got, frame.end(), 0.0f);
        int sz = 0;
        const uint8_t* data = glint_aac_encode_float(enc, ch, &sz);
        if (data && sz > 0)
            out.append((const char*)data, (size_t)sz);
    }
    // The MDCT looks back one block; flush emits the final 1024 samples.
    int flush_sz = 0;
    const uint8_t* flush_data = glint_aac_flush(enc, &flush_sz);
    if (flush_data && flush_sz > 0)
        out.append((const char*)flush_data, (size_t)flush_sz);
    glint_aac_destroy(enc);
    return out;
}

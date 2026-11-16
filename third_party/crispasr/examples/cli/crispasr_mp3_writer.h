// crispasr_mp3_writer.h — MP3 serializer for TTS float32 PCM, built on
// our in-tree glint encoder (glint/). Header-only like
// crispasr_wav_writer.h so the unit tests can exercise it without
// linking the server translation unit (consumers link the `glint`
// static library).
//
// glint is the default and always available. When the build found
// libmp3lame (CRISPASR_HAVE_LAME), it serves as an optional fallback:
// used automatically if glint fails, or forced with
// CRISPASR_MP3_ENCODER=lame (A/B comparisons).
//
// AI-generated provenance: the encoded stream is prefixed with the
// ID3v2 TXXX tag from crispasr_make_id3v2_ai_tag() (crispasr_wav_writer.h).

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <glint/glint.h>

#include "core/crispasr_wav_writer.h"

#ifdef CRISPASR_HAVE_LAME
#include <lame/lame.h>
#endif

// MP3 supports a fixed set of sample rates (MPEG-1: 32/44.1/48 kHz,
// MPEG-2: 16/22.05/24 kHz, MPEG-2.5: 8/11.025/12 kHz). Return `sr`
// when directly encodable, otherwise the nearest supported rate at or
// above it (keeps full bandwidth), capped at 48 kHz. TTS backends emit
// 16/22.05/24/44.1/48 kHz — all native — so resampling is the rare path.
inline int crispasr_mp3_target_rate(int sr) {
    static const int rates[] = {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000};
    for (int r : rates)
        if (r == sr)
            return r;
    for (int r : rates)
        if (r > sr)
            return r;
    return 48000;
}

// Encode float32 mono PCM in [-1, 1] to a CBR MP3 blob via glint, with
// the ID3v2 AI-provenance tag prepended. Samples outside [-1, 1] are
// clamped (same semantics as crispasr_make_wav_int16). Non-MP3 sample
// rates are linearly resampled to the nearest supported rate first
// (good enough for speech; mirrors the server's Opus path). 128 kbps
// mono is a valid bitrate for every MPEG version. Returns empty on
// failure.
inline std::string crispasr_make_mp3_glint(const float* pcm, int n_samples, int sample_rate, int bitrate_kbps = 128) {
    if (!pcm || n_samples <= 0 || sample_rate <= 0)
        return {};

    const int enc_rate = crispasr_mp3_target_rate(sample_rate);
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

    glint_config cfg = {};
    cfg.sample_rate = enc_rate;
    cfg.num_channels = 1;
    cfg.mode = GLINT_MONO;
    cfg.bitrate = bitrate_kbps;
    cfg.quality = GLINT_QUALITY_NORMAL;
    if (glint_check_config(cfg.sample_rate, cfg.bitrate) != 0)
        return {};
    glint_t enc = glint_create(&cfg);
    if (!enc)
        return {};

    const int spf = glint_samples_per_frame(enc);
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
        const uint8_t* data = glint_encode_float(enc, ch, &sz);
        if (data && sz > 0)
            out.append((const char*)data, (size_t)sz);
    }
    int flush_sz = 0;
    const uint8_t* flush_data = glint_flush(enc, &flush_sz);
    if (flush_data && flush_sz > 0)
        out.append((const char*)flush_data, (size_t)flush_sz);
    glint_destroy(enc);
    return out;
}

#ifdef CRISPASR_HAVE_LAME
// Optional libmp3lame path (fallback / CRISPASR_MP3_ENCODER=lame).
// Same contract as the glint encoder: ID3v2 AI-provenance tag
// prepended, empty on failure. Note lame_init_params rejects non-MP3
// output rates, so non-native input rates fail here (the glint path
// resamples instead).
inline std::string crispasr_make_mp3_lame(const float* pcm, int n_samples, int sample_rate, int bitrate_kbps = 128) {
    if (!pcm || n_samples <= 0 || sample_rate <= 0)
        return {};
    lame_t lame = lame_init();
    if (!lame)
        return {};
    lame_set_in_samplerate(lame, sample_rate);
    lame_set_num_channels(lame, 1);
    lame_set_out_samplerate(lame, sample_rate);
    lame_set_brate(lame, bitrate_kbps);
    lame_set_quality(lame, 2); // 2 = high quality
    lame_set_mode(lame, MONO);
    if (lame_init_params(lame) < 0) {
        lame_close(lame);
        return {};
    }

    // Convert float [-1,1] → int16
    std::vector<short> s16(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float v = pcm[i];
        if (v > 1.0f)
            v = 1.0f;
        else if (v < -1.0f)
            v = -1.0f;
        s16[i] = (short)(v * 32767.0f);
    }

    // Worst-case: 1.25 * n + 7200 (lame docs)
    size_t mp3_buf_size = (size_t)(1.25f * (float)n_samples) + 7200;
    std::vector<unsigned char> mp3_buf(mp3_buf_size);

    int written = lame_encode_buffer(lame, s16.data(), nullptr, n_samples, mp3_buf.data(), (int)mp3_buf_size);
    if (written < 0) {
        lame_close(lame);
        return {};
    }
    int flushed = lame_encode_flush(lame, mp3_buf.data() + written, (int)(mp3_buf_size - (size_t)written));
    lame_close(lame);
    if (flushed < 0)
        return {};

    std::string id3 = crispasr_make_id3v2_ai_tag();
    id3.append((const char*)mp3_buf.data(), (size_t)(written + flushed));
    return id3;
}
#endif // CRISPASR_HAVE_LAME

// Encoder dispatch: glint by default; libmp3lame (when compiled in) as
// automatic fallback on glint failure, or forced via
// CRISPASR_MP3_ENCODER=lame. A forced-lame failure still falls back to
// glint rather than returning empty.
inline std::string crispasr_make_mp3(const float* pcm, int n_samples, int sample_rate, int bitrate_kbps = 128) {
    if (!pcm || n_samples <= 0 || sample_rate <= 0)
        return {};
#ifdef CRISPASR_HAVE_LAME
    const char* pref = std::getenv("CRISPASR_MP3_ENCODER");
    if (pref && std::strcmp(pref, "lame") == 0) {
        std::string out = crispasr_make_mp3_lame(pcm, n_samples, sample_rate, bitrate_kbps);
        if (!out.empty())
            return out;
    }
#endif
    std::string out = crispasr_make_mp3_glint(pcm, n_samples, sample_rate, bitrate_kbps);
#ifdef CRISPASR_HAVE_LAME
    if (out.empty())
        out = crispasr_make_mp3_lame(pcm, n_samples, sample_rate, bitrate_kbps);
#endif
    return out;
}

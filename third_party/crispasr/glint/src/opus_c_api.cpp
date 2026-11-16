// glint - Opus C ABI (decoder, multistream decoder, CELT encoder)
// MIT License - Clean-room implementation

#include <cstdint>
#include <cstring>
#include <new>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "glint/glint.h"
#include "opus_celt_encoder.hpp"
#include "opus_decoder.hpp"
#include "opus_ms_decoder.hpp"
#include "opus_ogg.hpp"

using glint::opus::CeltEncoder;
using glint::opus::OpusDecoder;
using glint::opus::OpusMsDecoder;

struct glint_opus_dec_context {
    OpusDecoder dec;
};

struct glint_opus_ms_dec_context {
    OpusMsDecoder dec;
};

struct glint_opus_enc_context {
    CeltEncoder enc;
    int channels;
    int bitrate;
    int vbr;
};

extern "C" {

glint_opus_dec_t glint_opus_dec_create(int channels, int sample_rate) {
    if (channels < 1 || channels > 2) return nullptr;
    if (sample_rate != 48000 && sample_rate != 24000 &&
        sample_rate != 16000 && sample_rate != 12000 &&
        sample_rate != 8000)
        return nullptr;
    auto* ctx = new (std::nothrow) glint_opus_dec_context();
    if (!ctx) return nullptr;
    ctx->dec.init(channels, sample_rate);
    return ctx;
}

int glint_opus_decode(glint_opus_dec_t dec, const uint8_t* packet, int len,
                      float* pcm, int max_samples) {
    if (!dec || !packet || len <= 0 || !pcm) return -1;
    return dec->dec.decode(packet, len, pcm, max_samples);
}

int glint_opus_decode_fec(glint_opus_dec_t dec, const uint8_t* packet,
                          int len, float* pcm, int frame_size) {
    if (!dec || !pcm || frame_size <= 0) return -1;
    return dec->dec.decode_fec(packet, len, pcm, frame_size);
}

uint32_t glint_opus_dec_final_range(glint_opus_dec_t dec) {
    return dec ? dec->dec.final_range() : 0;
}

void glint_opus_dec_destroy(glint_opus_dec_t dec) { delete dec; }

glint_opus_ms_dec_t glint_opus_ms_dec_create(int channels, int streams,
                                             int coupled,
                                             const uint8_t* mapping,
                                             int sample_rate) {
    if (!mapping) return nullptr;
    auto* ctx = new (std::nothrow) glint_opus_ms_dec_context();
    if (!ctx) return nullptr;
    if (ctx->dec.init(channels, streams, coupled, mapping, sample_rate)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

int glint_opus_ms_decode(glint_opus_ms_dec_t dec, const uint8_t* packet,
                         int len, float* pcm, int max_samples) {
    if (!dec || !packet || len <= 0 || !pcm) return -1;
    return dec->dec.decode(packet, len, pcm, max_samples);
}

void glint_opus_ms_dec_destroy(glint_opus_ms_dec_t dec) { delete dec; }

glint_opus_enc_t glint_opus_enc_create(int channels, int bitrate_bps,
                                       int vbr) {
    if (channels < 1 || channels > 2) return nullptr;
    if (bitrate_bps < 6000 || bitrate_bps > 510000) return nullptr;
    auto* ctx = new (std::nothrow) glint_opus_enc_context();
    if (!ctx) return nullptr;
    ctx->enc.init(channels);
    ctx->channels = channels;
    ctx->bitrate = bitrate_bps;
    ctx->vbr = vbr ? 1 : 0;
    if (ctx->vbr) ctx->enc.set_vbr(bitrate_bps);
    return ctx;
}

int glint_opus_encode(glint_opus_enc_t enc, const float* pcm,
                      int frame_size, uint8_t* out, int max_bytes) {
    if (!enc || !pcm || !out) return -1;
    int lm;
    for (lm = 0; lm <= 3; lm++)
        if (120 << lm == frame_size) break;
    if (lm > 3) return -1;
    // CELT-only fullband TOC: configs 28..31 for 2.5/5/10/20 ms.
    const int config = 28 + lm;
    // CBR bytes per frame (excluding TOC); VBR uses the max as a cap.
    int nbytes;
    if (enc->vbr) {
        nbytes = 1275;
    } else {
        nbytes = enc->bitrate * frame_size / 48000 / 8 - 1;
        if (nbytes < 2) nbytes = 2;
        if (nbytes > 1275) nbytes = 1275;
    }
    if (max_bytes < 1 + nbytes && !enc->vbr) return -2;
    if (enc->vbr && max_bytes < 1276) {
        nbytes = max_bytes - 1;
        if (nbytes < 2) return -2;
    }
    out[0] = static_cast<uint8_t>((config << 3) |
                                  ((enc->channels == 2) << 2));
    int ret = enc->enc.encode_frame(pcm, frame_size, out + 1, nbytes);
    if (ret < 0) return ret;
    return 1 + ret;
}

uint32_t glint_opus_enc_final_range(glint_opus_enc_t enc) {
    return enc ? enc->enc.final_range() : 0;
}

void glint_opus_enc_destroy(glint_opus_enc_t enc) { delete enc; }

uint8_t* glint_opus_encode_file(const float* pcm, int frames, int channels,
                                int bitrate_bps, int vbr, int* out_size) {
    if (out_size) *out_size = 0;
    if (!pcm || frames <= 0 || channels < 1 || channels > 2) return nullptr;
    if (bitrate_bps < 6000 || bitrate_bps > 510000) return nullptr;

    CeltEncoder enc;
    enc.init(channels);
    if (vbr) enc.set_vbr(bitrate_bps);
    glint::opus::OggOpusWriter w;
    w.begin(channels, 120, 48000);  // pre-skip 120 (one CELT overlap)

    const int frame = 960;  // 20 ms at 48 kHz
    uint8_t pkt[1500];
    std::vector<float> buf(static_cast<size_t>(frame) * channels);
    for (long p = 0; p < frames; p += frame) {
        int got = static_cast<int>(std::min<long>(frame, frames - p));
        for (int i = 0; i < frame * channels; i++) buf[i] = 0.0f;
        for (int i = 0; i < got * channels; i++)
            buf[i] = pcm[static_cast<size_t>(p) * channels + i];
        int nb = vbr ? 1275 : bitrate_bps * frame / 48000 / 8 - 1;
        if (nb < 2) nb = 2;
        if (nb > 1275) nb = 1275;
        pkt[0] = static_cast<uint8_t>((31 << 3) | ((channels == 2) << 2));
        int r = enc.encode_frame(buf.data(), frame, pkt + 1, nb);
        if (r < 0) continue;
        w.add_packet(pkt, 1 + r, frame);
    }
    const auto& bytes = w.finish();
    uint8_t* out = static_cast<uint8_t*>(std::malloc(bytes.size()));
    if (!out) return nullptr;
    if (!bytes.empty()) std::memcpy(out, bytes.data(), bytes.size());
    if (out_size) *out_size = static_cast<int>(bytes.size());
    return out;
}

}  // extern "C"

// glint - MP3 + AAC-LC decoder C ABI
// MIT License - Clean-room implementation

#include <cstdint>
#include <new>

#include "aac_decoder.hpp"
#include "glint/glint.h"
#include "mp3_decoder.hpp"
#include "resample.hpp"

#include <cstdlib>
#include <cstring>
#include <vector>

struct glint_mp3_dec_context {
    glint::mp3::Mp3Decoder dec;
};
struct glint_aac_dec_context {
    glint::aac::AacDecoder dec;
};

static void copy_info(const glint::mp3::Mp3FrameInfo& s,
                      glint_dec_frame_info* d) {
    d->sample_rate = s.sample_rate;
    d->channels = s.channels;
    d->samples = s.samples;
    d->frame_bytes = s.frame_bytes;
}
static void copy_info(const glint::aac::AacFrameInfo& s,
                      glint_dec_frame_info* d) {
    d->sample_rate = s.sample_rate;
    d->channels = s.channels;
    d->samples = s.samples;
    d->frame_bytes = s.frame_bytes;
}

extern "C" {

int glint_mp3_frame_info(const uint8_t* data, int len,
                         glint_dec_frame_info* info) {
    glint::mp3::Mp3FrameInfo fi;
    if (glint::mp3::mp3_frame_info(data, len, &fi) < 0) return -1;
    if (info) copy_info(fi, info);
    return 0;
}

int glint_aac_frame_info(const uint8_t* data, int len,
                         glint_dec_frame_info* info) {
    glint::aac::AacFrameInfo fi;
    if (glint::aac::aac_frame_info(data, len, &fi) < 0) return -1;
    if (info) copy_info(fi, info);
    return 0;
}

glint_mp3_dec_t glint_mp3_dec_create(void) {
    auto* c = new (std::nothrow) glint_mp3_dec_context();
    if (c) c->dec.init();
    return c;
}

int glint_mp3_decode(glint_mp3_dec_t dec, const uint8_t* data, int len,
                     float* pcm, glint_dec_frame_info* info) {
    if (!dec || !data || !pcm) return -1;
    glint::mp3::Mp3FrameInfo fi;
    int n = dec->dec.decode_frame(data, len, pcm, &fi);
    if (info) copy_info(fi, info);
    return n;
}

void glint_mp3_dec_destroy(glint_mp3_dec_t dec) { delete dec; }

glint_aac_dec_t glint_aac_dec_create(void) {
    auto* c = new (std::nothrow) glint_aac_dec_context();
    if (c) c->dec.init();
    return c;
}

int glint_aac_decode(glint_aac_dec_t dec, const uint8_t* data, int len,
                     float* pcm, glint_dec_frame_info* info) {
    if (!dec || !data || !pcm) return -1;
    glint::aac::AacFrameInfo fi;
    int n = dec->dec.decode_frame(data, len, pcm, &fi);
    if (info) copy_info(fi, info);
    return n;
}

void glint_aac_dec_destroy(glint_aac_dec_t dec) { delete dec; }

float* glint_resample(const float* in, int in_frames, int channels,
                      int sr_in, int sr_out, int* out_frames) {
    if (!in || in_frames <= 0 || channels <= 0) {
        if (out_frames) *out_frames = 0;
        return nullptr;
    }
    int nf = 0;
    std::vector<float> r =
        glint::resample(in, in_frames, channels, sr_in, sr_out, &nf);
    float* buf = static_cast<float*>(
        std::malloc(sizeof(float) * static_cast<size_t>(nf) * channels));
    if (!buf) {
        if (out_frames) *out_frames = 0;
        return nullptr;
    }
    std::memcpy(buf, r.data(),
                sizeof(float) * static_cast<size_t>(nf) * channels);
    if (out_frames) *out_frames = nf;
    return buf;
}

void glint_free(void* p) { std::free(p); }

}  // extern "C"

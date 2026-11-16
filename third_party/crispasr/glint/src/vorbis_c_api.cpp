// glint - Vorbis decoder C ABI (whole-buffer Ogg-Vorbis I -> PCM)
// MIT License - Clean-room implementation.
//
// Mirrors glint_decode_audio / _ex for a single in-memory Ogg-Vorbis logical
// stream (the .sf3 sample case). glint_vorbis_decode returns interleaved
// float PCM; glint_vorbis_decode_ex adds an optional resample + int16 path.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "glint/glint.h"
#include "resample.hpp"
#include "vorbis_decoder.hpp"

extern "C" {

void* glint_vorbis_decode_ex(const uint8_t* ogg, int len, int out_rate,
                             int want_int16, int* out_sr, int* out_ch,
                             int* out_frames) {
    if (out_sr) *out_sr = 0;
    if (out_ch) *out_ch = 0;
    if (out_frames) *out_frames = 0;
    if (!ogg || len <= 0) return nullptr;

    std::vector<float> pcm;
    int sr = 0, ch = 0;
    if (glint::vorbis::decode_ogg(ogg, static_cast<size_t>(len), pcm, sr,
                                  ch) != 0 ||
        ch <= 0)
        return nullptr;

    if (out_rate > 0 && out_rate != sr && !pcm.empty()) {
        int in_frames = static_cast<int>(pcm.size() / static_cast<size_t>(ch));
        int nf = 0;
        std::vector<float> rs = glint::resample(pcm.data(), in_frames, ch, sr,
                                                out_rate, &nf);
        pcm.swap(rs);
        sr = out_rate;
    }

    int frames = static_cast<int>(pcm.size() / static_cast<size_t>(ch));
    void* buf = nullptr;
    if (want_int16) {
        int16_t* b = static_cast<int16_t*>(
            std::malloc(sizeof(int16_t) * (pcm.empty() ? 1 : pcm.size())));
        if (!b) return nullptr;
        for (size_t i = 0; i < pcm.size(); i++) {
            double s = pcm[i];
            if (s > 1.0) s = 1.0;
            if (s < -1.0) s = -1.0;
            int v = static_cast<int>(s * 32767.0 + (s >= 0 ? 0.5 : -0.5));
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            b[i] = static_cast<int16_t>(v);
        }
        buf = b;
    } else {
        float* b = static_cast<float*>(
            std::malloc(sizeof(float) * (pcm.empty() ? 1 : pcm.size())));
        if (!b) return nullptr;
        if (!pcm.empty())
            std::memcpy(b, pcm.data(), sizeof(float) * pcm.size());
        buf = b;
    }
    if (out_sr) *out_sr = sr;
    if (out_ch) *out_ch = ch;
    if (out_frames) *out_frames = frames;
    return buf;
}

float* glint_vorbis_decode(const uint8_t* ogg, int len, int* out_sr,
                           int* out_ch, int* out_frames) {
    return static_cast<float*>(glint_vorbis_decode_ex(
        ogg, len, 0, 0, out_sr, out_ch, out_frames));
}

}  // extern "C"

// glint - WAV I/O C ABI (any bit depth). Thin wrapper over src/wav_io.*.
// MIT License - Clean-room implementation.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "glint/glint.h"
#include "wav_io.hpp"

extern "C" {

float* glint_wav_read(const uint8_t* data, int len, int* out_sr,
                      int* out_ch, int* out_frames) {
    if (out_sr) *out_sr = 0;
    if (out_ch) *out_ch = 0;
    if (out_frames) *out_frames = 0;
    if (!data || len <= 0) return nullptr;

    std::vector<float> pcm;
    int sr = 0, ch = 0;
    if (!glint::wav_read(data, static_cast<size_t>(len), pcm, sr, ch) ||
        ch <= 0)
        return nullptr;

    int frames = static_cast<int>(pcm.size() / static_cast<size_t>(ch));
    float* buf = static_cast<float*>(std::malloc(
        sizeof(float) * (pcm.empty() ? 1 : pcm.size())));
    if (!buf) return nullptr;
    if (!pcm.empty())
        std::memcpy(buf, pcm.data(), sizeof(float) * pcm.size());
    if (out_sr) *out_sr = sr;
    if (out_ch) *out_ch = ch;
    if (out_frames) *out_frames = frames;
    return buf;
}

uint8_t* glint_wav_write(const float* pcm, int frames, int channels,
                         int sample_rate, int bits, int is_float,
                         int* out_size) {
    if (out_size) *out_size = 0;
    if (!pcm || frames < 0 || channels < 1) return nullptr;
    std::vector<uint8_t> bytes = glint::wav_write(
        pcm, frames, channels, sample_rate, bits, is_float != 0);
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(bytes.size()));
    if (!buf) return nullptr;
    if (!bytes.empty()) std::memcpy(buf, bytes.data(), bytes.size());
    if (out_size) *out_size = static_cast<int>(bytes.size());
    return buf;
}

}  // extern "C"

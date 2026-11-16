// glint - WAV / raw-PCM I/O to the universal interleaved-float format.
// Reads PCM 8/16/24/32, IEEE float 32/64, A-law, mu-law and
// WAVE_FORMAT_EXTENSIBLE; writes PCM 8/16/24/32 and float 32/64.
// Shared by the CLI (cli/audio_io.hpp) and the C ABI (wav_c_api.cpp).
// MIT License - Clean-room implementation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace glint {

// Parse a WAV buffer into interleaved float PCM (±1.0). On success fills
// `pcm`, `sample_rate`, `channels` and returns true; false if malformed
// or an unsupported sample format.
bool wav_read(const uint8_t* data, size_t len, std::vector<float>& pcm,
              int& sample_rate, int& channels);

// Interpret a headerless PCM buffer (little-endian signed int, `bits` =
// 8/16/24/32) as interleaved float. Returns false on bad parameters.
bool pcm_read(const uint8_t* data, size_t len, int sample_rate, int channels,
              int bits, std::vector<float>& pcm);

// Encode interleaved float PCM to a WAV buffer. `bits`: 8/16/24/32 for
// integer PCM, or 32/64 with `is_float` for IEEE float. Out-of-range
// combinations fall back to 16-bit PCM.
std::vector<uint8_t> wav_write(const float* pcm, long frames, int channels,
                               int sample_rate, int bits, bool is_float);

}  // namespace glint

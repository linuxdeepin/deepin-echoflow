// glint - windowed-sinc sample-rate converter (PLAN § B6)
// MIT License - Clean-room implementation.
//
// Arbitrary-ratio resampling of interleaved float PCM. A Kaiser-windowed
// sinc kernel with an anti-alias cutoff at the lower of the two Nyquist
// rates; unity passband gain. Not the fastest possible (per-output
// direct convolution), but dependency-free, accurate, and fine for a
// codec front-end (offline transcode / encode preprocessing).

#pragma once

#include <cstdint>
#include <vector>

namespace glint {

// Resample `in` (n_in frames, `channels` interleaved) from sr_in to
// sr_out. Returns the interleaved output; out_frames (optional) receives
// the frame count. quality = half-width of the sinc kernel in taps
// (16 is transparent for music; larger = sharper transition, slower).
std::vector<float> resample(const float* in, int n_in, int channels,
                            int sr_in, int sr_out, int* out_frames = nullptr,
                            int quality = 16);

}  // namespace glint

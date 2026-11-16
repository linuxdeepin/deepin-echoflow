// audio_window.h — shared --offset-t / --duration time-window math (#91).
//
// Both dispatch surfaces (the CLI in examples/cli/crispasr_run.cpp and the
// HTTP server in examples/cli/crispasr_server.cpp) restrict processing to a
// [offset, offset+duration) window of the decoded PCM before slicing, then
// shift reported timestamps back by the offset. The window arithmetic was
// duplicated across both — factor it here so the two can't drift, and so the
// (model-free) logic is unit-testable.
//
// Header-only, no ggml / no dependencies.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace core_audio_window {

struct Window {
    bool active = false;   // true when a window was requested (offset or duration > 0)
    bool past_end = false; // true when the offset starts at/after end-of-buffer
    int64_t start = 0;     // first sample index of the window
    int64_t len = 0;       // number of samples in the window
};

// Compute the sample window for the given offset/duration (both in ms) against
// a `total_samples` buffer at `sample_rate`. `active` is false when neither is
// set (no windowing). `past_end` is true when the offset lands at or beyond the
// end of the buffer (caller should treat as "nothing to process"). `len` is
// clamped to the available tail; a duration of 0 means "to the end".
inline Window compute(int64_t total_samples, int offset_ms, int duration_ms, int sample_rate) {
    Window w;
    if (offset_ms <= 0 && duration_ms <= 0)
        return w; // inactive
    w.active = true;
    if (total_samples <= 0 || sample_rate <= 0) {
        w.past_end = true;
        return w;
    }
    int64_t off = (int64_t)offset_ms * sample_rate / 1000;
    if (off < 0)
        off = 0;
    if (off >= total_samples) {
        w.past_end = true;
        return w;
    }
    int64_t len = (duration_ms > 0) ? (int64_t)duration_ms * sample_rate / 1000 : (total_samples - off);
    if (len < 0)
        len = 0;
    if (len > total_samples - off)
        len = total_samples - off;
    w.start = off;
    w.len = len;
    return w;
}

// Trim a PCM vector in place to the window (renamed from apply to avoid
// colliding with std::apply under a using-declaration). No-op when the window is inactive,
// past-end, or the vector is empty. Re-clamps to the vector's own size so it is
// safe to apply to a channel buffer that is shorter than the primary buffer.
inline void trim(std::vector<float>& v, const Window& w) {
    if (!w.active || w.past_end || v.empty())
        return;
    const int64_t o = std::min<int64_t>(w.start, (int64_t)v.size());
    const int64_t e = std::min<int64_t>(o + w.len, (int64_t)v.size());
    std::vector<float> trimmed(v.begin() + o, v.begin() + e);
    v.swap(trimmed);
}

} // namespace core_audio_window

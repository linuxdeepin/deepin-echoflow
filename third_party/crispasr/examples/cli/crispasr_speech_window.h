// crispasr_speech_window.h — what duration a min/max speech-token request
// actually yields, and when it cannot yield what was asked for.
//
// `min_speech_tokens` + `max_speech_tokens` (PR #330) exist to pin synthesis to
// an exact audio window: set both to N and the MOSS decode loop is forbidden
// from stopping before N frames and forced to stop at N, so the output is N
// frames (~12.5 frames/sec) with no post-hoc tempo change. That is what
// lip-sync and game dubbing need — audio that lands in the slot it was cut for.
//
// THE FAILURE THIS EXISTS TO MAKE VISIBLE
// ---------------------------------------
// The floor is not the only bound on the loop. The decode runs
// `for (f = 0; f < max_frames; f++)` where max_frames comes from
// `max_new_tokens` when the caller set it explicitly, else the backend default
// (4096). So a request pairing a 250-frame floor with `max_new_tokens: 100`
// silently produces ~100 frames — 8 seconds where 20 were asked for. Nothing
// errors: the floor is simply unreachable, the loop hits its own bound first,
// and the caller gets short audio that does NOT fit the slot.
//
// For a feature whose entire purpose is exact duration, silently returning a
// different duration is the worst available behaviour. It is also easy to hit:
// `max_new_tokens` is a general AR knob, so a caller who set it once for a
// different reason carries it into every later request.
//
// Weight-free and IO-free so the arithmetic is unit-testable without a model
// (tests/test-speech-window.cpp), like the other policy headers here.

#pragma once

#include <string>

namespace crispasr_speech_window {

// The bound the MOSS decode loop actually uses when `max_new_tokens` was not
// explicitly set. Mirrors `max_frames = sp.max_new_frames > 0 ? ... : 4096` in
// src/moss_tts_local.cpp — kept here so the check and the loop cannot drift
// apart silently.
inline constexpr int kDefaultFrameCap = 4096;

// Frames per second of the MOSS codec, for turning a frame count into the
// seconds a caller actually reasons in.
inline constexpr double kFramesPerSecond = 12.5;

inline double frames_to_seconds(int frames) {
    return frames <= 0 ? 0.0 : (double)frames / kFramesPerSecond;
}

struct Window {
    // The floor and ceiling as requested (-1 = unset).
    int min_frames = -1;
    int max_frames = -1;
    // The loop's own bound: max_new_tokens when explicit, else kDefaultFrameCap.
    int frame_cap = kDefaultFrameCap;

    // True when the request pins an exact duration (both set and equal).
    bool is_exact() const { return min_frames > 0 && max_frames > 0 && min_frames == max_frames; }

    // The floor cannot be honoured: the loop stops before reaching it. The
    // caller asked for at least `min_frames` and will receive at most this.
    bool floor_unreachable() const { return min_frames > 0 && min_frames > frame_cap; }

    // A floor above the ceiling is NOT an error — the ceiling wins, because the
    // max break runs right after each frame is appended while the floor only
    // suppresses the stop head. Worth reporting, because the caller plainly
    // meant something else.
    bool floor_above_ceiling() const { return min_frames > 0 && max_frames > 0 && min_frames > max_frames; }

    // Frames the caller will actually get, at most.
    int effective_ceiling() const {
        int c = frame_cap;
        if (max_frames > 0 && max_frames < c)
            c = max_frames;
        return c;
    }
};

// One diagnostic line, or "" when the request is coherent. Deliberately a
// WARNING and not a refusal: the synthesis is still useful, the caller may
// genuinely want a truncated take, and hard-failing a TTS request over a
// duration hint would be worse than saying so. Returning the text (rather than
// printing) keeps this testable and lets each surface route it — stderr on the
// CLI, a response header or log line on the server.
inline std::string diagnose(const Window& w) {
    if (w.floor_unreachable()) {
        return "min_speech_tokens=" + std::to_string(w.min_frames) + " (" +
               std::to_string(frames_to_seconds(w.min_frames)).substr(0, 5) +
               "s) cannot be reached: the decode stops at " + std::to_string(w.frame_cap) + " frames (" +
               std::to_string(frames_to_seconds(w.frame_cap)).substr(0, 5) +
               "s). Raise max_new_tokens to at least the floor, or lower min_speech_tokens — "
               "the output will otherwise be SHORTER than requested.";
    }
    if (w.floor_above_ceiling()) {
        return "min_speech_tokens=" + std::to_string(w.min_frames) +
               " exceeds max_speech_tokens=" + std::to_string(w.max_frames) +
               "; the ceiling wins, so the output will be " + std::to_string(w.max_frames) + " frames.";
    }
    return {};
}

} // namespace crispasr_speech_window

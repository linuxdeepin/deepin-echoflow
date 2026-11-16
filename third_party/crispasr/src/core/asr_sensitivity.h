// core/asr_sensitivity.h — named decode-threshold bundles (PLAN.md §W7).
//
// The whisper decoder's fallback behaviour is governed by four floats that
// interact: `entropy_thold`, `logprob_thold`, `no_speech_thold` and
// `temperature_inc`. Individually each is meaningless to anyone who has not
// read the decoder, and tuning them one at a time is how you end up with a
// combination nobody has tried — raise `no_speech_thold` alone and you suppress
// the very fallback that `logprob_thold` was supposed to trigger, because
// crispasr.cpp:8499 requires BOTH to be crossed before a decode is retried.
//
// This maps a name to a coherent set. It is presentation only: nothing here
// changes decoder behaviour that the four knobs could not already express, and
// an explicitly-passed knob always wins over the preset (the caller applies the
// preset first, then its own overrides).
//
// Weight-free and pure — see `tests/test-asr-sensitivity.cpp`.

#pragma once

#include <string>

namespace core_sensitivity {

struct Thresholds {
    float entropy_thold;
    float logprob_thold;
    float no_speech_thold;
    float temperature_inc;
};

// The shipped defaults (crispasr.cpp:6523-6526). `balanced` is exactly this, so
// naming it is always a no-op — that is the point: it gives a user a word for
// "leave it alone" and a fixed point to compare the other two against.
inline Thresholds defaults() {
    return Thresholds{2.40f, -1.00f, 0.60f, 0.20f};
}

enum class Preset { Balanced, Conservative, Aggressive };

// conservative — fewer hallucinations, at the cost of dropping marginal speech.
//   Tighter entropy and logprob bars mean a decode is rejected and retried
//   sooner; a LOWER no_speech_thold means the "this is silence" verdict is
//   reached more readily, so borderline segments are discarded rather than
//   guessed at. Use on noisy or music-heavy material where a confident wrong
//   sentence is worse than a missing one.
//
// aggressive — keeps marginal speech, at the cost of more hallucination.
//   Looser bars and a HIGHER no_speech_thold, so quiet or whispered audio still
//   produces text instead of being written off as silence. Use when the audio
//   is quiet by nature and you would rather post-filter than lose content.
inline Thresholds preset(Preset p) {
    switch (p) {
    case Preset::Conservative:
        return Thresholds{2.20f, -0.70f, 0.45f, 0.20f};
    case Preset::Aggressive:
        return Thresholds{2.80f, -1.50f, 0.80f, 0.20f};
    case Preset::Balanced:
    default:
        return defaults();
    }
}

// Parse a preset name. Returns false (and leaves `out` untouched) for anything
// unrecognised, so a caller can report the bad value rather than silently
// applying a default the user did not ask for.
inline bool parse_preset(const std::string& name, Preset& out) {
    if (name == "balanced" || name == "default") {
        out = Preset::Balanced;
        return true;
    }
    if (name == "conservative" || name == "strict") {
        out = Preset::Conservative;
        return true;
    }
    if (name == "aggressive" || name == "loose") {
        out = Preset::Aggressive;
        return true;
    }
    return false;
}

inline const char* preset_name(Preset p) {
    switch (p) {
    case Preset::Conservative:
        return "conservative";
    case Preset::Aggressive:
        return "aggressive";
    case Preset::Balanced:
    default:
        return "balanced";
    }
}

// Comma-separated list for help text and error messages, so the accepted values
// are never out of sync with `parse_preset`.
inline const char* preset_list() {
    return "conservative, balanced, aggressive";
}

} // namespace core_sensitivity

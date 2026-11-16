// btc_chord_vocab.h — BTC chord vocabulary and positional encoding.
//
// Extracted from btc_chords.cpp so it can be unit-tested WITHOUT the weights.
// Everything here is pure: no ggml, no I/O, no model. That matters because
// these are exactly the parts where a silent one-entry mistake produces
// plausible-looking but wrong output — a chord mislabelled `:min` reads as a
// model quality issue, not as a table typo, and no cosine check would ever
// flag it (the logits are identical; only the NAME is wrong).
//
// See docs/music-transcription/BTC_BLUEPRINT.md.

#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace btc_vocab {

inline const char* const* roots() {
    static const char* kRoots[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    return kRoots;
}

inline const char* const* qualities() {
    static const char* kQualities[14] = {"min",     "maj",  "dim", "aug",  "min6",  "maj6", "min7",
                                         "minmaj7", "maj7", "7",   "dim7", "hdim7", "sus2", "sus4"};
    return kQualities;
}

// 170-class: idx 0..167 = root*14 + quality, 168 = X (unknown), 169 = N (none).
// Quality index 1 ("maj") renders as the bare root, matching idx2voca_chord().
inline std::string voca_name(int i) {
    if (i == 169)
        return "N";
    if (i == 168)
        return "X";
    const int root = i / 14, q = i % 14;
    if (root < 0 || root >= 12)
        return "X";
    return q == 1 ? std::string(roots()[root]) : std::string(roots()[root]) + ":" + qualities()[q];
}

// 25-class: C, C:min, C#, C#:min, ... B, B:min, N.
inline std::string maj_min_name(int i) {
    if (i >= 24)
        return "N";
    return (i % 2 == 0) ? std::string(roots()[i / 2]) : std::string(roots()[i / 2]) + ":min";
}

// Which of the 14 qualities carry a minor third. Suspended and augmented
// chords have NO third at all, so their placement is a convention rather than
// a fact; the convention here is mir_eval's (they go to major).
inline const bool* quality_is_minor() {
    static const bool kIsMinor[14] = {
        true,  // min
        false, // maj
        true,  // dim
        false, // aug      -- no third; convention
        true,  // min6
        false, // maj6
        true,  // min7
        true,  // minmaj7
        false, // maj7
        false, // 7
        true,  // dim7
        true,  // hdim7
        false, // sus2     -- no third; convention
        false, // sus4     -- no third; convention
    };
    return kIsMinor;
}

// Collapse a 170-class label to the 25-class maj/min vocabulary. This is why
// 170 ships as the DEFAULT: it is strictly more expressive and can be reduced
// on demand, whereas a 25-class model can never be expanded.
inline int voca_to_maj_min(int i) {
    if (i >= 168)
        return 24; // X and N both become N
    const int root = i / 14, q = i % 14;
    return root * 2 + (quality_is_minor()[q] ? 1 : 0);
}

// Transformer positional encoding. The two halves are CONCATENATED
// ([sin... | cos...]), NOT interleaved — one of the details in BTC_BLUEPRINT.md
// that is a silent accuracy bug if assumed the other way round.
inline void timing_signal(int length, int channels, std::vector<float>& out) {
    const int n = channels / 2;
    const float inc = std::log(1.0e4f / 1.0f) / (float)(n - 1);
    out.assign((size_t)length * channels, 0.0f);
    for (int t = 0; t < length; t++)
        for (int i = 0; i < n; i++) {
            const float scaled = (float)t * std::exp((float)i * -inc);
            out[(size_t)t * channels + i] = std::sin(scaled);
            out[(size_t)t * channels + n + i] = std::cos(scaled);
        }
}

// ---------------------------------------------------------------------------
// Pipeline geometry
//
// BTC's reference front end (utils/mir_eval_modules.audio_file_to_features)
// CQTs each `inst_len`-second chunk INDEPENDENTLY and concatenates. Because
// librosa centres every call, each chunk carries its own edge padding, so this
// does NOT equal a continuous transform of the whole signal -- it yields more
// frames, and the difference grows with duration. Measured on the upstream
// 257 s test clip: 2778 frames chunked vs 2770 continuous.
// ---------------------------------------------------------------------------

// Frames a centred CQT emits for `n_samples` at `hop`. Mirrors librosa's
// centre convention (core_cqt::n_frames).
inline int centred_n_frames(int n_samples, int hop) {
    if (n_samples <= 0 || hop <= 0)
        return 0;
    return 1 + n_samples / hop;
}

// Total frames the CHUNKED pipeline emits. The reference always makes a final
// call for the remainder, even when it is empty.
inline int chunked_n_frames(int n_samples, int hop, int chunk) {
    if (chunk <= 0)
        return centred_n_frames(n_samples, hop);
    int total = 0, cur = 0;
    while (n_samples > cur + chunk) {
        total += centred_n_frames(chunk, hop);
        cur += chunk;
    }
    return total + centred_n_frames(n_samples - cur, hop);
}

// Seconds per output frame. The reference derives this from the CHUNK geometry
// (feature_per_second = inst_len / timestep), NOT from hop/sample_rate. They
// differ by 0.31 % -- 10/108 = 0.0925926 vs 2048/22050 = 0.0928798 -- which is
// 0.79 s of accumulated drift over a 4-minute song.
inline double frame_seconds(float inst_len_sec, int timestep, int hop, int sample_rate) {
    if (inst_len_sec > 0.0f && timestep > 0)
        return (double)inst_len_sec / (double)timestep;
    return (double)hop / (double)sample_rate;
}

} // namespace btc_vocab

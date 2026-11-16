// core/vad_failover.h — decide when VAD output is obviously wrong (PLAN.md §W4).
//
// A VAD that returns almost nothing on a long recording is indistinguishable,
// downstream, from a recording that genuinely contains almost no speech: both
// produce a handful of slices and a nearly-empty transcript. The difference is
// that one of them is a bug (wrong sample rate, a model that failed to warm, a
// threshold mistuned for the material) and the user just loses their audio.
//
// This predicate says "that cannot be right, transcribe the whole thing
// instead". It is deliberately hard to trigger: the cost of a false positive is
// transcribing silence (slow, some hallucination risk), while the cost of a
// false negative is losing the entire transcript. The asymmetry justifies the
// failover, but only at thresholds no plausible real recording reaches.
//
// Shape from WhisperJAV's `modules/vad_failover.py`, with one deliberate
// correction — see `few_segments` below. Weight-free and pure; the caller
// decides what "transcribe the whole thing" means for its pipeline
// (`crispasr_fixed_chunk_slices`, not one giant slice).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace core_vad_failover {

// One detected speech region, in seconds. Callers convert from whatever their
// slice type uses (sample indices, centiseconds).
struct Span {
    double t0;
    double t1;
};

struct Config {
    // Short clips keep the old behaviour unconditionally: on a 10 s file
    // "VAD found 0.2 s of speech" is entirely plausible, and re-running the
    // whole clip costs little anyway. The failure this guards is the long-file
    // one, where the loss is large and the evidence is strong.
    double min_audio_sec = 120.0;
    // Speech covering less than this fraction of a long recording is not a
    // recording, it is a broken VAD. 1% of a 2-minute file is 1.2 s.
    double min_coverage = 0.01;
    // A clip this many times `min_audio_sec` with only a couple of segments is
    // suspicious on segment COUNT rather than coverage.
    double very_long_multiple = 4.0;
    int few_segment_count = 2;
    // ...but only if those few segments also cover little of the audio.
    //
    // WhisperJAV's version omits this second condition and fails over on
    // `len(segments) <= 2 and duration >= 480` alone. That misfires on a
    // perfectly good result: an 8-minute continuous monologue is legitimately
    // ONE segment covering ~100% of the file, and forcing a full re-transcribe
    // there throws away correct work and re-runs the model over everything.
    // Requiring low coverage too keeps the signal ("barely any speech found")
    // and drops the accident ("speech found, in one piece").
    double few_segment_max_coverage = 0.10;
};

struct Verdict {
    bool failover = false;
    double speech_sec = 0.0;
    double coverage = 0.0;
    int segment_count = 0;
    std::string reason; // empty iff !failover
};

// `audio_sec <= 0` means unknown — never fail over, since every signal here is
// a ratio against the clip length.
inline Verdict assess(const std::vector<Span>& spans, double audio_sec, const Config& cfg = {}) {
    Verdict v;
    v.segment_count = (int)spans.size();
    if (audio_sec <= 0.0)
        return v;

    for (const Span& s : spans)
        if (s.t1 > s.t0)
            v.speech_sec += s.t1 - s.t0;
    v.coverage = v.speech_sec / audio_sec;

    if (audio_sec < cfg.min_audio_sec)
        return v;

    // No speech at all on a long file. The strongest signal there is.
    if (spans.empty()) {
        v.failover = true;
        v.reason = "VAD returned no speech at all on a " + std::to_string((long)audio_sec) + "s clip";
        return v;
    }

    if (v.coverage < cfg.min_coverage) {
        v.failover = true;
        v.reason = "VAD found " + std::to_string(v.speech_sec) + "s of speech in " + std::to_string((long)audio_sec) +
                   "s (" + std::to_string(v.coverage * 100.0) + "%, under " + std::to_string(cfg.min_coverage * 100.0) +
                   "%)";
        return v;
    }

    if ((int)spans.size() <= cfg.few_segment_count && audio_sec >= cfg.very_long_multiple * cfg.min_audio_sec &&
        v.coverage < cfg.few_segment_max_coverage) {
        v.failover = true;
        v.reason = "VAD returned only " + std::to_string(spans.size()) + " segment(s) covering " +
                   std::to_string(v.coverage * 100.0) + "% of a " + std::to_string((long)audio_sec) + "s clip";
        return v;
    }

    return v;
}

} // namespace core_vad_failover

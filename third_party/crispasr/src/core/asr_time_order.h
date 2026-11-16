// src/core/asr_time_order.h — "is this transcript in time order?" (#356).
//
// #356 shipped because nothing between the producer and the consumer ever
// checked it: the CLI's merge_segments() concatenates slice results with no
// check, the core_seg_hygiene pass is opt-in and documents itself as "never
// reorders", and the session ABI reimplements every backend's transcribe
// inline so nothing wired into the CLI reaches it. A gap-fill second pass
// emitted its recoveries inside the span of the segment they came from, and
// the first anyone heard of it was a downstream subtitle complaint.
//
// **Which timestamp orders the output is the whole subtlety.** The CLI's
// crispasr_make_disp_segments builds every SRT/VTT cue from the WORD
// timestamps of a segment that carries words, and only falls back to
// seg.t0/seg.t1 for a text-only segment. A guard written against segment
// spans passes the exact list that caused #356 — so this walks words first
// and spans only as the fallback, exactly as the writers do.
//
// Templated over the segment type on purpose: the three surfaces carry three
// different structs (crispasr_segment in the CLI/server, crispasr_session_seg
// in the C ABI) and this repo's #1 recurring bug is a fix that reached one of
// them. Any type with `.text`, `.t0` and a `.words` range of `{.text, .t0}`
// fits.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace core_time_order {

// Index of the first segment that starts before a position already emitted, or
// -1 when the list is in order. `prev_cs` / `cur_cs` (optional) receive the two
// offending centisecond timestamps.
//
// Empty-text segments are skipped rather than treated as position 0 — the
// writers drop them, so counting them would fake a backward jump on whatever
// follows.
template <typename SegT>
int first_backward(const std::vector<SegT>& segments, int64_t* prev_cs = nullptr, int64_t* cur_cs = nullptr) {
    int64_t prev = INT64_MIN;
    for (size_t i = 0; i < segments.size(); i++) {
        const auto& seg = segments[i];
        if (seg.words.empty()) {
            if (seg.text.empty())
                continue;
            if (seg.t0 < prev) {
                if (prev_cs)
                    *prev_cs = prev;
                if (cur_cs)
                    *cur_cs = seg.t0;
                return (int)i;
            }
            prev = seg.t0;
            continue;
        }
        for (const auto& w : seg.words) {
            if (w.text.empty())
                continue;
            if (w.t0 < prev) {
                if (prev_cs)
                    *prev_cs = prev;
                if (cur_cs)
                    *cur_cs = w.t0;
                return (int)i;
            }
            prev = w.t0;
        }
    }
    return -1;
}

// Centiseconds -> HH:MM:SS,mmm, so the warning reads in the same units as the
// SRT the user is looking at. Kept local rather than reaching for the CLI's
// crispasr_to_timestamp: this header has to be includable from src/, which
// cannot see examples/cli.
inline void format_cs(int64_t cs, char* buf, size_t n) {
    int64_t ms = cs * 10;
    const int64_t hr = ms / 3600000;
    ms -= hr * 3600000;
    const int64_t mn = ms / 60000;
    ms -= mn * 60000;
    const int64_t sc = ms / 1000;
    ms -= sc * 1000;
    snprintf(buf, n, "%02d:%02d:%02d,%03d", (int)hr, (int)mn, (int)sc, (int)ms);
}

// The once-per-process flag. It lives in a NON-template function on purpose: a
// function-local static inside a template is one object per instantiation, so
// putting it in warn_if_backward() below would give the CLI's crispasr_segment
// and the session's crispasr_session_seg a flag each and warn twice in a host
// that drives both.
inline bool& warn_once_flag() {
    static bool warned = false;
    return warned;
}

// One warning per process, shared across surfaces. `where` names the producing
// stage. CRISPASR_ORDER_WARN=0 silences it.
//
// Note what this deliberately does NOT flag: cues that merely OVERLAP, i.e. one
// ending after the next begins. A gap-fill refill window re-hears the audio
// either side of a hole, so a recovered word can end a couple of hundred ms
// after the following word starts; that is word-timestamp jitter, not a
// reordering, and flagging it would make this noise on every gap-filled file.
template <typename SegT> void warn_if_backward(const std::vector<SegT>& segments, const char* where) {
    if (warn_once_flag())
        return;
    if (const char* e = getenv("CRISPASR_ORDER_WARN"))
        if (atoi(e) == 0)
            return;
    int64_t prev = 0, cur = 0;
    const int i = first_backward(segments, &prev, &cur);
    if (i < 0)
        return;
    warn_once_flag() = true;
    char pbuf[32], cbuf[32];
    format_cs(prev, pbuf, sizeof(pbuf));
    format_cs(cur, cbuf, sizeof(cbuf));
    fprintf(stderr,
            "crispasr: warning: transcript is not in time order after %s -- segment %d starts at %s, "
            "after %s was already emitted. Subtitle output will contain overlapping cues; please report "
            "this with the audio (see issue #356).\n",
            where, i, cbuf, pbuf);
}

} // namespace core_time_order

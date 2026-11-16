// core/align_sentinel.h — detect forced-alignment collapse (PLAN.md §W3).
//
// `ctc_forced_align()` returns per-word timestamps and cannot fail loudly: it
// returns {} only when the whole call is unusable (T<=0, no words). Two paths
// inside it silently emit t0 == t1 == 0 for individual words, mixed into an
// otherwise plausible-looking result:
//
//   * `wranges[wi].cs < 0` — the word's characters are entirely absent from the
//     CTC vocabulary. align.h documents this in its @return block. Feed a
//     Chinese transcript to a Latin-vocab CTC model and EVERY word comes back
//     at (0,0) with a successful return.
//   * `t0_frame < 0` — the Viterbi path never visited that word's labels.
//
// The text is right and the timestamps are garbage, which is the worst shape a
// bug can take: subtitles render, they are merely all wrong. Nothing downstream
// checks for it.
//
// This header is the check. It is DETECTION ONLY by default — it does not
// rewrite timestamps, because we have never measured how often this fires in
// production and a silent auto-repair would replace one invisible wrong answer
// with another. `redistribute()` is available for callers that opt in.
//
// Shape borrowed from WhisperJAV's `modules/alignment_sentinel.py`; the
// thresholds are re-derived for `ctc_word_stamp` (seconds, float) and each one
// carries its own precondition so a short or sparse clip is not mistaken for a
// collapse. Weight-free and pure — see `tests/test-align-sentinel.cpp`.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace core_align_sentinel {

// One word's span. Mirrors `ctc_word_stamp` (align.h) without depending on it,
// so callers holding any word-timestamp type can feed this header.
struct Word {
    std::string text;
    float t0; // seconds
    float t1; // seconds, end-exclusive
};

struct Thresholds {
    // Below this many characters there is not enough text to judge: a 2-word
    // alignment legitimately spans a fraction of a second.
    int min_chars_to_assess = 10;
    // Span covering less than this fraction of the audio, WITH substantial
    // text, means the words piled up somewhere instead of spreading out.
    float min_coverage = 0.05f;
    // Characters per second across the whole span. ~15-20 is fast Latin
    // speech, ~10 is conversational Japanese; 50 is not physically utterable
    // in any script, so exceeding it means the span is too short for the text.
    float max_chars_per_sec = 50.0f;
    // A span this short carrying substantial text is the classic ~100 ms
    // pile-up. Only applied when the audio is meaningfully longer (below).
    float min_span_sec = 0.5f;
    // The span check is skipped unless the audio is at least this many times
    // the span — otherwise a genuinely short clip trips it.
    float short_span_audio_ratio = 4.0f;
    // Fraction of words sitting at exactly (0,0) — align.cpp's two documented
    // silent-zero paths. A real alignment puts at most the leading word there.
    float max_zero_position_ratio = 0.10f;
    // Fraction of words with t1 <= t0 (empty span). Distinct from the above:
    // these carry a nonzero position but no duration.
    float max_degenerate_ratio = 0.40f;
};

struct Assessment {
    bool collapsed = false;
    int word_count = 0;
    int char_count = 0;
    int zero_position_words = 0; // t0 == 0 && t1 == 0
    int degenerate_words = 0;    // t1 <= t0, excluding the zero-position ones
    float span_sec = 0.0f;       // last t1 - first t0, over positioned words
    float coverage = 0.0f;       // span_sec / audio_duration_sec
    float chars_per_sec = 0.0f;
    // One entry per signal that fired, each naming its measured value and the
    // threshold it crossed. Empty iff !collapsed.
    std::vector<std::string> reasons;
};

namespace detail {

inline std::string fmt2(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", (double)v);
    return std::string(buf);
}

// Count Unicode code points, not bytes: a chars-per-second rate computed on
// UTF-8 bytes would read 3x high on CJK and fire on every correct Japanese
// alignment.
inline int count_codepoints(const std::string& s) {
    int n = 0;
    for (unsigned char c : s)
        if ((c & 0xC0) != 0x80)
            n++;
    return n;
}

} // namespace detail

// Assess a word list against the audio it was aligned to.
// `audio_duration_sec <= 0` means "unknown" — the coverage and short-span
// signals are skipped, the rest still run.
inline Assessment assess(const std::vector<Word>& words, float audio_duration_sec, const Thresholds& th = {}) {
    Assessment a;
    a.word_count = (int)words.size();
    if (words.empty())
        return a;

    bool have_position = false;
    float first_t0 = 0.0f, last_t1 = 0.0f;

    for (const Word& w : words) {
        a.char_count += detail::count_codepoints(w.text);
        const bool zero_pos = (w.t0 == 0.0f && w.t1 == 0.0f);
        if (zero_pos) {
            a.zero_position_words++;
            continue; // a (0,0) word contributes no span and is not "degenerate"
        }
        if (w.t1 <= w.t0)
            a.degenerate_words++;
        if (!have_position) {
            first_t0 = w.t0;
            last_t1 = w.t1;
            have_position = true;
        } else {
            if (w.t0 < first_t0)
                first_t0 = w.t0;
            if (w.t1 > last_t1)
                last_t1 = w.t1;
        }
    }

    if (have_position && last_t1 > first_t0)
        a.span_sec = last_t1 - first_t0;
    if (audio_duration_sec > 0.0f)
        a.coverage = a.span_sec / audio_duration_sec;
    if (a.span_sec > 0.0f)
        a.chars_per_sec = (float)a.char_count / a.span_sec;

    // --- Signals. Each is independent; any one of them condemns the result. ---

    // (1) Silent-zero words. This one needs no minimum text: even a handful of
    // words all at (0,0) is align.cpp's out-of-vocab path, never real output.
    const float zero_ratio = (float)a.zero_position_words / (float)a.word_count;
    if (zero_ratio > th.max_zero_position_ratio) {
        a.collapsed = true;
        a.reasons.push_back("zero-position words " + std::to_string(a.zero_position_words) + "/" +
                            std::to_string(a.word_count) + " (" + detail::fmt2(zero_ratio) + " > " +
                            detail::fmt2(th.max_zero_position_ratio) + ")");
    }

    // (2) Empty spans — positioned but with no duration.
    const float degen_ratio = (float)a.degenerate_words / (float)a.word_count;
    if (degen_ratio > th.max_degenerate_ratio) {
        a.collapsed = true;
        a.reasons.push_back("zero-length spans " + std::to_string(a.degenerate_words) + "/" +
                            std::to_string(a.word_count) + " (" + detail::fmt2(degen_ratio) + " > " +
                            detail::fmt2(th.max_degenerate_ratio) + ")");
    }

    // The remaining signals all measure "text too big for its span", so they
    // need enough text to be meaningful.
    if (a.char_count >= th.min_chars_to_assess) {
        if (a.span_sec > 0.0f && a.chars_per_sec > th.max_chars_per_sec) {
            a.collapsed = true;
            a.reasons.push_back("chars/sec " + detail::fmt2(a.chars_per_sec) + " > " +
                                detail::fmt2(th.max_chars_per_sec) + " (physically impossible)");
        }
        if (audio_duration_sec > 0.0f) {
            if (a.coverage < th.min_coverage) {
                a.collapsed = true;
                a.reasons.push_back("coverage " + detail::fmt2(a.coverage) + " < " + detail::fmt2(th.min_coverage) +
                                    " of " + detail::fmt2(audio_duration_sec) + "s");
            }
            // Only meaningful when the audio is much longer than the span;
            // otherwise a short clip is correctly a short span.
            if (a.span_sec > 0.0f && a.span_sec < th.min_span_sec &&
                audio_duration_sec >= th.short_span_audio_ratio * a.span_sec) {
                a.collapsed = true;
                a.reasons.push_back("span " + detail::fmt2(a.span_sec) + "s < " + detail::fmt2(th.min_span_sec) +
                                    "s with " + std::to_string(a.char_count) + " chars");
            }
        }
    }

    return a;
}

// One-line summary for a log or a warning. Empty string when not collapsed.
inline std::string describe(const Assessment& a) {
    if (!a.collapsed)
        return "";
    std::string s = "alignment collapse suspected:";
    for (size_t i = 0; i < a.reasons.size(); i++)
        s += (i ? "; " : " ") + a.reasons[i];
    return s;
}

// OPT-IN recovery. Spread `words` evenly across [t_start, t_end] in proportion
// to each word's character count, which is a better prior than equal spacing
// (long words take longer to say). Callers must decide whether a synthesised
// timing is better than a collapsed one for their surface — for subtitles it
// usually is; for a word-level API contract it may not be. Never called
// automatically by `assess()`.
inline std::vector<Word> redistribute(const std::vector<Word>& words, float t_start, float t_end) {
    std::vector<Word> out = words;
    if (out.empty() || t_end <= t_start)
        return out;

    int total = 0;
    for (const Word& w : out)
        total += detail::count_codepoints(w.text) > 0 ? detail::count_codepoints(w.text) : 1;
    if (total <= 0)
        return out;

    const float span = t_end - t_start;
    float cursor = t_start;
    for (Word& w : out) {
        const int n = detail::count_codepoints(w.text) > 0 ? detail::count_codepoints(w.text) : 1;
        const float dur = span * (float)n / (float)total;
        w.t0 = cursor;
        w.t1 = cursor + dur;
        cursor += dur;
    }
    // Pin the last end exactly, so accumulated float error cannot overrun.
    out.back().t1 = t_end;
    return out;
}

} // namespace core_align_sentinel

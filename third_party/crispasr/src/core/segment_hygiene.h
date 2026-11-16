// core/segment_hygiene.h — post-decode segment cleanup (PLAN.md §W2, §W5, §W6).
//
// Three transforms that live downstream of the logits, in the zone the
// crispasr-diff harness cannot see: a per-stage cosine of 1.000000 says nothing
// about whether a runaway line got truncated or a duplicate segment got merged.
// They therefore get hermetic unit tests of their own
// (`tests/test-segment-hygiene.cpp`), which is the only place this behaviour is
// checkable at all.
//
// All three are OFF unless the caller supplies a config that enables them.
// Nothing here runs by default on a surface that has not opted in, because each
// one can delete or alter user-visible text, and a wrong deletion is far worse
// than a surviving artifact.
//
// Shapes from WhisperJAV (`repetition_cleaner` Layer 3, `cross_subtitle_
// processor`, `segment_filters`); thresholds and structure re-derived here.
// Weight-free, pure, no ggml.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace core_seg_hygiene {

// Minimal view of a segment. Callers adapt from `crispasr_segment` (CLI) or
// `crispasr_session_seg` (session ABI) — deliberately not coupled to either, so
// one implementation serves both surfaces (the multi-surface trap: a cleanup
// written against one struct silently misses the other).
struct Seg {
    std::string text;
    int64_t t0 = 0; // centiseconds, absolute
    int64_t t1 = 0;
    float avg_logprob = 0.0f;
    bool has_logprob = false;
};

// ---------------------------------------------------------------------------
// Shared UTF-8 helpers
// ---------------------------------------------------------------------------

namespace detail {

inline size_t count_codepoints(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s)
        if ((c & 0xC0) != 0x80)
            n++;
    return n;
}

// Byte offset of code point `idx`, or s.size() if beyond the end.
inline size_t byte_offset_of_codepoint(const std::string& s, size_t idx) {
    size_t n = 0, i = 0;
    while (i < s.size()) {
        if ((((unsigned char)s[i]) & 0xC0) != 0x80) {
            if (n == idx)
                return i;
            n++;
        }
        i++;
    }
    return s.size();
}

inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r'))
        a++;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r'))
        b--;
    return s.substr(a, b - a);
}

} // namespace detail

// ---------------------------------------------------------------------------
// §W2 — absolute line-length cutoff
// ---------------------------------------------------------------------------

struct LengthCapConfig {
    // 0 disables. A CJK subtitle line is normally 10-40 code points; a line
    // past this is essentially always a repetition hallucination that survived
    // the n-gram collapse. Latin lines run longer, so a caller that mixes
    // scripts should set this generously or leave it off.
    size_t max_codepoints = 0;
    // Never cut below this fraction of the cap when backing up to a sentence
    // boundary — otherwise an early "。" throws away most of a legitimate line.
    double min_keep_fraction = 0.75;
};

// Truncate an over-long line, preferring a sentence boundary. Returns the text
// unchanged when it is within the cap or the cap is disabled.
inline std::string cap_length(const std::string& text, const LengthCapConfig& cfg) {
    if (cfg.max_codepoints == 0)
        return text;
    const size_t n = detail::count_codepoints(text);
    if (n <= cfg.max_codepoints)
        return text;

    std::string cut = text.substr(0, detail::byte_offset_of_codepoint(text, cfg.max_codepoints));
    const size_t floor_cp = (size_t)((double)cfg.max_codepoints * cfg.min_keep_fraction);

    // Back up to the last sentence-ending mark, strongest first. Only accept it
    // if enough text survives.
    for (const char* sep : {"。", "．", ".", "！", "!", "？", "?", "、", ","}) {
        const size_t p = cut.rfind(sep);
        if (p == std::string::npos)
            continue;
        const std::string cand = cut.substr(0, p + std::string(sep).size());
        if (detail::count_codepoints(cand) >= floor_cp)
            return cand;
    }
    return cut;
}

// ---------------------------------------------------------------------------
// §W6 — segment filtering (low confidence / non-verbal)
// ---------------------------------------------------------------------------

struct FilterConfig {
    bool enabled = false;
    // Segments below this average log-probability are dropped. The whisper
    // decoder already applies `logprob_thold` during its temperature fallback;
    // this is the post-hoc pass over whatever survived, and it is OFF by
    // default so it cannot double-filter a backend that already gated.
    bool use_logprob = false;
    float logprob_threshold = -1.0f;
    // Short segments have noisier avg_logprob — a handful of tokens gives the
    // mean nothing to regress to — so a flat threshold under-filters them.
    // Tighten by this much for segments at or under `short_segment_sec`.
    float short_segment_margin = 0.0f;
    double short_segment_sec = 1.6;
    // Drop music/laughter/moan markers and bare vocalisations.
    bool drop_nonverbal = false;
};

enum class DropReason { None, LowLogprob, NonVerbal };

namespace detail {

inline std::string lower_ascii(const std::string& s) {
    std::string o = s;
    for (char& c : o)
        if (c >= 'A' && c <= 'Z')
            c = (char)(c - 'A' + 'a');
    return o;
}

// Opening/closing bracket forms. The CJK ones are NOT optional: a Japanese
// non-verbal marker is written （喘ぎ声）or【笑い】, never with ASCII parens, so
// an ASCII-only bracket test would fire on exactly zero real Japanese markers
// while claiming to support them.
inline const std::vector<std::string>& open_brackets() {
    static const std::vector<std::string> v = {"[", "(", "{", "<", "（", "【", "〔", "「", "『", "〈", "《", "［"};
    return v;
}
inline const std::vector<std::string>& close_brackets() {
    static const std::vector<std::string> v = {"]", ")", "}", ">", "）", "】", "〕", "」", "』", "〉", "》", "］"};
    return v;
}

inline bool starts_with_bracket(const std::string& s) {
    for (const auto& b : open_brackets())
        if (s.compare(0, b.size(), b) == 0)
            return true;
    return false;
}

// Strip one layer of surrounding brackets: "[Music]" / "(laughs)" / "（喘ぎ声）".
inline std::string strip_brackets(const std::string& s) {
    std::string t = trim(s);
    bool changed = true;
    while (changed && !t.empty()) {
        changed = false;
        for (const auto& b : open_brackets())
            if (t.compare(0, b.size(), b) == 0) {
                t.erase(0, b.size());
                changed = true;
                break;
            }
    }
    changed = true;
    while (changed && !t.empty()) {
        changed = false;
        for (const auto& b : close_brackets())
            if (t.size() >= b.size() && t.compare(t.size() - b.size(), b.size(), b) == 0) {
                t.erase(t.size() - b.size());
                changed = true;
                break;
            }
    }
    return trim(t);
}

inline bool is_note_only(const std::string& s) {
    // A line of nothing but musical notes / punctuation / spaces.
    static const char* kNotes[] = {"♪", "♫", "♬", "🎵", "🎶"};
    std::string t = trim(s);
    if (t.empty())
        return false;
    bool saw_note = false;
    size_t i = 0;
    while (i < t.size()) {
        bool matched = false;
        for (const char* nt : kNotes) {
            const size_t L = std::string(nt).size();
            if (t.compare(i, L, nt) == 0) {
                saw_note = true;
                matched = true;
                i += L;
                break;
            }
        }
        if (matched)
            continue;
        // cppcheck flags this as a possible out-of-bounds read because it does
        // not model `std::string::compare(pos, len, s)`: that compares the
        // substring [pos, pos + min(len, size() - pos)), so when fewer than L
        // bytes remain the compared substring is SHORTER than `nt` and the
        // result cannot be 0. A match therefore implies `i + L <= t.size()`,
        // `i += L` keeps `i <= t.size()`, and the loop condition re-checks
        // before we get here — so `i < t.size()` holds at this line. Suppressed
        // rather than guarded, because a guard here would be dead code.
        // cppcheck-suppress containerOutOfBounds
        const unsigned char c = (unsigned char)t[i];
        if (c == ' ' || c == '.' || c == ',' || c == '-' || c == '~')
            i++;
        else
            return false;
    }
    return saw_note;
}

} // namespace detail

// Keywords that mark a non-verbal event rather than speech. Matched only
// against text that was ENTIRELY inside brackets or is note-only — never
// against a substring of running speech, so "the applause died down" survives.
inline bool looks_nonverbal(const std::string& text) {
    if (detail::is_note_only(text))
        return true;

    const std::string raw = detail::trim(text);
    if (raw.empty())
        return false;

    // Only a fully-bracketed descriptor is eligible. This is the load-bearing
    // restriction: WhisperJAV substring-matches its keyword list against the
    // whole line, which deletes any sentence containing the word "music".
    if (!detail::starts_with_bracket(raw))
        return false;
    const std::string inner = detail::lower_ascii(detail::strip_brackets(raw));
    if (inner.empty())
        return false;

    static const char* kNonVerbal[] = {"music",  "applause", "laugh",   "laughter", "sfx",     "sound effect",
                                       "noise",  "silence",  "ambient", "moan",     "groan",   "sigh",
                                       "breath", "cough",    "sniff",   "grunt",    "chuckle", "gasp",
                                       "喘ぎ",   "うめき",   "笑い",    "拍手",     "音楽"};
    for (const char* kw : kNonVerbal)
        if (inner.find(kw) != std::string::npos)
            return true;
    return false;
}

// Decide whether a segment should be dropped. `duration_sec` may be <= 0 when
// unknown, in which case the short-segment margin is not applied.
inline DropReason should_drop(const Seg& s, double duration_sec, const FilterConfig& cfg) {
    if (!cfg.enabled)
        return DropReason::None;

    if (cfg.use_logprob && s.has_logprob) {
        float thr = cfg.logprob_threshold;
        if (cfg.short_segment_margin > 0.0f && duration_sec > 0.0 && duration_sec <= cfg.short_segment_sec)
            thr -= cfg.short_segment_margin;
        if (s.avg_logprob < thr)
            return DropReason::LowLogprob;
    }
    if (cfg.drop_nonverbal && looks_nonverbal(s.text))
        return DropReason::NonVerbal;
    return DropReason::None;
}

// ---------------------------------------------------------------------------
// §W5 — cross-segment duplicate merging
// ---------------------------------------------------------------------------

struct MergeConfig {
    bool enabled = false;
    // Similarity at or above this merges. 1.0 = byte-identical only.
    double similarity = 0.90;
    // Never merge across a gap wider than this (centiseconds): two identical
    // lines a minute apart are two real utterances, not one loop.
    int64_t max_gap_cs = 200;
    // Require at least this many consecutive similar segments before merging,
    // so an ordinary repeated "yes." pair survives.
    int min_run = 3;
};

namespace detail {

// Similarity over Unicode code points: |LCS| * 2 / (|a| + |b|).
// Code points, not bytes, so a one-kana difference in a short CJK line scores
// as one difference rather than three.
inline double similarity_ratio(const std::string& a, const std::string& b) {
    if (a == b)
        return 1.0;
    if (a.empty() || b.empty())
        return 0.0;

    std::vector<std::string> A, B;
    for (const std::string* s : {&a, &b}) {
        std::vector<std::string>& out = (s == &a) ? A : B;
        size_t i = 0;
        while (i < s->size()) {
            size_t len = 1;
            const unsigned char c = (unsigned char)(*s)[i];
            if ((c & 0xE0) == 0xC0)
                len = 2;
            else if ((c & 0xF0) == 0xE0)
                len = 3;
            else if ((c & 0xF8) == 0xF0)
                len = 4;
            if (len > s->size() - i)
                len = 1;
            out.push_back(s->substr(i, len));
            i += len;
        }
    }
    // Guard against a pathological line pair costing O(n*m) on huge inputs.
    constexpr size_t kMaxCp = 4096;
    if (A.size() > kMaxCp || B.size() > kMaxCp)
        return a == b ? 1.0 : 0.0;

    std::vector<int> prev(B.size() + 1, 0), cur(B.size() + 1, 0);
    for (size_t i = 1; i <= A.size(); i++) {
        for (size_t j = 1; j <= B.size(); j++)
            cur[j] = (A[i - 1] == B[j - 1]) ? prev[j - 1] + 1 : std::max(prev[j], cur[j - 1]);
        prev.swap(cur);
        std::fill(cur.begin(), cur.end(), 0);
    }
    const double lcs = (double)prev[B.size()];
    return 2.0 * lcs / (double)(A.size() + B.size());
}

} // namespace detail

inline double similarity(const std::string& a, const std::string& b) {
    return detail::similarity_ratio(detail::trim(a), detail::trim(b));
}

// Collapse runs of >= min_run consecutive near-identical segments separated by
// small gaps into one segment spanning the whole run. Keeps the FIRST text (the
// loop's later copies carry no new information) and the run's full time span.
inline std::vector<Seg> merge_repeats(const std::vector<Seg>& segs, const MergeConfig& cfg) {
    if (!cfg.enabled || (int)segs.size() < cfg.min_run)
        return segs;

    std::vector<Seg> out;
    size_t i = 0;
    while (i < segs.size()) {
        size_t j = i + 1;
        while (j < segs.size()) {
            const int64_t gap = segs[j].t0 - segs[j - 1].t1;
            if (gap > cfg.max_gap_cs)
                break;
            // Compare against the run's FIRST text, not its predecessor, so a
            // slowly-drifting chain cannot merge unboundedly.
            if (similarity(segs[i].text, segs[j].text) < cfg.similarity)
                break;
            j++;
        }
        const size_t run = j - i;
        if ((int)run >= cfg.min_run) {
            Seg merged = segs[i];
            merged.t1 = segs[j - 1].t1;
            out.push_back(std::move(merged));
        } else {
            for (size_t k = i; k < j; k++)
                out.push_back(segs[k]);
        }
        i = j;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Env-driven configuration + the single apply
// ---------------------------------------------------------------------------

struct Config {
    LengthCapConfig cap;
    FilterConfig filter;
    MergeConfig merge;
};

namespace detail {

inline const char* env_or_null(const char* k) {
    const char* v = std::getenv(k);
    return (v && v[0]) ? v : nullptr;
}

inline bool env_truthy(const char* k) {
    const char* v = env_or_null(k);
    return v && v[0] != '0';
}

inline double env_double(const char* k, double dflt) {
    const char* v = env_or_null(k);
    if (!v)
        return dflt;
    char* end = nullptr;
    const double d = std::strtod(v, &end);
    return (end && end != v) ? d : dflt;
}

} // namespace detail

// Read the opt-in knobs. Everything stays OFF unless explicitly enabled — each
// of these can delete or alter user-visible text, and a wrong deletion is worse
// than a surviving artifact, so none of them may switch on by surprise.
//
//   CRISPASR_SEG_MAX_CHARS=N     §W2 truncate lines over N code points
//   CRISPASR_SEG_DROP_NONVERBAL=1 §W6 drop [Music] / (laughs) / ♪ lines
//   CRISPASR_SEG_LOGPROB_THOLD=F §W6 drop segments below F average logprob
//   CRISPASR_SEG_LOGPROB_MARGIN=F §W6 tighten that by F for short segments
//   CRISPASR_SEG_MERGE_REPEATS=1 §W5 collapse runs of near-identical segments
//   CRISPASR_SEG_MERGE_SIMILARITY=F / _MERGE_GAP_CS=N / _MERGE_MIN_RUN=N
inline Config config_from_env() {
    Config c;

    const double max_chars = detail::env_double("CRISPASR_SEG_MAX_CHARS", 0.0);
    if (max_chars > 0.0)
        c.cap.max_codepoints = (size_t)max_chars;

    const bool nonverbal = detail::env_truthy("CRISPASR_SEG_DROP_NONVERBAL");
    const char* lp = detail::env_or_null("CRISPASR_SEG_LOGPROB_THOLD");
    if (nonverbal || lp) {
        c.filter.enabled = true;
        c.filter.drop_nonverbal = nonverbal;
        if (lp) {
            c.filter.use_logprob = true;
            c.filter.logprob_threshold = (float)detail::env_double("CRISPASR_SEG_LOGPROB_THOLD", -1.0);
            c.filter.short_segment_margin = (float)detail::env_double("CRISPASR_SEG_LOGPROB_MARGIN", 0.0);
        }
    }

    if (detail::env_truthy("CRISPASR_SEG_MERGE_REPEATS")) {
        c.merge.enabled = true;
        c.merge.similarity = detail::env_double("CRISPASR_SEG_MERGE_SIMILARITY", c.merge.similarity);
        c.merge.max_gap_cs = (int64_t)detail::env_double("CRISPASR_SEG_MERGE_GAP_CS", (double)c.merge.max_gap_cs);
        c.merge.min_run = (int)detail::env_double("CRISPASR_SEG_MERGE_MIN_RUN", (double)c.merge.min_run);
    }
    return c;
}

inline bool any_enabled(const Config& c) {
    return c.cap.max_codepoints > 0 || c.filter.enabled || c.merge.enabled;
}

// Apply all three stages in the only order that makes sense:
//   1. cap  — shorten runaway lines FIRST, so a truncated line can then match
//             its neighbours for merging. Capping after merging would leave the
//             merge comparing two different 400-char hallucinations that share
//             no prefix and so never merge.
//   2. filter — drop what should not be there, before spending merge work on it.
//   3. merge  — collapse what is left.
//
// `dropped_out` (optional) receives the number of segments removed by step 2,
// so a caller can report it rather than silently losing lines.
inline std::vector<Seg> apply_all(const std::vector<Seg>& in, const Config& cfg, int* dropped_out = nullptr) {
    if (dropped_out)
        *dropped_out = 0;
    if (!any_enabled(cfg))
        return in;

    std::vector<Seg> work;
    work.reserve(in.size());
    for (const Seg& s : in) {
        Seg t = s;
        if (cfg.cap.max_codepoints > 0)
            t.text = cap_length(t.text, cfg.cap);
        if (cfg.filter.enabled) {
            const double dur = (t.t1 > t.t0) ? (double)(t.t1 - t.t0) / 100.0 : -1.0;
            if (should_drop(t, dur, cfg.filter) != DropReason::None) {
                if (dropped_out)
                    (*dropped_out)++;
                continue;
            }
        }
        work.push_back(std::move(t));
    }
    return merge_repeats(work, cfg.merge);
}

} // namespace core_seg_hygiene

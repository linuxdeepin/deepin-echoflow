// src/core/asr_parity.h — canonical segment-parity predicate (improvements
// Phase 0).
//
// Different CrispASR surfaces (CLI adapter, HTTP server, session C-ABI) should
// produce the SAME transcription for the same audio. This header defines the
// one canonical rule for "do two segment lists match", used by the surface-
// parity harness and pinned by a unit test. It is the acceptance gate for the
// dispatch unification (Phase 1): the old (inline) and new (adapter) session
// paths must be `segments_equal` before the gate flips.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace core_parity {

struct SegView {
    std::string text;
    int64_t t0 = 0; // centiseconds
    int64_t t1 = 0;
};

// Whitespace-normalized text: trim ends, collapse each internal run of
// whitespace to a single space. Case- and punctuation-sensitive on purpose —
// two surfaces running the same decode must produce the same tokens; only
// incidental spacing (e.g. a trailing newline in one serializer) is ignored.
inline std::string norm_text(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_ws = true; // leading-trim: treat start as if preceded by ws
    for (char c : s) {
        const bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
        if (ws) {
            in_ws = true;
        } else {
            if (in_ws && !out.empty())
                out.push_back(' ');
            out.push_back(c);
            in_ws = false;
        }
    }
    return out;
}

// Two segment lists match when they have the same length, each pair's
// normalized text is identical, and each pair's t0/t1 differ by at most
// `tol_cs` centiseconds (0 = exact). tol_cs absorbs only serializer rounding,
// not real timing drift.
inline bool segments_equal(const std::vector<SegView>& a, const std::vector<SegView>& b, int64_t tol_cs = 0) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (norm_text(a[i].text) != norm_text(b[i].text))
            return false;
        if (std::llabs((long long)(a[i].t0 - b[i].t0)) > tol_cs)
            return false;
        if (std::llabs((long long)(a[i].t1 - b[i].t1)) > tol_cs)
            return false;
    }
    return true;
}

} // namespace core_parity

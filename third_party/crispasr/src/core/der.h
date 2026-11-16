// src/core/der.h — Diarization Error Rate (#324).
//
// DER = (missed speech + false alarm + speaker confusion) / total reference
// speech, after mapping hypothesis speakers onto reference speakers so as to
// maximise agreement. This is the acceptance gate for diarization: unlike ASR
// there is no transcript to compare, and unlike a ported model there is no
// per-stage reference to diff against (the clustering is weight-free and
// sklearn parity is unachievable — see core/spectral_diarize.h).
//
// Conventions follow NIST / dscore:
//   * a COLLAR is excluded around every reference boundary, because human
//     boundary annotation is not accurate to the millisecond and scoring
//     those edges measures the annotator, not the system;
//   * speaker labels are arbitrary, so the mapping is chosen to maximise
//     total overlap;
//   * the denominator is total REFERENCE speech, so DER can exceed 1 when a
//     system hallucinates a lot of speech.
//
// Overlapping speech is not modelled: each instant has at most one reference
// and one hypothesis speaker. Our pipeline cannot emit overlap, so scoring it
// would only add a constant.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <set>
#include <vector>

namespace core_der {

struct Turn {
    double start = 0.0;
    double end = 0.0;
    int speaker = 0;
};

struct Score {
    double missed = 0.0;      // reference speech with no hypothesis
    double false_alarm = 0.0; // hypothesis speech with no reference
    double confusion = 0.0;   // both present, mapped speakers disagree
    double total_ref = 0.0;   // scored reference speech (collar removed)
    double der() const { return total_ref > 0.0 ? (missed + false_alarm + confusion) / total_ref : 0.0; }
    int n_ref_speakers = 0;
    int n_hyp_speakers = 0;
};

namespace detail {

// Half-open intervals that survive collar removal around reference boundaries.
inline std::vector<std::pair<double, double>> collar_mask(const std::vector<Turn>& ref, double collar, double t_end) {
    if (collar <= 0.0)
        return {{0.0, t_end}};
    // Punch a hole of +/- collar around every reference boundary.
    std::vector<std::pair<double, double>> holes;
    for (const Turn& t : ref) {
        holes.push_back({t.start - collar, t.start + collar});
        holes.push_back({t.end - collar, t.end + collar});
    }
    std::sort(holes.begin(), holes.end());
    std::vector<std::pair<double, double>> merged;
    for (auto& h : holes) {
        if (!merged.empty() && h.first <= merged.back().second)
            merged.back().second = std::max(merged.back().second, h.second);
        else
            merged.push_back(h);
    }
    std::vector<std::pair<double, double>> keep;
    double cur = 0.0;
    for (auto& h : merged) {
        if (h.first > cur)
            keep.push_back({cur, std::min(h.first, t_end)});
        cur = std::max(cur, h.second);
    }
    if (cur < t_end)
        keep.push_back({cur, t_end});
    return keep;
}

inline double overlap(double a0, double a1, double b0, double b1) {
    return std::max(0.0, std::min(a1, b1) - std::max(a0, b0));
}

} // namespace detail

// Total duration of `hyp` speaker h overlapping `ref` speaker r, per pair.
inline std::map<std::pair<int, int>, double> overlap_matrix(const std::vector<Turn>& ref, const std::vector<Turn>& hyp,
                                                            const std::vector<std::pair<double, double>>& scored) {
    std::map<std::pair<int, int>, double> m;
    for (const Turn& r : ref)
        for (const Turn& h : hyp) {
            double acc = 0.0;
            for (const auto& s : scored) {
                const double lo = std::max({r.start, h.start, s.first});
                const double hi = std::min({r.end, h.end, s.second});
                if (hi > lo)
                    acc += hi - lo;
            }
            if (acc > 0.0)
                m[{r.speaker, h.speaker}] += acc;
        }
    return m;
}

// Best 1:1 hypothesis->reference speaker mapping by total overlap.
//
// Exact by enumeration for up to 8 hypothesis speakers (<= 40320 orders);
// above that it falls back to a greedy pass over the largest overlaps, which
// can be sub-optimal. Real diarization fixtures sit well under the exact
// bound, and the fallback is flagged via `out_exact` so a caller never
// mistakes an approximation for a guarantee.
inline std::map<int, int> best_mapping(const std::map<std::pair<int, int>, double>& ov, const std::set<int>& ref_ids,
                                       const std::set<int>& hyp_ids, bool* out_exact = nullptr) {
    const std::vector<int> rs(ref_ids.begin(), ref_ids.end());
    const std::vector<int> hs(hyp_ids.begin(), hyp_ids.end());
    auto get = [&](int r, int h) {
        auto it = ov.find({r, h});
        return it == ov.end() ? 0.0 : it->second;
    };

    if (hs.size() <= 8 && rs.size() <= 8) {
        if (out_exact)
            *out_exact = true;
        // Enumerate assignments of hypothesis speakers to reference slots
        // (or to "unmatched" when there are more hyp than ref speakers).
        std::vector<int> slots(hs.size());
        std::iota(slots.begin(), slots.end(), 0);
        std::vector<int> pad(rs.size());
        std::iota(pad.begin(), pad.end(), 0);
        while (pad.size() < hs.size())
            pad.push_back(-1); // unmatched slot
        std::sort(pad.begin(), pad.end());
        double best = -1.0;
        std::map<int, int> best_map;
        do {
            double total = 0.0;
            std::map<int, int> m;
            for (size_t i = 0; i < hs.size(); i++) {
                if (pad[i] < 0)
                    continue;
                total += get(rs[(size_t)pad[i]], hs[i]);
                m[hs[i]] = rs[(size_t)pad[i]];
            }
            if (total > best) {
                best = total;
                best_map = m;
            }
        } while (std::next_permutation(pad.begin(), pad.end()));
        return best_map;
    }

    if (out_exact)
        *out_exact = false;
    std::vector<std::pair<double, std::pair<int, int>>> pairs;
    for (const auto& kv : ov)
        pairs.push_back({kv.second, kv.first});
    std::sort(pairs.rbegin(), pairs.rend());
    std::set<int> used_r, used_h;
    std::map<int, int> m;
    for (const auto& p : pairs) {
        const int r = p.second.first, h = p.second.second;
        if (used_r.count(r) || used_h.count(h))
            continue;
        m[h] = r;
        used_r.insert(r);
        used_h.insert(h);
    }
    return m;
}

// Score `hyp` against `ref`. `collar` is the half-width in seconds excluded
// around each reference boundary (0.25 is the common convention).
inline Score score(const std::vector<Turn>& ref, const std::vector<Turn>& hyp, double collar = 0.25) {
    Score s;
    if (ref.empty())
        return s;

    double t_end = 0.0;
    for (const Turn& t : ref)
        t_end = std::max(t_end, t.end);
    for (const Turn& t : hyp)
        t_end = std::max(t_end, t.end);

    const auto scored = detail::collar_mask(ref, collar, t_end);

    std::set<int> ref_ids, hyp_ids;
    for (const Turn& t : ref)
        ref_ids.insert(t.speaker);
    for (const Turn& t : hyp)
        hyp_ids.insert(t.speaker);
    s.n_ref_speakers = (int)ref_ids.size();
    s.n_hyp_speakers = (int)hyp_ids.size();

    const auto ov = overlap_matrix(ref, hyp, scored);
    const auto mapping = best_mapping(ov, ref_ids, hyp_ids);

    // Walk a common timeline of all boundaries intersected with the scored
    // regions, and attribute each atomic slice.
    std::vector<double> pts;
    for (const auto& sc : scored) {
        pts.push_back(sc.first);
        pts.push_back(sc.second);
    }
    for (const Turn& t : ref) {
        pts.push_back(t.start);
        pts.push_back(t.end);
    }
    for (const Turn& t : hyp) {
        pts.push_back(t.start);
        pts.push_back(t.end);
    }
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

    for (size_t i = 0; i + 1 < pts.size(); i++) {
        const double a = pts[i], b = pts[i + 1];
        if (b <= a)
            continue;
        bool in_scored = false;
        for (const auto& sc : scored)
            if (a >= sc.first && b <= sc.second) {
                in_scored = true;
                break;
            }
        if (!in_scored)
            continue;
        const double mid = (a + b) * 0.5;
        const double dur = b - a;

        int r_spk = -1, h_spk = -1;
        for (const Turn& t : ref)
            if (mid >= t.start && mid < t.end) {
                r_spk = t.speaker;
                break;
            }
        for (const Turn& t : hyp)
            if (mid >= t.start && mid < t.end) {
                h_spk = t.speaker;
                break;
            }

        if (r_spk >= 0)
            s.total_ref += dur;
        if (r_spk >= 0 && h_spk < 0)
            s.missed += dur;
        else if (r_spk < 0 && h_spk >= 0)
            s.false_alarm += dur;
        else if (r_spk >= 0 && h_spk >= 0) {
            auto it = mapping.find(h_spk);
            if (it == mapping.end() || it->second != r_spk)
                s.confusion += dur;
        }
    }
    return s;
}

} // namespace core_der

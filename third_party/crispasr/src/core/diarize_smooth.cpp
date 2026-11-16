// src/core/diarize_smooth.cpp — see diarize_smooth.h.

#include "diarize_smooth.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace core_diarize_smooth {

namespace {
constexpr double kTiny = 1e-12;
} // namespace

int majority_label(const std::vector<int>& labels) {
    if (labels.empty())
        return -1;
    std::map<int, int> counts;
    for (int v : labels)
        counts[v]++;
    int best = -1, best_count = 0, n_at_best = 0;
    for (const auto& kv : counts) {
        if (kv.second > best_count) {
            best = kv.first;
            best_count = kv.second;
            n_at_best = 1;
        } else if (kv.second == best_count) {
            n_at_best++;
        }
    }
    // A tie is not a majority. Returning a winner here would let the filter
    // manufacture a label exactly at a speaker boundary, which is the one
    // place the sequence is genuinely ambiguous.
    return n_at_best > 1 ? -1 : best;
}

std::vector<int> smooth_window_labels(const std::vector<int>& labels) {
    if (labels.size() < 3)
        return labels;
    std::vector<int> out;
    out.reserve(labels.size());
    for (size_t i = 0; i < labels.size(); i++) {
        const size_t lo = i == 0 ? 0 : i - 1;
        const size_t hi = std::min(labels.size(), i + 2);
        std::vector<int> window(labels.begin() + (long)lo, labels.begin() + (long)hi);
        const int maj = majority_label(window);
        out.push_back(maj < 0 ? labels[i] : maj);
    }
    return out;
}

std::vector<float> speaker_centroids(const float* embeddings, int n, int d, const std::vector<int>& labels,
                                     std::vector<int>* out_label_values) {
    if (out_label_values)
        out_label_values->clear();
    if (n <= 0 || d <= 0 || (int)labels.size() != n)
        return {};

    std::set<int> uniq(labels.begin(), labels.end());
    std::vector<float> out;
    out.reserve(uniq.size() * (size_t)d);

    for (int label : uniq) {
        std::vector<double> acc((size_t)d, 0.0);
        int members = 0;
        for (int i = 0; i < n; i++) {
            if (labels[(size_t)i] != label)
                continue;
            // Mean of the NORMALISED members, not of the raw vectors: raw
            // magnitude varies with window energy and would let a loud window
            // dominate its speaker's centroid.
            double nrm = 0.0;
            for (int j = 0; j < d; j++)
                nrm += (double)embeddings[(size_t)i * d + j] * embeddings[(size_t)i * d + j];
            nrm = std::sqrt(nrm);
            if (nrm <= kTiny)
                continue;
            for (int j = 0; j < d; j++)
                acc[(size_t)j] += embeddings[(size_t)i * d + j] / nrm;
            members++;
        }
        if (members == 0)
            continue;
        double nrm = 0.0;
        for (int j = 0; j < d; j++) {
            acc[(size_t)j] /= members;
            nrm += acc[(size_t)j] * acc[(size_t)j];
        }
        nrm = std::sqrt(nrm);
        if (nrm <= kTiny)
            continue; // members cancelled out; no usable direction
        for (int j = 0; j < d; j++)
            out.push_back((float)(acc[(size_t)j] / nrm));
        if (out_label_values)
            out_label_values->push_back(label);
    }
    return out;
}

std::vector<int> viterbi_smooth(const float* scores, int n_frames, int n_labels, float switch_penalty) {
    if (n_frames <= 0)
        return {};
    if (n_labels <= 1)
        return std::vector<int>((size_t)n_frames, 0);

    std::vector<double> dp((size_t)n_frames * n_labels, 0.0);
    std::vector<int> back((size_t)n_frames * n_labels, 0);
    for (int l = 0; l < n_labels; l++)
        dp[(size_t)l] = scores[(size_t)l];

    for (int t = 1; t < n_frames; t++) {
        for (int cur = 0; cur < n_labels; cur++) {
            double best = -std::numeric_limits<double>::infinity();
            int arg = 0;
            for (int prev = 0; prev < n_labels; prev++) {
                const double v = dp[(size_t)(t - 1) * n_labels + prev] + (prev == cur ? 0.0 : -(double)switch_penalty);
                if (v > best) {
                    best = v;
                    arg = prev;
                }
            }
            dp[(size_t)t * n_labels + cur] = scores[(size_t)t * n_labels + cur] + best;
            back[(size_t)t * n_labels + cur] = arg;
        }
    }

    int cur = 0;
    double best = -std::numeric_limits<double>::infinity();
    for (int l = 0; l < n_labels; l++)
        if (dp[(size_t)(n_frames - 1) * n_labels + l] > best) {
            best = dp[(size_t)(n_frames - 1) * n_labels + l];
            cur = l;
        }
    std::vector<int> path((size_t)n_frames, 0);
    path[(size_t)(n_frames - 1)] = cur;
    for (int t = n_frames - 1; t > 0; t--) {
        cur = back[(size_t)t * n_labels + cur];
        path[(size_t)(t - 1)] = cur;
    }
    return path;
}

std::vector<int> smooth_segment_temporal(const std::vector<int>& labels, const float* embeddings, int n, int d,
                                         const std::vector<int>& label_values, const float* centroids) {
    const int n_labels = (int)label_values.size();
    if (labels.size() < 3 || n_labels <= 1 || (int)labels.size() != n)
        return labels;

    std::map<int, int> label_to_idx;
    for (int i = 0; i < n_labels; i++)
        label_to_idx[label_values[(size_t)i]] = i;

    std::vector<float> scores((size_t)n * n_labels, 0.0f);
    for (int i = 0; i < n; i++) {
        double nrm = 0.0;
        for (int j = 0; j < d; j++)
            nrm += (double)embeddings[(size_t)i * d + j] * embeddings[(size_t)i * d + j];
        nrm = std::sqrt(nrm);
        for (int l = 0; l < n_labels; l++) {
            double dot = 0.0;
            if (nrm > kTiny)
                for (int j = 0; j < d; j++)
                    dot += (double)embeddings[(size_t)i * d + j] / nrm * centroids[(size_t)l * d + j];
            scores[(size_t)i * n_labels + l] = (float)dot;
        }
        auto it = label_to_idx.find(labels[(size_t)i]);
        if (it != label_to_idx.end())
            scores[(size_t)i * n_labels + it->second] += kOriginalLabelAnchor;
    }

    std::vector<int> path = viterbi_smooth(scores.data(), n, n_labels);
    std::vector<int> out((size_t)n, 0);
    for (int i = 0; i < n; i++)
        out[(size_t)i] = label_values[(size_t)path[(size_t)i]];
    return out;
}

namespace {
// [start, end) index ranges of constant label, with each run's wall duration.
struct Run {
    size_t start, end;
    int label;
    float duration;
};

std::vector<Run> find_runs(const std::vector<int>& labels, const std::vector<Window>& windows) {
    std::vector<Run> runs;
    size_t start = 0;
    for (size_t i = 1; i <= labels.size(); i++) {
        if (i == labels.size() || labels[i] != labels[start]) {
            runs.push_back({start, i, labels[start], windows[i - 1].end - windows[start].start});
            start = i;
        }
    }
    return runs;
}
} // namespace

std::vector<int> collapse_short_islands(const std::vector<int>& labels, const std::vector<Window>& windows,
                                        float max_seconds) {
    if (labels.size() < 3 || labels.size() != windows.size())
        return labels;
    const std::vector<Run> runs = find_runs(labels, windows);
    if (runs.size() < 3)
        return labels;

    std::vector<int> out = labels;
    for (size_t r = 1; r + 1 < runs.size(); r++) {
        const Run& cur = runs[r];
        const int prev = runs[r - 1].label;
        const int next = runs[r + 1].label;
        // Only an A-B-A sandwich collapses. A-B-C is a genuine speaker
        // change and must survive.
        if (prev == next && cur.label != prev && cur.duration <= max_seconds)
            std::fill(out.begin() + (long)cur.start, out.begin() + (long)cur.end, prev);
    }
    return out;
}

std::vector<int> restore_sustained_runs(const std::vector<int>& original, const std::vector<int>& smoothed,
                                        const std::vector<Window>& windows, float min_seconds) {
    if (original.size() != smoothed.size() || original.size() != windows.size())
        return smoothed;
    std::vector<int> out = smoothed;
    for (const Run& run : find_runs(original, windows))
        if (run.duration > min_seconds)
            std::copy(original.begin() + (long)run.start, original.begin() + (long)run.end,
                      out.begin() + (long)run.start);
    return out;
}

} // namespace core_diarize_smooth

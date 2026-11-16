#pragma once
// Silero timestamp postprocessing: Copyright (c) 2020-present Silero Team.
// MIT License; see the repository LICENSE for the permission notice.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace core_index_echo {
struct Window {
    double start, end;
    bool hard_cut;
};

// Python round(seconds, 1) rounds the original binary value, including ties.
// Multiplying by ten first can lose that distinction (e.g. round(0.35, 1)).
inline double round_seconds(double seconds, int decimals) {
    char value[64];
    std::snprintf(value, sizeof(value), "%.*f", decimals, seconds);
    return std::strtod(value, nullptr);
}
inline double speech_seconds(int samples) {
    return round_seconds(samples / 16000.0, 1);
}
// Released ffmpeg seek/duration arguments use f"{seconds:.3f}". Round the
// original double before converting to samples; llround(seconds*1000) can
// differ at binary ties (1.0005 seconds formats to 1.000, not 1.001).
inline int64_t window_samples(double seconds) {
    return static_cast<int64_t>(std::llround(round_seconds(seconds, 3) * 16000));
}

// Silero v6.2 get_speech_timestamps with the released infer.py parameters:
// 16 kHz / 512 samples, threshold .5, minimum speech 250 ms, silence 300 ms,
// padding 30 ms, unlimited speech length, return_seconds=True (one decimal).
// Consume the shared native classifier's probabilities directly; its public
// segment accessor has already discarded the sample precision we need.
inline std::vector<std::pair<double, double>> speech_spans(const float* probs, int count, int samples) {
    if (!probs || count <= 0 || samples <= 0)
        return {};
    std::vector<std::pair<int, int>> spans;
    bool active = false;
    int start = 0, temporary_end = 0;
    count = std::min(count, (int)(((int64_t)samples + 511) / 512));
    for (int i = 0; i < count; ++i) {
        const int position = i * 512;
        if (probs[i] >= 0.5 && temporary_end)
            temporary_end = 0;
        if (probs[i] >= 0.5 && !active) {
            active = true;
            start = position;
            continue;
        }
        if (probs[i] < 0.35 && active) {
            if (!temporary_end)
                temporary_end = position;
            if (position - temporary_end < 4800)
                continue;
            if (temporary_end - start > 4000)
                spans.emplace_back(start, temporary_end);
            active = false;
            temporary_end = 0;
        }
    }
    if (active && samples - start > 4000)
        spans.emplace_back(start, samples);
    for (size_t i = 0; i < spans.size(); ++i) {
        if (i == 0)
            spans[i].first = std::max(0, spans[i].first - 480);
        if (i + 1 < spans.size()) {
            const int silence = spans[i + 1].first - spans[i].second;
            if (silence < 960) {
                spans[i].second += silence / 2;
                spans[i + 1].first = std::max(0, spans[i + 1].first - silence / 2);
            } else {
                spans[i].second = (int)std::min<int64_t>(samples, (int64_t)spans[i].second + 480);
                spans[i + 1].first = std::max(0, spans[i + 1].first - 480);
            }
        } else
            spans[i].second = (int)std::min<int64_t>(samples, (int64_t)spans[i].second + 480);
    }
    std::vector<std::pair<double, double>> result;
    for (const auto& span : spans)
        result.emplace_back(std::max(0.0, speech_seconds(span.first)),
                            std::min(samples / 16000.0, speech_seconds(span.second)));
    return result;
}

// Released infer.py vad_windows: split speech, greedily merge with lead/trail,
// then fold a short last window into its predecessor when it fits.
inline std::vector<Window> windows(const std::vector<std::pair<double, double>>& speech, double total,
                                   double maximum = 60) {
    if (maximum <= 0 || total <= 0)
        return {};
    std::vector<Window> flat, result;
    for (auto segment : speech) {
        double start = segment.first, end = segment.second;
        while (end - start > maximum) {
            flat.push_back({start, start + maximum, true});
            start += maximum;
        }
        flat.push_back({start, end, false});
    }
    for (std::size_t i = 0; i < flat.size();) {
        double start = std::max(flat[i].start - 0.3, result.empty() ? 0.0 : result.back().end);
        std::size_t j = i;
        bool hard = flat[i].hard_cut;
        while (j + 1 < flat.size() && flat[j + 1].end + 0.5 - start <= maximum && !flat[j].hard_cut)
            ++j;
        double end = std::min(flat[j].end + 0.5, j + 1 < flat.size() ? flat[j + 1].start : total);
        result.push_back({start, end, hard});
        i = j + 1;
    }
    if (result.size() >= 2 && result.back().end - result.back().start < 5 &&
        result.back().end - result[result.size() - 2].start <= maximum) {
        result[result.size() - 2].end = result.back().end;
        result.pop_back();
    }
    return result;
}
} // namespace core_index_echo

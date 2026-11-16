// src/core/foxnose_pipeline.cpp — see foxnose_pipeline.h.

#include "foxnose_pipeline.h"

#include "parallel_for.h"

#include "diarize_smooth.h"
#include "spectral_diarize.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>

namespace core_foxnose {

int windows_per_span() {
    static const int v = [] {
        if (const char* e = std::getenv("CRISPASR_DIARIZE_SPAN_WINDOWS")) {
            const int n = std::atoi(e);
            if (n > 0)
                return n;
        }
        return kWindowsPerSpan;
    }();
    return v;
}

std::vector<Speech> window_speech(const Speech& seg) {
    const double dur = seg.end - seg.start;
    if (dur < kMinSegmentSeconds)
        return {};
    if (dur <= kEmbeddingWindowSeconds * kSingleWindowFactor)
        return {seg};

    std::vector<Speech> out;
    double start = seg.start;
    // The guard is `start + kMinSegmentSeconds < end`, not `start < end`: a
    // trailing sliver shorter than the minimum would be embedded from almost
    // no audio and contribute a meaningless vector to the clustering.
    while (start + kMinSegmentSeconds < seg.end) {
        out.push_back({start, std::min(start + kEmbeddingWindowSeconds, seg.end)});
        start += kEmbeddingStepSeconds;
    }
    return out;
}

std::vector<Speech> window_boundaries(const Speech& seg, const std::vector<Speech>& windows) {
    if (windows.empty())
        return {};
    std::vector<double> centres;
    centres.reserve(windows.size());
    for (const auto& w : windows)
        centres.push_back((w.start + w.end) * 0.5);

    std::vector<double> bounds;
    bounds.reserve(centres.size() + 1);
    bounds.push_back(seg.start);
    for (size_t i = 0; i + 1 < centres.size(); i++) {
        const double mid = (centres[i] + centres[i + 1]) * 0.5;
        bounds.push_back(std::min(seg.end, std::max(seg.start, mid)));
    }
    bounds.push_back(seg.end);

    std::vector<Speech> out;
    out.reserve(bounds.size() - 1);
    for (size_t i = 0; i + 1 < bounds.size(); i++)
        out.push_back({bounds[i], bounds[i + 1]});
    return out;
}

Result diarize(const float* pcm, int n_samples, int sample_rate, const std::vector<Speech>& speech, EmbedFn embed,
               void* userdata, int embed_dim, const Params& params, EmbedWindowsFn embed_windows,
               EmbedBatchFn embed_batch) {
    Result res;
    if (!pcm || n_samples <= 0 || !embed || embed_dim <= 0 || speech.empty())
        return res;

    // ---- windows + embeddings ----
    std::vector<float> emb;        // (n_windows, embed_dim)
    std::vector<Speech> win_times; // window time spans
    std::vector<int> win_parent;   // index into `speech`
    std::vector<float> scratch((size_t)embed_dim);

    // Enumerate the windows first, then embed them concurrently. Each window is
    // an independent forward pass, so this is the axis with the most headroom:
    // measured interleaved on an 85 s file, threads x workers gave 4x1 10.27 s,
    // 8x1 10.80 s, 2x4 8.67 s, 1x8 8.09 s — i.e. spending cores on whole
    // windows beats spending them inside one graph.
    //
    // (An earlier version of this comment claimed intra-graph threads made it
    // 5x SLOWER. That came from a sequential -t 1/4/8 loop on a box whose load
    // was ramping, so it read the load as the thread count. Threads do help —
    // 20.5 s to 10.3 s from 1 to 4 — parallel windows just help more.)
    //
    // Order of results is kept identical to the serial version — the
    // clusterer's output depends on row order.
    struct Win {
        Speech w;
        int parent;
        long a, b;
    };
    std::vector<Win> wins;
    for (size_t p = 0; p < speech.size(); p++) {
        for (const Speech& w : window_speech(speech[p])) {
            const long a = std::lround(w.start * sample_rate);
            const long b = std::lround(w.end * sample_rate);
            if (a < 0 || b > n_samples || b <= a)
                continue;
            wins.push_back({w, (int)p, a, b});
        }
    }

    const int n_win = (int)wins.size();
    std::vector<float> all((size_t)n_win * embed_dim);
    std::vector<char> ok((size_t)n_win, 0);

    // Unit of work: a SPAN of consecutive windows from one speech region when
    // the caller offers a span embedder (they then share one trunk pass),
    // otherwise a single window. Spans are cut at a fixed size and never
    // straddle a region, so the split — and therefore the embeddings — do not
    // depend on how many workers happen to be available.
    struct Span {
        int first, last; // inclusive range into `wins`
    };
    std::vector<Span> spans;
    if (embed_windows) {
        for (int i = 0; i < n_win;) {
            int j = i;
            while (j + 1 < n_win && wins[(size_t)(j + 1)].parent == wins[(size_t)i].parent &&
                   (j + 1 - i) < windows_per_span())
                j++;
            spans.push_back({i, j});
            i = j + 1;
        }
    } else if (embed_batch) {
        // Batches are independent windows, so unlike spans they may straddle
        // parent regions — the chunking is pure scheduling. Fixed stride over
        // the window index keeps the calls (and any failure fallout) identical
        // for every worker count.
        for (int i = 0; i < n_win; i += kWindowsPerBatch)
            spans.push_back({i, std::min(i + kWindowsPerBatch, n_win) - 1});
    } else {
        spans.reserve((size_t)n_win);
        for (int i = 0; i < n_win; i++)
            spans.push_back({i, i});
    }

    const int n_spans = (int)spans.size();
    const int n_workers = std::max(1, std::min(params.n_workers, n_spans));

    auto do_span = [&](int si, int worker) {
        const Span& sp = spans[(size_t)si];
        if (!embed_windows && embed_batch) {
            const int cnt = sp.last - sp.first + 1;
            std::vector<int64_t> offs((size_t)cnt);
            std::vector<int> lens((size_t)cnt);
            for (int k = 0; k < cnt; k++) {
                offs[(size_t)k] = (int64_t)wins[(size_t)(sp.first + k)].a;
                lens[(size_t)k] = (int)(wins[(size_t)(sp.first + k)].b - wins[(size_t)(sp.first + k)].a);
            }
            if (embed_batch(userdata, worker, pcm, (int64_t)n_samples, offs.data(), lens.data(), cnt,
                            &all[(size_t)sp.first * embed_dim]) == 0) {
                for (int k = 0; k < cnt; k++)
                    ok[(size_t)(sp.first + k)] = true;
                return;
            }
            // A failed batch degrades to the per-window path so only the bad
            // window is skipped, not its 31 neighbours.
            for (int k = 0; k < cnt; k++) {
                const int i = sp.first + k;
                ok[(size_t)i] = embed(userdata, worker, pcm + wins[(size_t)i].a,
                                      (int)(wins[(size_t)i].b - wins[(size_t)i].a), &all[(size_t)i * embed_dim]) == 0;
            }
            return;
        }
        if (!embed_windows) {
            const int i = sp.first;
            ok[(size_t)i] = embed(userdata, worker, pcm + wins[(size_t)i].a,
                                  (int)(wins[(size_t)i].b - wins[(size_t)i].a), &all[(size_t)i * embed_dim]) == 0;
            return;
        }
        const long base = wins[(size_t)sp.first].a;
        const long end = wins[(size_t)sp.last].b;
        const int cnt = sp.last - sp.first + 1;
        std::vector<int> ws((size_t)cnt), we((size_t)cnt);
        for (int k = 0; k < cnt; k++) {
            ws[(size_t)k] = (int)(wins[(size_t)(sp.first + k)].a - base);
            we[(size_t)k] = (int)(wins[(size_t)(sp.first + k)].b - base);
        }
        const bool good = embed_windows(userdata, worker, pcm + base, (int)(end - base), ws.data(), we.data(), cnt,
                                        &all[(size_t)sp.first * embed_dim]) == 0;
        for (int k = 0; k < cnt; k++)
            ok[(size_t)(sp.first + k)] = good;
    };

    // Spans are handed out from an atomic counter rather than split into
    // contiguous per-worker blocks: a span's cost depends on how many windows
    // it holds, and `slot` is what tells do_span WHICH embedder context to use
    // (one per worker, never two concurrent calls on the same one).
    core_parallel::for_each_task(n_spans, n_workers, [&](int si, int slot) { do_span(si, slot); });

    for (int i = 0; i < n_win; i++) {
        if (!ok[i]) {
            res.n_skipped++;
            continue;
        }
        emb.insert(emb.end(), all.begin() + (size_t)i * embed_dim, all.begin() + (size_t)(i + 1) * embed_dim);
        win_times.push_back(wins[i].w);
        win_parent.push_back(wins[i].parent);
    }
    res.n_windows = (int)win_times.size();
    if (win_times.empty())
        return res;

    // ---- cluster ----
    const int n = (int)win_times.size();
    core_spectral::SpeakerEstimate est;
    std::vector<int> labels =
        core_spectral::cluster_speakers(emb.data(), n, embed_dim, params.min_speakers, params.max_speakers,
                                        params.num_speakers, &est, params.seed, params.n_workers);
    res.reason = est.reason ? est.reason : "";

    // ---- per-segment temporal smoothing ----
    std::vector<int> label_values;
    std::vector<float> centroids =
        core_diarize_smooth::speaker_centroids(emb.data(), n, embed_dim, labels, &label_values);

    std::map<int, std::vector<int>> by_parent;
    for (int i = 0; i < n; i++)
        by_parent[win_parent[(size_t)i]].push_back(i);

    struct Raw {
        double start, end;
        int speaker;
    };
    std::vector<Raw> raw;

    for (auto& kv : by_parent) {
        std::vector<int>& idx = kv.second;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return win_times[(size_t)a].start < win_times[(size_t)b].start; });

        std::vector<int> seg_labels;
        std::vector<Speech> seg_windows;
        std::vector<float> seg_emb;
        seg_labels.reserve(idx.size());
        seg_windows.reserve(idx.size());
        for (int i : idx) {
            seg_labels.push_back(labels[(size_t)i]);
            seg_windows.push_back(win_times[(size_t)i]);
            seg_emb.insert(seg_emb.end(), emb.begin() + (size_t)i * embed_dim,
                           emb.begin() + (size_t)(i + 1) * embed_dim);
        }
        const std::vector<Speech> bounds = window_boundaries(speech[(size_t)kv.first], seg_windows);
        if (bounds.size() != seg_labels.size())
            continue; // shapes must line up or the labels would be misattributed

        std::vector<core_diarize_smooth::Window> w;
        w.reserve(bounds.size());
        for (const Speech& b : bounds)
            w.push_back({(float)b.start, (float)b.end});

        if (label_values.size() > 1) {
            const std::vector<int> original = seg_labels;
            seg_labels = core_diarize_smooth::smooth_segment_temporal(seg_labels, seg_emb.data(), (int)idx.size(),
                                                                      embed_dim, label_values, centroids.data());
            // Restore BEFORE collapsing: Viterbi is biased toward fewer
            // switches, so sustained evidence goes back first, and only then
            // are the remaining short islands absorbed.
            seg_labels = core_diarize_smooth::restore_sustained_runs(original, seg_labels, w);
            seg_labels = core_diarize_smooth::collapse_short_islands(seg_labels, w);
        } else {
            seg_labels = core_diarize_smooth::smooth_window_labels(seg_labels);
        }

        for (size_t i = 0; i < bounds.size(); i++)
            if (bounds[i].end > bounds[i].start)
                raw.push_back({bounds[i].start, bounds[i].end, seg_labels[i]});
    }

    // ---- speech regions that produced no window inherit the nearest turn ----
    for (size_t p = 0; p < speech.size(); p++) {
        if (by_parent.count((int)p))
            continue;
        if (raw.empty())
            continue;
        const double mid = (speech[p].start + speech[p].end) * 0.5;
        int best = raw.front().speaker;
        double best_dist = 1e300;
        for (const Raw& r : raw) {
            const double d = std::fabs(mid - (r.start + r.end) * 0.5);
            if (d < best_dist) {
                best_dist = d;
                best = r.speaker;
            }
        }
        raw.push_back({speech[p].start, speech[p].end, best});
    }

    std::sort(raw.begin(), raw.end(), [](const Raw& a, const Raw& b) { return a.start < b.start; });
    if (raw.empty())
        return res;

    // ---- merge adjacent turns of the same speaker ----
    std::vector<Turn> merged;
    merged.push_back({raw[0].start, raw[0].end, raw[0].speaker});
    for (size_t i = 1; i < raw.size(); i++) {
        Turn& prev = merged.back();
        if (raw[i].speaker == prev.speaker && raw[i].start - prev.end < kMergeGapSeconds)
            prev.end = std::max(prev.end, raw[i].end);
        else
            merged.push_back({raw[i].start, raw[i].end, raw[i].speaker});
    }

    // Renumber densely so the caller never sees a gap in speaker indices.
    std::set<int> used;
    for (const Turn& t : merged)
        used.insert(t.speaker);
    std::map<int, int> remap;
    int next = 0;
    for (int s : used)
        remap[s] = next++;
    for (Turn& t : merged)
        t.speaker = remap[t.speaker];

    res.turns = std::move(merged);
    res.n_speakers = (int)used.size();
    return res;
}

} // namespace core_foxnose

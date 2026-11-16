// core/canary_chunk_merge.h — canary-1b-v2 dynamic-chunking primitives.
//
// Exact ports of the NeMo blueprint that canary-1b-v2's own `.transcribe()`
// uses for long-form audio (verified against nemo 2.7.3):
//   - longest_common_subsequence_merge
//     (nemo/collections/asr/parts/utils/streaming_utils.py)
//   - PromptedAudioToTextLhotseDataset._find_optimal_chunk_size
//     (nemo/collections/asr/data/audio_to_text_lhotse_prompted.py)
//
// Weight-free and header-only so tests/test-canary-chunk-merge.cpp can pin
// both against vectors generated from the actual Python functions — the port
// is validated cross-implementation, not against itself.

#pragma once

#include <algorithm>
#include <vector>

namespace core_canary_chunk {

// longest_common_subsequence_merge(X, Y): X is the tail of the accumulated
// token buffer, Y the head of the next chunk's tokens. Returns the reference's
// (i, j, slice_len) triple: the alignment start indices into X and Y and the
// number of leading Y tokens covered by the alignment (0 = no dedup).
inline void nemo_lcs_merge(const std::vector<int>& X, const std::vector<int>& Y, int& out_i, int& out_j, int& out_len) {
    const int kMinMergeSubsequenceLen = 1; // streaming_utils.MIN_MERGE_SUBSEQUENCE_LEN
    const int m = (int)X.size();
    const int n = (int)Y.size();
    std::vector<std::vector<int>> S((size_t)m + 1, std::vector<int>((size_t)n + 1, 0));
    int result = 0;
    int ri = 0, rj = 0;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (X[(size_t)i - 1] == Y[(size_t)j - 1]) {
                S[(size_t)i][(size_t)j] = S[(size_t)i - 1][(size_t)j - 1] + 1;
                if (result <= S[(size_t)i][(size_t)j]) {
                    result = S[(size_t)i][(size_t)j];
                    ri = i;
                    rj = j;
                }
            }
        }
    }
    int i = ri, j = rj;
    int out = result;
    const bool is_complete_merge = (i == m);
    if (is_complete_merge) {
        // Backtrack to the origin point of the slice.
        int length = result;
        while (length >= 0 && i > 0 && j > 0) {
            if (S[(size_t)i - 1][(size_t)j - 1] > 0) {
                length--;
                i--;
                j--;
            } else {
                i--;
                j--;
                break;
            }
        }
    } else {
        // Partial mismatch: (1) leftmost LCS, (2) greedy diagonal expansion,
        // (3) backtrack counting diagonal skips.
        int max_j = 0;
        int max_j_idx = n;
        int i_partial = m;
        int j_partial = -1;
        int j_skip = 0;
        int slice_count = 0;
        for (int i_idx = m; i_idx >= 0; i_idx--) {
            for (int j_idx = 0; j_idx <= n; j_idx++) {
                if (S[(size_t)i_idx][(size_t)j_idx] > max_j && j_idx <= max_j_idx) {
                    max_j = S[(size_t)i_idx][(size_t)j_idx];
                    max_j_idx = j_idx;
                    i_partial = i_idx;
                    j_partial = j_idx;
                }
            }
        }
        if (max_j <= kMinMergeSubsequenceLen) {
            i = i_partial;
            j = 0;
            out = 0;
        } else {
            int j_temp = j_partial + 1;
            int j_exp = 0;
            j_skip = 0;
            for (int i_idx = i_partial + 1; i_idx <= m; i_idx++) {
                int j_any_skip = 0;
                for (int j_idx = j_temp; j_idx <= j_temp + j_skip; j_idx++) {
                    if (j_idx < n + 1) {
                        if (S[(size_t)i_idx][(size_t)j_idx] == 0)
                            j_any_skip = 1;
                        else
                            j_exp = 1 + j_skip + j_any_skip;
                    }
                }
                j_skip += j_any_skip;
                j_temp += 1;
            }
            j_skip = 0;
            j_partial += j_exp;
            while (i_partial > 0 && j_partial > 0) {
                if (S[(size_t)i_partial][(size_t)j_partial] == 0) {
                    j_partial -= 1;
                    j_skip += 1;
                }
                if (j_partial > 0) {
                    slice_count += 1;
                    i_partial -= 1;
                    j_partial -= 1;
                }
            }
            i = std::max(0, i_partial);
            j = std::max(0, j_partial);
            out = slice_count + j_skip;
        }
    }
    out_i = i;
    out_j = j;
    out_len = out;
}

// _find_optimal_chunk_size: dynamic 30..40 s chunk size chosen to maximize the
// last chunk's length. Returns total_len (single pass) when the audio fits a
// single max_sec chunk.
inline int optimal_chunk_samples(int total_len, int sr, int min_sec = 30, int max_sec = 40, double overlap_sec = 1.0) {
    int best_chunk = min_sec * sr;
    int best_last = 0;
    if (total_len < max_sec * sr)
        return total_len;
    for (int sec = min_sec; sec <= max_sec; sec++) {
        const int chunk = sec * sr;
        const int overlap = (int)(overlap_sec * sr);
        const int step = chunk - overlap;
        if (step <= 0)
            continue;
        if (chunk > total_len)
            continue;
        const int n_chunks = (total_len + step - 1) / step;
        const int last = total_len - step * (n_chunks - 1);
        if (last > best_last) {
            best_last = last;
            best_chunk = chunk;
        }
    }
    return best_chunk;
}

} // namespace core_canary_chunk

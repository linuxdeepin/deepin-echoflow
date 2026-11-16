// src/core/spm_bpe.h — SentencePiece-style BPE merge (LlamaTokenizer family).
//
// Hoisted verbatim from irodori_tts.cpp (sarashina2.2 path); confucius4_tts
// is the second consumer. This is the HuggingFace `tokenizers` BPE model as
// used by LlamaTokenizerFast:
//   1. caller applies Metaspace first (replace ' ' with ▁ U+2581; prepend ▁
//      when the tokenizer's prepend_scheme says so),
//   2. split into UTF-8 characters as initial symbols,
//   3. iteratively merge the lowest-rank adjacent pair per the merges table
//      (textual "left right" keys, rank = list index),
//   4. byte_fallback: symbols absent from the vocab emit <0xHH> tokens per
//      UTF-8 byte.
//
// This is NOT the GPT-2 byte-level BPE (see core/bpe.h — different alphabet,
// byte→unicode remap, regex pre-tokenizer) and NOT the SentencePiece unigram
// Viterbi (see core/sentencepiece.h — score-based, no merges table).

#pragma once

#include <cstdint>
#include <cstdio>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core_spm_bpe {

// BPE merge: split text into UTF-8 chars, iteratively merge by rank.
// Uses linked-list + priority queue for O(N log N) merging.
inline std::vector<int32_t> bpe_merge(const std::string& text,
                                      const std::unordered_map<std::string, int32_t>& token_to_id,
                                      const std::unordered_map<std::string, int32_t>& merge_rank) {
    if (text.empty())
        return {};

    // Split into individual UTF-8 characters as initial symbols
    struct Node {
        std::string text;
        int prev, next;
    };
    std::vector<Node> nodes;
    {
        size_t i = 0;
        while (i < text.size()) {
            size_t len = 1;
            unsigned char c = (unsigned char)text[i];
            if (c >= 0xC0) {
                if (c < 0xE0)
                    len = 2;
                else if (c < 0xF0)
                    len = 3;
                else
                    len = 4;
            }
            len = std::min(len, text.size() - i);
            int idx = (int)nodes.size();
            nodes.push_back({text.substr(i, len), idx - 1, -1});
            if (idx > 0)
                nodes[idx - 1].next = idx;
            i += len;
        }
    }
    if (nodes.empty())
        return {};

    // Priority-queue BPE: merge lowest-rank pairs first
    using PQE = std::pair<int, int>; // (rank, left_index)
    auto cmp = [](const PQE& a, const PQE& b) { return a.first > b.first; };
    std::priority_queue<PQE, std::vector<PQE>, decltype(cmp)> pq(cmp);

    auto try_add = [&](int i) {
        int j = nodes[i].next;
        if (j < 0)
            return;
        std::string pair = nodes[i].text + " " + nodes[j].text;
        auto it = merge_rank.find(pair);
        if (it != merge_rank.end())
            pq.push({it->second, i});
    };
    for (int i = 0; i < (int)nodes.size(); i++)
        try_add(i);

    while (!pq.empty()) {
        auto [rank, left] = pq.top();
        pq.pop();
        int right = nodes[left].next;
        if (right < 0)
            continue;
        // Verify the pair hasn't been invalidated by a prior merge
        std::string pair = nodes[left].text + " " + nodes[right].text;
        auto it = merge_rank.find(pair);
        if (it == merge_rank.end() || it->second != rank)
            continue;
        // Merge right into left
        nodes[left].text += nodes[right].text;
        nodes[left].next = nodes[right].next;
        if (nodes[right].next >= 0)
            nodes[nodes[right].next].prev = left;
        nodes[right].next = -1;
        nodes[right].prev = -1;
        // Re-check neighbors for new merge opportunities
        if (nodes[left].prev >= 0)
            try_add(nodes[left].prev);
        try_add(left);
    }

    // Collect symbols and convert to token IDs
    std::vector<int32_t> ids;
    for (int i = 0; i >= 0; i = nodes[i].next) {
        auto it = token_to_id.find(nodes[i].text);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            // Byte fallback: encode each UTF-8 byte as <0xHH>
            for (unsigned char byte : nodes[i].text) {
                char hex[16];
                std::snprintf(hex, sizeof(hex), "<0x%02X>", byte);
                auto bit = token_to_id.find(hex);
                if (bit != token_to_id.end()) {
                    ids.push_back(bit->second);
                }
            }
        }
    }
    return ids;
}

} // namespace core_spm_bpe

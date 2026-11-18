// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace core_audio_encoder_cache {

struct Block {
    std::vector<float> mel;
    std::vector<float> stem;
};

struct Region {
    int first_chunk = 0;
    int chunks = 0;
    int mel_frames = 0;
    int frames = 0;
    int dim = 0;
    std::vector<float> output;
};

// Cache dependencies are the normalized Mel bytes, not PCM duration. A later
// loud sample can change global normalization of every historical frame.
class State {
public:
    std::vector<Block> blocks;
    std::vector<Region> regions;

    void prepare(const float* mel, int n_mels, int frames, int chunk_frames, int window_chunks)
    {
        if (!mel || n_mels <= 0 || frames <= 0 || chunk_frames <= 0 || window_chunks < 0)
            throw std::invalid_argument("invalid encoder cache input");
        if (n_mels != n_mels_ || chunk_frames != chunk_frames_ || window_chunks != window_chunks_) {
            clear();
            n_mels_ = n_mels;
            chunk_frames_ = chunk_frames;
            window_chunks_ = window_chunks;
        }
        const int count = 1 + (frames - 1) / chunk_frames;
        blocks.resize(count);
        std::vector<bool> changed(count, false);
        for (int index = 0; index < count; ++index) {
            const int length = std::min(chunk_frames, frames - index * chunk_frames);
            std::vector<float> padded(static_cast<size_t>(n_mels) * chunk_frames, 0.0f);
            for (int feature = 0; feature < n_mels; ++feature) {
                std::memcpy(padded.data() + static_cast<size_t>(feature) * chunk_frames,
                    mel + static_cast<size_t>(feature) * frames + index * chunk_frames,
                    static_cast<size_t>(length) * sizeof(float));
            }
            auto& block = blocks[index];
            changed[index] = block.mel.size() != padded.size()
                || std::memcmp(block.mel.data(), padded.data(), padded.size() * sizeof(float)) != 0;
            if (changed[index]) {
                block.mel = std::move(padded);
                block.stem.clear();
            }
        }
        // Dense bidirectional attention is one region. Every changed/appended
        // frame invalidates all layers, even if historical stems are reusable.
        const int span = window_chunks > 0 ? window_chunks : count;
        regions.resize(1 + (count - 1) / span);
        for (size_t index = 0; index < regions.size(); ++index) {
            const int first = static_cast<int>(index) * span;
            const int chunks = std::min(span, count - first);
            const int valid = std::min(chunks * chunk_frames, frames - first * chunk_frames);
            auto& region = regions[index];
            const bool dirty = region.first_chunk != first || region.chunks != chunks
                || region.mel_frames != valid
                || std::any_of(changed.begin() + first, changed.begin() + first + chunks,
                               [](bool value) { return value; });
            if (dirty) region = {first, chunks, valid, 0, 0, {}};
        }
    }

    void clear()
    {
        blocks.clear();
        regions.clear();
        n_mels_ = chunk_frames_ = window_chunks_ = 0;
    }

    size_t bytes() const
    {
        size_t bytes = 0;
        for (const auto& block : blocks) bytes += (block.mel.size() + block.stem.size()) * sizeof(float);
        for (const auto& region : regions) bytes += region.output.size() * sizeof(float);
        return bytes;
    }

private:
    int n_mels_ = 0;
    int chunk_frames_ = 0;
    int window_chunks_ = 0;
};

} // namespace core_audio_encoder_cache

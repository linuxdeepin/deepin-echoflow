// src/core/asr_segment_group.h — group a word list into fixed-window output
// segments (issue #257).
//
// Some ASR backends (parakeet/canary FastConformer with CAP_INTERNAL_CHUNKING)
// run ONE coherent decode over the whole clip and would otherwise emit a single
// giant transcription segment. When the caller asks for `--chunk-seconds N`,
// we keep that single high-quality decode but group the resulting words into
// ~N-second OUTPUT segments so `-ojf`/`-osrt` produce per-segment offsets like
// the whisper/cohere/granite backends — WITHOUT re-running the encoder on small
// windows (which degrades this full-attention model, see parakeet.cpp).
//
// `group_by_window` is a pure boundary computation over ascending word start
// times; it is unit-tested independently of the ASR runtime.
#pragma once

#include <cstdint>
#include <vector>

namespace core_segment {

// Group ascending word start-times (centiseconds) into output segments each
// spanning at most `win_cs` centiseconds, snapped to word boundaries. A new
// segment starts at the first word whose start is `win_cs` or more past the
// current segment's first-word start. Every segment holds at least one word,
// so a single word longer than the window still gets its own segment (no
// infinite loop / no dropped words).
//
// Returns the word index at which each output segment begins. Segment k spans
// word indices [starts[k], starts[k+1]) with the final segment running to the
// end of the list. An empty input yields an empty result; a non-positive
// `win_cs` collapses everything into one segment ({0}).
inline std::vector<int> group_by_window(const std::vector<int64_t>& word_t0, int64_t win_cs) {
    std::vector<int> starts;
    if (word_t0.empty())
        return starts;
    starts.push_back(0);
    if (win_cs <= 0)
        return starts; // degenerate window → single segment
    int64_t seg_start = word_t0[0];
    for (int i = 1; i < (int)word_t0.size(); ++i) {
        if (word_t0[i] - seg_start >= win_cs) {
            starts.push_back(i);
            seg_start = word_t0[i];
        }
    }
    return starts;
}

} // namespace core_segment

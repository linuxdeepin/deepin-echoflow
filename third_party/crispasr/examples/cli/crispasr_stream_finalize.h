// crispasr_stream_finalize.h — helpers shared between the streaming
// finalize_utterance / EOF flush paths in crispasr_run.cpp and their
// unit tests.
//
// The logic here is intentionally pure (no I/O, no global state) so it
// can be tested in isolation. The streaming loop in crispasr_run.cpp
// owns the surrounding state (`prefix_committed`, `last_partial_text`,
// `utterance_pcm`, …) and feeds it into these helpers.

#pragma once

#include "core/audio_chunking.h"

#include <algorithm>
#include <string>

namespace crispasr {

// Build a `final.text` value by stitching the prefix-mode accumulator:
//   - `committed_prefix` is the longest-common-prefix of all observed
//     partials, captured before the rolling window evicted them.
//   - `last_partial` is the most recent partial for the open utterance.
//
// Cases (mirrors the historical inline implementation in
// `examples/cli/crispasr_run.cpp`):
//   * `last_partial` starts with `committed_prefix`  → `last_partial`
//     alone (it already extends the prefix).
//   * `committed_prefix` empty                       → `last_partial`.
//   * `last_partial` empty                           → `committed_prefix`.
//   * otherwise (divergent strings)                  → `committed_prefix + " " + last_partial`.
//
// This is the only place that decides what a finalized utterance's text
// reads as when no `redecode` pass produced one — either because the
// caller is in `--stream-final-mode prefix`, or because `redecode` was
// skipped (sub-2-s utterance) or returned empty in `redecode` mode (the
// fallback added in commit cf1de878, fixing #84 round 4).
inline std::string stitch_partial_accumulator(const std::string& committed_prefix, const std::string& last_partial) {
    if (!last_partial.empty() && last_partial.size() >= committed_prefix.size() &&
        last_partial.compare(0, committed_prefix.size(), committed_prefix) == 0) {
        return last_partial;
    }
    if (committed_prefix.empty()) {
        return last_partial;
    }
    if (last_partial.empty()) {
        return committed_prefix;
    }
    return committed_prefix + " " + last_partial;
}

// --stream-partial-tail-sec (#404): bound the audio a PARTIAL decode covers.
//
// The partial decode of an open utterance grows with the utterance (up to
// --stream-length), so live-preview cost grows O(utterance) per step even
// though the preview only needs to move at its tail. Activation-level reuse
// cannot fix this exactly: the cohere Conformer runs unmasked rel-pos
// self-attention over the WHOLE window in every layer, so every cached
// frame depends on audio that arrives later. Incrementality therefore lives
// at the TEXT level: decode a bounded tail, keep the already-decoded text as
// a committed prefix, and let `final.text` stay exact via the untouched
// full-utterance redecode.
//
// plan_partial_tail decides, for one step's partial decode over the window-
// relative range [range_start, range_end):
//   * where this step's decode should START (the anchor — the cut of the
//     previous commit, clamped into the range), and
//   * whether the region between the anchor and the tail cap has grown past
//     `slack` and must first be COMMITTED: decoded once, its text appended
//     to the caller's committed-prefix string, and the anchor advanced.
// The commit cut lands on the quietest 100 ms via find_energy_min_split —
// the same boundary policy as the long-audio chunker — so the seam does not
// split a word. Both the commit range and the remaining tail are kept at or
// above `min_decode` so conv-frontend backends never see a sub-minimum clip.
struct PartialTailPlan {
    int decode_start = 0; // window-relative start for this step's partial decode
    bool commit = false;  // decode [commit_start, commit_end) first, append its
    int commit_start = 0; // text to the committed prefix, then decode from
    int commit_end = 0;   // decode_start (== commit_end)
};

inline PartialTailPlan plan_partial_tail(const float* window_pcm, int anchor, int range_start, int range_end,
                                         int tail_cap, int slack, int min_decode) {
    PartialTailPlan plan;
    plan.decode_start = range_start;
    if (tail_cap <= 0 || range_end <= range_start)
        return plan; // disabled or degenerate — decode the full range
    const int eff_anchor = std::max(anchor, range_start);
    if (eff_anchor > range_end - min_decode)
        return plan; // stale/foreign anchor (this is not the region it was
                     // set in, or the range shrank) — decode the full range
    plan.decode_start = eff_anchor;
    if (range_end - plan.decode_start <= tail_cap + slack)
        return plan; // tail still small — no commit needed
    // Advance: cut at the quietest 100 ms so ~tail_cap remains after it.
    const size_t win_100ms = 1600; // 100 ms @ 16 kHz, matches split_at_energy_minima
    const size_t search_lo = (size_t)(range_end - tail_cap - slack / 2);
    const size_t search_hi = (size_t)(range_end - tail_cap);
    int cut = (int)audio_chunking::find_energy_min_split(window_pcm, search_lo, search_hi, win_100ms);
    cut = std::max(cut, plan.decode_start + min_decode); // commit region stays decodable
    if (range_end - cut < min_decode)
        return plan; // remaining tail would be sub-minimum — wait for more audio
    plan.commit = true;
    plan.commit_start = plan.decode_start;
    plan.commit_end = cut;
    plan.decode_start = cut;
    return plan;
}

// Threshold used by the streaming `finalize_utterance` to decide
// whether the `redecode`-mode extra backend pass is safe to invoke.
// Convolutional encoders (moonshine, parakeet, voxtral, …) abort with
// `OW > 0` from `ggml_im2col` when given less audio than the first
// conv kernel needs — empirically ~2 s at 16 kHz mono. When the
// VAD-trimmed `[t0..t1]` buffer is shorter than this, redecode is
// skipped and the caller falls back to `stitch_partial_accumulator`.
constexpr int kStreamRedecodeMinSamples = 32000;

} // namespace crispasr

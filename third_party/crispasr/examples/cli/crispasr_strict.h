// crispasr_strict.h — shared strict-pipeline requirement logic (issue #311).
//
// Both the CLI (crispasr_run.cpp) and the HTTP server (crispasr_server.cpp)
// auto-run the VAD / forced-aligner / punctuation stages and, by default,
// degrade gracefully when one fails to load. Strict mode makes an explicitly
// requested stage that fails to load/produce a hard error instead. This header
// holds the surface-agnostic pieces so the two front-ends can't drift (the
// session C-ABI is a lower-level, caller-driven primitive and is out of scope —
// see docs/cli.md#strict-pipeline).
#pragma once

#include "crispasr_backend.h" // crispasr_segment
#include "whisper_params.h"

#include <cctype>
#include <vector>

// Which stages must succeed for this invocation.
struct crispasr_strict_reqs {
    bool vad = false;   // a loaded, running VAD stage is required
    bool words = false; // every non-empty output segment must carry word timestamps
    bool punc = false;  // a loaded punctuation model is required
};

// Resolve the effective per-stage requirements from the flags. `--strict-pipeline`
// requires each stage that was explicitly requested on this command line (or in
// the request); the per-stage `--require-*` flags force one requirement
// regardless. `require_word_timestamps` is a property of the OUTPUT (native or
// aligned), so it needs no aligner precondition; the others need their stage to
// actually be requested.
inline crispasr_strict_reqs crispasr_compute_strict_reqs(const whisper_params& p) {
    const bool vad_requested = p.vad || !p.vad_model.empty();
    const bool align_requested = !p.aligner_model.empty() || p.force_aligner;
    const bool punc_requested = !p.punc_model.empty();
    crispasr_strict_reqs r;
    r.vad = p.require_vad || (p.strict_pipeline && vad_requested);
    r.words = p.require_word_timestamps || (p.strict_pipeline && align_requested);
    r.punc = p.require_punctuation || (p.strict_pipeline && punc_requested);
    return r;
}

// Count segments whose text has non-whitespace content but carry no word
// timestamps. Zero = the word-timestamp requirement is satisfied (or there is
// nothing to transcribe). Each surface formats its own error from the count.
inline int crispasr_count_missing_word_ts(const std::vector<crispasr_segment>& segs) {
    int missing = 0;
    for (const auto& s : segs) {
        bool has_text = false;
        for (unsigned char c : s.text) {
            if (c > ' ') {
                has_text = true;
                break;
            }
        }
        if (has_text && s.words.empty())
            missing++;
    }
    return missing;
}

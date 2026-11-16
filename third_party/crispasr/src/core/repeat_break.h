// src/core/repeat_break.h — decode-time repetition-loop detector.
//
// Greedy autoregressive decoders (moonshine) can get stuck emitting a short
// token cycle on hard audio ("I'm sorry, I'm sorry, …"), burning decode steps
// until max_len. This detects a period-p block (p <= max_period) repeated at
// least min_reps times at the TAIL of the generated tokens, so the caller can
// stop early. Post-hoc core_ngram::fix_loops still cleans any residue; this
// saves the wasted compute. Pure + unit-tested (test-repeat-break).
#pragma once

#include <cstdint>
#include <vector>

namespace core_repeat {

// True iff `tok` ends with a block of length p (1..max_period) repeated at
// least `min_reps` times consecutively. Conservative by design: a small cycle
// repeated many times is a decode loop, not natural speech (which rarely
// repeats a verbatim phrase 4×).
template <typename T> inline bool tail_is_repetition(const std::vector<T>& tok, int min_reps = 4, int max_period = 8) {
    const long n = (long)tok.size();
    if (min_reps < 2)
        min_reps = 2;
    for (int p = 1; p <= max_period; ++p) {
        if (n < (long)p * min_reps)
            continue;
        bool loop = true;
        // Compare the last block against the (min_reps-1) preceding blocks.
        for (int k = 1; k < min_reps && loop; ++k)
            for (int j = 0; j < p; ++j)
                if (tok[n - 1 - j] != tok[n - 1 - j - (long)k * p]) {
                    loop = false;
                    break;
                }
        if (loop)
            return true;
    }
    return false;
}

} // namespace core_repeat

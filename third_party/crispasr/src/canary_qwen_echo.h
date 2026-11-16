// canary_qwen_echo.h — instruction-echo detection for canary-qwen (#247).
//
// When a canary-qwen audio window is too short/low-content to ground a
// transcription, the Qwen3 LLM decoder falls back to its language prior and
// echoes the task framing as a meta word instead of transcribing. Verified
// against the NeMo SALM reference (nvidia/canary-qwen-2.5b): on a 0.1 s clip
// both NeMo and CrispASR emit "Okay"; on ~0.3 s clips both emit "Transcript".
// These exact leading words are never a real ASR transcript under the
// "Transcribe the following:" prompt, so the decode path strips them.
//
// This header holds the pure, tokenizer-independent word-level test so it can
// be unit-tested without a model. The multi-token accumulation that feeds it
// lives in canary_qwen.cpp.
#pragma once

#include <cctype>
#include <string>

namespace canary_qwen_echo {

// Normalise `s` (trim surrounding whitespace, drop a single trailing ':',
// lowercase) and return whether it is one of the known instruction-echo meta
// words the SALM decoder emits on a degenerate window.
inline bool is_meta_echo(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\n' || s[b] == '\t' || s[b] == '\r'))
        b++;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\n' || s[e - 1] == '\t' || s[e - 1] == '\r'))
        e--;
    if (e > b && s[e - 1] == ':')
        e--;
    std::string w = s.substr(b, e - b);
    for (auto& c : w)
        c = (char)std::tolower((unsigned char)c);
    // Unambiguous meta words only. "Okay"/"And"/etc. are legitimate transcript
    // words, so they are NOT listed here — the degenerate-window length gate
    // (not this strip) is what suppresses those hallucinations.
    return w == "transcript" || w == "transcription" || w == "pass";
}

} // namespace canary_qwen_echo

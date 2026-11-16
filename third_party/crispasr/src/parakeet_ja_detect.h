// parakeet_ja_detect.h — issue #257: detect a Japanese-only parakeet model by
// vocab CONTENT, not vocab size. The prior `vocab_size <= 4096` heuristic
// misclassified small-vocab ENGLISH models (parakeet-tdt-1.1b, vocab 1024) as
// Japanese, forcing them onto the JA short-chunk decode path and corrupting
// long / chunked output. JA-only parakeet vocabs are ~97% kana/kanji;
// multilingual v3 (vocab 8192) and English models are ~0%.
//
// Header-only + pure so tests/test-parakeet-ja-detect.cpp can pin it without a
// model.

#pragma once

#include <string>
#include <vector>

namespace crispasr_parakeet {

// True iff the UTF-8 string `s` contains a Japanese-script character:
// Hiragana/Katakana (U+3040–U+30FF, UTF-8 `E3 [81..83] xx`) or a CJK ideograph
// (U+4E00–U+9FFF and neighbours, UTF-8 3-byte lead `E4..E9`).
inline bool token_has_japanese(const std::string& s) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(s.c_str());
    for (std::size_t i = 0; p[i]; ++i) {
        const unsigned char c = p[i];
        if (c == 0xE3 && p[i + 1] >= 0x81 && p[i + 1] <= 0x83)
            return true; // kana
        if (c >= 0xE4 && c <= 0xE9)
            return true; // CJK ideographs
    }
    return false;
}

// True iff more than half of the non-special tokens contain Japanese script,
// i.e. a Japanese-only model. Special tokens (`<blank>`, `<|ja|>`, …) are
// skipped so a handful of language tags on a multilingual model don't count.
inline bool vocab_looks_japanese(const std::vector<std::string>& tokens) {
    std::size_t ja = 0, total = 0;
    for (const std::string& t : tokens) {
        if (t.empty() || (t.front() == '<' && t.back() == '>'))
            continue;
        ++total;
        if (token_has_japanese(t))
            ++ja;
    }
    return total > 0 && ja * 2 > total;
}

} // namespace crispasr_parakeet

#pragma once

// NFKD Unicode normalization for the Chatterbox multilingual text
// preprocessor. Reproduces upstream `MTLTokenizer.preprocess_text`, which
// runs `unicodedata.normalize("NFKD", text.lower())` before tokenizing.
// Without it, precomposed graphemes (e.g. Arabic U+0623 ALEF-WITH-HAMZA,
// accented Latin) tokenize to a different id than the model was trained on,
// producing spurious onset phonemes (issue #170).
//
// The decomposition + combining-class tables are generated from the Python
// unicodedata database by tools/gen_nfkd_table.py. Hangul syllables are
// decomposed arithmetically (the standard NFKD algorithm) rather than tabled.

#include "chatterbox_nfkd_data.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace chatterbox_text_prep {

namespace nfkd_detail {

constexpr uint32_t kHangulBase = 0xAC00;
constexpr uint32_t kHangulEnd = 0xD7A3;
constexpr uint32_t kLBase = 0x1100;
constexpr uint32_t kVBase = 0x1161;
constexpr uint32_t kTBase = 0x11A7;
constexpr uint32_t kVCount = 21;
constexpr uint32_t kTCount = 28;
constexpr uint32_t kNCount = kVCount * kTCount; // 588

// Decode UTF-8 into Unicode codepoints. Invalid bytes pass through as-is
// (one codepoint per byte) so the routine never throws on malformed input.
inline void utf8_decode(const std::string& s, std::vector<uint32_t>& out) {
    out.clear();
    out.reserve(s.size());
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        int len;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c >> 5) == 0x6 && i + 1 < n) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c >> 4) == 0xE && i + 2 < n) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c >> 3) == 0x1E && i + 3 < n) {
            cp = c & 0x07;
            len = 4;
        } else {
            out.push_back(c); // invalid lead byte — pass through
            ++i;
            continue;
        }
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            const unsigned char cc = (unsigned char)s[i + k];
            if ((cc >> 6) != 0x2) {
                ok = false;
                break;
            }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) {
            out.push_back(c); // malformed continuation — pass lead byte through
            ++i;
            continue;
        }
        out.push_back(cp);
        i += len;
    }
}

inline void utf8_encode_cp(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

inline uint8_t combining_class(uint32_t cp) {
    const uint32_t* keys = nfkd_data::kCccKeys;
    const int n = nfkd_data::kCccCount;
    const uint32_t* it = std::lower_bound(keys, keys + n, cp);
    if (it != keys + n && *it == cp) {
        return nfkd_data::kCccVals[it - keys];
    }
    return 0;
}

// Append the full NFKD decomposition of `cp` to `out` (Hangul handled
// arithmetically; everything else via the generated table; codepoints with
// no decomposition append unchanged).
inline void decompose_cp(uint32_t cp, std::vector<uint32_t>& out) {
    if (cp >= kHangulBase && cp <= kHangulEnd) {
        const uint32_t s = cp - kHangulBase;
        out.push_back(kLBase + s / kNCount);
        out.push_back(kVBase + (s % kNCount) / kTCount);
        const uint32_t t = s % kTCount;
        if (t != 0) {
            out.push_back(kTBase + t);
        }
        return;
    }
    const uint32_t* keys = nfkd_data::kDecompKeys;
    const int n = nfkd_data::kDecompCount;
    const uint32_t* it = std::lower_bound(keys, keys + n, cp);
    if (it != keys + n && *it == cp) {
        const size_t idx = it - keys;
        const uint32_t off = nfkd_data::kDecompOffset[idx];
        const uint32_t len = nfkd_data::kDecompLen[idx];
        for (uint32_t k = 0; k < len; ++k) {
            out.push_back(nfkd_data::kDecompData[off + k]);
        }
        return;
    }
    out.push_back(cp);
}

// Canonical ordering: stable-sort each maximal run of non-starters
// (combining class != 0) by combining class. Starters break runs.
inline void canonical_order(std::vector<uint32_t>& cps) {
    std::vector<uint8_t> ccc(cps.size());
    for (size_t i = 0; i < cps.size(); ++i) {
        ccc[i] = combining_class(cps[i]);
    }
    size_t i = 0;
    while (i < cps.size()) {
        if (ccc[i] == 0) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < cps.size() && ccc[j] != 0) {
            ++j;
        }
        // Stable insertion sort of [i, j) by ccc (runs are tiny).
        for (size_t a = i + 1; a < j; ++a) {
            const uint32_t cp = cps[a];
            const uint8_t cc = ccc[a];
            size_t b = a;
            while (b > i && ccc[b - 1] > cc) {
                cps[b] = cps[b - 1];
                ccc[b] = ccc[b - 1];
                --b;
            }
            cps[b] = cp;
            ccc[b] = cc;
        }
        i = j;
    }
}

} // namespace nfkd_detail

inline std::string nfkd(const std::string& text) {
    std::vector<uint32_t> cps;
    nfkd_detail::utf8_decode(text, cps);

    std::vector<uint32_t> decomposed;
    decomposed.reserve(cps.size() * 2);
    for (uint32_t cp : cps) {
        nfkd_detail::decompose_cp(cp, decomposed);
    }

    nfkd_detail::canonical_order(decomposed);

    std::string out;
    out.reserve(text.size() * 2);
    for (uint32_t cp : decomposed) {
        nfkd_detail::utf8_encode_cp(cp, out);
    }
    return out;
}

} // namespace chatterbox_text_prep

// src/core/pinyin_g2p.cpp — see pinyin_g2p.h.
#include "core/pinyin_g2p.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Embedded tables: PY_CHAR_CP[], PY_CHAR_PY[], PY_CHAR_N, PY_PHRASE_K[],
// PY_PHRASE_V[], PY_PHRASE_N, PY_MAX_PHRASE_CHARS. Generated from pypinyin.
#include "pinyin_data.inc"

constexpr uint32_t HAN_LO = 0x3100, HAN_HI = 0x9FFF; // F5-TTS is_chinese range
constexpr uint32_t CP_BU = 0x4E0D;                   // 不
constexpr uint32_t CP_YI = 0x4E00;                   // 一

struct Tables {
    std::unordered_map<uint32_t, const char*> ch;    // codepoint -> TONE3
    std::unordered_map<std::string, const char*> ph; // phrase (UTF-8) -> "syl syl ..."
    int max_phrase_chars = 1;
};

const Tables& tables() {
    static const Tables t = [] {
        Tables t;
        t.ch.reserve(PY_CHAR_N * 2);
        for (int i = 0; i < PY_CHAR_N; ++i)
            t.ch.emplace(PY_CHAR_CP[i], PY_CHAR_PY[i]);
        t.ph.reserve(PY_PHRASE_N * 2);
        for (int i = 0; i < PY_PHRASE_N; ++i)
            t.ph.emplace(PY_PHRASE_K[i], PY_PHRASE_V[i]);
        t.max_phrase_chars = PY_MAX_PHRASE_CHARS;
        return t;
    }();
    return t;
}

// Decode one UTF-8 codepoint at byte offset i; returns codepoint and advances n.
inline uint32_t utf8_decode(const std::string& s, size_t i, int& n) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
        n = 1;
        return c;
    }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        n = 2;
        return ((c & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F);
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        n = 3;
        return ((c & 0x0F) << 12) | (((unsigned char)s[i + 1] & 0x3F) << 6) | ((unsigned char)s[i + 2] & 0x3F);
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        n = 4;
        return ((uint32_t)(c & 0x07) << 18) | (((unsigned char)s[i + 1] & 0x3F) << 12) |
               (((unsigned char)s[i + 2] & 0x3F) << 6) | ((unsigned char)s[i + 3] & 0x3F);
    }
    n = 1;
    return c;
}

inline void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

inline bool is_han(uint32_t cp) {
    return cp >= HAN_LO && cp <= HAN_HI;
}

// TONE3 tone digit of a syllable: 1..5, or 0 for neutral (no trailing digit).
inline int tone_of(const std::string& syl) {
    if (!syl.empty()) {
        char c = syl.back();
        if (c >= '1' && c <= '5')
            return c - '0';
    }
    return 0;
}

// Set the tone digit of a TONE3 syllable (rewrites/removes the trailing digit).
inline void set_tone(std::string& syl, int t) {
    if (!syl.empty() && syl.back() >= '1' && syl.back() <= '5')
        syl.pop_back();
    if (t >= 1 && t <= 5)
        syl += (char)('0' + t);
}

// ── tone-sandhi, mirroring pypinyin.contrib.tone_sandhi (order: 3rd, 不, 一) ──
// Operates within one segment (word), on parallel (han codepoint, syllable).

void sandhi_third_tone(std::vector<std::string>& syl) {
    bool any3 = false;
    for (auto& s : syl)
        if (tone_of(s) == 3) {
            any3 = true;
            break;
        }
    if (!any3)
        return;
    // third_num = length of the trailing run of 3rd tones.
    int third_num = 0;
    for (auto& s : syl) {
        if (tone_of(s) == 3)
            ++third_num;
        else
            third_num = 0;
    }
    if (third_num == 2) {
        for (auto& s : syl)
            if (tone_of(s) == 3) {
                set_tone(s, 2);
                break;
            }
    } else if (third_num > 2) {
        int n = 1;
        for (auto& s : syl) {
            if (tone_of(s) == 3) {
                if (n == third_num)
                    break;
                set_tone(s, 2);
                ++n;
            }
        }
    }
}

void sandhi_bu_yi(const std::vector<uint32_t>& han, std::vector<std::string>& syl, uint32_t which) {
    // which = CP_BU (不: →2 before 4th tone, else →4) or CP_YI (一: →2 before
    // 4th, →4 before 1/2/3, →1 when final in the segment).
    bool present = false;
    for (uint32_t h : han)
        if (h == which) {
            present = true;
            break;
        }
    if (!present)
        return;
    const int n = (int)han.size();
    for (int i = 0; i < n; ++i) {
        if (han[i] != which)
            continue;
        if (i < n - 1) {
            if (tone_of(syl[i + 1]) == 4)
                set_tone(syl[i], 2);
            else
                set_tone(syl[i], 4);
        } else {
            set_tone(syl[i], which == CP_BU ? 4 : 1);
        }
    }
}

void apply_sandhi(const std::vector<uint32_t>& han, std::vector<std::string>& syl) {
    sandhi_third_tone(syl);
    sandhi_bu_yi(han, syl, CP_BU);
    sandhi_bu_yi(han, syl, CP_YI);
}

// Split "syl1 syl2 ..." into syllables.
std::vector<std::string> split_ws(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = s; *p; ++p) {
        if (*p == ' ') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur += *p;
        }
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

// Default single-char TONE3 (raw, pre-sandhi). Unknown → the char itself.
std::string char_pinyin(uint32_t cp) {
    const auto& t = tables();
    auto it = t.ch.find(cp);
    if (it != t.ch.end())
        return it->second;
    std::string s;
    utf8_append(s, cp);
    return s;
}

// Normalize a handful of OOV punctuation exactly like the reference custom_trans:
//   ; -> ,   “ ” -> "   ‘ ’ -> '
std::string custom_trans(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    size_t i = 0;
    while (i < in.size()) {
        int n = 1;
        uint32_t cp = utf8_decode(in, i, n);
        if (cp == ';')
            out += ',';
        else if (cp == 0x201C || cp == 0x201D)
            out += '"';
        else if (cp == 0x2018 || cp == 0x2019)
            out += '\'';
        else
            out.append(in, i, n);
        i += n;
    }
    return out;
}

} // namespace

namespace core_pinyin {

bool has_han(const std::string& utf8) {
    size_t i = 0;
    while (i < utf8.size()) {
        int n = 1;
        uint32_t cp = utf8_decode(utf8, i, n);
        if (is_han(cp))
            return true;
        i += n;
    }
    return false;
}

std::vector<std::string> convert_char_to_pinyin(const std::string& utf8_in) {
    const std::string text = custom_trans(utf8_in);
    const auto& t = tables();

    // Decode to codepoints (with byte spans for non-Han passthrough).
    std::vector<uint32_t> cps;
    std::vector<std::string> raw; // the UTF-8 bytes of each codepoint
    {
        size_t i = 0;
        while (i < text.size()) {
            int n = 1;
            uint32_t cp = utf8_decode(text, i, n);
            cps.push_back(cp);
            raw.emplace_back(text, i, n);
            i += n;
        }
    }
    const int N = (int)cps.size();

    // Per-Han-char syllable, computed by segmenting Han runs (forward max-match
    // over the phrase table) and applying tone-sandhi per segment.
    std::vector<std::string> syl(N);
    for (int i = 0; i < N;) {
        if (!is_han(cps[i])) {
            ++i;
            continue;
        }
        // Extent of this Han run.
        int j = i;
        while (j < N && is_han(cps[j]))
            ++j;
        // Segment [i, j) by longest phrase match, else single char.
        int p = i;
        while (p < j) {
            int maxL = std::min(t.max_phrase_chars, j - p);
            bool matched = false;
            for (int L = maxL; L >= 2 && !matched; --L) {
                std::string key;
                for (int k = 0; k < L; ++k)
                    key += raw[p + k];
                auto it = t.ph.find(key);
                if (it == t.ph.end())
                    continue;
                std::vector<std::string> word_syl = split_ws(it->second);
                if ((int)word_syl.size() != L)
                    continue; // guard: reading count must match char count
                std::vector<uint32_t> word_han(cps.begin() + p, cps.begin() + p + L);
                apply_sandhi(word_han, word_syl);
                for (int k = 0; k < L; ++k)
                    syl[p + k] = word_syl[k];
                p += L;
                matched = true;
            }
            if (!matched) {
                std::vector<std::string> one = {char_pinyin(cps[p])};
                std::vector<uint32_t> oneh = {cps[p]};
                apply_sandhi(oneh, one);
                syl[p] = one[0];
                ++p;
            }
        }
        i = j;
    }

    // Emit tokens in the reference's element structure.
    std::vector<std::string> out;
    out.reserve(N * 2);
    for (int i = 0; i < N; ++i) {
        uint32_t cp = cps[i];
        if (is_han(cp)) {
            out.emplace_back(" ");
            out.push_back(syl[i]);
        } else if (cp < 256) {
            out.push_back(raw[i]); // single ASCII/Latin-1 byte(s) as-is
        } else {
            out.push_back(raw[i]); // other (CJK punctuation, etc.) as-is
        }
    }
    return out;
}

} // namespace core_pinyin

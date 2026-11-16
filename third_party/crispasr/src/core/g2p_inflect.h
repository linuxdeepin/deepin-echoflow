// core/g2p_inflect.h — pronounce a regular inflection from its stem.
//
// A pronunciation lexicon that stores STEMS cannot pronounce "believes",
// "abdicated" or "airbases" — and misaki's does exactly that: only 46% of the
// inflected forms in a 1508-word corpus appear in it verbatim, against 100% in
// CMUdict. Those misses used to fall through to CMUdict, inheriting its vowels
// and losing the agreement with Kokoro's training data that the lexicon exists
// to provide (#316). Recovering them by rule took whole-word phoneme agreement
// with misaki from 88.5% to 98.2%.
//
// The rules are the regular English ones, and they are audible:
//
//   -s / -'s   after a sibilant  (s z ʃ ʒ ʧ ʤ)   →  <reduced>z   "airbases"
//              after voiceless   (p t k f θ)     →  s            "cats"
//              otherwise                          →  z            "believes"
//   -ed        after t or d                       →  <reduced>d   "abdicated"
//              after voiceless                    →  t            "walked"
//              otherwise                          →  d            "believed"
//   -ing                                          →  ɪŋ
//   -ly                                           →  li
//
// …plus one thing a naive concatenation gets wrong: attaching a suffix can put
// a /t/ between vowels, where American English FLAPS it. "abdicate" + "ᵻd" is
// `ˈæbdəkˌAtᵻd`, but misaki says `ˈæbdəkˌATᵻd`. Flapping has to run after
// attachment, not before.
//
// DIALECT-PARAMETERISED because the symbols differ (see core/phoneme_dialect.h):
// misaki writes the reduced vowel `ᵻ` and the flap `T`; espeak writes `ɪ` and
// `ɾ`. The rules are identical; only the alphabet changes.
//
// Weight-free and header-only; tests/test-kokoro-g2p-316.cpp pins it.

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace core_g2p_inflect {

struct Params {
    // The reduced vowel inserted before z/d ("ᵻ" for misaki, "ɪ" for espeak).
    const char* reduced_vowel = "ɪ";
    // The alveolar flap ("T" for misaki, "ɾ" for espeak). Empty disables
    // post-attachment flapping.
    const char* flap = "ɾ";
};

namespace detail {

// Last UTF-8 codepoint of `s`, as a string (phoneme symbols are multi-byte).
inline std::string last_cp(const std::string& s) {
    if (s.empty())
        return "";
    size_t i = s.size() - 1;
    while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80)
        i--;
    return s.substr(i);
}

inline bool is_one_of(const std::string& cp, const std::vector<std::string>& set) {
    for (const auto& x : set)
        if (cp == x)
            return true;
    return false;
}

inline const std::vector<std::string>& sibilants() {
    static const std::vector<std::string> v = {"s", "z", "ʃ", "ʒ", "ʧ", "ʤ", "tʃ", "dʒ"};
    return v;
}
inline const std::vector<std::string>& voiceless() {
    static const std::vector<std::string> v = {"p", "t", "k", "f", "θ"};
    return v;
}
inline const std::vector<std::string>& vowels() {
    static const std::vector<std::string> v = {"ɑ", "æ", "ɛ", "ɪ", "i", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ",
                                               "A", "I", "O", "W", "Y", "ᵊ", "ᵻ", "ɚ", "e", "a", "ɐ"};
    return v;
}

// American flapping: /t/ between a vowel and an unstressed vowel. Runs AFTER
// the suffix is attached, which is the whole point — "abdicate"+"ᵻd" only
// creates the intervocalic context once joined.
inline std::string flap_intervocalic(const std::string& s, const char* flap) {
    if (!flap || !*flap)
        return s;
    std::string out;
    out.reserve(s.size());
    std::vector<std::pair<size_t, std::string>> cps;
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        const unsigned char c = (unsigned char)s[i];
        if ((c & 0xF8) == 0xF0)
            n = 4;
        else if ((c & 0xF0) == 0xE0)
            n = 3;
        else if ((c & 0xE0) == 0xC0)
            n = 2;
        cps.push_back({i, s.substr(i, n)});
        i += n;
    }
    for (size_t k = 0; k < cps.size(); k++) {
        const std::string& cp = cps[k].second;
        const bool interv = cp == "t" && k > 0 && k + 1 < cps.size() && is_one_of(cps[k - 1].second, vowels()) &&
                            is_one_of(cps[k + 1].second, vowels());
        out += interv ? std::string(flap) : cp;
    }
    return out;
}

} // namespace detail

// Attach a plural / 3rd-person / possessive -s to a stem pronunciation.
inline std::string add_s(const std::string& stem, const Params& p) {
    const std::string l = detail::last_cp(stem);
    if (detail::is_one_of(l, detail::sibilants()))
        return stem + p.reduced_vowel + "z";
    if (detail::is_one_of(l, detail::voiceless()))
        return stem + "s";
    return stem + "z";
}

// Attach a past-tense / participle -ed.
inline std::string add_ed(const std::string& stem, const Params& p) {
    const std::string l = detail::last_cp(stem);
    if (l == "t" || l == "d" || (p.flap && l == p.flap))
        return stem + p.reduced_vowel + "d";
    if (detail::is_one_of(l, detail::voiceless()) || l == "s" || l == "ʃ" || l == "ʧ")
        return stem + "t";
    return stem + "d";
}

// Candidate orthographic stems for `word` minus `suffix`, in preference order:
// bare stem, +e (abdicate), un-doubled (running→run), i→y (carries→carry).
inline std::vector<std::string> stem_candidates(const std::string& word, const std::string& suffix) {
    std::vector<std::string> out;
    if (word.size() <= suffix.size() + 2)
        return out;
    const std::string base = word.substr(0, word.size() - suffix.size());
    out.push_back(base);
    out.push_back(base + "e");
    if (base.size() > 2 && base[base.size() - 1] == base[base.size() - 2])
        out.push_back(base.substr(0, base.size() - 1));
    if (!base.empty() && base.back() == 'i')
        out.push_back(base.substr(0, base.size() - 1) + "y");
    return out;
}

// Pronounce `word` from its stem. `lookup` returns the stem's phonemes, or ""
// when the stem is unknown. Returns "" when no rule applies — the caller then
// continues to its next G2P tier.
inline std::string inflect(const std::string& word, const std::function<std::string(const std::string&)>& lookup,
                           const Params& p) {
    struct Rule {
        const char* suffix;
        int kind; // 0 = -s, 1 = -ed, 2 = -ing, 3 = -ly
    };
    // Longest suffix first so "'s" and "es" win over "s".
    static const Rule rules[] = {{"'s", 0}, {"es", 0}, {"ed", 1}, {"ing", 2}, {"ly", 3}, {"s", 0}, {"d", 1}};
    for (const auto& r : rules) {
        const std::string suf = r.suffix;
        if (word.size() <= suf.size() + 2 || word.compare(word.size() - suf.size(), suf.size(), suf) != 0)
            continue;
        for (const auto& cand : stem_candidates(word, suf)) {
            const std::string stem = lookup(cand);
            if (stem.empty())
                continue;
            std::string joined;
            switch (r.kind) {
            case 0:
                joined = add_s(stem, p);
                break;
            case 1:
                joined = add_ed(stem, p);
                break;
            case 2:
                joined = stem + "ɪŋ";
                break;
            default:
                joined = stem + "li";
                break;
            }
            return detail::flap_intervocalic(joined, p.flap);
        }
    }
    return "";
}

} // namespace core_g2p_inflect

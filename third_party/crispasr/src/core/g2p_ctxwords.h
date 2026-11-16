// core/g2p_ctxwords.h — English function words whose pronunciation depends on
// what comes NEXT.
//
// A pronunciation lexicon can only store one form per word, so it stores the
// citation form: "the"→`ði`, "to"→`tu`, "a"→`A`. In running speech those reduce,
// and which reduction you get depends on whether the FOLLOWING word starts with
// a vowel:
//
//     the apple   ði ˈæpᵊl        the box   ðə bˈɑks
//     to open     tʊ ˈOpᵊn        to walk   tə wˈɔk
//     a cat       ɐ kˈæt          (the article is never `A` = "eɪ")
//
// Reading the citation form aloud in every position is exactly the "old English"
// diction reported in #316 — "ðiː bɑks" instead of "ðə bɑks". Measured on a
// 400-sentence corpus of real prose, `the`/`to`/`a` alone accounted for 91 of
// 198 sampled word errors against misaki.
//
// The rules are misaki's own, read out of misaki/en.py `get_special_case`:
//
//     the   ->  ði  when a vowel follows, else ðə
//     to    ->  tʊ  when a vowel follows, tə before a consonant,
//               tu  in isolation (no following word)
//     a/an  ->  ɐ / ɐn as an article; `ˈA` only as the letter
//     in    ->  ɪn in running text; ˈɪn in isolation
//
// misaki also special-cases `am`, `by` and `used`, all of which need a
// part-of-speech tag to disambiguate ("by" as adverb vs preposition). We have no
// tagger, so those are deliberately left to the lexicon rather than guessed —
// see the follow-ups in PLAN.md.
//
// Weight-free and header-only.

#pragma once

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <vector>

namespace core_g2p_ctxwords {

// Does the next word begin with a vowel? Unknown means "no next word", which
// misaki treats as its own case rather than as `false`.
enum class NextVowel { Unknown, No, Yes };

inline bool starts_with_vowel(const std::string& ipa) {
    // First CODEPOINT, skipping any leading stress mark.
    size_t i = 0;
    while (i < ipa.size()) {
        if (ipa.compare(i, 2, "ˈ") == 0 || ipa.compare(i, 2, "ˌ") == 0) {
            i += 2;
            continue;
        }
        break;
    }
    if (i >= ipa.size())
        return false;
    static const char* kVowels[] = {"ɑ", "æ", "ɛ", "ɪ", "i", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ",
                                    "A", "I", "O", "W", "Y", "ᵊ", "ᵻ", "ɚ", "e", "a", "ɐ"};
    for (const char* v : kVowels) {
        const size_t n = std::char_traits<char>::length(v);
        if (ipa.compare(i, n, v) == 0)
            return true;
    }
    return false;
}

// Returns the contextual pronunciation for `word`, or "" when the word is not
// context-dependent and the lexicon's entry should stand.
//
// `word` must already be lowercased. `lexical` is what the lexicon returned,
// used for the isolation case where misaki falls back to it.
inline std::string lookup(const std::string& word, NextVowel next, const std::string& lexical) {
    if (word == "the")
        return next == NextVowel::Yes ? "ði" : "ðə";
    if (word == "to") {
        if (next == NextVowel::Unknown)
            return lexical.empty() ? "tu" : lexical;
        return next == NextVowel::Yes ? "tʊ" : "tə";
    }
    if (word == "a")
        return "ɐ"; // the article; the LETTER "A" is `ˈA`, but a lone lowercase
                    // "a" in running text is the determiner in ~every case
    if (word == "an")
        return "ɐn";
    if (word == "in")
        return next == NextVowel::Unknown ? "ˈɪn" : "ɪn";
    if (word == "am")
        return next == NextVowel::Unknown ? "ˈæm" : "ɐm";
    if (word == "i")
        return "ˌI"; // misaki special-cases the pronoun to SECONDARY stress
    return "";
}

// ── capitalisation stress ───────────────────────────────────────────────────
//
// misaki derives a token's stress from its CAPITALISATION, not from a
// part-of-speech tag (Lexicon.__call__):
//
//     lowercase   ->  None  — the lexicon form stands
//     Titlecase   ->  0.5   — insert SECONDARY stress if the form has none
//     ALLCAPS     ->  2     — insert PRIMARY stress if the form has none
//
// That single rule is what produced "And"->`ˌænd`, "It"->`ˌɪt`, "She"->`ʃˌi`
// in the corpus — sentence-initial capitals, not grammar. It accounted for the
// large majority of the residual stress disagreements (#316), and it needs no
// tagger.
//
// The mark goes immediately BEFORE the first vowel, matching misaki's
// `restress`, which relocates a leading mark to just ahead of the vowel it
// belongs to: "she" is `ʃˌi`, not `ˌʃi`.
enum class Caps { Lower, Title, Upper };

inline Caps classify_caps(const std::string& word) {
    bool has_upper = false, has_lower = false;
    for (unsigned char c : word) {
        if (c >= 'A' && c <= 'Z')
            has_upper = true;
        else if (c >= 'a' && c <= 'z')
            has_lower = true;
    }
    if (!has_upper)
        return Caps::Lower;
    return has_lower ? Caps::Title : Caps::Upper;
}

inline bool has_stress_mark(const std::string& ps) {
    return ps.find("ˈ") != std::string::npos || ps.find("ˌ") != std::string::npos;
}

// Replace every occurrence of `from` with `to`.
inline std::string replace_all(std::string s, const char* from, const char* to) {
    const size_t nf = std::char_traits<char>::length(from);
    for (size_t p = s.find(from); p != std::string::npos; p = s.find(from, p + std::char_traits<char>::length(to)))
        s = s.substr(0, p) + to + s.substr(p + nf);
    return s;
}

// Insert `mark` directly before the first vowel of `ps`.
inline std::string insert_stress(const std::string& ps, const char* mark) {
    static const char* kVowels[] = {"ɑ", "æ", "ɛ", "ɪ", "i", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ",
                                    "A", "I", "O", "W", "Y", "ᵊ", "ᵻ", "ɚ", "e", "a", "ɐ"};
    for (size_t i = 0; i < ps.size();) {
        size_t n = 1;
        const unsigned char c = (unsigned char)ps[i];
        if ((c & 0xF8) == 0xF0)
            n = 4;
        else if ((c & 0xF0) == 0xE0)
            n = 3;
        else if ((c & 0xE0) == 0xC0)
            n = 2;
        for (const char* v : kVowels) {
            if (ps.compare(i, std::char_traits<char>::length(v), v) == 0)
                return ps.substr(0, i) + mark + ps.substr(i);
        }
        i += n;
    }
    return ps; // no vowel — misaki leaves these alone
}

// Apply the capitalisation rule to one already-phonemized word.
inline std::string apply_caps_stress(const std::string& word, const std::string& ps) {
    const Caps c = classify_caps(word);
    if (c == Caps::Lower || ps.empty())
        return ps;
    if (has_stress_mark(ps)) {
        // An existing mark wins — with one exception misaki spells out
        // (`apply_stress`, the `stress >= 1` branch): ALLCAPS demands PRIMARY
        // stress, so a form carrying only a secondary mark is promoted.
        if (c == Caps::Upper && ps.find("ˈ") == std::string::npos)
            return replace_all(ps, "ˌ", "ˈ");
        return ps;
    }
    return insert_stress(ps, c == Caps::Upper ? "ˈ" : "ˌ");
}

// ── de-stressing a compound's subtokens ─────────────────────────────────────
//
// misaki's `apply_stress(ps, -0.5)`: demote. A form that carries primary stress
// loses its secondary marks and its primary becomes secondary; a form with no
// primary is left alone.
inline std::string destress(const std::string& ps) {
    if (ps.find("ˈ") == std::string::npos)
        return ps;
    return replace_all(replace_all(ps, "ˌ", ""), "ˈ", "ˌ");
}

// misaki's `stress_weight` — the tie-break for which half of a compound keeps
// its primary stress. Diphthongs and affricates (one codepoint in misaki's
// alphabet) count double.
inline int stress_weight(const std::string& ps) {
    static const char* kHeavy[] = {"A", "I", "O", "Q", "W", "Y", "ʤ", "ʧ"};
    int w = 0;
    for (size_t i = 0; i < ps.size();) {
        size_t n = 1;
        const unsigned char c = (unsigned char)ps[i];
        if ((c & 0xF8) == 0xF0)
            n = 4;
        else if ((c & 0xF0) == 0xE0)
            n = 3;
        else if ((c & 0xE0) == 0xC0)
            n = 2;
        bool heavy = false;
        for (const char* h : kHeavy)
            if (ps.compare(i, std::char_traits<char>::length(h), h) == 0)
                heavy = true;
        w += heavy ? 2 : 1;
        i += n;
    }
    return w;
}

// ── spelling a word out letter by letter ────────────────────────────────────
//
// misaki's `get_NNP`: an out-of-lexicon ALLCAPS word, and any dotted acronym,
// is read as its letters — `U.S.A.` is `jˌuˈɛsˈA`, not whatever the
// letter-to-sound rules make of "u.s.a.". The stress pattern is the whole
// point: every letter is demoted to secondary and the LAST one is promoted
// back to primary, so the run reads as one word with one peak.
//
// `letter_ipa` returns the reading of a single lowercase letter ("a" -> "ˈA")
// or "" when the caller has no letters table, in which case this returns "" and
// the caller keeps its existing fallback chain.
inline std::string spell_out(const std::string& word,
                             const std::function<std::string(const std::string&)>& letter_ipa) {
    std::string ps;
    for (char c : word) {
        if (!isalpha((unsigned char)c))
            continue; // misaki drops the dots and keeps the letters
        const std::string one = letter_ipa(std::string(1, (char)tolower((unsigned char)c)));
        if (one.empty())
            return std::string(); // an unknown letter aborts the whole word
        ps += one;
    }
    if (ps.empty())
        return ps;
    // apply_stress(ps, 0): with a primary somewhere, drop every secondary and
    // demote the primaries.
    if (ps.find("ˈ") != std::string::npos)
        ps = replace_all(replace_all(ps, "ˌ", ""), "ˈ", "ˌ");
    // …then the LAST secondary becomes primary (misaki's rsplit/join).
    const size_t last = ps.rfind("ˌ");
    if (last != std::string::npos)
        ps = ps.substr(0, last) + "ˈ" + ps.substr(last + std::char_traits<char>::length("ˌ"));
    return ps;
}

// Is this token the dotted-acronym shape misaki spells out? Its rule
// (`get_special_case`): there is a dot INSIDE the word, everything that is not
// a dot is a letter, and the longest run between dots is under 3 characters —
// so `U.S.`, `e.g.` and `Ph.D.` qualify and `Mr.` (no interior dot) does not.
inline bool is_dotted_acronym(const std::string& word) {
    size_t first = word.find_first_not_of('.');
    size_t last = word.find_last_not_of('.');
    if (first == std::string::npos || word.find('.', first) > last)
        return false;
    size_t run = 0, longest = 0;
    for (char c : word) {
        if (c == '.') {
            longest = run > longest ? run : longest;
            run = 0;
            continue;
        }
        if (!isalpha((unsigned char)c))
            return false;
        run++;
    }
    longest = run > longest ? run : longest;
    return longest > 0 && longest < 3;
}

// Join the parts of a hyphenated compound the way misaki's `resolve_tokens`
// does: no space between them, and when more than half the parts carry primary
// stress the lightest half is demoted — so "high-contrast" is `hˌIkˈɑntɹˌæst`,
// one word with one primary stress, not two stressed words with a gap between.
inline std::string join_compound(std::vector<std::string> parts) {
    std::vector<size_t> voiced;
    for (size_t i = 0; i < parts.size(); i++)
        if (!parts[i].empty())
            voiced.push_back(i);
    size_t n_primary = 0;
    for (size_t i : voiced)
        if (parts[i].find("ˈ") != std::string::npos)
            n_primary++;
    // misaki: a single-CHARACTER first part ("e-mail", "x-ray") always yields to
    // the second; otherwise demote only when primary stress is in the majority.
    if (voiced.size() == 2 && parts[voiced[0]].size() <= 2) {
        parts[voiced[1]] = destress(parts[voiced[1]]);
    } else if (voiced.size() >= 2 && n_primary > (voiced.size() + 1) / 2) {
        // Sort by (has-primary, weight, index) and demote the lighter half —
        // `sorted(indices)[:len(indices)//2]` in misaki.
        std::vector<size_t> order = voiced;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const bool pa = parts[a].find("ˈ") != std::string::npos;
            const bool pb = parts[b].find("ˈ") != std::string::npos;
            if (pa != pb)
                return !pa && pb;
            const int wa = stress_weight(parts[a]), wb = stress_weight(parts[b]);
            if (wa != wb)
                return wa < wb;
            return a < b;
        });
        for (size_t k = 0; k < voiced.size() / 2; k++)
            parts[order[k]] = destress(parts[order[k]]);
    }
    std::string out;
    for (const auto& p : parts)
        out += p;
    return out;
}

} // namespace core_g2p_ctxwords

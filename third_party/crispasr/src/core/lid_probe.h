// lid_probe.h — scoring for probe-based spoken-language identification.
//
// Some ASR models perform no language detection but ACCEPT a language as a
// decoder prompt slot, and answer a wrong one fluently rather than failing
// (Cohere Transcribe is the case in hand). For those, the model is its own
// language detector: transcribe a short clip once per candidate language and
// keep the candidate whose output looks most like real speech in that language.
//
// The scoring below is the piece worth getting right, and it is pure string
// logic, so it lives here — weight-free and unit-tested against fixed strings
// (tests/test-lid-probe.cpp) rather than only through a 1 GB model.
//
// Ported from the approach in bakrianoo/cohereX (Apache-2.0), whose langid.py
// scores `len(text) * (1 + 3*agree) * diversity**2`. Two porting details that
// matter and are easy to get wrong in C++:
//   * Python's len() counts CODEPOINTS. Using byte length instead would hand
//     Arabic (2 bytes/char) and CJK (3) a 2-3x bonus over Latin purely from
//     UTF-8 encoding, which is exactly the comparison the score is making.
//   * `list(text)` for the no-space languages is codepoints too, not bytes.
// Both are why this file carries its own small UTF-8 splitter.

#ifndef CORE_LID_PROBE_H
#define CORE_LID_PROBE_H

#include <string>
#include <unordered_set>
#include <vector>

namespace core_lid_probe {

// Languages written without word spaces — token = character, not whitespace run.
inline bool is_no_space_language(const std::string& lang) {
    return lang == "zh" || lang == "ja" || lang == "ko" || lang == "th" || lang == "lo" || lang == "my";
}

// Split a UTF-8 string into codepoints. Malformed bytes become one-byte
// "codepoints" — this is a scoring heuristic, not a validator.
inline std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        size_t len = 1;
        if ((c & 0x80) == 0x00)
            len = 1;
        else if ((c & 0xE0) == 0xC0)
            len = 2;
        else if ((c & 0xF0) == 0xE0)
            len = 3;
        else if ((c & 0xF8) == 0xF0)
            len = 4;
        if (i + len > s.size())
            len = 1;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

// Codepoint count — the length term of the score (see the note above).
inline size_t utf8_length(const std::string& s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); i++)
        if (((unsigned char)s[i] & 0xC0) != 0x80)
            n++;
    return n;
}

// First `max_bytes` bytes, cut back to a codepoint boundary.
//
// For logging a probe transcript. printf's "%.60s" truncates at 60 BYTES and
// will happily slice a multi-byte character in half, putting invalid UTF-8 on
// stderr — which is not cosmetic: it crashed a Kaggle run outright, because
// Python's subprocess decodes stderr as UTF-8 and raised UnicodeDecodeError
// on the Greek candidate's severed lead byte (0xce). Any consumer that reads
// our stderr as text hits the same thing, and only for non-Latin languages.
inline std::string utf8_prefix(const std::string& s, size_t max_bytes) {
    if (s.size() <= max_bytes)
        return s;
    size_t cut = max_bytes;
    // A continuation byte is 10xxxxxx; walk back to the lead byte.
    while (cut > 0 && ((unsigned char)s[cut] & 0xC0) == 0x80)
        cut--;
    return s.substr(0, cut);
}

// Whitespace-separated tokens, for languages that use spaces.
inline std::vector<std::string> whitespace_tokens(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

// Fraction of DISTINCT tokens in the output. This is the hallucination
// detector: a model prompted with the wrong language does not fail, it
// produces fluent-looking but repetitive text ("the the the the"), so a low
// distinct-token ratio is the signal that the candidate is wrong. Returns
// 0.0 for empty text and 1.0 for text with no repetition.
inline double diversity(const std::string& text, const std::string& lang) {
    const std::vector<std::string> tokens = is_no_space_language(lang) ? utf8_chars(text) : whitespace_tokens(text);
    if (tokens.empty())
        return 0.0;
    const std::unordered_set<std::string> distinct(tokens.begin(), tokens.end());
    return (double)distinct.size() / (double)tokens.size();
}

// Score one candidate language's probe transcript.
//
//   length            — a candidate the model cannot transcribe emits little.
//   (1 + 3 * agree)   — a TEXT language detector run on the output, agreeing
//                       that it is in fact `lang`, is the strongest single
//                       signal; `agree` is its probability in [0,1], and 0
//                       when no text LID is available (the score still works,
//                       just more weakly among same-script languages).
//   diversity^2       — squared so repetition is punished hard: a hallucinated
//                       wrong-language decode scores near zero even when long.
//
// Higher is better; the caller takes the argmax over candidates.
//
// Split from score() so a test can feed the three measured quantities
// directly — a probe log records (length, agree, diversity), not the full
// transcript, and reconstructing a string with a given diversity to test the
// formula would be testing the reconstruction.
inline double score_from(size_t length, double agree, double div) {
    if (agree < 0.0)
        agree = 0.0;
    if (agree > 1.0)
        agree = 1.0;
    return (double)length * (1.0 + 3.0 * agree) * div * div;
}

inline double score(const std::string& text, const std::string& lang, double agree) {
    return score_from(utf8_length(text), agree, diversity(text, lang));
}

} // namespace core_lid_probe

#endif // CORE_LID_PROBE_H

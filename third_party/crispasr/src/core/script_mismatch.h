// core/script_mismatch.h — detect wrong-script ASR output (#419).
//
// canary-1b-v2 conditioned on the WRONG language renders speech in the wrong
// SCRIPT: Russian audio decoded with <|en|><|en|> comes out as clean Latin
// transliteration ("vikingi, otvazhnye voyny" instead of "викинги, отважные
// воины") — correct words, unusable text, and nothing in the pipeline said a
// word. Reproduced exactly with `-l en` on Russian audio; with `-l ru` every
// backend/config combination produced proper Cyrillic (CPU, CUDA P100,
// Vulkan/llvmpipe, streamed, sub-second VAD-sized slices — the #419 matrix).
// So when the transcript's script contradicts the requested target language,
// the language conditioning did not arrive (a frontend not passing `-l`
// through, an LID fallback to 'en', a wrong model file) and the user should
// hear about it instead of shipping translit subtitles.
//
// Header-only and string-pure so tests/test-script-mismatch.cpp can pin the
// decision table without a model.

#pragma once

#include <cstdint>
#include <string>

namespace core_script {

enum class Script {
    Latin,
    Cyrillic,
    Greek,
    Unknown, // language whose script we don't classify — never warns
};

// Languages canary-1b-v2 supports whose native script is NOT Latin. Everything
// else returns Unknown (Latin-script languages are left alone: English loanwords
// and code-switching make a "too much Cyrillic in French" warning noise).
inline Script expected_for_lang(const std::string& lang) {
    if (lang == "ru" || lang == "uk" || lang == "bg")
        return Script::Cyrillic;
    if (lang == "el")
        return Script::Greek;
    return Script::Unknown;
}

struct ScriptCounts {
    int latin = 0;
    int cyrillic = 0;
    int greek = 0;
};

// Count letters per script in UTF-8 text. Only unambiguous letter ranges are
// counted; digits/punctuation/whitespace are ignored.
inline ScriptCounts count_scripts(const std::string& utf8) {
    ScriptCounts c;
    const unsigned char* s = (const unsigned char*)utf8.data();
    size_t n = utf8.size();
    for (size_t i = 0; i < n;) {
        uint32_t cp = 0;
        int len = 1;
        if (s[i] < 0x80) {
            cp = s[i];
        } else if ((s[i] >> 5) == 0x6 && i + 1 < n) {
            cp = ((uint32_t)(s[i] & 0x1F) << 6) | (s[i + 1] & 0x3F);
            len = 2;
        } else if ((s[i] >> 4) == 0xE && i + 2 < n) {
            cp = ((uint32_t)(s[i] & 0x0F) << 12) | ((uint32_t)(s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
            len = 3;
        } else if ((s[i] >> 3) == 0x1E && i + 3 < n) {
            len = 4; // outside all ranges we count
        }
        i += len;
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z'))
            c.latin++;
        else if (cp >= 0x0400 && cp <= 0x04FF)
            c.cyrillic++;
        else if ((cp >= 0x0370 && cp <= 0x03FF) && cp != 0x0387 && cp != 0x0374 && cp != 0x0375)
            c.greek++;
    }
    return c;
}

// True when `text` contradicts the script `lang` should produce. Conservative:
// needs >= 20 letters of evidence, and fires only when the expected script
// holds less than a quarter of the (expected + Latin) letter mass — normal
// code-switched fragments ("iPhone", an English name) never trip it.
inline bool mismatch(const std::string& lang, const std::string& text, ScriptCounts* out_counts = nullptr) {
    const Script want = expected_for_lang(lang);
    if (want == Script::Unknown)
        return false;
    const ScriptCounts c = count_scripts(text);
    if (out_counts)
        *out_counts = c;
    const int expected = want == Script::Cyrillic ? c.cyrillic : c.greek;
    const int total = expected + c.latin;
    if (total < 20)
        return false;
    return expected * 4 < total;
}

} // namespace core_script

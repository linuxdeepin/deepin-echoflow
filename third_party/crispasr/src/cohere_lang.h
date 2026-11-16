// cohere_lang.h — language-code resolution for Cohere Transcribe.
//
// Weight-free and header-only on purpose: the decision "which language token
// goes into the decoder prompt" is pure string logic, so it is unit-tested
// hermetically (tests/test-cohere-lang.cpp) instead of only through a 1 GB
// GGUF. See the btc_chord_vocab.h precedent.
//
// Why this exists at all: Cohere Transcribe supports a FIXED language set
// (`supported_languages` in config.json — 14 codes for the base model, but
// only {en, ar} for the Arabic finetune). The model performs no language
// detection and does not fail on a wrong language: it transcribes fluently in
// whatever language the prompt names, so an unsupported code is never a
// decode-time error, just a quietly different transcript.
//
// The config list is the ONLY signal available. Measured on the published
// Arabic GGUF: its tokenizer carries 183 `<|xx|>` tokens — the whole of ISO
// 639-1 — while the model supports two. So `<|ru|>`, `<|de|>`, `<|ja|>` all
// exist, the prompt is well-formed, and a vocab-membership check catches
// nothing at all. Observed on one 8 s Arabic clip: `-l ru` added a
// hallucinated leading word, `-l ja` swapped the quotation marks, `-l de`
// changed the diacritics — every one of them silent and plausible.
//
// (The vocab check IS still worth keeping as a backstop, but only for
// non-ISO input: `<|auto|>` is genuinely absent, and used to be deleted from
// the prompt outright, leaving the decoder with no language slot.)

#ifndef COHERE_LANG_H
#define COHERE_LANG_H

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace cohere_lang {

// Lowercase, trim, and drop any region / script suffix: "en-US" -> "en",
// "zh_Hans" -> "zh". Cohere's prompt tokens are bare ISO-639-1 (`<|en|>`).
inline std::string normalize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '-' || c == '_')
            break;
        if (std::isspace((unsigned char)c))
            continue;
        out += (char)std::tolower((unsigned char)c);
    }
    return out;
}

inline bool contains(const std::vector<std::string>& langs, const std::string& code) {
    return std::find(langs.begin(), langs.end(), code) != langs.end();
}

struct Resolution {
    std::string lang;   // the code to actually put in the decoder prompt
    bool substituted;   // true when `requested` was unusable and `lang` is a fallback
    std::string reason; // human-readable explanation; empty when !substituted
};

// Pick the language token for the decoder prompt.
//
// `supported` is the model's own list, read from the GGUF key
// `cohere_transcribe.supported_languages`. An EMPTY list means "unknown" — a
// GGUF converted before that key existed — in which case the request passes
// through untouched and the caller falls back to its vocab-membership check.
//
// The fallback is "en" when the model supports it (true for every Cohere
// Transcribe release so far), otherwise the model's first listed language.
inline Resolution resolve(const std::vector<std::string>& supported, const std::string& requested) {
    Resolution r;
    r.substituted = false;
    r.lang = normalize(requested);

    if (supported.empty())
        return r; // unknown set — caller validates against the vocab instead

    if (!r.lang.empty() && contains(supported, r.lang))
        return r;

    const std::string fallback = contains(supported, "en") ? std::string("en") : supported.front();

    std::string list;
    for (size_t i = 0; i < supported.size(); i++) {
        if (i)
            list += ", ";
        list += supported[i];
    }

    r.reason =
        (r.lang.empty() || r.lang == "auto")
            ? ("no language was resolved; this model does not detect one. Using '" + fallback + "'. Supported: " + list)
            : ("language '" + r.lang + "' is not supported by this model — using '" + fallback +
               "' instead. Supported: " + list);
    r.lang = fallback;
    r.substituted = true;
    return r;
}

} // namespace cohere_lang

#endif // COHERE_LANG_H

#pragma once

// OmniVoice target-language resolution.
//
// The OmniVoice prompt carries the language as a literal string between
// `<|lang_start|>` and `<|lang_end|>`, so what goes in there has to be a token
// sequence the model saw in training: the ISO 639-3 IDs from
// `omnivoice/utils/lang_map.py`. Anything else conditions the model on noise
// while still looking like it worked.
//
// `resolve()` mirrors `_resolve_language()` in the blueprint
// (`omnivoice/models/omnivoice.py`) exactly:
//
//     if language is None or language.lower() == "none": return None
//     if language in LANG_IDS:                           return language
//     key = language.lower()
//     if key in LANG_NAME_TO_ID:                         return LANG_NAME_TO_ID[key]
//     warn(...); return None
//
// Two details of that order are load-bearing and deliberately reproduced:
//   * the ID check is CASE-SENSITIVE, so "DE" is not an ID, is not a name, and
//     falls through to language-agnostic mode. `suggest()` exists so the caller
//     can say "did you mean 'de'?" without deviating from the blueprint.
//   * an unrecognized value resolves to None (language-agnostic), never to
//     itself — passing it through is what silently poisons the prompt.
//
// One addition on top of the blueprint: "auto" clears, because that is
// CrispASR's own sentinel for `-l` and the server forwards it verbatim.
// Upstream would reach the same None by the unrecognized path — this only
// suppresses a warning that would be noise, never changes the prompt.
//
// Weight-free and header-only so it is unit-testable without a model
// (tests/test-omnivoice-lang.cpp).

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "omnivoice_lang_table.h"
#include "tts_lang.h" // core_tts_lang::detect — the auto_detect() fallback below

namespace core_omnivoice_lang {

enum class Status {
    cleared,        // "" / "none" — language-agnostic on purpose
    id_passthrough, // already a valid ISO 639-3 ID
    name_resolved,  // English name mapped to its ID
    unrecognized,   // not an ID and not a name — falls back to language-agnostic
};

struct Resolved {
    std::string id; // empty means "None" in the prompt
    Status status = Status::cleared;
};

namespace detail {

struct Index {
    std::unordered_map<std::string, std::string> name_to_id;
    std::unordered_set<std::string> ids;
};

inline const Index& index() {
    static const Index idx = [] {
        Index built;
        built.name_to_id.reserve(kLangTableN);
        built.ids.reserve(kLangTableN);
        for (int i = 0; i < kLangTableN; i++) {
            built.name_to_id.emplace(kLangTable[i].name, kLangTable[i].id);
            built.ids.emplace(kLangTable[i].id);
        }
        return built;
    }();
    return idx;
}

inline std::string ascii_lower(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z')
            c = (char)(c + 32);
    }
    return out;
}

} // namespace detail

// True when `s` is one of the ISO 639-3 IDs the model was trained on. Exact
// match — the blueprint's `language in LANG_IDS` is case-sensitive.
inline bool is_valid_id(const std::string& s) {
    return detail::index().ids.count(s) > 0;
}

inline bool is_valid_name(const std::string& lowercase_name) {
    return detail::index().name_to_id.count(lowercase_name) > 0;
}

inline Resolved resolve(const std::string& language) {
    if (language.empty())
        return {std::string(), Status::cleared};

    const std::string low = detail::ascii_lower(language);
    if (low == "none" || low == "auto")
        return {std::string(), Status::cleared};

    const auto& idx = detail::index();
    if (idx.ids.count(language) > 0)
        return {language, Status::id_passthrough};

    auto it = idx.name_to_id.find(low);
    if (it != idx.name_to_id.end())
        return {it->second, Status::name_resolved};

    return {std::string(), Status::unrecognized};
}

// Best-effort "did you mean" for an unrecognized value, so the warning can be
// actionable. Covers the three ways a caller gets this wrong in practice:
// wrong case ("DE"), a BCP-47 tag ("de-DE", "en_US"), and a capitalised name
// ("German" is already handled by resolve(), but "GERMAN" is not). Returns ""
// when nothing plausible is nearby.
inline std::string suggest(const std::string& language) {
    const std::string low = detail::ascii_lower(language);
    if (is_valid_id(low))
        return low;

    const size_t cut = low.find_first_of("-_");
    if (cut != std::string::npos && cut > 0) {
        const std::string base = low.substr(0, cut);
        if (is_valid_id(base))
            return base;
    }

    auto it = detail::index().name_to_id.find(low);
    if (it != detail::index().name_to_id.end())
        return it->second;

    return std::string();
}

// Last-resort fallback for when NOBODY supplied a language: guess one from the
// text about to be spoken, and use it only if it lands on an id the model
// actually knows. Returns "" whenever it is not confident — "" is the status
// quo (language-agnostic), so a silent no-answer is always safe.
//
// Why guessing is acceptable here, when it usually is not: measured on an
// English reference cloned onto German text, a *wrong* tag (`-l en`) costs
// nothing detectable — the ASR round-trip is word-perfect and whisper still
// LIDs the output as German at 0.98, the same band as the correct tag and no
// tag. So the downside of a bad guess is bounded, while the upside is the one
// upstream documents ("performance is slightly better if you specify the
// language"). That asymmetry is the whole argument; if a future measurement
// shows a wrong tag DOES degrade output, this fallback should go back to
// opt-in, because its failure mode is the thing that justifies it.
//
// `core_tts_lang::detect` is deliberately conservative: script detection first
// (Hangul/Kana/Han/Cyrillic), then a Latin function-word detector that needs
// several words of evidence and returns "" when thin. Short subtitle lines
// therefore mostly land on "" and behave exactly as they do today.
//
// ⚠ Pass the TARGET text only. Passing the combined stream (reference
// transcript + target) would let an English reference clip drag the guess to
// English on German subtitles — the exact failure this exists to avoid.
inline std::string auto_detect(const std::string& target_text) {
    const std::string guess = core_tts_lang::detect(target_text);
    if (guess.empty())
        return std::string();

    // The detector speaks ISO-639-1-ish 2-letter codes; the model speaks its
    // own 639-3 id set. Most common codes are in both, but not all (there is
    // no "ar"), and a code the model never saw is worse than no tag.
    const Resolved r = resolve(guess);
    if (r.status == Status::id_passthrough || r.status == Status::name_resolved)
        return r.id;
    return std::string();
}

} // namespace core_omnivoice_lang

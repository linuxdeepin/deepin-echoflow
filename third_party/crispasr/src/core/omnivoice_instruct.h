#pragma once

// OmniVoice voice-design instruct resolution.
//
// The sibling of core/omnivoice_lang.h, and the same trap: the instruct string
// goes into the prompt LITERALLY, between `<|instruct_start|>` and
// `<|instruct_end|>`, and the model was trained on a CLOSED 48-item vocabulary
// in a fixed spelling. Feeding it anything else is silent — the graph computes,
// the audio is speech, the voice design just quietly does something other than
// what was asked. The measurement that makes this concrete:
//
//     'Male, British Accent'  -> [151672, 36421, 11,  7855, 81809, 151673]
//     'male, british accent'  -> [151672, 36476, 11, 93927, 29100, 151673]
//
// Not one shared token id, for a string a user would consider the same request.
//
// Mirrors `_resolve_instruct()` (omnivoice/models/omnivoice.py), which:
//   1. splits on half- OR full-width commas, with surrounding whitespace
//   2. lowercases each item and REJECTS anything outside the vocabulary,
//      offering a close match
//   3. rejects mixing a Chinese dialect with an English accent
//   4. unifies every item to one language — a dialect forces Chinese, an accent
//      forces English, otherwise it follows whether the TARGET TEXT is Chinese
//   5. rejects two items from one category (gender, age, pitch, style, accent,
//      dialect)
//   6. joins with "，" if the result is Chinese, else ", "
//
// Upstream RAISES on 2/3/5 rather than degrading, and so do we: a voice-design
// request that silently does nothing is the exact failure mode this file exists
// to remove.
//
// ⚠ Step 4 depends on the text being synthesised, so resolution is TWO-PHASE:
// `parse()` does everything text-independent (validation, the conflict checks)
// and can run when the instruct is set; `render()` needs the text and must run
// per synthesis. Collapsing them would either validate too late or bake one
// line's language choice into every later line on a reused server context.
//
// Weight-free and header-only; tests/test-omnivoice-instruct.cpp.

#include <string>
#include <unordered_map>
#include <vector>

#include "omnivoice_instruct_table.h"

namespace core_omnivoice_instruct {

enum class Status {
    cleared,              // empty/whitespace — no voice design, same as upstream's None
    ok,                   // every item valid and consistent
    unknown_item,         // not in the 48-item vocabulary
    mixed_dialect_accent, // a Chinese dialect and an English accent together
    category_conflict,    // two items from one category
};

struct Parsed {
    Status status = Status::cleared;
    std::vector<std::string> items; // validated + lowercased, in input order
    std::string error;              // populated on every non-ok, non-cleared status
    bool has_dialect = false;
    bool has_accent = false;
};

namespace detail {

struct Row {
    const char* counterpart;
    int group;
    bool is_zh;
};

inline const std::unordered_map<std::string, Row>& index() {
    static const std::unordered_map<std::string, Row> idx = [] {
        std::unordered_map<std::string, Row> m;
        m.reserve(kInstructTableN);
        for (int i = 0; i < kInstructTableN; i++) {
            const InstructItem& e = kInstructTable[i];
            m.emplace(e.item, Row{e.counterpart, e.group, e.is_zh});
        }
        return m;
    }();
    return idx;
}

// ASCII-only lowercase. UTF-8 safe by construction: continuation bytes all have
// the high bit set and can never fall in 'A'..'Z', so Chinese items pass through
// untouched — which is what Python's .lower() does to them too.
inline std::string ascii_lower(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z')
            c = (char)(c + 32);
    }
    return out;
}

// Exactly Python's `[一-鿿]`, decoded properly rather than sniffed from
// the lead byte. The lead-byte shortcut (0xE4..0xE9) looks equivalent — those
// are the leads of U+4000..U+9FFF — but it over-matches U+4000..U+4DFF by 3584
// codepoints (CJK Ext-A and the Yijing hexagrams). That window decides whether
// an instruct renders as "male, elderly" or "男，老年", so it has to be exact.
inline bool has_cjk(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= s.size())
                break; // truncated sequence — nothing more to classify
            const unsigned cp = ((unsigned)(c & 0x0F) << 12) | ((unsigned)((unsigned char)s[i + 1] & 0x3F) << 6) |
                                (unsigned)((unsigned char)s[i + 2] & 0x3F);
            if (cp >= 0x4E00 && cp <= 0x9FFF)
                return true;
            i += 3;
        } else {
            i += 4;
        }
    }
    return false;
}

inline bool ends_with(const std::string& s, const char* suffix) {
    const size_t n = std::char_traits<char>::length(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

inline std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return std::string();
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Split on ',' or the full-width '，' (U+FF0C = EF BC 8C), mirroring the
// blueprint's `re.split(r"\s*[,，]\s*", ...)`. Accepting the wrong separator for
// the language is deliberate upstream ("minor issues, auto-fixed").
inline std::vector<std::string> split_items(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < s.size();) {
        if (s[i] == ',') {
            out.push_back(trim(cur));
            cur.clear();
            i++;
        } else if (i + 2 < s.size() && (unsigned char)s[i] == 0xEF && (unsigned char)s[i + 1] == 0xBC &&
                   (unsigned char)s[i + 2] == 0x8C) {
            out.push_back(trim(cur));
            cur.clear();
            i += 3;
        } else {
            cur += s[i++];
        }
    }
    out.push_back(trim(cur));

    std::vector<std::string> kept;
    for (auto& x : out) {
        if (!x.empty())
            kept.push_back(x);
    }
    return kept;
}

// Best-effort "did you mean". The ACCEPT/REJECT decision and the normalised
// output are blueprint-exact; this string is not — upstream uses difflib's
// Ratcliff/Obershelp ratio and we use a cheaper LCS ratio at the same 0.6
// cutoff, so we may occasionally name a different near-miss. It changes no
// model input, only the error text.
inline double similarity(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty())
        return 0.0;
    std::vector<size_t> prev(b.size() + 1, 0), cur(b.size() + 1, 0);
    for (size_t i = 1; i <= a.size(); i++) {
        for (size_t j = 1; j <= b.size(); j++)
            cur[j] = (a[i - 1] == b[j - 1]) ? prev[j - 1] + 1 : std::max(prev[j], cur[j - 1]);
        prev = cur;
    }
    return 2.0 * (double)prev[b.size()] / (double)(a.size() + b.size());
}

inline std::string suggest(const std::string& bad) {
    std::string best;
    double best_score = 0.6; // upstream's difflib cutoff
    for (int i = 0; i < kInstructTableN; i++) {
        const double s = similarity(bad, kInstructTable[i].item);
        if (s > best_score) {
            best_score = s;
            best = kInstructTable[i].item;
        }
    }
    return best;
}

} // namespace detail

inline bool is_valid_item(const std::string& item) {
    return detail::index().count(item) > 0;
}

// True when the synthesis text contains Chinese — upstream's `_ZH_RE.search`,
// which is what decides the instruct language when nothing forces it.
inline bool text_is_zh(const std::string& text) {
    return detail::has_cjk(text);
}

// Everything that does not depend on the text being synthesised.
inline Parsed parse(const std::string& instruct) {
    Parsed p;
    const std::string trimmed = detail::trim(instruct);
    if (trimmed.empty())
        return p; // Status::cleared

    for (const std::string& raw : detail::split_items(trimmed)) {
        const std::string n = detail::ascii_lower(raw);
        if (!is_valid_item(n)) {
            p.status = Status::unknown_item;
            p.error = "unsupported instruct item '" + raw + "'";
            const std::string sug = detail::suggest(n);
            if (!sug.empty())
                p.error += " — did you mean '" + sug + "'?";
            p.error += ". Valid items name a gender, age, pitch, style, accent or "
                       "Chinese dialect (e.g. 'female, elderly, british accent').";
            return p;
        }
        p.items.push_back(n);
        if (detail::ends_with(n, "话"))
            p.has_dialect = true;
        if (n.find(" accent") != std::string::npos)
            p.has_accent = true;
    }

    if (p.has_dialect && p.has_accent) {
        p.status = Status::mixed_dialect_accent;
        p.error = "cannot mix a Chinese dialect and an English accent in one instruct — "
                  "dialects are for Chinese speech, accents for English.";
        return p;
    }

    // Conflicts are checked on the unified form upstream, but every
    // exclusivity group holds BOTH language forms of its items, so the answer
    // is the same before unification — and checking here keeps the whole
    // rejection path text-independent.
    for (int g = 0; g < kInstructGroups; g++) {
        std::vector<std::string> hits;
        for (const std::string& n : p.items) {
            auto it = detail::index().find(n);
            if (it != detail::index().end() && it->second.group == g)
                hits.push_back(n);
        }
        if (hits.size() > 1) {
            p.status = Status::category_conflict;
            p.error = "conflicting instruct items in one category: ";
            for (size_t i = 0; i < hits.size(); i++)
                p.error += (i ? " vs " : "") + std::string("'") + hits[i] + "'";
            p.error += ". Each category (gender, age, pitch, style, accent, dialect) "
                       "allows at most one item.";
            return p;
        }
    }

    // A string of nothing but separators ("," / "，") parses to zero items.
    // Upstream reaches the same place by a different route — it never raises,
    // joins an empty list to "", and the caller's `instruct if instruct else
    // "None"` turns that back into "None" — so report it as cleared rather than
    // as an ok-with-no-items that callers would have to special-case.
    p.status = p.items.empty() ? Status::cleared : Status::ok;
    return p;
}

// The text-dependent half: unify to one language, then join. `text_has_zh`
// only decides when neither a dialect nor an accent has already forced it.
inline std::string render(const Parsed& p, bool text_has_zh) {
    if (p.status != Status::ok || p.items.empty())
        return std::string();

    bool use_zh = text_has_zh;
    if (p.has_dialect)
        use_zh = true;
    else if (p.has_accent)
        use_zh = false;

    std::vector<std::string> unified;
    unified.reserve(p.items.size());
    for (const std::string& n : p.items) {
        auto it = detail::index().find(n);
        const bool translatable = it != detail::index().end() && it->second.counterpart != nullptr;
        // Map only when the item is in the language we are moving AWAY from —
        // upstream's dict .get(n, n), where the other language's items simply
        // are not keys.
        if (translatable && it->second.is_zh != use_zh)
            unified.push_back(it->second.counterpart);
        else
            unified.push_back(n);
    }

    bool any_zh = false;
    for (const std::string& n : unified)
        any_zh = any_zh || detail::has_cjk(n);

    const std::string sep = any_zh ? "，" : ", ";
    std::string out;
    for (size_t i = 0; i < unified.size(); i++)
        out += (i ? sep : "") + unified[i];
    return out;
}

} // namespace core_omnivoice_instruct

// core/currency_symbols.h — recognise currency symbols around a number.
//
// The #316 silent-drop class again, one layer out. Spelling digits out fixed
// "82" vanishing; the UNIT beside them was still disappearing:
//
//   "€50"  ->  "€ fünfzig"   ->  fʏnftsɪç          (the € is gone from the audio)
//   "50€"  ->  "fünfzig€"    ->  …and glued on, so the token stops matching
//                                 the pronunciation dictionary at all
//
// A currency symbol is in no pronunciation dictionary and no letter-to-sound
// rule, so it phonemizes to nothing exactly like a digit did. For a TTS used to
// dub subtitles this drops prices — the single most common number in the
// material.
//
// `num2words_en.h` already handled a leading `$`. This is that idea generalised:
// the SYMBOL detection (UTF-8 byte matching, prefix and postfix) is shared here
// because it is identical in every language, while the WORDS stay in each
// language's own file — they do not merely translate:
//
//   de   50 Euro     — invariable, never "Euros"; likewise Dollar, Pfund
//   fr   50 euros    — regular plural
//   es   50 dólares  — and the singular carries an accent, dólar
//   en   50 euros / 50 dollars
//
// Weight-free and header-only. Covered by each language's num2words test.
#pragma once

#include <cstddef>
#include <string>

namespace core_currency {

enum class sym { none, eur, usd, gbp, jpy };

namespace detail {

// UTF-8 byte sequences. € is 3 bytes, £ and ¥ are 2, $ is ASCII.
inline bool is_eur(const std::string& s, size_t i) {
    return i + 2 < s.size() && (unsigned char)s[i] == 0xE2 && (unsigned char)s[i + 1] == 0x82 &&
           (unsigned char)s[i + 2] == 0xAC;
}
inline bool is_gbp(const std::string& s, size_t i) {
    return i + 1 < s.size() && (unsigned char)s[i] == 0xC2 && (unsigned char)s[i + 1] == 0xA3;
}
inline bool is_jpy(const std::string& s, size_t i) {
    return i + 1 < s.size() && (unsigned char)s[i] == 0xC2 && (unsigned char)s[i + 1] == 0xA5;
}

} // namespace detail

// A symbol STARTING at `i` (the "50 €" / "50€" form). `len` receives its byte
// length so the caller can step over it.
inline sym at_postfix(const std::string& s, size_t i, size_t& len) {
    if (i >= s.size())
        return sym::none;
    if (detail::is_eur(s, i)) {
        len = 3;
        return sym::eur;
    }
    if (detail::is_gbp(s, i)) {
        len = 2;
        return sym::gbp;
    }
    if (detail::is_jpy(s, i)) {
        len = 2;
        return sym::jpy;
    }
    if (s[i] == '$') {
        len = 1;
        return sym::usd;
    }
    return sym::none;
}

// A symbol already emitted into `out` just before the number (the "€50" form).
// On a hit, erases it from `out` — including one separating space, so
// "es kostet € 50" does not leave a double space — and returns which it was.
inline sym take_prefix(std::string& out) {
    size_t end = out.size();
    while (end > 0 && out[end - 1] == ' ')
        end--;
    if (end == 0)
        return sym::none;

    size_t start = 0;
    sym found = sym::none;
    if (end >= 3 && detail::is_eur(out, end - 3)) {
        start = end - 3;
        found = sym::eur;
    } else if (end >= 2 && detail::is_gbp(out, end - 2)) {
        start = end - 2;
        found = sym::gbp;
    } else if (end >= 2 && detail::is_jpy(out, end - 2)) {
        start = end - 2;
        found = sym::jpy;
    } else if (out[end - 1] == '$') {
        start = end - 1;
        found = sym::usd;
    }
    if (found != sym::none)
        out.erase(start);
    return found;
}

} // namespace core_currency

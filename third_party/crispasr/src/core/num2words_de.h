// core/num2words_de.h — spell German numbers out before phonemization.
//
// The German/French/Spanish half of #316. `core/num2words_en.h` fixed English;
// `g2p_de` never had a number path at all, so a numeric token produced NO
// phonemes and vanished from the audio:
//
//   "Ich habe 82 Euro"  ->  ɪç hɑːbə  ɔʏ̯roː      (the 82 is simply gone)
//   "82"                ->  ""
//
// Nothing errors. German kokoro/piper silently drops every price, date and
// quantity. Reproduced on main before this file existed.
//
// German is not English with different words — four rules change the shape:
//
//   1. UNITS COME FIRST, joined by "und", as ONE word:
//        82   zweiundachtzig            (two-and-eighty)
//        21   einundzwanzig             ("ein", never "eins", inside a compound)
//      Getting the order wrong is audible and wrong, not merely accented.
//
//   2. STEM CHANGES that are not the obvious concatenation:
//        16   sechzehn   (not sechszehn)     60  sechzig
//        17   siebzehn   (not siebenzehn)    70  siebzig
//        30   dreißig    (ß, not dreissig/dreizig)
//
//   3. DECIMAL SEPARATORS ARE INVERTED vs English. In German text "1.000" is
//      one thousand and "3,14" is three point one four — the comma is the
//      decimal mark and the period groups thousands. Feeding German text to the
//      English expander reads "3,14" as two numbers and "1.000" as a decimal.
//      This is the trap most likely to produce plausible-but-wrong audio.
//
//   4. YEARS use the hundreds form only below 2000:
//        1984 neunzehnhundertvierundachtzig    2026 zweitausendsechsundzwanzig
//      German switched to the cardinal reading at the millennium, so the
//      English "twenty twenty six" pattern has no German equivalent.
//
// Weight-free and header-only, like its English sibling — tests/test-num2words-de.cpp
// covers every line above without a model, because a wrong entry here is
// invisible to every numeric check we have: the phonemes it produces are
// perfectly well-formed, just for the wrong word.
#pragma once

#include "core/currency_symbols.h" // #316: the unit beside the number

#include <cctype>
#include <cstdint>
#include <string>

namespace core_num2words_de {

namespace detail {

inline bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

// 0-19. Note "eins" here is the STANDALONE form; inside a compound German uses
// "ein" (einundzwanzig, einhundert), handled at each use site below.
inline const char* const kOnes[20] = {"null",     "eins",     "zwei",     "drei",     "vier",     "fünf",    "sechs",
                                      "sieben",   "acht",     "neun",     "zehn",     "elf",      "zwölf",   "dreizehn",
                                      "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"};

// Tens. dreißig carries the ß; sechzig/siebzig drop the stem consonants.
inline const char* const kTens[10] = {"",        "",        "zwanzig", "dreißig", "vierzig",
                                      "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"};

// 1-99 as the German compound. `standalone_one` picks "eins" vs "ein": a bare 1
// is "eins", but the 1 in 21 and the 1 in 100 are both "ein".
inline std::string under_hundred(int64_t n, bool standalone_one) {
    if (n < 20) {
        if (n == 1 && !standalone_one)
            return "ein";
        return kOnes[n];
    }
    const int64_t tens = n / 10, ones = n % 10;
    if (ones == 0)
        return kTens[tens];
    // units-first, joined by "und", written as one word
    const std::string u = (ones == 1) ? "ein" : kOnes[ones];
    return u + "und" + kTens[tens];
}

// 1-999.
inline std::string under_thousand(int64_t n, bool standalone_one) {
    if (n < 100)
        return under_hundred(n, standalone_one);
    const int64_t h = n / 100, rest = n % 100;
    std::string out = ((h == 1) ? std::string("ein") : std::string(kOnes[h])) + "hundert";
    if (rest)
        out += under_hundred(rest, /*standalone_one=*/true); // "einhunderteins"
    return out;
}

// Read a long digit run one digit at a time (phone numbers, hashes) rather than
// inventing a quadrillion — same guard as the English side.
inline std::string digits_each(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (!is_digit(c))
            continue;
        if (!out.empty())
            out += ' ';
        out += kOnes[c - '0'];
    }
    return out;
}

} // namespace detail

// Cardinal reading of a non-negative integer.
//
// Scale words are separate lexical words and inflect: "eine Million" /
// "zwei Millionen", against "eintausend" which is glued on. Emitted lowercase —
// the G2P lowercases before dictionary lookup, and casing carries no
// pronunciation.
inline std::string cardinal(int64_t n) {
    if (n == 0)
        return "null";
    std::string out;
    auto scale = [&out](int64_t count, const char* singular, const char* plural, bool feminine) {
        if (count == 0)
            return;
        if (!out.empty())
            out += ' ';
        if (count == 1)
            out += feminine ? std::string("eine ") + singular : std::string("ein ") + singular;
        else
            out += detail::under_thousand(count, /*standalone_one=*/true) + " " + plural;
    };

    const int64_t billions = n / 1000000000;
    const int64_t millions = (n / 1000000) % 1000;
    const int64_t thousands = (n / 1000) % 1000;
    const int64_t rest = n % 1000;

    scale(billions, "Milliarde", "Milliarden", /*feminine=*/true);
    scale(millions, "Million", "Millionen", /*feminine=*/true);
    if (thousands) {
        if (!out.empty())
            out += ' ';
        // "eintausend", not "ein tausend" — tausend glues, unlike Million.
        out += detail::under_thousand(thousands, /*standalone_one=*/false) + "tausend";
    }
    if (rest) {
        // Glue onto tausend/hundert ("eintausendfünf"), but keep a space after a
        // free-standing scale word ("zwei Millionen fünf").
        if (!out.empty() && thousands == 0)
            out += ' ';
        out += detail::under_thousand(rest, /*standalone_one=*/true);
    }
    return out;
}

// Ordinal ("3." -> "dritte"). German ordinals are written with a trailing
// period, and three stems are irregular: erste, dritte, siebte, achte.
inline std::string ordinal(int64_t n) {
    if (n == 1)
        return "erste";
    if (n == 3)
        return "dritte";
    if (n == 7)
        return "siebte";
    if (n == 8)
        return "achte";
    const std::string c = cardinal(n);
    // < 20 takes -te, from 20 up takes -ste.
    return c + (n < 20 ? "te" : "ste");
}

// Four-digit year reading. Below 2000 German says "<hi>hundert<rest>"
// (neunzehnhundertvierundachtzig); from 2000 it uses the plain cardinal
// (zweitausendsechsundzwanzig). Whole hundreds drop the tail.
inline std::string year(int64_t n) {
    if (n < 1100 || n >= 2000)
        return cardinal(n);
    const int64_t hi = n / 100, lo = n % 100;
    std::string out = detail::under_hundred(hi, /*standalone_one=*/false) + "hundert";
    if (lo)
        out += detail::under_hundred(lo, /*standalone_one=*/true);
    return out;
}

// Currency unit as it is SPOKEN, after the amount.
// German currency units are INVARIABLE: "ein Euro", "fuenfzig Euro" —
// never "Euros". Same for Dollar, Pfund and Yen.
inline std::string currency_word(core_currency::sym s, bool one) {
    switch (s) {
    case core_currency::sym::eur:
        return one ? "Euro" : "Euro";
    case core_currency::sym::usd:
        return one ? "Dollar" : "Dollar";
    case core_currency::sym::gbp:
        return one ? "Pfund" : "Pfund";
    case core_currency::sym::jpy:
        return one ? "Yen" : "Yen";
    default:
        return std::string();
    }
}

// Rewrite every numeric token in `text` as words, leaving the rest untouched.
//
// German separator convention (rule 3 above): '.' groups thousands, ',' is the
// decimal mark. Both are only consumed when the surrounding digits make them
// unambiguous — a '.' is a thousands separator only before exactly three
// digits, so "3.14" stays a period-terminated "3" rather than becoming 314, and
// a sentence-final "Es kostet 5." is not read as five hundred.
inline std::string expand(const std::string& text) {
    using detail::is_digit;
    std::string out;
    out.reserve(text.size() + 32);
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        if (!is_digit(text[i])) {
            out += text[i++];
            continue;
        }
        // Only a token start is a number: leave "mp3" / "x64" to the word rules.
        const bool at_token_start = (i == 0) || !(std::isalnum((unsigned char)text[i - 1]) || text[i - 1] == '\'');
        if (!at_token_start) {
            out += text[i++];
            continue;
        }

        const size_t start = i;
        std::string int_part;
        while (i < n) {
            if (is_digit(text[i])) {
                int_part += text[i++];
                continue;
            }
            // Thousands separator: a '.' followed by EXACTLY three digits that
            // are not themselves followed by another digit.
            if (text[i] == '.' && i + 3 < n + 1 && i + 3 <= n - 1 + 1) {
                const bool three = (i + 3 < n + 1) && is_digit(text[i + 1]) && (i + 2 < n) && is_digit(text[i + 2]) &&
                                   (i + 3 < n) && is_digit(text[i + 3]);
                const bool no_fourth = three && (i + 4 >= n || !is_digit(text[i + 4]));
                if (three && no_fourth) {
                    i++; // skip '.'
                    continue;
                }
            }
            break;
        }
        std::string frac_part;
        if (i < n && text[i] == ',' && i + 1 < n && is_digit(text[i + 1])) {
            i++;
            while (i < n && is_digit(text[i]))
                frac_part += text[i++];
        }
        // Ordinal: a trailing '.' that is NOT a thousands separator. Require a
        // following space or end so "1. Mai" is ordinal while "Es kostet 5."
        // keeps its full stop as punctuation — decided by the caller's text, so
        // only treat it as ordinal when a letter follows the space.
        bool is_ordinal = false;
        if (frac_part.empty() && i < n && text[i] == '.') {
            size_t j = i + 1;
            while (j < n && text[j] == ' ')
                j++;
            if (j < n && std::isalpha((unsigned char)text[j])) {
                is_ordinal = true;
                i++;
            }
        }

        if (int_part.size() > 15) {
            if (!out.empty() && out.back() != ' ')
                out += ' ';
            out += detail::digits_each(int_part);
            continue;
        }
        int64_t value = 0;
        for (char c : int_part)
            value = value * 10 + (c - '0');

        bool negative = false;
        if (start > 0 && text[start - 1] == '-') {
            const bool prev_is_word = start >= 2 && std::isalnum((unsigned char)text[start - 2]);
            if (!prev_is_word) {
                negative = true;
                if (!out.empty() && out.back() == '-')
                    out.pop_back();
            }
        }

        std::string words;
        if (is_ordinal) {
            words = ordinal(value);
        } else if (!frac_part.empty()) {
            words = cardinal(value) + " Komma " + detail::digits_each(frac_part);
        } else if (int_part.size() == 4 && int_part[0] != '0') {
            words = year(value);
        } else {
            words = cardinal(value);
        }
        if (negative)
            words = "minus " + words;

        if (i < n && text[i] == '%') {
            words += " Prozent";
            i++;
        }

        // #316: the currency UNIT, which used to vanish exactly like the digits
        // did. Both written forms occur: "€50" and "50 €".
        {
            const bool one = (value == 1 && frac_part.empty());
            size_t sym_len = 0;
            core_currency::sym cs = core_currency::at_postfix(text, i, sym_len);
            if (cs == core_currency::sym::none && i + 1 < n && text[i] == ' ')
                cs = core_currency::at_postfix(text, i + 1, sym_len);
            if (cs != core_currency::sym::none) {
                i += sym_len + (text[i] == ' ' ? 1 : 0);
            } else {
                cs = core_currency::take_prefix(out);
            }
            const std::string unit = currency_word(cs, one);
            if (!unit.empty()) {
                // Attributive form before the unit noun: "ein Euro", not "eins
                // Euro". Only a TRAILING standalone "eins" changes — 21 is
                // "einundzwanzig Euro" and must not be touched.
                if (words.size() >= 4 && words.compare(words.size() - 4, 4, "eins") == 0)
                    words = words.substr(0, words.size() - 1);
                words += " " + unit;
            }
        }

        if (!out.empty() && out.back() != ' ')
            out += ' ';
        out += words;
    }
    return out;
}

} // namespace core_num2words_de

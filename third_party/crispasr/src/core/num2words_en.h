// core/num2words_en.h — spell English numbers out before phonemization.
//
// Every builtin-G2P consumer (kokoro, piper, …) looked words up in a
// pronunciation dictionary and fell back to letter-to-sound rules. Digits are
// in neither, so a numeric token produced NO phonemes at all and the word
// vanished from the audio — "…with 82 million parameters" was spoken as
// "…with million parameters" (#316). Nothing errored; the number was simply
// gone.
//
// The spec here is misaki's, Kokoro's own G2P, read off its behaviour rather
// than guessed — including the parts that are not obvious:
//
//   82        eighty two            1000     one thousand
//   101       one hundred one       1005     one thousand five
//   1100      eleven hundred        1090     ten ninety
//   1984      nineteen eighty four  1900     nineteen hundred
//   2026      twenty twenty six     2100     twenty one hundred
//   9999      ninety nine ninety nine        12345  twelve thousand three
//                                                   hundred forty five
//   3.14      three point one four  1,000    one thousand
//   1st       first                 -5       minus five
//   50%       fifty percent         $20      twenty dollars
//
// So a four-digit number is read as a YEAR (pair of two-digit halves) unless
// its last two digits are 00 (→ "<hi> hundred", or a plain cardinal when it is
// a whole thousand) or below 10 (→ plain cardinal). Five digits and up are
// always cardinal. Getting that wrong is audible: "twelve thirty four" vs "one
// thousand two hundred thirty four".
//
// Weight-free and header-only — tests/test-num2words-en.cpp covers it without a
// model, and the table above is the test's expectation list.

#pragma once

#include "core/currency_symbols.h" // #316: the unit beside the number

#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace core_num2words_en {

namespace detail {

inline const char* const kOnes[20] = {"zero",     "one",     "two",     "three",     "four",     "five",    "six",
                                      "seven",    "eight",   "nine",    "ten",       "eleven",   "twelve",  "thirteen",
                                      "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
inline const char* const kTens[10] = {"",      "",      "twenty",  "thirty", "forty",
                                      "fifty", "sixty", "seventy", "eighty", "ninety"};

inline void append_word(std::string& out, const std::string& w) {
    if (!out.empty() && out.back() != ' ')
        out += ' ';
    out += w;
}

// 0..999 → words.
inline void under_thousand(std::string& out, int64_t n) {
    if (n >= 100) {
        append_word(out, kOnes[n / 100]);
        append_word(out, "hundred");
        n %= 100;
        if (n == 0)
            return;
    }
    if (n < 20) {
        append_word(out, kOnes[n]);
        return;
    }
    std::string t = kTens[n / 10];
    if (n % 10)
        t += std::string(" ") + kOnes[n % 10];
    append_word(out, t);
}

} // namespace detail

// Plain cardinal for any non-negative integer up to ~10^15.
inline std::string cardinal(int64_t n) {
    if (n == 0)
        return "zero";
    std::string out;
    if (n < 0) {
        out = "minus";
        n = -n;
    }
    struct Scale {
        int64_t value;
        const char* name;
    };
    static const Scale scales[] = {
        {1000000000000LL, "trillion"}, {1000000000LL, "billion"}, {1000000LL, "million"}, {1000LL, "thousand"}};
    for (const auto& s : scales) {
        if (n >= s.value) {
            detail::under_thousand(out, n / s.value);
            detail::append_word(out, s.name);
            n %= s.value;
        }
    }
    if (n > 0)
        detail::under_thousand(out, n);
    return out;
}

// A four-digit integer the way misaki reads it (see the header table). Callers
// use this for bare 4-digit tokens; everything else goes through cardinal().
inline std::string four_digit(int64_t n) {
    const int64_t hi = n / 100, lo = n % 100;
    if (lo == 0) {
        // 1000/2000 stay cardinal; 1100/1900/2100 become "<hi> hundred".
        if (n % 1000 == 0)
            return cardinal(n);
        return cardinal(hi) + " hundred";
    }
    if (lo < 10)
        return cardinal(n); // 1005 → "one thousand five", never "ten five"
    return cardinal(hi) + " " + cardinal(lo);
}

// "first", "second", … for <digits>st/nd/rd/th.
inline std::string ordinal(int64_t n) {
    std::string c = cardinal(n);
    size_t sp = c.find_last_of(' ');
    std::string head = (sp == std::string::npos) ? "" : c.substr(0, sp + 1);
    std::string last = (sp == std::string::npos) ? c : c.substr(sp + 1);
    struct Irr {
        const char* from;
        const char* to;
    };
    static const Irr irr[] = {{"one", "first"},          {"two", "second"},       {"three", "third"},
                              {"five", "fifth"},         {"eight", "eighth"},     {"nine", "ninth"},
                              {"twelve", "twelfth"},     {"twenty", "twentieth"}, {"thirty", "thirtieth"},
                              {"forty", "fortieth"},     {"fifty", "fiftieth"},   {"sixty", "sixtieth"},
                              {"seventy", "seventieth"}, {"eighty", "eightieth"}, {"ninety", "ninetieth"}};
    for (const auto& r : irr)
        if (last == r.from)
            return head + r.to;
    // Compound tens ("twenty one" → "twenty first") already split above.
    return head + last + "th";
}

namespace detail {

inline bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

// Digit-by-digit, for the fractional part of a decimal.
inline std::string digits_each(const std::string& s) {
    std::string out;
    for (char c : s)
        if (is_digit(c))
            append_word(out, kOnes[c - '0']);
    return out;
}

} // namespace detail

// Currency unit as it is SPOKEN, after the amount.
inline std::string currency_word(core_currency::sym s, bool one) {
    switch (s) {
    case core_currency::sym::eur:
        return one ? "euro" : "euros";
    case core_currency::sym::usd:
        return one ? "dollar" : "dollars";
    case core_currency::sym::gbp:
        return one ? "pound" : "pounds";
    case core_currency::sym::jpy:
        return "yen"; // invariable in English
    default:
        return std::string();
    }
}

// Rewrite every numeric token in `text` as words, leaving the rest untouched.
// Handles a leading -, thousands separators, decimals, ordinal suffixes, a
// trailing %, and a leading $.
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
        // Only treat this as a number if it starts a token — otherwise leave
        // alphanumerics like "mp3" or "x64" for the existing token rules.
        // isalNUM, not isalpha: in "x64" the '6' is skipped for having an alpha
        // predecessor, and the '4' must then be skipped too — with isalpha it
        // saw a digit before it, called itself a token start, and produced
        // "x6 four".
        const bool at_token_start = (i == 0) || !(std::isalnum((unsigned char)text[i - 1]) || text[i - 1] == '\'');
        if (!at_token_start) {
            out += text[i++];
            continue;
        }

        const size_t start = i;
        std::string int_part;
        while (i < n && (is_digit(text[i]) || (text[i] == ',' && i + 1 < n && is_digit(text[i + 1])))) {
            if (text[i] != ',')
                int_part += text[i];
            i++;
        }
        std::string frac_part;
        if (i < n && text[i] == '.' && i + 1 < n && is_digit(text[i + 1])) {
            i++;
            while (i < n && is_digit(text[i]))
                frac_part += text[i++];
        }
        // Ordinal suffix directly attached: 1st, 2nd, 3rd, 4th.
        bool is_ordinal = false;
        if (frac_part.empty() && i + 1 < n) {
            const char a = (char)std::tolower((unsigned char)text[i]);
            const char b = (char)std::tolower((unsigned char)text[i + 1]);
            if ((a == 's' && b == 't') || (a == 'n' && b == 'd') || (a == 'r' && b == 'd') || (a == 't' && b == 'h')) {
                is_ordinal = true;
                i += 2;
            }
        }

        // Guard against absurd lengths (phone numbers, hashes): read those
        // digit-by-digit rather than inventing a quadrillion.
        if (int_part.size() > 15) {
            detail::append_word(out, detail::digits_each(int_part));
            continue;
        }
        int64_t value = 0;
        for (char c : int_part)
            value = value * 10 + (c - '0');

        // A leading '-' immediately before the digits is a minus sign, but only
        // when it is not hyphenating a word ("state-of-the-art 5" vs "-5").
        bool negative = false;
        if (start > 0 && text[start - 1] == '-') {
            const bool prev_is_word = start >= 2 && (std::isalnum((unsigned char)text[start - 2]));
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
            words = cardinal(value) + " point " + detail::digits_each(frac_part);
        } else if (int_part.size() == 4 && int_part[0] != '0') {
            words = four_digit(value);
        } else {
            words = cardinal(value);
        }
        if (negative)
            words = "minus " + words;

        // Trailing unit symbols.
        if (i < n && text[i] == '%') {
            words += " percent";
            i++;
        }
        // #316: the currency UNIT, which vanishes exactly like a digit does —
        // it is in no dictionary and no letter-to-sound rule. This used to
        // handle a leading `$` only, so "€50" and "£50" lost their unit and
        // "50$" lost it in the postfix form every European locale writes.
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
            if (!unit.empty())
                words += " " + unit;
        }

        detail::append_word(out, words);
    }
    return out;
}

} // namespace core_num2words_en

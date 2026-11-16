// core/num2words_fr.h — spell French numbers out before phonemization.
//
// Third of the #316 set, after num2words_en.h and num2words_de.h. `g2p_fr` had
// no number path, so a numeric token produced NO phonemes and vanished:
//
//   "J'ai 82 euros"  ->  ʒ ɛ  øʁɔ        (the 82 is simply gone)
//   "82"             ->  ""
//
// Reproduced on main before this file existed.
//
// French counting is VIGESIMAL above 60 — the single reason this is not a
// table lookup, and the thing a naive port gets wrong:
//
//   70  soixante-dix          (sixty-ten, NOT septante)
//   71  soixante et onze      (…and eleven — the "et" survives here)
//   79  soixante-dix-neuf
//   80  quatre-vingts         (four-twenties, WITH the plural s)
//   81  quatre-vingt-un       (…and the s DROPS as soon as anything follows)
//   90  quatre-vingt-dix
//   99  quatre-vingt-dix-neuf
//
// Three more agreement rules that are easy to miss:
//
//   * "et" joins only 21/31/41/51/61 and 71 — never 81 or 91
//     (vingt et un, but quatre-vingt-un).
//   * `cent` takes an s only when a multiplier precedes it AND nothing follows:
//     deux cents, but deux cent un.
//   * `mille` is invariable. "deux milles" is always wrong.
//
// Separators follow the German convention, not the English one: the comma is
// the decimal mark ("3,14" is three point one four). French groups thousands
// with a space rather than a period, and a period after digits is far more
// likely to be a full stop, so only the space form is consumed.
//
// Weight-free and header-only — tests/test-num2words-fr.cpp covers every line
// above without a model, because a wrong entry produces perfectly well-formed
// phonemes for the wrong word and no numeric check can see it.
#pragma once

#include "core/currency_symbols.h" // #316: the unit beside the number

#include <cctype>
#include <cstdint>
#include <string>

namespace core_num2words_fr {

namespace detail {

inline bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

// 0-16 are lexical; 17-19 are already compounds (dix-sept …).
inline const char* const kOnes[17] = {"zéro", "un",  "deux", "trois", "quatre", "cinq",     "six",    "sept", "huit",
                                      "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize"};

// Only the tens French actually names. 70/80/90 are built, not named.
inline const char* const kTens[7] = {"", "", "vingt", "trente", "quarante", "cinquante", "soixante"};

// 0-99, the whole vigesimal mess in one place.
//
// `final` = nothing follows this group inside the number. It controls the one
// plural in French numerals: `quatre-vingts` keeps its s only when it ENDS the
// number, so 80 is "quatre-vingts" but 80 000 is "quatre-vingt mille". Same
// rule as `cent` in under_thousand below. Note this is about position within
// the NUMBER, not the sentence — "quatre-vingts euros" keeps the s.
inline std::string under_hundred(int64_t n, bool final = true) {
    if (n < 17)
        return kOnes[n];
    if (n < 20) // 17,18,19
        return std::string("dix-") + kOnes[n - 10];
    if (n < 70) {
        const int64_t t = n / 10, u = n % 10;
        if (u == 0)
            return kTens[t];
        if (u == 1) // vingt et un, trente et un, … (but see 81/91 below)
            return std::string(kTens[t]) + " et un";
        return std::string(kTens[t]) + "-" + kOnes[u];
    }
    if (n < 80) { // 70-79: soixante + 10..19
        const int64_t r = n - 60;
        if (r == 11) // soixante et onze — the last "et" in the language
            return "soixante et onze";
        return std::string("soixante-") + under_hundred(r);
    }
    // 80-99: quatre-vingt(s) + 0..19. The plural s survives ONLY on a bare 80,
    // and no "et" ever appears here.
    const int64_t r = n - 80;
    if (r == 0)
        return final ? "quatre-vingts" : "quatre-vingt";
    return std::string("quatre-vingt-") + under_hundred(r);
}

// 0-999. `cent` pluralises only when multiplied AND nothing follows it —
// including a following scale word, so 200 is "deux cents" but 200 000 is
// "deux cent mille". `final` carries that through from cardinal().
inline std::string under_thousand(int64_t n, bool final = true) {
    if (n < 100)
        return under_hundred(n, final);
    const int64_t h = n / 100, rest = n % 100;
    std::string out;
    if (h == 1)
        out = "cent";
    else
        out = std::string(kOnes[h]) + " cent" + ((rest == 0 && final) ? "s" : "");
    if (rest)
        out += " " + under_hundred(rest, final);
    return out;
}

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
inline std::string cardinal(int64_t n) {
    if (n == 0)
        return "zéro";
    std::string out;
    auto add = [&out](const std::string& w) {
        if (!out.empty())
            out += ' ';
        out += w;
    };

    const int64_t billions = n / 1000000000;
    const int64_t millions = (n / 1000000) % 1000;
    const int64_t thousands = (n / 1000) % 1000;
    const int64_t rest = n % 1000;

    // Each group is followed by its scale word, so none of them is `final` —
    // that is what keeps 80 000 at "quatre-vingt mille" and 200 000 at
    // "deux cent mille".
    if (billions)
        add(detail::under_thousand(billions, false) + (billions == 1 ? " milliard" : " milliards"));
    if (millions)
        add(detail::under_thousand(millions, false) + (millions == 1 ? " million" : " millions"));
    if (thousands) {
        // "mille", never "un mille" and never "milles" — it is invariable.
        add(thousands == 1 ? std::string("mille") : detail::under_thousand(thousands, false) + " mille");
    }
    if (rest)
        add(detail::under_thousand(rest));
    return out;
}

// Ordinal. 1 is the irregular one ("premier"); the rest take -ième off the
// cardinal, with the usual spelling repairs: a final silent e is dropped
// (quatre -> quatrième), cinq gains a u, neuf turns f into v.
inline std::string ordinal(int64_t n) {
    if (n == 1)
        return "premier";
    std::string c = cardinal(n);
    if (c.size() >= 4 && c.compare(c.size() - 4, 4, "cinq") == 0)
        return c + "uième";
    if (c.size() >= 4 && c.compare(c.size() - 4, 4, "neuf") == 0)
        return c.substr(0, c.size() - 1) + "vième";
    if (!c.empty() && c.back() == 'e')
        c.pop_back();
    // "quatre-vingts" -> "quatre-vingtième": the plural s goes too.
    if (!c.empty() && c.back() == 's')
        c.pop_back();
    return c + "ième";
}

// Currency unit as it is SPOKEN, after the amount.
// French pluralises regularly.
inline std::string currency_word(core_currency::sym s, bool one) {
    switch (s) {
    case core_currency::sym::eur:
        return one ? "euro" : "euros";
    case core_currency::sym::usd:
        return one ? "dollar" : "dollars";
    case core_currency::sym::gbp:
        return one ? "livre" : "livres";
    case core_currency::sym::jpy:
        return one ? "yen" : "yens";
    default:
        return std::string();
    }
}

// Rewrite every numeric token as words.
//
// Separators: ',' is the decimal mark; thousands are grouped with a SPACE
// (French convention) — a period after digits is treated as punctuation, since
// "Il coûte 5." is far more common than a period-grouped thousand.
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
            // Space-grouped thousands: consume only before EXACTLY three more
            // digits, so "3 euros" keeps its space.
            if (text[i] == ' ' && i + 3 < n && is_digit(text[i + 1]) && is_digit(text[i + 2]) &&
                is_digit(text[i + 3]) && (i + 4 >= n || !is_digit(text[i + 4]))) {
                i++;
                continue;
            }
            break;
        }
        std::string frac_part;
        if (i < n && text[i] == ',' && i + 1 < n && is_digit(text[i + 1])) {
            i++;
            while (i < n && is_digit(text[i]))
                frac_part += text[i++];
        }
        // Ordinal marks: 1er / 1re / 2e / 2ème.
        bool is_ordinal = false;
        if (frac_part.empty() && i < n) {
            const std::string tail = text.substr(i, 3);
            if (tail.rfind("ème", 0) == 0) {
                is_ordinal = true;
                i += 3;
            } else if (tail.rfind("er", 0) == 0 || tail.rfind("re", 0) == 0) {
                is_ordinal = true;
                i += 2;
            } else if (text[i] == 'e' && (i + 1 >= n || !std::isalpha((unsigned char)text[i + 1]))) {
                is_ordinal = true;
                i += 1;
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
        if (is_ordinal)
            words = ordinal(value);
        else if (!frac_part.empty())
            words = cardinal(value) + " virgule " + detail::digits_each(frac_part);
        else
            words = cardinal(value);
        if (negative)
            words = "moins " + words;

        if (i < n && text[i] == '%') {
            words += " pour cent";
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
            if (!unit.empty())
                words += " " + unit;
        }

        if (!out.empty() && out.back() != ' ')
            out += ' ';
        out += words;
    }
    return out;
}

} // namespace core_num2words_fr

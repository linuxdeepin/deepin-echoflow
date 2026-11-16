// core/num2words_es.h — spell Spanish numbers out before phonemization.
//
// Last of the #316 set (en, de, fr, es). `g2p_es` had no number path, so a
// numeric token produced NO phonemes and vanished:
//
//   "Tengo 82 euros"  ->  tenɡo  euɾos      (the 82 is simply gone)
//   "82"              ->  ""
//
// Reproduced on main before this file existed.
//
// Spanish looks regular and then is not, in three specific places:
//
//   1. 16-19 and 21-29 FUSE into one word, and the fusion carries an accent
//      that a naive concatenation loses:
//        16  dieciséis      (not "diez y seis", and the é is not optional)
//        22  veintidós      26  veintiséis      23  veintitrés
//      From 31 up it splits again into three words with "y":
//        31  treinta y uno  82  ochenta y dos
//      So "y" appears from 31 onward and NEVER below 30.
//
//   2. FOUR of the hundreds are irregular, and they are the ones a pattern
//      would get wrong:
//        500  quinientos   (not "cincocientos")
//        700  setecientos  (not "sietecientos")
//        900  novecientos  (not "nuevecientos")
//        100  cien standing alone, but ciento when anything follows
//             (cien euros / ciento uno)
//
//   3. `mil` is invariable and takes no article: 1000 is "mil", never
//      "un mil". But a million does: "un millón" / "dos millones".
//
// Separators match German and French, not English: ',' is the decimal mark.
// Spanish officially groups thousands with a space; a period is still common,
// so both are accepted where they are unambiguous.
//
// Weight-free and header-only — tests/test-num2words-es.cpp covers every line
// above without a model, because a wrong entry produces perfectly well-formed
// phonemes for the wrong word and no numeric check can see it.
#pragma once

#include "core/currency_symbols.h" // #316: the unit beside the number

#include <cctype>
#include <cstdint>
#include <string>

namespace core_num2words_es {

namespace detail {

inline bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

// 0-29 written out, because 16-19 and 21-29 are single fused words with
// accents. Listing them beats generating them and losing the é/ó/á.
inline const char* const kUnder30[30] = {
    "cero",         "uno",         "dos",        "tres",        "cuatro",     "cinco",      "seis",      "siete",
    "ocho",         "nueve",       "diez",       "once",        "doce",       "trece",      "catorce",   "quince",
    "dieciséis",    "diecisiete",  "dieciocho",  "diecinueve",  "veinte",     "veintiuno",  "veintidós", "veintitrés",
    "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve"};

inline const char* const kTens[10] = {"",          "",        "veinte",  "treinta", "cuarenta",
                                      "cincuenta", "sesenta", "setenta", "ochenta", "noventa"};

// The four irregular hundreds are why this is a table, not a formula.
inline const char* const kHundreds[10] = {"",           "ciento",      "doscientos",  "trescientos", "cuatrocientos",
                                          "quinientos", "seiscientos", "setecientos", "ochocientos", "novecientos"};

// 0-99. "y" only from 31 up.
inline std::string under_hundred(int64_t n) {
    if (n < 30)
        return kUnder30[n];
    const int64_t t = n / 10, u = n % 10;
    if (u == 0)
        return kTens[t];
    return std::string(kTens[t]) + " y " + kUnder30[u];
}

// 0-999. 100 alone is "cien"; with anything after it, "ciento".
inline std::string under_thousand(int64_t n) {
    if (n < 100)
        return under_hundred(n);
    const int64_t h = n / 100, rest = n % 100;
    if (h == 1 && rest == 0)
        return "cien";
    std::string out = kHundreds[h];
    if (rest)
        out += " " + under_hundred(rest);
    return out;
}

inline std::string digits_each(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (!is_digit(c))
            continue;
        if (!out.empty())
            out += ' ';
        out += kUnder30[c - '0'];
    }
    return out;
}

// APOCOPATION: before a masculine noun — and `mil`/`millones` count — a final
// "uno" shortens to "un", and "veintiuno" takes an accent when it does:
//   21      veintiuno        21 000  veintiún mil
//   31      treinta y uno    31 000  treinta y un mil
// Standing alone the full form is correct, so this is applied only where a
// scale word follows.
inline std::string apocopate(const std::string& w) {
    const std::string veintiuno = "veintiuno";
    if (w.size() >= veintiuno.size() && w.compare(w.size() - veintiuno.size(), veintiuno.size(), veintiuno) == 0)
        return w.substr(0, w.size() - veintiuno.size()) + "veintiún";
    if (w.size() >= 3 && w.compare(w.size() - 3, 3, "uno") == 0)
        return w.substr(0, w.size() - 3) + "un";
    return w;
}

} // namespace detail

// Cardinal reading of a non-negative integer.
inline std::string cardinal(int64_t n) {
    if (n == 0)
        return "cero";
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

    // Millions take the article and inflect; mil does neither.
    // Every group here is followed by a scale word, so each apocopates:
    // 21 000 is "veintiún mil", not "veintiuno mil".
    if (billions)
        add(billions == 1 ? std::string("mil millones")
                          : detail::apocopate(detail::under_thousand(billions)) + " mil millones");
    if (millions)
        add(millions == 1 ? std::string("un millón")
                          : detail::apocopate(detail::under_thousand(millions)) + " millones");
    if (thousands)
        add(thousands == 1 ? std::string("mil") : detail::apocopate(detail::under_thousand(thousands)) + " mil");
    if (rest)
        add(detail::under_thousand(rest));
    return out;
}

// Ordinal. The first ten are lexical; above that Spanish overwhelmingly uses
// the cardinal in speech, so that is what we emit rather than inventing
// "vigésimo primero" forms a TTS voice will rarely have been trained on.
inline std::string ordinal(int64_t n) {
    switch (n) {
    case 1:
        return "primero";
    case 2:
        return "segundo";
    case 3:
        return "tercero";
    case 4:
        return "cuarto";
    case 5:
        return "quinto";
    case 6:
        return "sexto";
    case 7:
        return "séptimo";
    case 8:
        return "octavo";
    case 9:
        return "noveno";
    case 10:
        return "décimo";
    default:
        return cardinal(n);
    }
}

// Currency unit as it is SPOKEN, after the amount.
// Spanish: the dollar singular carries an accent the plural loses
// (dólar / dólares).
inline std::string currency_word(core_currency::sym s, bool one) {
    switch (s) {
    case core_currency::sym::eur:
        return one ? "euro" : "euros";
    case core_currency::sym::usd:
        return one ? "dólar" : "dólares";
    case core_currency::sym::gbp:
        return one ? "libra" : "libras";
    case core_currency::sym::jpy:
        return one ? "yen" : "yenes";
    default:
        return std::string();
    }
}

// Rewrite every numeric token as words.
//
// Separators: ',' is the decimal mark. Thousands group with a space (the RAE
// recommendation) or a period (still widespread) — each consumed only before
// exactly three digits, so "Cuesta 5." keeps its full stop.
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
            if ((text[i] == '.' || text[i] == ' ') && i + 3 < n && is_digit(text[i + 1]) && is_digit(text[i + 2]) &&
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
        // Ordinal marks: 1º / 1ª / 1er.
        bool is_ordinal = false;
        if (frac_part.empty() && i < n) {
            if (text.compare(i, 2, "º") == 0 || text.compare(i, 2, "ª") == 0) {
                is_ordinal = true;
                i += 2;
            } else if (text.compare(i, 2, "er") == 0) {
                is_ordinal = true;
                i += 2;
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
            words = cardinal(value) + " coma " + detail::digits_each(frac_part);
        else
            words = cardinal(value);
        if (negative)
            words = "menos " + words;

        if (i < n && text[i] == '%') {
            words += " por ciento";
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
                // A unit noun is masculine-or-not but always a NOUN, so `uno`
                // apocopates here the same way it does before `mil`.
                words = detail::apocopate(words);
                words += " " + unit;
            }
        }

        if (!out.empty() && out.back() != ' ')
            out += ' ';
        out += words;
    }
    return out;
}

} // namespace core_num2words_es

#pragma once

#include "chatterbox_nfkd.h"

#include <cctype>
#include <string>
#include <string_view>

namespace chatterbox_text_prep {

inline void replace_all(std::string& s, std::string_view from, std::string_view to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

inline void collapse_whitespace(std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool pending_space = false;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
            pending_space = true;
            ++i;
            continue;
        }
        if (pending_space && !out.empty()) {
            out.push_back(' ');
        }
        pending_space = false;
        out.push_back(s[i++]);
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    s.swap(out);
}

inline void lowercase_ascii_inplace(std::string& s) {
    for (char& ch : s) {
        const unsigned char c = (unsigned char)ch;
        if (c >= 'A' && c <= 'Z') {
            ch = (char)(c - 'A' + 'a');
        }
    }
}

inline std::string normalize(std::string text, bool multilingual = false) {
    if (text.empty()) {
        return "You need to add some text for me to talk.";
    }

    // Match the upstream reference path as closely as possible without
    // depending on a heavier Unicode normalization library.
    replace_all(text, "\xC2\xA0", " ");     // NBSP
    replace_all(text, "\xE2\x80\x80", " "); // en quad
    replace_all(text, "\xE2\x80\x81", " "); // em quad
    replace_all(text, "\xE2\x80\x82", " "); // en space
    replace_all(text, "\xE2\x80\x83", " "); // em space
    replace_all(text, "\xE2\x80\x84", " "); // three-per-em space
    replace_all(text, "\xE2\x80\x85", " "); // four-per-em space
    replace_all(text, "\xE2\x80\x86", " "); // six-per-em space
    replace_all(text, "\xE2\x80\x87", " "); // figure space
    replace_all(text, "\xE2\x80\x89", " "); // thin space
    replace_all(text, "\xE2\x80\x8A", " "); // hair space
    replace_all(text, "\xE2\x80\xAF", " "); // narrow no-break space
    replace_all(text, "\xE3\x80\x80", " "); // ideographic space

    replace_all(text, "\xE2\x80\xA6", ", "); // ellipsis
    replace_all(text, "...", ", ");
    replace_all(text, ":", ",");
    replace_all(text, " - ", ", ");
    replace_all(text, ";", ", ");
    replace_all(text, " ,", ",");
    replace_all(text, "\xE2\x80\x94", "-");  // em dash
    replace_all(text, "\xE2\x80\x93", "-");  // en dash
    replace_all(text, "\xE2\x80\x98", "'");  // left single quote
    replace_all(text, "\xE2\x80\x99", "'");  // right single quote / apostrophe
    replace_all(text, "\xE2\x80\x9C", "\""); // left double quote
    replace_all(text, "\xE2\x80\x9D", "\""); // right double quote

    collapse_whitespace(text);
    if (text.empty()) {
        return "You need to add some text for me to talk.";
    }

    if (!multilingual && text[0] >= 'a' && text[0] <= 'z') {
        text[0] = (char)(text[0] - 'a' + 'A');
    }
    if (multilingual) {
        // Match upstream MTLTokenizer.preprocess_text: lowercase + NFKD
        // (issue #170). NFKD must run first so accented Latin uppercase
        // decomposes to an ASCII base letter that lowercase_ascii then
        // folds (e.g. "É" -> "E" + combining acute -> "e" + combining
        // acute), and so precomposed Arabic graphemes (e.g. U+0623
        // ALEF-WITH-HAMZA) split into base + combining mark exactly as the
        // tokenizer was trained on. Non-Latin uppercase casefolding (Greek,
        // Cyrillic) is out of scope — those languages route through their
        // own normalizers upstream.
        text = nfkd(text);
        lowercase_ascii_inplace(text);
    }

    if (!text.empty()) {
        const char last = text.back();
        if (last != '.' && last != '!' && last != '?' && last != '-' && last != ',') {
            text.push_back('.');
        }
    }
    return text;
}

} // namespace chatterbox_text_prep

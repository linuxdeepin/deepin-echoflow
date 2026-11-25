// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "AsrHotwords.h"

#include <algorithm>
#include <iterator>
#include <string_view>
#include <vector>

namespace echoflow {
namespace {

constexpr size_t kMaxHotwordBytes = 512;

std::vector<std::string> splitTerms(std::string list)
{
    for (const std::string_view separator : {u8"\uFF0C", u8"\u3001", u8"\uFF1B"}) {
        for (size_t pos = 0; (pos = list.find(separator, pos)) != std::string::npos; ++pos) {
            list.replace(pos, separator.size(), ",");
        }
    }
    std::vector<std::string> terms;
    std::string_view remaining = list;
    while (!remaining.empty()) {
        const size_t separator = remaining.find_first_of(",;\r\n");
        auto term = remaining.substr(0, separator);
        remaining.remove_prefix(separator == std::string_view::npos ? remaining.size() : separator + 1);
        const size_t first = term.find_first_not_of(" \t");
        if (first == std::string_view::npos) continue;
        terms.emplace_back(term.substr(first, term.find_last_not_of(" \t") - first + 1));
    }
    return terms;
}

std::string normalizeTerm(std::string_view term)
{
    // Ignore sentence punctuation at the end of a term and keep its interior intact.
    constexpr std::string_view kSuffixes[] = {".", "!", "?", u8"\u3002", u8"\uFF01", u8"\uFF1F"};
    for (;;) {
        const auto mark = std::find_if(std::begin(kSuffixes), std::end(kSuffixes),
            [term](std::string_view suffix) {
                return term.size() >= suffix.size() && term.substr(term.size() - suffix.size()) == suffix;
            });
        if (mark == std::end(kSuffixes)) break;
        term.remove_suffix(mark->size());
    }
    std::string result(term);
    for (char& ch : result) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return result;
}

} // namespace

AsrHotwords::AsrHotwords(const std::string& value)
{
    for (const auto& term : splitTerms(value)) {
        if (term.find("<|") != std::string::npos || term.find("|>") != std::string::npos) continue;
        const size_t separatorBytes = echoTerms_.empty() ? 0 : 2;
        if (term.size() > kMaxHotwordBytes - prompt_.size() - separatorBytes) continue;
        const auto key = normalizeTerm(term);
        if (key.empty() || std::find(echoTerms_.begin(), echoTerms_.end(), key) != echoTerms_.end()) continue;
        if (!echoTerms_.empty()) prompt_ += ", ";
        prompt_.append(term);
        echoTerms_.push_back(key);
        if (prompt_.size() >= kMaxHotwordBytes - 2) break;
    }
}

std::string AsrHotwords::filterEcho(std::string text) const
{
    // A single term may be the entire utterance. Require a list in the output.
    if (echoTerms_.size() < 2) return text;
    const auto terms = splitTerms(text);
    if (std::equal(terms.begin(), terms.end(), echoTerms_.begin(), echoTerms_.end(),
            [](const std::string& term, const std::string& key) { return normalizeTerm(term) == key; })) {
        text.clear();
    }
    return text;
}

} // namespace echoflow

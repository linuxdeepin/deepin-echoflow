// core/g2p_de_unstressed.h — German function words that carry no stress in
// running speech.
//
// GENERATED — do not hand-edit. Regenerate with
// tools/gen-g2p-de-unstressed.py, which asks espeak-ng directly.
//
// Why this file exists. Our German dictionary (`espeak_de.tsv`) was generated
// by running espeak-ng over a word list ONE WORD AT A TIME, so every entry is
// the CITATION form — and espeak stresses a word in isolation that it leaves
// unstressed in a sentence:
//
//     espeak "sie"                       ->  zˈiː
//     espeak "sie ging dann nach Hause"  ->  ziː ɡˈɪŋ dan nɑːx hˈaʊzə
//
// So we were emitting a primary stress on every article, pronoun, preposition
// and auxiliary in the sentence — the German shape of exactly the #316 English
// bug, where a per-word lexicon can only store `the` as `ði` and never `ðə`.
// Both the German Kokoro (dida-80b hui) and the German piper voices are trained
// on espeak's SENTENCE output, so the citation form is a token sequence they
// never saw.
//
// Measured against espeak's sentence output over 8 sentences: token agreement
// 45.9% -> 87.1% with this table applied.
//
// The rule is purely LEXICAL, not positional — espeak reads even a
// sentence-initial "Der" as `dɛɾ`. Every entry below was verified in two
// independent carrier frames and accepted only when the in-frame form is the
// citation form with its stress marks removed and nothing else changed.
//
// Weight-free and header-only.

#pragma once

#include <map>
#include <string>

namespace core_g2p_de_unstressed {

// word (lowercase) -> the unstressed reading espeak gives it in a sentence.
inline const std::map<std::string, std::string>& table() {
    static const std::map<std::string, std::string> t = {
        {"ab", "ap"},    {"als", "als"},   {"am", "am"},        {"an", "an"},     {"ans", "ans"},      {"auf", "aʊf"},
        {"bei", "baɪ"},  {"beim", "baɪm"}, {"bin", "bɪn"},      {"bis", "bɪs"},   {"bist", "bɪst"},    {"da", "dɑː"},
        {"dann", "dan"}, {"das", "das"},   {"dass", "das"},     {"dein", "daɪn"}, {"dem", "deːm"},     {"den", "deːn"},
        {"denn", "dɛn"}, {"der", "dɛɾ"},   {"des", "dɛs"},      {"dich", "dɪç"},  {"die", "diː"},      {"dir", "diːɾ"},
        {"doch", "dɔx"}, {"du", "duː"},    {"durch", "dʊɐç"},   {"ein", "aɪn"},   {"er", "ɛɾ"},        {"es", "ɛs"},
        {"euch", "ɔøç"}, {"für", "fyːɾ"},  {"habe", "hɑːbə"},   {"hast", "hast"}, {"hat", "hat"},      {"ich", "ɪç"},
        {"ihm", "iːm"},  {"ihn", "iːn"},   {"ihr", "iːɾ"},      {"im", "ɪm"},     {"in", "ɪn"},        {"ins", "ɪns"},
        {"ist", "ɪst"},  {"je", "jeː"},    {"man", "man"},      {"mein", "maɪn"}, {"mich", "mɪç"},     {"mir", "miːɾ"},
        {"mit", "mɪt"},  {"nach", "nɑːx"}, {"noch", "nɔx"},     {"pro", "pɾoː"},  {"sein", "zaɪn"},    {"seit", "zaɪt"},
        {"sich", "zɪç"}, {"sie", "ziː"},   {"sind", "zɪnt"},    {"so", "zoː"},    {"um", "ʊm"},        {"und", "ʊnt"},
        {"uns", "ʊns"},  {"vom", "fɔm"},   {"von", "fɔn"},      {"war", "vɑːɾ"},  {"waren", "vɑːrən"}, {"wie", "viː"},
        {"will", "vɪl"}, {"wir", "viːɾ"},  {"wollen", "vɔlən"}, {"zu", "tsuː"},   {"zum", "tsʊm"},     {"zur", "tsuːɾ"},
    };
    return t;
}

// The unstressed reading, or "" when the word is not in the closed class.
inline std::string lookup(const std::string& lower_word) {
    const auto& t = table();
    auto it = t.find(lower_word);
    return it == t.end() ? std::string() : it->second;
}

} // namespace core_g2p_de_unstressed

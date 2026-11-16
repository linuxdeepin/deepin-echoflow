// src/core/pinyin_g2p.h — Chinese grapheme→pinyin for F5-TTS (#294).
//
// Reproduces F5-TTS's `convert_char_to_pinyin` (jieba + pypinyin, TONE3 style,
// tone-sandhi) in C++ so Han text tokenizes to the same pinyin-syllable tokens
// the model was trained on. Data (single-char + phrase readings) is embedded
// from pypinyin via tools/gen_pinyin_data.py → src/core/pinyin_data.inc.
//
// Segmentation is a "min jieba": forward maximum-matching over the phrase table
// (not full jieba) — enough to resolve polyphones and scope tone-sandhi. Exact
// jieba parity is not attainable without embedding jieba's dictionary; parity
// vs the reference is measured by tests/test-pinyin-g2p.cpp.
#pragma once

#include <string>
#include <vector>

namespace core_pinyin {

// True if the UTF-8 string contains any Han character (U+3100..U+9FFF) — the
// same range F5-TTS's `is_chinese` uses. Callers can keep the pure-ASCII/Latin
// path unchanged and only invoke the g2p when this returns true.
bool has_han(const std::string& utf8);

// Convert UTF-8 text to the F5-TTS token list: each Han char becomes a space
// token followed by its TONE3 pinyin syllable (e.g. " ", "zhong1"); ASCII and
// other characters pass through per character. Mirrors the element structure of
// the reference `convert_char_to_pinyin(text)[0]`.
std::vector<std::string> convert_char_to_pinyin(const std::string& utf8);

} // namespace core_pinyin

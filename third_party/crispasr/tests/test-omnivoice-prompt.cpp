// test-omnivoice-prompt.cpp — the exact OmniVoice style prefix (#13273).
//
// Closes the last gap in the #13273 work. The language and instruct RESOLVERS
// were unit-tested, and the assembled prompt was checked against the real HF
// tokenizer by hand — but by hand only: it needed a 1.2 GB model, a debug
// print and an eyeball. That is precisely the kind of check that gets done once
// and then silently rots, and it is how the instruct defect survived the first
// review of the language fix.
//
// Every string asserted below was verified byte-for-byte against
// `AutoTokenizer.from_pretrained(<omnivoice-lm-src>)(style).input_ids`, and the
// ids are recorded next to each case. So this file pins the LAST thing the
// model sees before tokenization, hermetically, with no weights.
//
//   style                                                              -> ids
//   <|lang_start|>de<|lang_end|><|instruct_start|>None<|instruct_end|>
//       151670 450 151671 151672 4064 151673
//   <|denoise|> + same
//       151669 151670 450 151671 151672 4064 151673
//   <|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>
//       151670 4064 151671 151672 4064 151673
//   <|lang_start|>en<|lang_end|><|instruct_start|>male, british accent<|instruct_end|>
//       151670 268 151671 151672 36476 11 93927 29100 151673
//   <|lang_start|>zh<|lang_end|><|instruct_start|>男，老年<|instruct_end|>
//       151670 23815 151671 151672 70108 3837 106487 151673

#include "core/omnivoice_instruct.h"
#include "core/omnivoice_lang.h"
#include "core/omnivoice_prompt.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

using core_omnivoice_prompt::build_style_text;

TEST_CASE("omnivoice prompt: the tokenizer-verified style strings", "[unit][omnivoice]") {
    REQUIRE(build_style_text(false, "de", "") == "<|lang_start|>de<|lang_end|><|instruct_start|>None<|instruct_end|>");
    REQUIRE(build_style_text(true, "de", "") ==
            "<|denoise|><|lang_start|>de<|lang_end|><|instruct_start|>None<|instruct_end|>");
    REQUIRE(build_style_text(false, "", "") == "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>");
    REQUIRE(build_style_text(false, "en", "male, british accent") ==
            "<|lang_start|>en<|lang_end|><|instruct_start|>male, british accent<|instruct_end|>");
    REQUIRE(build_style_text(false, "zh", "男，老年") ==
            "<|lang_start|>zh<|lang_end|><|instruct_start|>男，老年<|instruct_end|>");
}

// Both empty slots become the literal "None" — the token the model was trained
// on for "unspecified". An empty tag (`<|lang_start|><|lang_end|>`) or a dropped
// tag would be a different prompt shape entirely, and would still synthesise.
TEST_CASE("omnivoice prompt: an unset slot is the literal None, not an empty tag", "[unit][omnivoice]") {
    const std::string s = build_style_text(false, "", "");
    REQUIRE(s.find("<|lang_start|>None<|lang_end|>") != std::string::npos);
    REQUIRE(s.find("<|instruct_start|>None<|instruct_end|>") != std::string::npos);
    REQUIRE(s.find("<|lang_start|><|lang_end|>") == std::string::npos);
}

// Order and presence are fixed by the blueprint: denoise (when cloning), then
// language, then instruct — every tag always emitted.
TEST_CASE("omnivoice prompt: tag order is fixed", "[unit][omnivoice]") {
    const std::string s = build_style_text(true, "fr", "whisper");
    const size_t d = s.find("<|denoise|>");
    const size_t l = s.find("<|lang_start|>");
    const size_t i = s.find("<|instruct_start|>");
    REQUIRE(d != std::string::npos);
    REQUIRE(l != std::string::npos);
    REQUIRE(i != std::string::npos);
    REQUIRE(d < l);
    REQUIRE(l < i);
    // <|denoise|> appears only when there IS a reference to denoise from.
    REQUIRE(build_style_text(false, "fr", "whisper").find("<|denoise|>") == std::string::npos);
}

// The end-to-end contract that the three headers exist to uphold: whatever a
// caller types, the prompt only ever carries a resolved id and a normalised
// instruct.
TEST_CASE("omnivoice prompt: resolvers feed the builder, raw values cannot reach it", "[unit][omnivoice]") {
    // A name resolves to its id before assembly.
    const auto lang = core_omnivoice_lang::resolve("German");
    // Natural capitalisation normalises before assembly.
    const auto instruct = core_omnivoice_instruct::parse("Male, British Accent");
    REQUIRE(instruct.status == core_omnivoice_instruct::Status::ok);
    const std::string rendered = core_omnivoice_instruct::render(instruct, /*text_has_zh=*/false);

    REQUIRE(build_style_text(false, lang.id, rendered) ==
            "<|lang_start|>de<|lang_end|><|instruct_start|>male, british accent<|instruct_end|>");

    // And an unrecognized language resolves to empty, i.e. "None" — never to
    // itself. This is the assembled-prompt view of the original #13273 defect.
    const auto bad = core_omnivoice_lang::resolve("de-DE");
    REQUIRE(build_style_text(false, bad.id, "") ==
            "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>");
}

// Python's `_ZH_RE` is exactly [U+4E00, U+9FFF]. Sniffing the UTF-8 lead byte
// (0xE4..0xE9) looks equivalent and over-matches U+4000..U+4DFF by 3584
// codepoints — CJK Ext-A and the Yijing hexagrams. That window decides whether
// an instruct renders as "male, elderly" or "男，老年".
TEST_CASE("omnivoice prompt: CJK detection matches Python's range exactly", "[unit][omnivoice]") {
    using core_omnivoice_instruct::text_is_zh;
    REQUIRE(text_is_zh("我们明天一起去公园散步吧")); // U+4E00..U+9FFF
    REQUIRE(text_is_zh("Hello 世界"));               // mixed still counts
    REQUIRE(!text_is_zh("Hello there."));
    REQUIRE(!text_is_zh(""));
    REQUIRE(!text_is_zh("Ich möchte Brötchen.")); // 2-byte UTF-8, not CJK
    REQUIRE(!text_is_zh("こんにちは"));           // kana is outside the range
    REQUIRE(!text_is_zh("안녕하세요"));           // hangul too
    REQUIRE(!text_is_zh("䀀䷿"));                 // U+4000 / U+4DFF — the over-match window
    REQUIRE(text_is_zh("一"));                    // U+4E00, the first in range
    REQUIRE(text_is_zh("鿿"));                    // U+9FFF, the last in range
    REQUIRE(!text_is_zh("䓀"));                   // U+44C0, inside the window, not Chinese to Python
    // Emoji are 4-byte sequences; the decoder must step over them, not into them.
    REQUIRE(!text_is_zh("nice 🎉 work"));
}

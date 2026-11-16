// test-cohere-lang.cpp — hermetic tests for Cohere Transcribe language
// resolution. No GGUF required.
//
// The defect these guard: the runtime had no notion of a supported set, so an
// unsupported `-l` was accepted in full silence. The vocab cannot substitute
// for the config list — the published Arabic GGUF ships 183 `<|xx|>` tokens
// (all of ISO 639-1) while supporting two languages, so `<|de|>` is present
// and well-formed and the model simply transcribes differently. Every case
// below except the pass-throughs failed before this list existed.

#include <catch2/catch_test_macros.hpp>

#include "cohere_lang.h"

using namespace cohere_lang;

// The real lists, from config.json of each published model.
static const std::vector<std::string> kBase = {"en", "fr", "de", "es", "it", "pt", "nl",
                                               "pl", "el", "ar", "ja", "zh", "vi", "ko"};
static const std::vector<std::string> kArabic = {"en", "ar"};

TEST_CASE("cohere_lang::normalize strips case, region and script", "[unit][cohere]") {
    REQUIRE(normalize("en") == "en");
    REQUIRE(normalize("EN") == "en");
    REQUIRE(normalize("en-US") == "en");
    REQUIRE(normalize("zh_Hans") == "zh");
    REQUIRE(normalize(" de ") == "de");
    REQUIRE(normalize("") == "");
}

TEST_CASE("cohere_lang: a supported language passes through untouched", "[unit][cohere]") {
    for (const auto& code : kBase) {
        auto r = resolve(kBase, code);
        REQUIRE(r.lang == code);
        REQUIRE_FALSE(r.substituted);
        REQUIRE(r.reason.empty());
    }
    auto r = resolve(kArabic, "ar");
    REQUIRE(r.lang == "ar");
    REQUIRE_FALSE(r.substituted);
}

TEST_CASE("cohere_lang: an unsupported language falls back to en, loudly", "[unit][cohere]") {
    // Whisper-tiny LID knows 99 languages; this model knows 14. Before the fix
    // 'ru' produced a prompt with NO language slot at all.
    auto r = resolve(kBase, "ru");
    REQUIRE(r.substituted);
    REQUIRE(r.lang == "en");
    REQUIRE(r.reason.find("ru") != std::string::npos);
    REQUIRE(r.reason.find("not supported") != std::string::npos);
}

TEST_CASE("cohere_lang: the Arabic finetune rejects the base model's languages", "[unit][cohere]") {
    // The Arabic finetune SHARES the base tokenizer, so `<|de|>` is present in
    // its vocab and a vocab-membership check alone would wave this through.
    // Only config.json's supported_languages says {en, ar}.
    auto r = resolve(kArabic, "de");
    REQUIRE(r.substituted);
    REQUIRE(r.lang == "en");

    auto fr = resolve(kArabic, "fr");
    REQUIRE(fr.substituted);
    REQUIRE(fr.lang == "en");
}

TEST_CASE("cohere_lang: 'auto' and empty resolve to the fallback, not to a dropped slot", "[unit][cohere]") {
    for (const char* req : {"auto", "", "  "}) {
        auto r = resolve(kBase, req);
        REQUIRE(r.substituted);
        REQUIRE(r.lang == "en");
        REQUIRE(r.reason.find("does not detect") != std::string::npos);
    }
}

TEST_CASE("cohere_lang: region-tagged codes resolve against the supported set", "[unit][cohere]") {
    auto us = resolve(kBase, "en-US");
    REQUIRE(us.lang == "en");
    REQUIRE_FALSE(us.substituted);

    auto hans = resolve(kBase, "zh-Hans");
    REQUIRE(hans.lang == "zh");
    REQUIRE_FALSE(hans.substituted);

    // 'de-DE' is supported by the base model but not by the Arabic finetune.
    REQUIRE_FALSE(resolve(kBase, "de-DE").substituted);
    REQUIRE(resolve(kArabic, "de-DE").substituted);
}

TEST_CASE("cohere_lang: an empty supported list passes through (pre-metadata GGUF)", "[unit][cohere]") {
    // GGUFs converted before cohere_transcribe.supported_languages existed carry
    // no list. Pass through and let the caller's vocab check decide, rather than
    // inventing a restriction the model may not have.
    const std::vector<std::string> none;
    auto r = resolve(none, "ru");
    REQUIRE(r.lang == "ru");
    REQUIRE_FALSE(r.substituted);
}

TEST_CASE("cohere_lang: a model without en falls back to its first language", "[unit][cohere]") {
    const std::vector<std::string> ar_only = {"ar"};
    auto r = resolve(ar_only, "de");
    REQUIRE(r.substituted);
    REQUIRE(r.lang == "ar");
}

// test-ngram-loop-fix-cjk.cpp — Catch2 unit tests for the code-point-level
// half of core_ngram::fix_loops (PLAN.md §W1).
//
// Why a second file: `test-ngram-loop-fix.cpp` covers the whitespace/word-level
// collapse and every one of its fixtures is Latin. That is not an accident of
// coverage — it is the bug. `split_words()` splits on ASCII whitespace only, so
// on Japanese/Chinese the entire segment is ONE "word", `collapse()` can never
// fire, and the guard is inert on exactly the backends that need it most —
// `moss_transcribe` (zh/en) and `glm-asr` (Mandarin + Chinese dialects +
// Cantonese, via the c_api) are two of this header's own callers.
//
// Every degenerate case below returned its input UNCHANGED before the fix.
// If you are here because one of them went red, do not "fix" the fixture:
// re-read PLAN.md §W1 and check whether the CJK gate stopped firing.
//
// Pure text transform — no model load.

#include <catch2/catch_test_macros.hpp>

#include "core/ngram_loop_fix.h"

#include <string>
#include <vector>
#include "portable_env.h"

using core_ngram::fix_loops;

// Repeat a UTF-8 string `n` times. Fixtures are written as explicit repeat
// counts so the degenerate length is visible at the call site.
static std::string rep(const std::string& unit, int n) {
    std::string out;
    for (int i = 0; i < n; i++)
        out += unit;
    return out;
}

// Count occurrences of `needle` in `hay` (overlapping scan, needles here are
// whole code-point units so overlap is not a concern).
static size_t count_of(const std::string& hay, const std::string& needle) {
    size_t c = 0;
    for (size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + 1))
        c++;
    return c;
}

TEST_CASE("single-kana flood collapses", "[ngram-loop][cjk]") {
    // The classic Whisper ja degenerate output on non-verbal audio: one kana
    // emitted until the token cap. No whitespace anywhere, so the word-level
    // path sees a single token and returns it verbatim.
    const std::string loop = rep("あ", 40);
    const std::string out = fix_loops(loop);

    REQUIRE(out != loop);
    REQUIRE(out.size() < loop.size());
    REQUIRE(count_of(out, "あ") <= 3);
}

TEST_CASE("kana phrase with separator collapses", "[ngram-loop][cjk]") {
    // "はい、" ×12 — a trigram cycle in code points, invisible to a whitespace
    // split.
    const std::string loop = rep("はい、", 12);
    const std::string out = fix_loops(loop);

    REQUIRE(out != loop);
    REQUIRE(count_of(out, "はい") <= 2);
}

TEST_CASE("the canonical zh closing-phrase loop collapses", "[ngram-loop][cjk]") {
    // "謝謝觀看，" ("thanks for watching") is *the* canonical Whisper Chinese
    // hallucination and the one that motivated §W1. Reducing the run is this
    // header's job; deleting the phrase outright is a blacklist's, and we
    // deliberately do not ship one (PLAN.md §"Deliberately NOT porting").
    const std::string loop = rep("謝謝觀看，", 8);
    const std::string out = fix_loops(loop);

    REQUIRE(out != loop);
    REQUIRE(count_of(out, "謝謝觀看") <= 2);
}

TEST_CASE("a good CJK prefix survives the collapse of a trailing loop", "[ngram-loop][cjk]") {
    // The transform must trim the degenerate tail without eating the real
    // transcript in front of it — the failure mode of a "replace the whole
    // line with the dominant unit" detector.
    const std::string prefix = "今日はいい天気ですね。";
    const std::string loop = prefix + rep("あ", 30);
    const std::string out = fix_loops(loop);

    REQUIRE(out.find(prefix) == 0);
    REQUIRE(out.size() < loop.size());
}

TEST_CASE("mixed CJK/Latin: each side is collapsed by its own path", "[ngram-loop][cjk]") {
    // A whitespace-delimited token that is majority CJK gets the code-point
    // path; the Latin tokens around it keep the word-level path.
    const std::string loop = "OK OK OK OK OK " + rep("うん", 20) + " done";
    const std::string out = fix_loops(loop);

    REQUIRE(out.find("done") != std::string::npos);
    REQUIRE(count_of(out, "OK") <= 3);
    REQUIRE(count_of(out, "うん") <= 2);
}

TEST_CASE("natural CJK text passes through byte-identical", "[ngram-loop][cjk]") {
    // The whole value of this transform is that it is a no-op on good text.
    // These are ordinary sentences; a collapse here is a false positive that
    // corrupts a correct transcript.
    const std::vector<std::string> clean = {
        "今日はいい天気ですね。",
        "私は日本語を勉強しています。",
        "这是一个测试句子。",
        "ありがとうございました。",
        // Genuine short reduplication — common and correct in both languages.
        "はいはい、わかりました。",
        "そこそこ美味しいですね。",
        "人人都可以学习。",
    };
    for (const auto& s : clean)
        REQUIRE(fix_loops(s) == s);
}

TEST_CASE("Latin behaviour is unchanged by the CJK path", "[ngram-loop][cjk]") {
    // The gate must not fire on Latin text, including words with doubled or
    // tripled letters that a naive code-point collapse would mangle.
    const std::vector<std::string> clean = {
        "Mississippi committee bookkeeper",
        "Fast! This is some sort of a test.",
        "no no no more games",
        "aaa",
    };
    for (const auto& s : clean)
        REQUIRE(fix_loops(s) == s);
}

TEST_CASE("CJK edge cases are safe", "[ngram-loop][cjk]") {
    REQUIRE(fix_loops("あ") == "あ");
    REQUIRE(fix_loops("ああ") == "ああ");
    // Short degenerate text stays under the minimum length for the CJK path —
    // too little evidence to call it a loop.
    REQUIRE(fix_loops("ええ") == "ええ");
    // Invalid/truncated UTF-8 must not crash or corrupt surviving bytes.
    const std::string truncated = "\xe3\x81";
    REQUIRE_NOTHROW(fix_loops(truncated));
}

TEST_CASE("CRISPASR_NGRAM_LOOPFIX_OFF disables the CJK path too", "[ngram-loop][cjk]") {
    // The diagnostic opt-out exists to expose the RAW decode. A CJK collapse
    // that ignored it would hide the very thing the flag is set to see.
    const std::string loop = rep("あ", 40);
    REQUIRE(fix_loops(loop) != loop); // gate is on by default

    setenv("CRISPASR_NGRAM_LOOPFIX_OFF", "1", 1);
    REQUIRE(fix_loops(loop) == loop);
    unsetenv("CRISPASR_NGRAM_LOOPFIX_OFF");

    REQUIRE(fix_loops(loop) != loop); // and back on
}

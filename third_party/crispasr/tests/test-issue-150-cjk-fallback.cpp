// test-issue-150-cjk-fallback.cpp — regression guard for issue #150.
//
// When the parakeet library returns n_words=0, crispasr_make_disp_segments
// falls back to split_text_at_punct.  Before the fix, that function had a
// CJK subtitle fallback that re-split any single-sentence text longer than
// 42 codepoints at raw character boundaries — regardless of whether the text
// was actually CJK.  On the reporter's English clip the word "about" was
// split across two SRT entries as "abo" / "ut".
//
// The fix: text_has_cjk() gates the 42-char fallback so it only fires for
// text containing CJK/kana/Hangul characters.

#include "crispasr_output.h"

#include <catch2/catch_test_macros.hpp>
#include <string>

// Helper: build a no-words segment spanning [t0, t1] with given text.
static crispasr_segment make_wordless(int64_t t0, int64_t t1, const std::string& text) {
    crispasr_segment s;
    s.t0   = t0;
    s.t1   = t1;
    s.text = text;
    // words intentionally empty — simulates parakeet n_words=0
    return s;
}

// The exact sentence from the issue reporter's clip (22.8 s, ~289 chars).
static const char* kIssue150Text =
    "Because what we have seen in the video about conversation threads is there "
    "is the agent thread uh that can keep track of this information and then you "
    "can serialize and deserialize that uh so you can store uh the various "
    "conversations you have, or you can do a self uh implemented way.";

// ─── English text must NOT be split at 42-char boundaries ────────────────────

TEST_CASE("issue #150: long English sentence with trailing period → one display segment",
          "[unit][output][issue-150]") {
    // Simulate the parakeet backend: one segment, no words, full transcript.
    crispasr_segment seg = make_wordless(16, 2240, kIssue150Text);
    std::vector<crispasr_segment> segs{seg};

    // The failing condition: split_on_punct=true, max_len=0, words empty.
    auto disp = crispasr_make_disp_segments(segs, /*max_len=*/0, /*split_on_punct=*/true);

    // Before fix: 7 segments of ~42 chars each, splitting "about" → "abo"/"ut".
    // After fix: exactly 1 segment (the whole sentence).
    REQUIRE(disp.size() == 1);
    REQUIRE(disp[0].text == kIssue150Text);
    REQUIRE(disp[0].t0 == 16);
    REQUIRE(disp[0].t1 == 2240);
}

TEST_CASE("issue #150: English text longer than 42 chars, no words, no split_on_punct → one segment",
          "[unit][output][issue-150]") {
    // Without split_on_punct the easy-exit path fires; still must be 1 segment.
    crispasr_segment seg = make_wordless(0, 2000, kIssue150Text);
    auto disp = crispasr_make_disp_segments({seg}, 0, false);
    REQUIRE(disp.size() == 1);
}

TEST_CASE("issue #150: English text with multiple sentence ends is split at sentence boundaries",
          "[unit][output][issue-150]") {
    // Two proper sentences → still two display segments, but at sentence
    // boundaries (not raw character positions).
    const char* two_sentences =
        "She said hello. He replied goodbye and walked away from the building.";
    crispasr_segment seg = make_wordless(0, 700, two_sentences);
    auto disp = crispasr_make_disp_segments({seg}, 0, true);

    REQUIRE(disp.size() == 2);
    // First segment must end with the period after "hello"
    REQUIRE(disp[0].text.back() == '.');
    // Second segment starts with "He"
    REQUIRE(disp[1].text.substr(0, 2) == "He");
}

TEST_CASE("issue #150: very short English text (≤42 chars) → one segment",
          "[unit][output][issue-150]") {
    const char* short_text = "Hello world.";
    crispasr_segment seg = make_wordless(0, 100, short_text);
    auto disp = crispasr_make_disp_segments({seg}, 0, true);
    REQUIRE(disp.size() == 1);
    REQUIRE(disp[0].text == short_text);
}

// ─── CJK text must STILL be split at ~42 char boundaries ─────────────────────

TEST_CASE("issue #150: long Japanese text without sentence-end → CJK fallback still fires",
          "[unit][output][issue-150][cjk]") {
    // A plausible long Japanese transcript with no 。 markers (some ASR
    // backends omit them).  utf8_len > 42 AND text_has_cjk → fallback fires.
    const char* ja_text =
        "会話スレッドで見たように代理スレッドがその情報を追跡し続けシリアライズしてデシリアライズ"
        "することができますそしてさまざまな会話を保存することもできます";
    crispasr_segment seg = make_wordless(0, 2000, ja_text);
    auto disp = crispasr_make_disp_segments({seg}, 0, true);

    // Should be split into multiple segments (CJK fallback active).
    REQUIRE(disp.size() > 1);
}

TEST_CASE("issue #150: short Japanese text (≤42 codepoints) → no CJK split",
          "[unit][output][issue-150][cjk]") {
    const char* ja_short = "こんにちは世界"; // 7 codepoints
    crispasr_segment seg = make_wordless(0, 100, ja_short);
    auto disp = crispasr_make_disp_segments({seg}, 0, true);
    REQUIRE(disp.size() == 1);
}

TEST_CASE("issue #150: Japanese text WITH sentence-end marker → sentence split, not 42-char",
          "[unit][output][issue-150][cjk]") {
    // Two Japanese sentences separated by 。 — should split at the boundary.
    const char* ja_two = "会話スレッドで見たように代理スレッドが追跡します。シリアライズすることができます。";
    crispasr_segment seg = make_wordless(0, 2000, ja_two);
    auto disp = crispasr_make_disp_segments({seg}, 0, true);
    REQUIRE(disp.size() == 2);
}

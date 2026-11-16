// tests/test-cosyvoice3-prompt-policy.cpp — #334.
//
// The diff harness stops at the logits, so nothing it checks can see these
// two laws: they live in the decode loop and in the advice the runtime gives
// about `--ref-text`. Both are pure arithmetic over token counts, so they get
// hermetic tests here.

#include "cosyvoice3_prompt_policy.h"

#include <catch2/catch_test_macros.hpp>

using namespace cosyvoice3_policy;

TEST_CASE("decode floor is 2 speech tokens per target text token", "[unit][cosyvoice3]") {
    // Upstream: min_len = int((text_len - prompt_text_len) * min_token_text_ratio).
    REQUIRE(decode_min_tokens(30) == 60);
    REQUIRE(decode_min_tokens(1) == 2);
    // No text, no floor — a floor here would mask the stop token forever.
    REQUIRE(decode_min_tokens(0) == 0);
    REQUIRE(decode_min_tokens(-5) == 0);
}

TEST_CASE("decode floor never swallows the step budget", "[unit][cosyvoice3]") {
    // A floor at or past max_steps masks the stop token for the whole decode,
    // so a short line would come out as max_steps frames of filler instead of
    // ending. It must always leave at least one step where stopping is legal.
    REQUIRE(clamp_min_tokens(60, 600) == 60);
    REQUIRE(clamp_min_tokens(600, 600) == 599);
    REQUIRE(clamp_min_tokens(5000, 16) == 15);
    REQUIRE(clamp_min_tokens(0, 600) == 0);
    REQUIRE(clamp_min_tokens(60, 0) == 0);
}

TEST_CASE("reference transcript length is judged against the model's own band", "[unit][cosyvoice3]") {
    // Real pairings measured on the #334 reproduction. A 4.26 s clip is 106
    // speech tokens; "However, due to the slow communication channels," is 9
    // text tokens → 11.8, comfortably inside.
    REQUIRE(prompt_length_plausible(106, 9));
    // A 17.7 s clip (442 tokens) with its full 60-token transcript → 7.4.
    REQUIRE(prompt_length_plausible(442, 60));
    // The same clip labelled with only "Scientists have discovered the
    // northern lights." (7 tokens) → 63, which is what actually collapsed the
    // decode. This is the case the warning exists for.
    REQUIRE_FALSE(prompt_length_plausible(442, 7));
    // Transcript far longer than the audio can hold — the mirror-image
    // mistake, and the one that makes the clone rush.
    REQUIRE_FALSE(prompt_length_plausible(50, 40));

    // Exactly on the boundaries is inside the band, not outside: upstream's
    // ratios are the range the decode itself works in.
    REQUIRE(prompt_length_plausible(20, 10));  // ratio 2
    REQUIRE(prompt_length_plausible(200, 10)); // ratio 20
    REQUIRE_FALSE(prompt_length_plausible(19, 10));
    REQUIRE_FALSE(prompt_length_plausible(201, 10));

    // Nothing to judge — never warn on an empty side.
    REQUIRE(prompt_length_plausible(0, 10));
    REQUIRE(prompt_length_plausible(200, 0));
}

TEST_CASE("the band and the token rate are upstream's constants", "[unit][cosyvoice3]") {
    // A typo in either silently rescales every judgement above, and nothing
    // downstream would notice.
    REQUIRE(kMinTokenTextRatio == 2);
    REQUIRE(kMaxTokenTextRatio == 20);
    // 25 Hz speech tokens; the flow's token_mel_ratio=2 puts mel at 50 Hz and
    // HiFT's hop of 480 at 24 kHz puts audio back at 50 frames/s.
    REQUIRE(kSecondsPerSpeechToken * 25.0 == 1.0);
}

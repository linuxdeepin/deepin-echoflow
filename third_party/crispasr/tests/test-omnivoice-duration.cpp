// #363 — OmniVoice "fragments of the ends of phrases".
//
// OmniVoice is a masked iterative generator: it is handed a target length T and
// fills those frames. There is no EOS and no way to ask for more room, so a T
// that is too small for the text does not produce a rushed clip — it produces
// the start of the utterance and a fragment where the end belongs.
//
// T comes from a speaking rate, and with a voice prompt that rate is measured
// from the caller's reference: weight(ref_text) / ref_T. Nothing checked that
// the pair was self-consistent, so a transcript that did not match the audio
// went straight into the length. Measured on an M1 with omnivoice-f16, cloning
// a 2.60 s reference for a 22-word line — a matching ref_text gave 6.60 s and
// the complete sentence; a ref_text 3.2x too long for the same audio gave
// 2.04 s of "and then continued running through the field for f and then
// continued running the field for a very long time", i.e. fragments of the end
// of the *reference*.
//
// These cases pin the guard that rejects such a pair, and the weights it rests
// on. The weights are a port of OmniVoice's rule-based estimator and are what
// makes the rate comparable across scripts at all.
#include "core/omnivoice_duration.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace core_omnivoice_duration;

namespace {
// The reference actually used in the measurement above: OmniVoice's own default
// voice saying this line, 65 frames at 40 ms = 2.60 s.
const std::string kRefText = "The quick brown fox jumps over the lazy dog.";
constexpr int kRefT = 65;

// The 22-word line that came out as fragments.
const std::string kTarget = "The report is finished. I left a copy on your desk this morning. "
                            "It covers the last three quarters in some detail.";
} // namespace

TEST_CASE("omnivoice duration: a matching reference is accepted", "[omnivoice][duration]") {
    REQUIRE(ref_rate_is_plausible(kRefText, kRefT));
    // 0.571 in the measurement; assert the band rather than the digits so a
    // weight-table tweak does not fail this for the wrong reason.
    const double rate = duration_text_weight(kRefText) / (double)kRefT;
    REQUIRE(rate > kMinRefRate);
    REQUIRE(rate < kMaxRefRate);
}

TEST_CASE("omnivoice duration: the reference that produced #363 is rejected", "[omnivoice][duration]") {
    // Same audio, transcript 3.2x too long — the exact input that rendered as
    // fragments before the guard existed.
    const std::string too_long = kRefText + " and then continued running through the field for a very long "
                                            "time indeed without stopping once.";
    REQUIRE_FALSE(ref_rate_is_plausible(too_long, kRefT));
    REQUIRE(duration_text_weight(too_long) / (double)kRefT > kMaxRefRate);
}

TEST_CASE("omnivoice duration: a rejected reference falls back, not truncates", "[omnivoice][duration]") {
    // The point of the fix. A bad reference must not shorten T — the whole
    // failure mode is a T too small for the text.
    const std::string too_long = kRefText + " and then continued running through the field for a very long "
                                            "time indeed without stopping once.";
    const int with_bad_ref = estimate_target_tokens(kTarget, too_long, kRefT);
    const int with_no_ref = estimate_target_tokens(kTarget);

    // Falling back means landing exactly on the no-reference anchor.
    REQUIRE(with_bad_ref == with_no_ref);

    // And that anchor must be enough room for the line. The measured good
    // render was 6.60 s = 165 frames; anything near the 2.04 s (51 frames)
    // failure is what we are ruling out.
    REQUIRE(with_bad_ref > 120);
}

TEST_CASE("omnivoice duration: a good reference is still honoured", "[omnivoice][duration]") {
    // The guard must not flatten every clone back onto the built-in anchor —
    // tracking the reference speaker's rate is why #254 wanted this path.
    const int with_ref = estimate_target_tokens(kTarget, kRefText, kRefT);
    REQUIRE(with_ref > 120);
    // A slower reference (same words, more frames) must ask for more room.
    const int slower = estimate_target_tokens(kTarget, kRefText, kRefT * 2);
    REQUIRE(slower > with_ref);
}

TEST_CASE("omnivoice duration: degenerate references do not divide by zero", "[omnivoice][duration]") {
    REQUIRE_FALSE(ref_rate_is_plausible("", 65));
    REQUIRE_FALSE(ref_rate_is_plausible(kRefText, 0));
    REQUIRE_FALSE(ref_rate_is_plausible(kRefText, -1));
    // Punctuation-only carries weight 0.5/char, so this is a real rate, just an
    // implausible one — it must be refused rather than divided by.
    REQUIRE_FALSE(ref_rate_is_plausible("...", 65));
    // And each still yields a usable length via the anchor.
    REQUIRE(estimate_target_tokens(kTarget, "", 65) >= 10);
    REQUIRE(estimate_target_tokens(kTarget, kRefText, 0) >= 10);
}

TEST_CASE("omnivoice duration: speed scales the target length", "[omnivoice][duration]") {
    const int normal = estimate_target_tokens(kTarget, kRefText, kRefT, 1.0f);
    const int faster = estimate_target_tokens(kTarget, kRefText, kRefT, 2.0f);
    const int slower = estimate_target_tokens(kTarget, kRefText, kRefT, 0.5f);
    REQUIRE(faster < normal);
    REQUIRE(slower > normal);
}

TEST_CASE("omnivoice duration: weights are script-aware", "[omnivoice][duration]") {
    // The rate is only comparable across languages because the weights differ
    // per script. If these collapsed to "count the bytes", a CJK reference
    // would read as an absurd rate and every clone in those languages would be
    // pushed onto the anchor — the guard would become a regression.
    REQUIRE(duration_cp_weight(U'a') == 1.0);
    REQUIRE(duration_cp_weight(U'7') > duration_cp_weight(U'a'));  // digits are spoken as words
    REQUIRE(duration_cp_weight(U'中') > duration_cp_weight(U'a')); // CJK ideograph
    REQUIRE(duration_cp_weight(U'あ') > duration_cp_weight(U'a')); // kana
    REQUIRE(duration_cp_weight(U' ') < duration_cp_weight(U'a'));  // separator

    // A CJK reference at a normal rate must pass. Ten ideographs at weight 3.0
    // is 30 weighted chars; over 40 frames that is 0.75 — inside the band.
    REQUIRE(ref_rate_is_plausible("你好世界今天天气很好", 40));

    // Multi-byte decoding must actually work: same string, weight must exceed
    // what a naive per-byte count of ASCII weight would give.
    REQUIRE(duration_text_weight("你好") == 6.0);
}

// ===========================================================================
// Parity with upstream's RuleDurationEstimator (k2-fsa/OmniVoice)
// ===========================================================================
//
// The weights are a port of omnivoice/utils/duration.py, and the port had
// drifted. Upstream dispatches on the Unicode CATEGORY first — P*/S* -> 0.5,
// Z* -> 0.2, M* -> 0.0, N* -> 3.5 — and only then consults its script ranges.
// This port hardcoded ranges, so anything punctuation-like outside ASCII/CJK
// fell through to the 1.0 "letter-ish" default and was costed as a spoken
// character.
//
// The expected values below are what upstream's own Python produces for these
// exact strings, checked by running it — not by reading our implementation
// back to itself. What it cost in practice:
//
//   "Москва — большой …"       51.3 here vs 50.8 upstream  (em-dash)
//   "मैं आज बाज़ार …"           41.6 vs 40.3                (danda)
//   "مرحبا، كيف حالك اليوم؟"   29.1 vs 27.1                (Arabic , and ?)
//
// Devanagari matras and Arabic/Hebrew vowel marks were the worst of it: costed
// as full characters instead of silent, so those languages were systematically
// over-length.

static double W(const char* s) {
    return core_omnivoice_duration::duration_text_weight(s);
}

TEST_CASE("omnivoice duration: weights match upstream on real prose", "[unit][duration][omnivoice]") {
    CHECK(W("Hello, world.") == Catch::Approx(11.2));
    CHECK(W("Папа у Васи силён в математике, учится папа за Васю весь год.") == Catch::Approx(51.2));
    CHECK(W("Москва — большой и очень красивый город с богатой историей.") == Catch::Approx(50.8));
    CHECK(W("Привет! Как дела? Всё хорошо…") == Catch::Approx(24.3));
    CHECK(W("Цена — 1500 рублей (со скидкой).") == Catch::Approx(36.0));
    CHECK(W("मैं आज बाज़ार जा रहा हूँ और वहाँ से फल खरीदूँगा।") == Catch::Approx(40.3));
    CHECK(W("שלום, מה שלומך היום?") == Catch::Approx(24.1));
    CHECK(W("مرحبا، كيف حالك اليوم؟") == Catch::Approx(27.1));
    CHECK(W("今日はいい天気ですね。散歩に行きましょう。") == Catch::Approx(48.4));
}

TEST_CASE("omnivoice duration: the character classes that had drifted", "[unit][duration][omnivoice]") {
    using core_omnivoice_duration::duration_cp_weight;
    // Punctuation outside ASCII is a pause, not a syllable.
    CHECK(duration_cp_weight(0x2014) == Catch::Approx(0.5)); // em dash
    CHECK(duration_cp_weight(0x2026) == Catch::Approx(0.5)); // ellipsis
    CHECK(duration_cp_weight(0x00AB) == Catch::Approx(0.5)); // « guillemet, ubiquitous in Russian
    CHECK(duration_cp_weight(0x0964) == Catch::Approx(0.5)); // danda — ends every Hindi sentence
    CHECK(duration_cp_weight(0x060C) == Catch::Approx(0.5)); // Arabic comma

    // Combining marks are silent modifiers.
    CHECK(duration_cp_weight(0x0941) == Catch::Approx(0.0)); // Devanagari vowel sign U
    CHECK(duration_cp_weight(0x05B4) == Catch::Approx(0.0)); // Hebrew hiriq
    CHECK(duration_cp_weight(0x064E) == Catch::Approx(0.0)); // Arabic fatha

    // …but a letter sitting inside a mark block is NOT silent. The hand-written
    // range chain silenced these; the generated table cannot, because it asks
    // upstream rather than guessing at block boundaries.
    CHECK(duration_cp_weight(0x093D) == Catch::Approx(1.8)); // Devanagari avagraha (Lo)
    CHECK(duration_cp_weight(0x00B5) == Catch::Approx(1.0)); // µ micro sign (Ll)
    CHECK(duration_cp_weight(0x00AA) == Catch::Approx(1.0)); // ª ordinal indicator (Lo)

    // Digits are spoken as words in any script.
    CHECK(duration_cp_weight(0x0966) == Catch::Approx(3.5)); // Devanagari zero
    CHECK(duration_cp_weight(0x0660) == Catch::Approx(3.5)); // Arabic-Indic zero
}

TEST_CASE("omnivoice duration: upstream-exact mode is available and differs only where documented",
          "[unit][duration][omnivoice]") {
    using core_omnivoice_duration::duration_cp_weight;
    using core_omnivoice_duration::duration_cp_weight_upstream;

    // The ONE deliberate divergence: format characters. Upstream's dispatch
    // handles categories M/P/S/Z/N but not Cf, so zero-width and bidi controls
    // fall through to its script search and land in the kana block — a
    // zero-width joiner costed as a spoken Japanese syllable. ZWJ and ZWNJ are
    // orthographic in Indic and Arabic text, so we silence them by default.
    CHECK(duration_cp_weight_upstream(0x200D) == Catch::Approx(2.2)); // upstream: ZWJ
    CHECK(duration_cp_weight_upstream(0x202B) == Catch::Approx(2.2)); // upstream: RLE
    CHECK(duration_cp_weight(0x200D) == Catch::Approx(0.0));          // ours
    CHECK(duration_cp_weight(0x202B) == Catch::Approx(0.0));          // ours

    // Everywhere else the two agree, because ours IS the generated table.
    for (uint32_t cp : {0x0041u, 0x0430u, 0x2014u, 0x0941u, 0x4E00u, 0x3042u, 0x0966u, 0x00B5u}) {
        INFO("cp=" << cp);
        CHECK(duration_cp_weight(cp) == duration_cp_weight_upstream(cp));
    }
}

TEST_CASE("omnivoice duration: the run table is well formed", "[unit][duration][omnivoice]") {
    // A binary search over an unsorted or overlapping table returns silently
    // wrong weights rather than failing, so the shape is asserted here.
    using core_omnivoice_duration::kUpstreamWeightRunCount;
    using core_omnivoice_duration::kUpstreamWeightRuns;
    REQUIRE(kUpstreamWeightRunCount > 1000);
    for (int i = 0; i < kUpstreamWeightRunCount; i++) {
        INFO("run " << i);
        REQUIRE(kUpstreamWeightRuns[i].lo <= kUpstreamWeightRuns[i].hi);
        if (i > 0)
            REQUIRE(kUpstreamWeightRuns[i].lo > kUpstreamWeightRuns[i - 1].hi);
    }
    REQUIRE(kUpstreamWeightRuns[0].lo == 0);
}

TEST_CASE("omnivoice duration: target-token count matches upstream, edges included", "[unit][duration][omnivoice]") {
    using core_omnivoice_duration::estimate_target_tokens;
    const std::string ru = "Папа у Васи силён в математике, учится папа за Васю весь год.";

    // Values produced by running upstream's _estimate_target_tokens with the
    // same built-in anchor ("Nice to meet you." / 25 tokens).
    CHECK(estimate_target_tokens(ru) == 90);
    CHECK(estimate_target_tokens("Hello, world.") == 36);
    CHECK(estimate_target_tokens("今日はいい天気ですね。") == 48);
    CHECK(estimate_target_tokens(" ") == 9); // upstream floors at 1, not 10

    // Upstream applies the speed divisor only for speed > 0 && != 1.0.
    CHECK(estimate_target_tokens("Hello, world.", {}, 0, 0.5f) == 73);

    // The guard is why these are here. Dividing unconditionally made
    // --tts-speed 0 yield est=inf and a target of 2147483647 frames, and a
    // negative speed yield a negative estimate that only the old floor caught.
    CHECK(estimate_target_tokens("Hello, world.", {}, 0, 0.0f) == 36);
    CHECK(estimate_target_tokens("Hello, world.", {}, 0, -1.0f) == 36);
}

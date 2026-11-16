// test-session-introspection.cpp — unit tests for the exception-safe session
// introspection accessors added in PR #259 (CTC vocab + whisper no_speech_prob
// + detected_language).
//
// Strategy: exercise the null-handle / bad-arg / out-of-range guard paths with
// NO model loaded — sub-millisecond, no network, safe on every CI tier. The
// point of these accessors is a *sentinel contract*: a caller must be able to
// tell "unavailable" (null / wrong backend / OOB index) apart from a genuine
// value. A regression here silently breaks that contract (e.g. a consumer reads
// a garbage no_speech_prob as a real posterior, or token_text OOBs). The
// with-model behaviour (real posteriors, CTC vocab round-trip, language
// fallback) is covered by the [.live] Rust integration tests.
//
// Coverage: crispasr_session_detected_language,
// crispasr_session_result_segment_no_speech_prob, crispasr_session_n_vocab,
// crispasr_session_token_text — declared in include/crispasr_session.h.

#include <catch2/catch_test_macros.hpp>

#include <cstring>

#include "crispasr_session.h"

// ─── detected_language: bad-arg guards return -1 ────────────────────────────

TEST_CASE("session introspection: detected_language null-session → -1", "[unit][introspection]") {
    char buf[8] = {0};
    REQUIRE(crispasr_session_detected_language(nullptr, buf, sizeof(buf)) == -1);
}

TEST_CASE("session introspection: detected_language null-buffer → -1", "[unit][introspection]") {
    // A non-null session is not needed to hit the arg guard: out_buf == nullptr
    // must be rejected before any session field is touched.
    REQUIRE(crispasr_session_detected_language(nullptr, nullptr, 8) == -1);
}

TEST_CASE("session introspection: detected_language non-positive capacity → -1", "[unit][introspection]") {
    char buf[8] = {0};
    REQUIRE(crispasr_session_detected_language(nullptr, buf, 0) == -1);
    REQUIRE(crispasr_session_detected_language(nullptr, buf, -1) == -1);
}

// ─── no_speech_prob: null result / OOB segment → -1.0 sentinel ──────────────

TEST_CASE("session introspection: no_speech_prob null-result → -1.0 sentinel", "[unit][introspection]") {
    // -1.0 is the "no data" sentinel; a real posterior is always in [0, 1], so a
    // caller can branch on `< 0` to detect unavailability.
    REQUIRE(crispasr_session_result_segment_no_speech_prob(nullptr, 0) == -1.0f);
}

TEST_CASE("session introspection: no_speech_prob negative segment index → -1.0", "[unit][introspection]") {
    REQUIRE(crispasr_session_result_segment_no_speech_prob(nullptr, -1) == -1.0f);
}

// ─── CTC vocab: null-session guards (0 / "") ────────────────────────────────

TEST_CASE("session introspection: n_vocab null-session → 0", "[unit][introspection]") {
    // 0 means "this backend exposes no CTC vocab" — same value a non-CTC backend
    // returns, so a caller iterating [0, n_vocab) simply gets an empty range.
    REQUIRE(crispasr_session_n_vocab(nullptr) == 0);
}

TEST_CASE("session introspection: token_text null-session → non-null empty string", "[unit][introspection]") {
    // Must return a valid C string (""), never nullptr — callers pass it
    // straight to strlen/printf.
    const char* t = crispasr_session_token_text(nullptr, 0);
    REQUIRE(t != nullptr);
    REQUIRE(std::strcmp(t, "") == 0);
}

TEST_CASE("session introspection: token_text negative id → empty string", "[unit][introspection]") {
    // The top-level id < 0 guard must fire before any per-backend vocab index
    // (some backend branches only bounds-check the upper end).
    const char* t = crispasr_session_token_text(nullptr, -1);
    REQUIRE(t != nullptr);
    REQUIRE(std::strcmp(t, "") == 0);
}

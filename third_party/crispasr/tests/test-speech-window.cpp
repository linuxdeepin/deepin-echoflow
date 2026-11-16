// tests/test-speech-window.cpp — the min/max speech-token window arithmetic.
//
// PR #330 added `min_speech_tokens` so a caller can pin synthesis to an exact
// audio window (set min == max). These guard the part of that contract which is
// pure arithmetic and therefore checkable without a model: WHEN the requested
// window is deliverable, and when it silently is not.
//
// The decode loop itself (src/moss_tts_local.cpp) needs weights and is covered
// by the live tests; what is tested here is the reasoning a caller depends on.
//
// Run: ctest -R test-speech-window --output-on-failure

#include <catch2/catch_test_macros.hpp>

#include "crispasr_speech_window.h"

using crispasr_speech_window::diagnose;
using crispasr_speech_window::Window;

namespace {
Window w(int min_f, int max_f, int cap = crispasr_speech_window::kDefaultFrameCap) {
    Window x;
    x.min_frames = min_f;
    x.max_frames = max_f;
    x.frame_cap = cap;
    return x;
}
} // namespace

// ── The feature working as advertised ──────────────────────────────────────

TEST_CASE("min == max is an exact window and is deliverable by default", "[unit][pr330]") {
    // The PR's own documented example: 250 frames = 20 s, with max_new_tokens
    // 600 leaving the loop plenty of room.
    const Window x = w(250, 250, 600);
    REQUIRE(x.is_exact());
    REQUIRE_FALSE(x.floor_unreachable());
    REQUIRE_FALSE(x.floor_above_ceiling());
    REQUIRE(x.effective_ceiling() == 250);
    REQUIRE(diagnose(x).empty()); // nothing to warn about
}

TEST_CASE("frames convert to the seconds callers actually reason in", "[unit][pr330]") {
    REQUIRE(crispasr_speech_window::frames_to_seconds(250) == 20.0);
    REQUIRE(crispasr_speech_window::frames_to_seconds(0) == 0.0);
    REQUIRE(crispasr_speech_window::frames_to_seconds(-5) == 0.0);
}

TEST_CASE("an unset window is unconstrained and warns about nothing", "[unit][pr330]") {
    const Window x = w(-1, -1);
    REQUIRE_FALSE(x.is_exact());
    REQUIRE_FALSE(x.floor_unreachable());
    REQUIRE(x.effective_ceiling() == crispasr_speech_window::kDefaultFrameCap);
    REQUIRE(diagnose(x).empty());
}

// ── The failure the warning exists for ─────────────────────────────────────

TEST_CASE("a floor above the loop bound is unreachable and says so", "[unit][pr330]") {
    // The trap: max_new_tokens is a general AR knob, so a caller who set it low
    // for some other reason carries it in. The floor is then never reached, the
    // loop stops first, and the audio is SHORTER than the slot it was cut for —
    // silently, because nothing errors.
    const Window x = w(250, 250, 100);
    REQUIRE(x.floor_unreachable());
    const std::string msg = diagnose(x);
    REQUIRE_FALSE(msg.empty());
    REQUIRE(msg.find("cannot be reached") != std::string::npos);
    REQUIRE(msg.find("250") != std::string::npos); // the floor asked for
    REQUIRE(msg.find("100") != std::string::npos); // the bound that wins
    // And the caller really does get the smaller number.
    REQUIRE(x.effective_ceiling() == 100);
}

TEST_CASE("the default 4096-frame cap is the bound when max_new_tokens is unset", "[unit][pr330]") {
    // Mirrors moss_tts_local.cpp: max_frames = sp.max_new_frames > 0 ? … : 4096.
    // A floor beyond it is unreachable even though the caller set no cap.
    REQUIRE_FALSE(w(4096, 4096).floor_unreachable());
    REQUIRE(w(4097, 4097).floor_unreachable());
}

TEST_CASE("a floor above the ceiling degrades to the ceiling, and reports it", "[unit][pr330]") {
    // The PR documents this as graceful degradation, which it is — the max
    // break runs right after each frame is appended, while the floor only
    // suppresses the stop head. Still worth a line: the caller meant something
    // else, and would otherwise never learn which bound won.
    const Window x = w(300, 200, 4096);
    REQUIRE(x.floor_above_ceiling());
    REQUIRE(x.effective_ceiling() == 200);
    const std::string msg = diagnose(x);
    REQUIRE(msg.find("the ceiling wins") != std::string::npos);
    REQUIRE(msg.find("200") != std::string::npos);
}

TEST_CASE("unreachable outranks floor-above-ceiling in the diagnosis", "[unit][pr330]") {
    // Both wrong at once: report the one that changes the outcome most, and
    // report exactly one line so a log rule can match it.
    const Window x = w(500, 300, 100);
    REQUIRE(x.floor_unreachable());
    REQUIRE(x.floor_above_ceiling());
    REQUIRE(diagnose(x).find("cannot be reached") != std::string::npos);
}

// ── Boundaries, where an off-by-one would quietly change the contract ──────

TEST_CASE("the floor is reachable exactly at the bound, not one short", "[unit][pr330]") {
    REQUIRE_FALSE(w(100, 100, 100).floor_unreachable()); // equal is fine
    REQUIRE(w(101, 101, 100).floor_unreachable());       // one over is not
}

TEST_CASE("a zero or negative floor is not a constraint", "[unit][pr330]") {
    // The adapters gate on `>= 0`, so 0 reaches the sampler and means "no
    // floor". It must never be reported as unreachable.
    REQUIRE_FALSE(w(0, -1, 10).floor_unreachable());
    REQUIRE_FALSE(w(-1, -1, 10).floor_unreachable());
    REQUIRE(diagnose(w(0, -1, 10)).empty());
}

TEST_CASE("a ceiling alone still bounds the output", "[unit][pr330]") {
    const Window x = w(-1, 50, 4096);
    REQUIRE_FALSE(x.is_exact());
    REQUIRE(x.effective_ceiling() == 50);
    REQUIRE(diagnose(x).empty());
}

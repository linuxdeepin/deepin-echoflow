// test-marking-policy.cpp — unit tests for the spoken-AI-disclaimer opt-out gate.
//
// Pins the policy #312 fixed: a voice-clone request that sets
// "spoken_disclaimer": false without an attestation must be DENIED THE OPT-OUT,
// not refused outright — v0.8.22/v0.8.23 returned 400 there, which took out
// voice cloning for every Subtitle Edit build up to v5.1.0-rc16.
//
// Header-only decision logic — no server, no socket, no models.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_marking_policy.h"

using crispasr_marking::decide;

TEST_CASE("non-clone output is not policed", "[unit][marking]") {
    // Nothing prepends a disclaimer to non-clone output, so the field is a
    // no-op rather than an opt-out — never a denial, never a disclaimer.
    for (bool asked : {true, false}) {
        auto d = decide(/*needs_spoken_disclaimer=*/false, /*requested_spoken_disclaimer=*/asked, "", false, "");
        REQUIRE_FALSE(d.apply_spoken_disclaimer);
        REQUIRE_FALSE(d.optout_denied);
        REQUIRE_FALSE(d.optout_honored);
    }
}

TEST_CASE("a clone that does not ask to skip gets the disclaimer", "[unit][marking]") {
    auto d = decide(true, /*requested_spoken_disclaimer=*/true, "", false, "");
    REQUIRE(d.apply_spoken_disclaimer);
    REQUIRE_FALSE(d.optout_denied);
    REQUIRE_FALSE(d.optout_honored);
}

TEST_CASE("an attested opt-out is honored", "[unit][marking]") {
    auto d = decide(true, false, "I will disclose this is AI-generated", false, "");
    REQUIRE(d.optout_honored);
    REQUIRE_FALSE(d.apply_spoken_disclaimer);
    REQUIRE_FALSE(d.optout_denied);
    REQUIRE(d.attestation == "I will disclose this is AI-generated");
    REQUIRE(d.scope == "request");
}

TEST_CASE("#312: an unattested opt-out is denied, not refused", "[unit][marking]") {
    auto d = decide(true, /*requested_spoken_disclaimer=*/false, "", false, "");
    // The load-bearing assertion: the request still produces audio.
    REQUIRE(d.optout_denied);
    REQUIRE(d.apply_spoken_disclaimer); // served with the default, not rejected
    REQUIRE_FALSE(d.optout_honored);
    REQUIRE(d.attestation.empty());
    REQUIRE(d.scope.empty());
}

TEST_CASE("the server's --accept-marking-responsibility satisfies the gate", "[unit][marking]") {
    // The operator accepted the duty for every response the process serves,
    // which subsumes the per-request field.
    auto d = decide(true, false, "", /*server_accepted=*/true, "CLI --accept-marking-responsibility flag");
    REQUIRE(d.optout_honored);
    REQUIRE_FALSE(d.apply_spoken_disclaimer);
    REQUIRE(d.scope == "server");
    REQUIRE(d.attestation == "CLI --accept-marking-responsibility flag");
}

TEST_CASE("an operator attestation with no prose still names its source", "[unit][marking]") {
    auto d = decide(true, false, "", true, "");
    REQUIRE(d.optout_honored);
    REQUIRE(d.scope == "server");
    REQUIRE(d.attestation == "(unspecified)"); // audit line always has a source
}

TEST_CASE("the request attestation wins over the server one", "[unit][marking]") {
    auto d = decide(true, false, "per-request words", true, "server words");
    REQUIRE(d.optout_honored);
    REQUIRE(d.scope == "request");
    REQUIRE(d.attestation == "per-request words");
}

TEST_CASE("an attestation on a request that keeps the disclaimer changes nothing", "[unit][marking]") {
    // Sending marking_attestation without spoken_disclaimer:false is not an
    // opt-out; the disclaimer stays on and nothing is logged as honored.
    auto d = decide(true, /*requested_spoken_disclaimer=*/true, "I will disclose this is AI-generated", true, "x");
    REQUIRE(d.apply_spoken_disclaimer);
    REQUIRE_FALSE(d.optout_honored);
    REQUIRE_FALSE(d.optout_denied);
}

// ---------------------------------------------------------------------------
// decide_raw_surface — container-less surfaces (Wyoming audio-chunk PCM).
//
// The Wyoming server shipped for four releases with no watermark, no clone
// classification, no spoken disclaimer and no consent gate, because the marking
// work was organised from a list of surfaces in prose that it was never on.
// These cases pin the rule so it can go red here rather than only in a live
// server with a model loaded.
// ---------------------------------------------------------------------------

using crispasr_marking::decide_raw_surface;

TEST_CASE("a container-less surface always forces the watermark", "[unit][marking]") {
    // No container ⇒ no C2PA manifest is possible ⇒ the watermark is the only
    // mark obtainable, so --no-watermark must never reach it. True on every
    // arm, including the refusal.
    REQUIRE(
        decide_raw_surface(/*is_clone=*/false, /*operator_consent=*/false, /*needs_disclosure=*/false).force_watermark);
    REQUIRE(decide_raw_surface(false, true, false).force_watermark);
    REQUIRE(decide_raw_surface(true, true, true).force_watermark);
    REQUIRE(decide_raw_surface(true, false, true).force_watermark);
}

TEST_CASE("a preset voice is marked but neither gated nor disclaimed", "[unit][marking]") {
    auto d = decide_raw_surface(/*is_clone=*/false, /*operator_consent=*/false, /*needs_disclosure=*/false);
    REQUIRE_FALSE(d.refuse);
    REQUIRE_FALSE(d.apply_spoken_disclaimer);
    REQUIRE(d.force_watermark);
}

TEST_CASE("a clone with no operator consent is refused", "[unit][marking]") {
    // The protocol carries no consent field, so the operator's launch-time
    // --i-have-rights is the only attestation there can be. Without it the
    // alternative is emitting an ungated clone of a real person.
    auto d = decide_raw_surface(/*is_clone=*/true, /*operator_consent=*/false, /*needs_disclosure=*/true);
    REQUIRE(d.refuse);
    REQUIRE_FALSE(d.apply_spoken_disclaimer);
}

TEST_CASE("a consented clone is served WITH the audible disclosure", "[unit][marking]") {
    // Art. 50(4). Not opt-out-able on this surface: opting out requires an
    // attestation (#312) and there is no request field to carry one.
    auto d = decide_raw_surface(/*is_clone=*/true, /*operator_consent=*/true, /*needs_disclosure=*/true);
    REQUIRE_FALSE(d.refuse);
    REQUIRE(d.apply_spoken_disclaimer);
    REQUIRE(d.force_watermark);
}

TEST_CASE("a real-person PRESET is disclosed but never refused", "[unit][marking]") {
    // The case the identity split exists for. `is_clone` is false — no
    // recording passed through a baker, so there is nothing for this operator
    // to attest to and refusing would be theatre. But the voice IS an
    // identifiable person, so the audio is a deep fake under Art. 3(60) and the
    // audible label is owed. Before the split this served silently.
    auto d = decide_raw_surface(/*is_clone=*/false, /*operator_consent=*/false, /*needs_disclosure=*/true);
    REQUIRE_FALSE(d.refuse);
    REQUIRE(d.apply_spoken_disclaimer);
    REQUIRE(d.force_watermark);
}

TEST_CASE("a real-person preset does not need operator consent", "[unit][marking]") {
    // Same output with and without --i-have-rights: the consent gate keys on
    // cloning alone. Whether the donor agreed to the model being trained is a
    // licensing question settled upstream that this operator cannot answer.
    const auto without = decide_raw_surface(false, /*operator_consent=*/false, /*needs_disclosure=*/true);
    const auto with = decide_raw_surface(false, /*operator_consent=*/true, /*needs_disclosure=*/true);
    REQUIRE(without.refuse == with.refuse);
    REQUIRE(without.apply_spoken_disclaimer == with.apply_spoken_disclaimer);
    REQUIRE_FALSE(without.refuse);
}

TEST_CASE("consent and disclosure are decided independently", "[unit][marking]") {
    // All four corners, because collapsing these two questions into one boolean
    // is the bug this signature exists to prevent.
    REQUIRE_FALSE(decide_raw_surface(false, false, false).apply_spoken_disclaimer); // synthetic preset
    REQUIRE(decide_raw_surface(false, false, true).apply_spoken_disclaimer);        // real-person preset
    REQUIRE(decide_raw_surface(true, false, true).refuse);                          // ungated clone
    REQUIRE(decide_raw_surface(true, true, true).apply_spoken_disclaimer);          // consented clone
    // A clone is refused on consent grounds even if nothing would be disclosed;
    // the refusal must not depend on the disclosure answer.
    REQUIRE(decide_raw_surface(true, false, false).refuse);
}

TEST_CASE("operator consent never silences the disclosure", "[unit][marking]") {
    // --i-have-rights attests the RIGHT to clone; it says nothing about telling
    // the listener the audio is synthetic. Conflating the two would make every
    // Home Assistant clone undisclosed.
    REQUIRE(decide_raw_surface(true, true, true).apply_spoken_disclaimer);
}

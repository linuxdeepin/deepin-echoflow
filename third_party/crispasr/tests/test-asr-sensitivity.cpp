// test-asr-sensitivity.cpp — named decode-threshold bundles (PLAN.md §W7).
//
// The presets are only useful if they are COHERENT: the whisper fallback at
// crispasr.cpp:8499 requires avg_logprob < logprob_thold AND no_speech_prob <
// no_speech_thold together, so moving one knob without the other produces a
// combination that does not mean what its name says. These tests pin the
// direction of each knob against `balanced`, which is what stops a later edit
// quietly making "conservative" looser than "aggressive".

#include <catch2/catch_test_macros.hpp>

#include "core/asr_sensitivity.h"

#include <string>

using core_sensitivity::defaults;
using core_sensitivity::parse_preset;
using core_sensitivity::Preset;
using core_sensitivity::preset;

TEST_CASE("balanced is exactly the shipped defaults", "[sensitivity]") {
    // Naming the default must be a no-op. If this drifts, every user who set
    // --sensitivity balanced silently got a different decoder.
    const auto b = preset(Preset::Balanced);
    const auto d = defaults();
    REQUIRE(b.entropy_thold == d.entropy_thold);
    REQUIRE(b.logprob_thold == d.logprob_thold);
    REQUIRE(b.no_speech_thold == d.no_speech_thold);
    REQUIRE(b.temperature_inc == d.temperature_inc);

    // And those defaults are the ones the decoder actually ships
    // (crispasr.cpp:6523-6526). A change there without a change here is the
    // drift this pins.
    REQUIRE(d.entropy_thold == 2.40f);
    REQUIRE(d.logprob_thold == -1.00f);
    REQUIRE(d.no_speech_thold == 0.60f);
    REQUIRE(d.temperature_inc == 0.20f);
}

TEST_CASE("conservative is tighter than balanced on every axis that matters", "[sensitivity]") {
    const auto c = preset(Preset::Conservative);
    const auto b = preset(Preset::Balanced);

    // Lower entropy bar => a repetitive/degenerate decode is rejected sooner.
    REQUIRE(c.entropy_thold < b.entropy_thold);
    // HIGHER (less negative) logprob bar => a weakly-confident decode fails.
    REQUIRE(c.logprob_thold > b.logprob_thold);
    // LOWER no-speech bar => the "this is silence" verdict is easier to reach,
    // so borderline audio is dropped rather than guessed at.
    REQUIRE(c.no_speech_thold < b.no_speech_thold);
}

TEST_CASE("aggressive is looser than balanced on every axis that matters", "[sensitivity]") {
    const auto a = preset(Preset::Aggressive);
    const auto b = preset(Preset::Balanced);

    REQUIRE(a.entropy_thold > b.entropy_thold);
    REQUIRE(a.logprob_thold < b.logprob_thold);
    REQUIRE(a.no_speech_thold > b.no_speech_thold);
}

TEST_CASE("the two presets sit on opposite sides of balanced", "[sensitivity]") {
    // The property that makes the names meaningful. Asserting each against
    // balanced separately (above) would still allow both to drift the same way.
    const auto c = preset(Preset::Conservative);
    const auto a = preset(Preset::Aggressive);
    REQUIRE(c.entropy_thold < a.entropy_thold);
    REQUIRE(c.logprob_thold > a.logprob_thold);
    REQUIRE(c.no_speech_thold < a.no_speech_thold);
}

TEST_CASE("thresholds stay inside the ranges the decoder can use", "[sensitivity]") {
    // no_speech_thold is compared against a probability, so a value outside
    // (0,1) makes the gate unreachable in one direction — a silently dead knob.
    for (Preset p : {Preset::Conservative, Preset::Balanced, Preset::Aggressive}) {
        const auto t = preset(p);
        INFO(core_sensitivity::preset_name(p));
        REQUIRE(t.no_speech_thold > 0.0f);
        REQUIRE(t.no_speech_thold < 1.0f);
        REQUIRE(t.entropy_thold > 0.0f);
        REQUIRE(t.logprob_thold < 0.0f); // it is a log probability
        REQUIRE(t.temperature_inc > 0.0f);
    }
}

TEST_CASE("preset names parse, including the aliases", "[sensitivity]") {
    Preset p = Preset::Balanced;
    REQUIRE(parse_preset("conservative", p));
    REQUIRE(p == Preset::Conservative);
    REQUIRE(parse_preset("strict", p));
    REQUIRE(p == Preset::Conservative);
    REQUIRE(parse_preset("aggressive", p));
    REQUIRE(p == Preset::Aggressive);
    REQUIRE(parse_preset("loose", p));
    REQUIRE(p == Preset::Aggressive);
    REQUIRE(parse_preset("balanced", p));
    REQUIRE(p == Preset::Balanced);
    REQUIRE(parse_preset("default", p));
    REQUIRE(p == Preset::Balanced);
}

TEST_CASE("an unknown name is rejected and leaves the value alone", "[sensitivity]") {
    // Must not silently fall back to balanced: a typo'd --sensitivity should be
    // reported, not quietly ignored.
    Preset p = Preset::Aggressive;
    REQUIRE_FALSE(parse_preset("agressive", p)); // typo
    REQUIRE(p == Preset::Aggressive);            // untouched
    REQUIRE_FALSE(parse_preset("", p));
    REQUIRE_FALSE(parse_preset("Conservative", p)); // case-sensitive by design
    REQUIRE(p == Preset::Aggressive);
}

TEST_CASE("every name in the help list actually parses", "[sensitivity]") {
    // Guards the classic drift: a value documented in --help that the parser
    // does not accept. Splits the advertised list and round-trips each entry.
    const std::string list = core_sensitivity::preset_list();
    size_t start = 0;
    int seen = 0;
    while (start <= list.size()) {
        size_t comma = list.find(',', start);
        if (comma == std::string::npos)
            comma = list.size();
        std::string name = list.substr(start, comma - start);
        while (!name.empty() && name.front() == ' ')
            name.erase(name.begin());
        while (!name.empty() && name.back() == ' ')
            name.pop_back();
        if (!name.empty()) {
            INFO("advertised preset: " << name);
            Preset p;
            REQUIRE(parse_preset(name, p));
            REQUIRE(std::string(core_sensitivity::preset_name(p)) == name);
            seen++;
        }
        if (comma == list.size())
            break;
        start = comma + 1;
    }
    REQUIRE(seen == 3);
}

// test-canary-qwen-params.cpp — Catch2 unit tests for canary-qwen.
//
// Covers the params/null-guard API surface plus the #247 instruction-echo
// detection logic (canary_qwen_echo::is_meta_echo). No GGUF required.
//
// Background (#247): on a too-short audio window the canary-qwen SALM decoder
// echoes its task framing as a meta word ("Transcript"/"Transcription"/"PASS")
// instead of transcribing. This is genuine model behaviour — the NeMo SALM
// reference emits the identical tokens ("Okay" on a 0.1 s clip, "Transcript" on
// a ~0.3 s clip) — so the runtime gates degenerate windows and strips these
// leading echoes from both the text and the tokens array.

#include <catch2/catch_test_macros.hpp>

#include "canary_qwen.h"
#include "canary_qwen_echo.h"

TEST_CASE("canary_qwen_params: default values are sensible", "[unit][canary-qwen]") {
    struct canary_qwen_context_params p = canary_qwen_context_default_params();
    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197 + PLAN #89). Pin
// the shipped gpu/flash defaults so a silent flip fails CI.
TEST_CASE("canary_qwen_params: gpu/flash defaults are pinned", "[unit][canary-qwen]") {
    struct canary_qwen_context_params p = canary_qwen_context_default_params();
    REQUIRE(p.use_gpu == true);
    REQUIRE(p.flash_attn == false);
}

TEST_CASE("canary_qwen_init_from_file: null path returns nullptr", "[unit][canary-qwen]") {
    struct canary_qwen_context_params p = canary_qwen_context_default_params();
    REQUIRE(canary_qwen_init_from_file(nullptr, p) == nullptr);
}

TEST_CASE("canary_qwen_init_from_file: empty path returns nullptr", "[unit][canary-qwen]") {
    struct canary_qwen_context_params p = canary_qwen_context_default_params();
    REQUIRE(canary_qwen_init_from_file("", p) == nullptr);
}

TEST_CASE("canary_qwen_free: NULL context is a no-op", "[unit][canary-qwen]") {
    canary_qwen_free(nullptr);
    SUCCEED("canary_qwen_free tolerated a NULL ctx.");
}

TEST_CASE("canary_qwen echo: meta words are detected (#247)", "[unit][canary-qwen]") {
    // The three instruction-echo meta words, in the casings/decorations the
    // Qwen3 BPE detokeniser produces.
    REQUIRE(canary_qwen_echo::is_meta_echo("Transcript"));
    REQUIRE(canary_qwen_echo::is_meta_echo("transcript"));
    REQUIRE(canary_qwen_echo::is_meta_echo("Transcription"));
    REQUIRE(canary_qwen_echo::is_meta_echo("PASS"));
    REQUIRE(canary_qwen_echo::is_meta_echo("Pass"));

    SECTION("normalisation: leading space, trailing colon, surrounding ws") {
        REQUIRE(canary_qwen_echo::is_meta_echo(" Transcript")); // BPE 'Ġ' → leading space
        REQUIRE(canary_qwen_echo::is_meta_echo("Transcript:"));
        REQUIRE(canary_qwen_echo::is_meta_echo(" Transcription: "));
        REQUIRE(canary_qwen_echo::is_meta_echo("\tPASS\n"));
    }
}

TEST_CASE("canary_qwen echo: legitimate transcript words are NOT stripped", "[unit][canary-qwen]") {
    // "Okay"/"And"/"So" are hallucinated by the model on ultra-short windows
    // too, but they are also legitimate transcript words — they must NOT be
    // stripped as echoes (the length gate suppresses those cases instead).
    REQUIRE_FALSE(canary_qwen_echo::is_meta_echo("Okay"));
    REQUIRE_FALSE(canary_qwen_echo::is_meta_echo("And"));
    REQUIRE_FALSE(canary_qwen_echo::is_meta_echo("So"));
    REQUIRE_FALSE(canary_qwen_echo::is_meta_echo("Ask"));
    REQUIRE_FALSE(canary_qwen_echo::is_meta_echo(""));
    REQUIRE_FALSE(canary_qwen_echo::is_meta_echo("transcripts")); // substring, not the word
    REQUIRE_FALSE(canary_qwen_echo::is_meta_echo("passport"));
    REQUIRE_FALSE(canary_qwen_echo::is_meta_echo("And so my fellow Americans"));
}

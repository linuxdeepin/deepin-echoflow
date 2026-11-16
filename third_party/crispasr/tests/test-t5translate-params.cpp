// test-t5translate-params.cpp — unit tests for t5_translate_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "t5_translate.h"

TEST_CASE("t5_translate_params: default values are sensible", "[unit][t5_translate]") {
    struct t5_translate_context_params p = t5_translate_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("t5_translate_init_from_file: null path returns nullptr", "[unit][t5_translate]") {
    struct t5_translate_context_params p = t5_translate_context_default_params();
    struct t5_translate_context* ctx = t5_translate_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("t5_translate_init_from_file: empty path returns nullptr", "[unit][t5_translate]") {
    struct t5_translate_context_params p = t5_translate_context_default_params();
    struct t5_translate_context* ctx = t5_translate_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("t5_translate_free: NULL context is a no-op", "[unit][t5_translate]") {
    t5_translate_free(nullptr);
    SUCCEED("t5_translate_free tolerated a NULL ctx.");
}

// ── #333: the decode-loop break ─────────────────────────────────────────────

#include "core/repeat_break.h"

TEST_CASE("t5: a degenerate greedy cycle is detected, ordinary output is not", "[unit][t5][translate]") {
    // MADLAD greedy-decodes into a digit loop on short inputs — fr→en
    // "Bonjour le monde!" translates correctly and then emits the same token to
    // the cap. Measured on the real model: with the break on, the output is
    // "Hello world! – 10"; with it off, "Hello world! – 10000000000…".
    //
    // ⚠ This DEVIATES FROM THE BLUEPRINT DELIBERATELY. The PyTorch reference was
    // run on the same input and runs away identically (60 tokens, no EOS,
    // byte-identical string), so the port already matches — this is a product
    // decision, not a parity fix, and CRISPASR_T5_REPEAT_BREAK=0 restores exact
    // HF behaviour for anyone diffing against it.
    //
    // What must NOT trigger: real translations. Both of the sentences the
    // reference terminates cleanly on were verified byte-identical with the
    // break on and off.
    const std::vector<int> looping = {0, 4531, 892, 137, 401, 401, 401, 401, 401, 401};
    CHECK(core_repeat::tail_is_repetition(looping));

    // "Hallo Welt, wie geht es dir heute?" — no repeated tail cycle.
    const std::vector<int> clean = {0, 12905, 8823, 261, 3016, 1204, 512, 9911, 1620, 293};
    CHECK_FALSE(core_repeat::tail_is_repetition(clean));

    // A doubled word is not a loop — the detector needs 4 reps by default, so
    // ordinary repetition ("sehr, sehr gut") must survive.
    const std::vector<int> doubled = {0, 771, 771, 4402, 88, 12};
    CHECK_FALSE(core_repeat::tail_is_repetition(doubled));
}

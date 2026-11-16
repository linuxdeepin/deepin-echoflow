// test-kyutaistt-params.cpp — unit tests for kyutai_stt_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "kyutai_stt.h"

#include <string>

TEST_CASE("kyutai_stt_params: default values are sensible", "[unit][kyutai_stt]") {
    struct kyutai_stt_context_params p = kyutai_stt_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). ASR ships greedy
// deterministic decode (temperature 0, beam 1); pin them so a silent drift fails
// CI.
TEST_CASE("kyutai_stt_params: decode knobs match the shipped defaults", "[unit][kyutai_stt]") {
    struct kyutai_stt_context_params p = kyutai_stt_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(0.0f)); // greedy
    REQUIRE(p.beam_size == 1);
    REQUIRE(p.use_gpu == true);
}

TEST_CASE("kyutai_stt_init_from_file: null path returns nullptr", "[unit][kyutai_stt]") {
    struct kyutai_stt_context_params p = kyutai_stt_context_default_params();
    struct kyutai_stt_context* ctx = kyutai_stt_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kyutai_stt_init_from_file: empty path returns nullptr", "[unit][kyutai_stt]") {
    struct kyutai_stt_context_params p = kyutai_stt_context_default_params();
    struct kyutai_stt_context* ctx = kyutai_stt_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kyutai_stt_free: NULL context is a no-op", "[unit][kyutai_stt]") {
    kyutai_stt_free(nullptr);
    SUCCEED("kyutai_stt_free tolerated a NULL ctx.");
}

// #366: CrispASR warned "English-only model; language='fr' ignored" for EVERY
// kyutai-stt GGUF — including stt-1b-en_fr, which is the model the registry
// downloads by default and which does support French. The warning was
// unconditional; nothing asked the model.
//
// A GGUF now records kyutai.languages, and for the checkpoints already
// published (which predate that key) the loader falls back to the layer count:
// stt-1b-en_fr is 16 layers, stt-2.6b-en is 48.
//
// These cases cover the null path, which is all that can be exercised without a
// GGUF. The language logic itself is pinned by test-kyutai-stt-jfk-live.
TEST_CASE("kyutai_stt: language queries are null-safe", "[unit][kyutai_stt][issue-366]") {
    // A null context must not crash and must not claim a language is
    // unsupported — the caller would print a misleading warning, which is the
    // bug being fixed.
    REQUIRE(kyutai_stt_supports_language(nullptr, "fr") == 1);
    REQUIRE(kyutai_stt_supports_language(nullptr, nullptr) == 1);
    REQUIRE(std::string(kyutai_stt_languages(nullptr)).empty());
}

TEST_CASE("kyutai_stt: an empty or auto language is never rejected", "[unit][kyutai_stt][issue-366]") {
    // The CLI passes params.language straight through; "" and "auto" mean
    // "caller did not ask", and warning about those would be noise.
    REQUIRE(kyutai_stt_supports_language(nullptr, "") == 1);
    REQUIRE(kyutai_stt_supports_language(nullptr, "auto") == 1);
}

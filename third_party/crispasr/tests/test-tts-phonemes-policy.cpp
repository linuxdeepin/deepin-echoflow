// test-tts-phonemes-policy.cpp — which backends --tts-phonemes may drive (#316).
//
// The flag hands phonemes straight to the acoustic model, bypassing the G2P.
// A backend without a phonemes-in entry point must REFUSE, never quietly
// synthesize the text: a silent fallback makes an A/B look like the phonemes
// changed nothing, which is the opposite of the truth for anyone debugging a
// pronunciation.
//
// This started as a hardcoded `backend_name != "kokoro"` in the CLI, which was
// wrong — piper's runtime has had piper_tts_synthesize_phonemes() all along; its
// adapter just never reached it except by accident. Extracted here so the list
// is reviewable and testable without a model.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_phonemes_policy.h"

using crispasr_phonemes_policy::backend_supports;
using crispasr_phonemes_policy::unsupported_message;

TEST_CASE("backends with a phonemes-in entry point are accepted", "[unit][tts]") {
    REQUIRE(backend_supports("kokoro")); // kokoro_synthesize_phonemes
    REQUIRE(backend_supports("piper"));  // piper_tts_synthesize_phonemes
}

TEST_CASE("every other TTS backend is refused", "[unit][tts]") {
    // Refusing is the point — these would otherwise synthesize `--tts` text and
    // look like the phonemes had no effect.
    for (const char* b : {"qwen3-tts", "chatterbox", "vibevoice-tts", "orpheus", "f5-tts", "dia", "bark",
                          "cosyvoice3-tts", "tada-tts", "melotts", ""})
        REQUIRE_FALSE(backend_supports(b));
}

TEST_CASE("the refusal names the backend and the alternatives", "[unit][tts]") {
    const std::string m = unsupported_message("chatterbox");
    REQUIRE(m.find("chatterbox") != std::string::npos);
    REQUIRE(m.find("kokoro") != std::string::npos);
    REQUIRE(m.find("piper") != std::string::npos);
    // It must say why it is not falling back, not just that it failed.
    REQUIRE(m.find("refusing") != std::string::npos);
}

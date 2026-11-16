// test-vibevoice-asr-prompt.cpp — decode the hardcoded VibeVoice-ASR prompt
// ids back into text (#369).
//
// The prompt is emitted as fixed Qwen2.5 token ids (see
// src/core/vibevoice_asr_prompt.h for why it cannot be tokenized at runtime).
// Hardcoded ids are only as good as the last person who checked them, and the
// table this guards was wrong for its whole life: it carried
//
//     6546, 7699, 11, 4587, 38840, 432, 449   // "seconds audio, please transcribe it with"
//
// where the comment was the intent and the ids were something else entirely.
// 6546 is "Ġconfiguration", 4587 is "thon", 38840 is "PEND", 449 is "iz", so
// every VibeVoice-ASR transcription ran with the instruction
//
//     "This is a 5.98 configuration audio,thonPEND itiz these keys: …"
//
// The word "transcribe" was not in the prompt. A per-stage encoder diff cannot
// see this (the encoder is fine), and the model transcribes well enough anyway
// that the output never looked structurally broken — it just answered in the
// wrong language on borderline audio. What finds it is turning the ids back
// into text, which is what this file does.
//
// The vocabulary slice below is the real Qwen2.5 table, read out of the shipped
// `vibevoice-asr-bitnet-embed-q8.gguf` (`tokenizer.ggml.tokens`) and confirmed
// against `AutoTokenizer.from_pretrained(<checkpoint>)`. Strings are in GPT-2
// byte-encoded form ("Ġ" = leading space, "Ċ" = newline), the same form the
// GGUF stores, so core_bpe::detokenize() is the exact decode path the runtime
// uses for model output.

#include "core/bpe.h"
#include "core/vibevoice_asr_prompt.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace {

// id -> byte-encoded token text, for every id the prompt tables use.
const std::vector<std::pair<int32_t, const char*>>& vocab_slice() {
    static const std::vector<std::pair<int32_t, const char*>> v = {
        {11, ","},
        {13, "."},
        {15, "0"},
        {16, "1"},
        {17, "2"},
        {18, "3"},
        {19, "4"},
        {20, "5"},
        {21, "6"},
        {22, "7"},
        {23, "8"},
        {24, "9"},
        {25, ":"},
        {198, "Ċ"},
        {220, "Ġ"},
        {264, "Ġa"},
        {271, "ĊĊ"},
        {304, "Ġin"},
        {374, "Ġis"},
        {429, "Ġthat"},
        {432, "Ġit"},
        {448, "Ġwith"},
        {525, "Ġare"},
        {872, "user"},
        {882, "Ġtime"},
        {1119, "Ġinto"},
        {1356, "Ġtrans"},
        {1467, "Ġtext"},
        {1493, "Ġthese"},
        {1946, "Ġinput"},
        {1986, "This"},
        {2550, "Ġoutput"},
        {2610, "You"},
        {3034, "ĠID"},
        {3114, "cribe"},
        {3546, "Ġinfo"},
        {3561, "Ġformat"},
        {3972, "ĠEnd"},
        {4486, "Ġplease"},
        {4718, "ĠJSON"},
        {4960, "Ġextra"},
        {5145, "ĠStart"},
        {5501, "Please"},
        {6486, "Ġseconds"},
        {6894, "Ġkeys"},
        {7699, "Ġaudio"},
        {8883, "ĠContent"},
        {8948, "system"},
        {10950, "Ġhelpful"},
        {17847, "Ġassistant"},
        {29073, "ĠSpeaker"},
        {55136, "cribes"},
        {77091, "assistant"},
        {151643, "<|endoftext|>"},
        {151644, "<|im_start|>"},
        {151645, "<|im_end|>"},
        // The three speech placeholders sit past the end of the GGUF's token
        // table (151646 entries) but inside the 151936-row embedding matrix, so
        // the runtime embeds them correctly and only the string is unavailable.
        // Names per Qwen2.5's vocabulary and all three upstream sources.
        {151646, "<|object_ref_start|>"},
        {151647, "<|object_ref_end|>"},
        {151648, "<|box_start|>"},
        // Present only so the legacy-table assertion below can show what the
        // old ids really said.
        {449, "iz"},
        {4587, "thon"},
        {6546, "Ġconfiguration"},
        {38840, "PEND"},
    };
    return v;
}

std::string decode(const std::vector<int32_t>& ids) {
    static std::vector<std::string> table;
    if (table.empty()) {
        table.resize(151649);
        for (const auto& kv : vocab_slice())
            table[(size_t)kv.first] = kv.second;
    }
    // An id outside the slice decodes to "", which shows up as missing text in
    // the comparison rather than passing silently.
    return core_bpe::detokenize(table, ids.data(), ids.size());
}

} // namespace

using namespace core_vibevoice_asr_prompt;

TEST_CASE("vibevoice-asr prompt: the system + user header decodes to upstream's text", "[unit][vibevoice]") {
    REQUIRE(decode(system_user_header()) == std::string(system_user_header_text()));
}

// The bug this file exists for. If any id in the no-context suffix is wrong,
// this comparison shows the wrong WORD, not a wrong number.
TEST_CASE("vibevoice-asr prompt: the no-context suffix decodes to upstream's text", "[unit][vibevoice]") {
    REQUIRE(decode(suffix_no_context("5.98")) ==
            "\nThis is a 5.98 seconds audio, please transcribe it with these keys: "
            "Start time, End time, Speaker ID, Content");
    REQUIRE(decode(suffix_no_context("123.45")) ==
            "\nThis is a 123.45 seconds audio, please transcribe it with these keys: "
            "Start time, End time, Speaker ID, Content");
}

// The instruction verb is the whole point of the prompt, and it is the token
// the old table dropped. Assert it by name so a future edit cannot quietly
// lose it again.
TEST_CASE("vibevoice-asr prompt: the suffix actually says 'transcribe'", "[unit][vibevoice]") {
    const std::string s = decode(suffix_no_context("5.98"));
    REQUIRE(s.find("please transcribe it with") != std::string::npos);
    REQUIRE(s.find("seconds audio") != std::string::npos);
    REQUIRE(s.find("configuration") == std::string::npos);
    REQUIRE(s.find("PEND") == std::string::npos);
}

// The 1.5B checkpoints (including VibeVoice-ASR-BitNet) take a PLAIN-TEXT
// instruction, not the JSON-keys one. Microsoft's own runtime says so —
// VibeASR.cpp utils/prompt_builder.h labels "text" the 1.5B format and "json"
// the 7B's, and defaults to text. We sent every checkpoint the 7B form because
// ours came from the Python processor, which only targets the 7B; the 1.5B was
// being asked for a JSON transcript it does not emit (#369).
TEST_CASE("vibevoice-asr prompt: the 1.5B plain-text suffix decodes correctly", "[unit][vibevoice]") {
    REQUIRE(decode(suffix_no_context_plain("3.26")) == "\nThis is a 3.26 seconds audio, please transcribe it.");
    REQUIRE(decode(suffix_no_context_plain("5.98")) == "\nThis is a 5.98 seconds audio, please transcribe it.");
    REQUIRE(decode(suffix_context_tail_plain()) == "\n\nPlease transcribe it.");
}

// The two instructions must stay distinguishable: the plain one must NOT ask for
// keys, and the JSON one must. Sending the wrong one is the whole of the BitNet
// half of #369, so it is asserted rather than left to reading.
TEST_CASE("vibevoice-asr prompt: plain and JSON instructions do not collide", "[unit][vibevoice]") {
    const std::string plain = decode(suffix_no_context_plain("5.98"));
    const std::string json = decode(suffix_no_context("5.98"));
    REQUIRE(plain.find("these keys") == std::string::npos);
    REQUIRE(json.find("these keys") != std::string::npos);
    REQUIRE(plain.find("please transcribe it.") != std::string::npos);
    REQUIRE(plain != json);
}

TEST_CASE("vibevoice-asr prompt: the context form brackets the user's text", "[unit][vibevoice]") {
    REQUIRE(decode(suffix_context_head("5.98")) == "\nThis is a 5.98 seconds audio, with extra info:");
    REQUIRE(decode(suffix_context_tail()) ==
            "\n\nPlease transcribe it with these keys: Start time, End time, Speaker ID, Content");
}

// Upstream's processor never forwards add_generation_prompt to
// apply_chat_template, the chat_template transformers ships has no assistant
// branch, and audio.cpp passes false. So the prompt ends at "<|im_end|>\n".
TEST_CASE("vibevoice-asr prompt: no assistant header by default", "[unit][vibevoice]") {
    REQUIRE(decode(prompt_tail(false)) == "<|im_end|>\n");
    REQUIRE(decode(prompt_tail(true)) == "<|im_end|>\n<|im_start|>assistant\n");
}

TEST_CASE("vibevoice-asr prompt: duration digits map to single-character ids", "[unit][vibevoice]") {
    REQUIRE(duration_ids("5.98") == std::vector<int32_t>{20, 13, 24, 23});
    REQUIRE(duration_ids("0.50") == std::vector<int32_t>{15, 13, 20, 15});
    REQUIRE(duration_ids("123.45") == std::vector<int32_t>{16, 17, 18, 13, 19, 20});
}

// Pin what the old table said, so the regression is documented in executable
// form rather than only in a commit message.
TEST_CASE("vibevoice-asr prompt: the pre-fix ids decoded to nonsense", "[unit][vibevoice]") {
    const std::vector<int32_t> legacy_mid = {6546, 7699, 11, 4587, 38840, 432, 449,   1493, 6894, 25,
                                             5145, 882,  11, 3972, 882,   11,  29073, 3034, 11,   8883};
    REQUIRE(decode(legacy_mid) == " configuration audio,thonPEND itiz these keys: "
                                  "Start time, End time, Speaker ID, Content");
}

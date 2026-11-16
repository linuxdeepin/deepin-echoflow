// src/core/vibevoice_asr_prompt.h — the exact VibeVoice-ASR chat prompt, as ids.
//
// The prompt is a fixed template around a duration string, so it is emitted as
// hardcoded Qwen2.5 token ids rather than tokenized at runtime: our
// `core_bpe::tokenize_simple` is a whitespace pre-tokenizer, and Qwen2 splits
// digits and punctuation with its own regex ("5.98" -> `Ġ 5 . 9 8`,
// "audio," -> `Ġaudio ,`), so a runtime pass would not reproduce it.
//
// ⚠ Hardcoded ids need a guard that DECODES them. The table this replaces had
// carried four wrong ids since it was written (#369): it read
//
//     6546, 7699, 11, 4587, 38840, 432, 449   // "seconds audio, please transcribe it with"
//
// and the comment was the intent, not the content. Those ids actually decode to
// " configuration audio,thonPEND itiz" — the word "transcribe" was not in the
// prompt at all, and nobody noticed because nothing ever turned the ids back
// into text. `tests/test-vibevoice-asr-prompt.cpp` now does exactly that,
// against an embedded slice of the real Qwen2.5 vocabulary.
//
// Every id below was produced by
// `AutoTokenizer.from_pretrained(<vibevoice-asr checkpoint>).encode(<text>)`
// and cross-checked against three independent implementations of the same
// prompt, which agree on all three points our old table got wrong (the extra
// newline, the corrupted suffix, and the assistant header):
//
//   * microsoft/VibeVoice `vibevoice/processor/vibevoice_asr_processor.py`
//     (`_encode_single`, SYSTEM_PROMPT)
//   * transformers `models/vibevoice_asr/convert_vibevoice_asr_to_hf.py`
//     (the chat_template it ships with the converted processor)
//   * 0xShug0/audio.cpp `src/models/vibevoice_asr/tokenizer_text.cpp`
//     (`build_prompt`, `chat_message(..., add_generation_prompt=false)`)

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace core_vibevoice_asr_prompt {

// Qwen2.5 chat control tokens.
constexpr int32_t IM_START = 151644; // <|im_start|>
constexpr int32_t IM_END = 151645;   // <|im_end|>
constexpr int32_t NEWLINE = 198;     // "\n"
constexpr int32_t EOS = 151643;      // <|endoftext|>

// Speech placeholders. VibeVoice-ASR reuses three Qwen2.5-VL control tokens
// (VibeVoiceTextTokenizerFast._add_vibevoice_special_tokens).
constexpr int32_t AUDIO_BOS = 151646; // <|object_ref_start|>
constexpr int32_t AUDIO_PAD = 151648; // <|box_start|>
constexpr int32_t AUDIO_EOS = 151647; // <|object_ref_end|>

// The texts the id tables below must decode to. Asserted by the unit test.
inline const char* system_user_header_text() {
    return "<|im_start|>system\n"
           "You are a helpful assistant that transcribes audio input into text output in JSON format."
           "<|im_end|>\n<|im_start|>user\n";
}
inline const char* suffix_no_context_plain_text() {
    return "\nThis is a %s seconds audio, please transcribe it.";
}
inline const char* suffix_context_tail_plain_text() {
    return "\n\nPlease transcribe it.";
}
inline const char* suffix_no_context_text() {
    return "\nThis is a %s seconds audio, please transcribe it with these keys: "
           "Start time, End time, Speaker ID, Content";
}
inline const char* suffix_context_head_text() {
    return "\nThis is a %s seconds audio, with extra info:";
}
inline const char* suffix_context_tail_text() {
    return "\n\nPlease transcribe it with these keys: Start time, End time, Speaker ID, Content";
}

// "<|im_start|>system\n…JSON format.<|im_end|>\n<|im_start|>user\n".
//
// Note there is NO newline between "format." and <|im_end|>: the chat template
// is `'<|im_start|>' + role + '\n' + content + '<|im_end|>' + '\n'`, and the
// system content ends at the full stop. We used to emit one.
inline std::vector<int32_t> system_user_header() {
    return {
        IM_START, 8948,    NEWLINE,                                   // <|im_start|>system\n
        2610,     525,     264,      10950, 17847,  429, 1356, 55136, // You are a helpful assistant that transcribes
        7699,     1946,    1119,     1467,  2550,   304, 4718, 3561,
        13,                                         // audio input into text output in JSON format.
        IM_END,   NEWLINE, IM_START, 872,   NEWLINE // <|im_end|>\n<|im_start|>user\n
    };
}

// The "%.2f" duration, digit by digit. Qwen2.5 maps '0'-'9' to 15-24 and '.'
// to 13 as single-character tokens, so no BPE pass is needed.
inline std::vector<int32_t> duration_ids(const std::string& dur) {
    std::vector<int32_t> out;
    out.reserve(dur.size());
    for (char c : dur) {
        if (c >= '0' && c <= '9')
            out.push_back(15 + (c - '0'));
        else if (c == '.')
            out.push_back(13);
    }
    return out;
}

// "\nThis is a " + duration — shared by both suffix forms.
inline std::vector<int32_t> suffix_head(const std::string& dur) {
    std::vector<int32_t> out = {NEWLINE, 1986, 374, 264, 220}; // \n This is a<space>
    const auto d = duration_ids(dur);
    out.insert(out.end(), d.begin(), d.end());
    return out;
}

// " seconds audio, please transcribe it with these keys: Start time, End time,
// Speaker ID, Content" — the no-context instruction.
inline std::vector<int32_t> suffix_no_context(const std::string& dur) {
    std::vector<int32_t> out = suffix_head(dur);
    const int32_t rest[] = {
        6486,  7699, 11,                    // " seconds audio,"
        4486,  1356, 3114, 432,  448, 1493, // " please transcribe it with these"
        6894,  25,                          // " keys:"
        5145,  882,  11,   3972, 882, 11,   // " Start time, End time,"
        29073, 3034, 11,   8883,            // " Speaker ID, Content"
    };
    out.insert(out.end(), std::begin(rest), std::end(rest));
    return out;
}

// " seconds audio, please transcribe it." — the PLAIN-TEXT instruction.
//
// ⚠ This form belongs to the 1.5B checkpoints (including VibeVoice-ASR-BitNet);
// the JSON-keys form above belongs to the 7B. Microsoft's own reference runtime
// says so in as many words — VibeASR.cpp, utils/prompt_builder.h:
//
//     // "text" format (1.5B model, plain text output):
//     //   "This is a X.XX seconds audio, please transcribe it."
//     // "json" format (7B model, JSON output with keys):
//     //   "This is a X.XX seconds audio, please transcribe it with these keys: ..."
//
// and it defaults to "text". We sent every checkpoint the JSON form, because our
// prompt was derived from microsoft/VibeVoice's PYTHON processor, which targets
// the 7B. So the 1.5B was being asked for a JSON transcript it was not trained
// to emit — which is the shape of #369's BitNet complaint: plausible but
// degraded output on the small model, while the 7B is exact through the same
// pipeline, and the official demo (VibeASR.cpp defaults, i.e. this form) is
// exact on the same clip.
inline std::vector<int32_t> suffix_no_context_plain(const std::string& dur) {
    std::vector<int32_t> out = suffix_head(dur);
    const int32_t rest[] = {6486, 7699, 11, 4486, 1356, 3114, 432, 13}; // " seconds audio, please transcribe it."
    out.insert(out.end(), std::begin(rest), std::end(rest));
    return out;
}

// "\n\nPlease transcribe it." — the context form's tail for the plain-text
// instruction (VibeASR.cpp uses the capitalised sentence after the blank line,
// exactly as it does for the JSON form).
inline std::vector<int32_t> suffix_context_tail_plain() {
    return {271, 5501, 1356, 3114, 432, 13};
}

// " seconds audio, with extra info:" — the context form stops here so the
// caller can splice in BPE-tokenized user text (which must carry its own
// leading space, i.e. be tokenized as " <context>").
inline std::vector<int32_t> suffix_context_head(const std::string& dur) {
    std::vector<int32_t> out = suffix_head(dur);
    const int32_t rest[] = {6486, 7699, 11, 448, 4960, 3546, 25}; // " seconds audio, with extra info:"
    out.insert(out.end(), std::begin(rest), std::end(rest));
    return out;
}

// "\n\nPlease transcribe it with these keys: …". 271 is the single `ĊĊ` token.
inline std::vector<int32_t> suffix_context_tail() {
    return {271, 5501, 1356, 3114, 432, 448, 1493, 6894, 25, 5145, 882, 11, 3972, 882, 11, 29073, 3034, 11, 8883};
}

// "<|im_end|>\n", plus the "<|im_start|>assistant\n" generation prompt.
//
// Upstream's token sequence stops at "<|im_end|>\n": `_encode_single` accepts
// `add_generation_prompt` and never forwards it to `apply_chat_template`, the
// chat_template transformers ships has no assistant branch, and audio.cpp
// passes `add_generation_prompt=false` explicitly. We still emit the header,
// and that is a deliberate divergence rather than an oversight: measured on
// vibevoice-asr-bitnet-embed-q8, ending the prompt at "<|im_end|>\n" makes the
// model generate ZERO tokens — its first sampled token is a stop token, so it
// closes the document instead of answering. The header only pre-seeds the turn
// the model would otherwise have to open itself.
inline std::vector<int32_t> prompt_tail(bool with_assistant_header) {
    std::vector<int32_t> out = {IM_END, NEWLINE};
    if (with_assistant_header) {
        out.push_back(IM_START);
        out.push_back(77091); // "assistant"
        out.push_back(NEWLINE);
    }
    return out;
}

} // namespace core_vibevoice_asr_prompt

#pragma once

// OmniVoice style-prefix assembly.
//
// This is the string that carries ALL of the model's conditioning other than
// the text itself — the target language and the voice-design instruct — and it
// reaches the model as literal tokens. Every defect in #13273 ended here: a
// wrong language id, an unnormalised instruct, a value that never arrived at
// all. Each one produced a prompt that tokenized cleanly and conditioned the
// model on something nobody asked for.
//
// It lives in its own weight-free header for one reason: so the exact string
// can be asserted in a unit test. Built inline in omnivoice.cpp it was only
// checkable by loading a 1.2 GB model, reading a debug print and eyeballing it
// against the HF tokenizer — which is how the instruct bug survived the first
// pass. tests/test-omnivoice-prompt.cpp now pins the strings that the real
// tokenizer was verified against.
//
// Mirrors `_prepare_inference_inputs` (omnivoice/models/omnivoice.py):
//
//     style_text = ""
//     if denoise and ref_audio_tokens is not None: style_text += "<|denoise|>"
//     lang_str     = lang     if lang     else "None"
//     instruct_str = instruct if instruct else "None"
//     style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
//     style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"
//
// Both empty values become the literal "None" — not an empty tag and not an
// omitted tag. The tags are always present, always in this order.

#include <string>

namespace core_omnivoice_prompt {

// Upstream's `x if x else "None"`, which is why an unresolved language or
// instruct must arrive here as an EMPTY string: "None" is a real token the
// model was trained on for "unspecified", and anything else in that slot is a
// silent miscondition.
inline const char* or_none(const std::string& s) {
    return s.empty() ? "None" : s.c_str();
}

// `has_reference` is the blueprint's `denoise and ref_audio_tokens is not None`
// — the <|denoise|> lead appears only when cloning from reference audio.
// `lang_id` must already be resolved (core_omnivoice_lang::resolve) and
// `instruct` already rendered (core_omnivoice_instruct::render); this function
// deliberately does no validation, so that the ONE place a raw value could slip
// into the prompt is the one place with no vocabulary to check it against.
inline std::string build_style_text(bool has_reference, const std::string& lang_id, const std::string& instruct) {
    std::string style;
    if (has_reference)
        style += "<|denoise|>";
    style += "<|lang_start|>";
    style += or_none(lang_id);
    style += "<|lang_end|>";
    style += "<|instruct_start|>";
    style += or_none(instruct);
    style += "<|instruct_end|>";
    return style;
}

} // namespace core_omnivoice_prompt

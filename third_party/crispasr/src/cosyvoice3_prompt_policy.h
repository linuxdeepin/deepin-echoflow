// src/cosyvoice3_prompt_policy.h — the two length laws CosyVoice3's talker
// decode depends on, kept weight-free so they can be unit-tested (#334).
//
// Both come from `Qwen2LM.inference` in the upstream CosyVoice repo, which
// `CosyVoice3LM` inherits unchanged:
//
//     min_len = int((text_len - prompt_text_len) * min_token_text_ratio)   # 2
//     max_len = int((text_len - prompt_text_len) * max_token_text_ratio)   # 20
//     top_ids = self.sampling_ids(..., ignore_eos=True if i < min_len else False)
//
// i.e. the model is only ever allowed to spend between 2 and 20 speech tokens
// per text token, and a stop token is masked out until the floor is reached.
// A speech token is 40 ms, so these are also the bounds on how fast the model
// may say a given piece of text.
//
// The same band answers a second question the runtime needs: is a
// user-supplied `--ref-text` plausibly a transcript of the reference clip?
// The clip's speech-token count and the transcript's text-token count are
// exactly the two quantities above, so a ratio outside the band means the two
// cannot describe the same utterance — which is the #334 failure (a clip
// labelled with a partial transcript makes the decode stop at once, or rush
// the requested line into far too few frames).

#pragma once

#include <cstddef>

namespace cosyvoice3_policy {

// Upstream min_token_text_ratio / max_token_text_ratio.
inline constexpr int kMinTokenTextRatio = 2;
inline constexpr int kMaxTokenTextRatio = 20;

// Seconds of audio one speech token covers (25 Hz token rate).
inline constexpr double kSecondsPerSpeechToken = 0.04;

// Floor on generated speech tokens for `n_target_text_ids` text tokens.
// `n_target_text_ids` is the TARGET text only — upstream's
// `text_len - prompt_text_len`, with the reference transcript excluded.
inline int decode_min_tokens(int n_target_text_ids) {
    return n_target_text_ids > 0 ? n_target_text_ids * kMinTokenTextRatio : 0;
}

// Clamp the floor so it can never reach or exceed the step budget — a floor
// at or past `max_steps` would mask the stop token for the whole decode and
// turn a short line into `max_steps` frames of filler.
inline int clamp_min_tokens(int min_tokens, int max_steps) {
    if (min_tokens <= 0 || max_steps <= 0)
        return 0;
    return min_tokens < max_steps ? min_tokens : max_steps - 1;
}

// Does `n_speech` speech tokens against `n_text` text tokens sit inside the
// band the model works in? Returns true when the pairing is plausible, or
// when either side is empty (nothing to judge).
inline bool prompt_length_plausible(std::size_t n_speech, std::size_t n_text) {
    if (n_speech == 0 || n_text == 0)
        return true;
    const double ratio = (double)n_speech / (double)n_text;
    return ratio >= (double)kMinTokenTextRatio && ratio <= (double)kMaxTokenTextRatio;
}

} // namespace cosyvoice3_policy

// generation_health.h — objective quality checks on ASR/TTS generation output.
//
// Catches perceptual pathologies that cosine-similarity diff harnesses miss:
// truncated outputs, n-gram loops, duration outliers. Each check returns a
// pass/fail + diagnostic string. Use from unit/integration tests or as a
// post-generation gate in live backends.
//
// Non-breaking: pure additive header, no existing code modified.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace core_generation_health {

struct CheckResult {
    bool pass;
    std::string detail;
};

// ---- ASR health checks ----

// Check that the transcript is not empty for audio that should contain speech.
inline CheckResult check_not_empty(const char* text) {
    if (!text || !text[0])
        return {false, "transcript is empty"};
    // Strip whitespace
    const char* p = text;
    while (*p == ' ' || *p == '\n' || *p == '\t')
        p++;
    if (!*p)
        return {false, "transcript is whitespace-only"};
    return {true, ""};
}

// Check that the transcript length is plausible for the audio duration.
// Heuristic: English speech is ~2-4 words/sec, ~5-7 chars/word → ~10-28 chars/sec.
// We use a generous 1-50 chars/sec band to avoid false positives across languages.
inline CheckResult check_duration_plausibility(const char* text, float audio_duration_s, float min_chars_per_sec = 1.0f,
                                               float max_chars_per_sec = 50.0f) {
    if (!text || audio_duration_s <= 0.0f)
        return {true, "skipped (no text or zero duration)"};
    size_t len = std::strlen(text);
    float chars_per_sec = (float)len / audio_duration_s;
    if (chars_per_sec < min_chars_per_sec) {
        return {false, "transcript too short for audio duration (" + std::to_string(chars_per_sec) +
                           " chars/sec, min " + std::to_string(min_chars_per_sec) + ")"};
    }
    if (chars_per_sec > max_chars_per_sec) {
        return {false, "transcript too long for audio duration (" + std::to_string(chars_per_sec) + " chars/sec, max " +
                           std::to_string(max_chars_per_sec) + ")"};
    }
    return {true, ""};
}

// Check for n-gram repetition loops in the transcript.
// A simple heuristic: if any word appears more than `max_repeats` times in a row,
// flag it. This catches "hey hey hey hey hey" style degenerate outputs.
inline CheckResult check_no_ngram_loop(const char* text, int max_consecutive_repeats = 4) {
    if (!text || !text[0])
        return {true, ""};
    std::string s(text);
    // Tokenize by whitespace
    size_t i = 0;
    std::string prev_word;
    int repeat_count = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t'))
            i++;
        size_t start = i;
        while (i < s.size() && s[i] != ' ' && s[i] != '\n' && s[i] != '\t')
            i++;
        if (i == start)
            break;
        std::string word = s.substr(start, i - start);
        if (word == prev_word) {
            repeat_count++;
            if (repeat_count >= max_consecutive_repeats) {
                return {false, "n-gram loop detected: \"" + word + "\" repeated " + std::to_string(repeat_count + 1) +
                                   " times"};
            }
        } else {
            repeat_count = 0;
            prev_word = word;
        }
    }
    return {true, ""};
}

// Check that a TTS output has a plausible number of audio samples for the text.
// Heuristic: English speech at 24 kHz is ~2-4 words/sec, so 1 word ≈ 6000-12000 samples.
inline CheckResult check_tts_duration(int n_samples, int sample_rate, int n_words, float min_sec_per_word = 0.1f,
                                      float max_sec_per_word = 2.0f) {
    if (n_words <= 0 || n_samples <= 0 || sample_rate <= 0)
        return {true, "skipped"};
    float duration_s = (float)n_samples / (float)sample_rate;
    float sec_per_word = duration_s / (float)n_words;
    if (sec_per_word < min_sec_per_word) {
        return {false, "TTS audio too short (" + std::to_string(sec_per_word) + " sec/word, min " +
                           std::to_string(min_sec_per_word) + ")"};
    }
    if (sec_per_word > max_sec_per_word) {
        return {false, "TTS audio too long (" + std::to_string(sec_per_word) + " sec/word, max " +
                           std::to_string(max_sec_per_word) + ")"};
    }
    return {true, ""};
}

// Check that a TTS waveform doesn't end with an excessive run of silence.
// Trailing dead air is a common TTS pathology — a mis-handled EOS, a vocoder that
// runs past the last phoneme, or a padded generation buffer that never got
// trimmed. Scans backward in 10 ms windows and stops at the first window whose RMS
// exceeds `silence_rms_threshold`; the untrimmed tail beyond that is the trailing
// silence. (A per-sample abs test would be fooled by a stray DC-offset sample, so
// use windowed RMS.) A fully-silent clip trips this too — its whole length is
// "trailing silence" — which is the desired signal for "the model emitted nothing".
inline CheckResult check_trailing_silence(const float* pcm, int n_samples, int sample_rate,
                                          float max_trailing_silence_s = 2.0f, float silence_rms_threshold = 1e-3f) {
    if (!pcm || n_samples <= 0 || sample_rate <= 0)
        return {true, "skipped (no audio or zero sample rate)"};
    const int win = std::max(1, sample_rate / 100); // 10 ms
    int trailing_silent = 0;
    for (int end = n_samples; end > 0; end -= win) {
        int start = std::max(0, end - win);
        double sumsq = 0.0;
        for (int i = start; i < end; ++i)
            sumsq += (double)pcm[i] * (double)pcm[i];
        double rms = std::sqrt(sumsq / (double)(end - start));
        if (rms > silence_rms_threshold)
            break;
        trailing_silent += (end - start);
    }
    float silence_s = (float)trailing_silent / (float)sample_rate;
    if (silence_s > max_trailing_silence_s) {
        return {false, "excessive trailing silence (" + std::to_string(silence_s) + " s, max " +
                           std::to_string(max_trailing_silence_s) + ")"};
    }
    return {true, ""};
}

// Check that the token count didn't hit the max (truncation).
inline CheckResult check_not_truncated(int n_tokens, int max_tokens) {
    if (max_tokens <= 0)
        return {true, "skipped (no max)"};
    if (n_tokens >= max_tokens) {
        return {false, "output hit max_tokens (" + std::to_string(n_tokens) + " >= " + std::to_string(max_tokens) +
                           ") — likely truncated"};
    }
    return {true, ""};
}

} // namespace core_generation_health

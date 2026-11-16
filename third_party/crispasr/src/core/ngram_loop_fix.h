// core/ngram_loop_fix.h — collapse degenerate greedy n-gram loops in decoded
// text.
//
// Autoregressive ASR decoders (higgs-audio-v3-stt, MOSS-Transcribe, ... — all
// Qwen3-1.7B-class LMs decoded greedily) occasionally fall into a repeated
// n-gram attractor and emit the same phrase until the max-token cap
// ("Hey, hey, hey, ..." / "run hey hey hey run ..."). higgs-audio ships an
// `ngram_loop_fix.py` post-process for exactly this; the collapse below is that
// algorithm, extracted here so multiple backends share one implementation.
//
// The transform is a pure text post-process: it never touches the token
// stream, so per-token / logit parity against a Python reference is unchanged.
// It is a no-op on non-degenerate text — only *immediately* repeated n-grams
// beyond `max_rep` reps are trimmed — so a clean transcript passes through
// byte-for-byte.
//
// TWO TOKENIZATIONS, ONE ALGORITHM (PLAN.md §W1). The collapse above is
// whitespace-delimited, which makes it structurally incapable of firing on
// Japanese or Chinese: those scripts have no inter-word spaces, so the whole
// segment is a single "word" and `collapse()` never sees a repeat. That is not
// a corner case — two of this header's own consumers are Chinese-first
// (`moss_transcribe` is zh/en; `glm-asr`, via the c_api, is Mandarin + Chinese
// dialects + Cantonese), so the guard was dead exactly where the failure mode
// is worst. The fix runs the *same* `collapse()` over Unicode code points
// instead of words, gated to tokens that are majority CJK so Latin text keeps
// the word-level path byte-for-byte. See `tests/test-ngram-loop-fix-cjk.cpp`.

#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace core_ngram {

// Collapse immediately-repeated n-grams (window size `n`) in `w` to at most
// `max_rep` consecutive reps. Walks left-to-right building `out`; whenever the
// next n words equal the tail of `out` and that tail already repeats >= max_rep
// times, the duplicate n-gram is dropped.
inline std::vector<std::string> collapse(const std::vector<std::string>& w, int n, int max_rep) {
    std::vector<std::string> out;
    const int L = (int)w.size();
    int i = 0;
    auto tail_eq = [&]() {
        for (int k = 0; k < n; k++)
            if (w[i + k] != out[out.size() - n + k])
                return false;
        return true;
    };
    while (i < L) {
        bool matched = false;
        if ((int)out.size() >= n && i + n <= L && tail_eq()) {
            int reps = 1;
            while ((int)out.size() >= n * (reps + 1)) {
                bool eq = true;
                const size_t b = out.size() - (size_t)n * (reps + 1);
                for (int k = 0; k < n; k++)
                    if (out[b + k] != out[out.size() - n + k]) {
                        eq = false;
                        break;
                    }
                if (!eq)
                    break;
                reps++;
            }
            if (reps >= max_rep) {
                i += n;
                matched = true;
            }
        }
        if (!matched) {
            out.push_back(w[i]);
            i++;
        }
    }
    return out;
}

// Split `text` on whitespace into words.
inline std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace((unsigned char)text[i]))
            i++;
        size_t j = i;
        while (j < text.size() && !std::isspace((unsigned char)text[j]))
            j++;
        if (j > i)
            words.push_back(text.substr(i, j - i));
        i = j;
    }
    return words;
}

// Same collapse algorithm as `collapse()`, but tracks original indices
// (into `words`) instead of copying strings. `idx` is the current
// surviving subsequence (indices into `words`) to collapse further;
// returns the subsequence of `idx` that survives this pass.
inline std::vector<int> collapse_indices(const std::vector<std::string>& words, const std::vector<int>& idx, int n,
                                         int max_rep) {
    std::vector<int> out; // indices into `words`
    const int L = (int)idx.size();
    int i = 0;
    auto tail_eq = [&]() {
        for (int k = 0; k < n; k++)
            if (words[idx[i + k]] != words[out[out.size() - n + k]])
                return false;
        return true;
    };
    while (i < L) {
        bool matched = false;
        if ((int)out.size() >= n && i + n <= L && tail_eq()) {
            int reps = 1;
            while ((int)out.size() >= n * (reps + 1)) {
                bool eq = true;
                const size_t b = out.size() - (size_t)n * (reps + 1);
                for (int k = 0; k < n; k++)
                    if (words[out[b + k]] != words[out[out.size() - n + k]]) {
                        eq = false;
                        break;
                    }
                if (!eq)
                    break;
                reps++;
            }
            if (reps >= max_rep) {
                i += n;
                matched = true;
            }
        }
        if (!matched) {
            out.push_back(idx[i]);
            i++;
        }
    }
    return out;
}

// Runs the same n=max_n..1 collapse passes as `fix_loops`, but returns the
// ascending subsequence of original indices into `words` that survive —
// i.e. which words `fix_loops` would keep. Callers with a parallel
// per-word array (timestamps, confidences) use this to filter that array
// in lockstep with the text collapse, instead of just cleaning the flat
// text and leaving duplicates in word-level output (issue #218 follow-up:
// `fix_loops` alone cleans `seg.text` but not `seg.words`/tokens, which
// are built independently from the raw token stream).
// Global diagnostic opt-out: CRISPASR_NGRAM_LOOPFIX_OFF=1 turns every
// fix_loops/fix_loops_keep_indices call into an identity pass, exposing the
// RAW decoded text. For A/B-ing whether a loop originates in the decode
// itself (quant drift, #218) or is merely being masked by the collapse.
inline bool loopfix_disabled() {
    const char* e = std::getenv("CRISPASR_NGRAM_LOOPFIX_OFF");
    return e && std::atoi(e) != 0;
}

inline std::vector<int> fix_loops_keep_indices(const std::vector<std::string>& words, int max_n = 16) {
    std::vector<int> idx(words.size());
    for (size_t i = 0; i < words.size(); i++)
        idx[i] = (int)i;
    if (loopfix_disabled())
        return idx;
    for (int n = max_n; n >= 1; n--)
        idx = collapse_indices(words, idx, n, n == 1 ? 3 : 2);
    return idx;
}

// ---------------------------------------------------------------------------
// Code-point path — for scripts with no inter-word spaces (PLAN.md §W1)
// ---------------------------------------------------------------------------

// Split `text` into UTF-8 code points, one string per code point. A malformed
// or truncated sequence is emitted as a single-byte unit rather than dropped,
// so the transform never corrupts bytes it does not deliberately collapse.
inline std::vector<std::string> split_codepoints(const std::string& text) {
    std::vector<std::string> cps;
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char c = (unsigned char)text[i];
        size_t len = 1;
        if ((c & 0x80) == 0x00)
            len = 1;
        else if ((c & 0xE0) == 0xC0)
            len = 2;
        else if ((c & 0xF0) == 0xE0)
            len = 3;
        else if ((c & 0xF8) == 0xF0)
            len = 4;
        // Truncated tail, or a stray continuation byte: take one byte.
        if (len > text.size() - i)
            len = 1;
        for (size_t k = 1; k < len; k++)
            if ((((unsigned char)text[i + k]) & 0xC0) != 0x80) {
                len = 1;
                break;
            }
        cps.push_back(text.substr(i, len));
        i += len;
    }
    return cps;
}

// Decode one UTF-8 code point (as produced by `split_codepoints`) to its scalar
// value. Returns 0 for a malformed single-byte unit that is not plain ASCII.
inline uint32_t decode_codepoint(const std::string& cp) {
    if (cp.empty())
        return 0;
    const unsigned char c0 = (unsigned char)cp[0];
    if (cp.size() == 1)
        return c0 < 0x80 ? (uint32_t)c0 : 0;
    if (cp.size() == 2)
        return ((uint32_t)(c0 & 0x1F) << 6) | (uint32_t)((unsigned char)cp[1] & 0x3F);
    if (cp.size() == 3)
        return ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)((unsigned char)cp[1] & 0x3F) << 6) |
               (uint32_t)((unsigned char)cp[2] & 0x3F);
    return ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)((unsigned char)cp[1] & 0x3F) << 12) |
           ((uint32_t)((unsigned char)cp[2] & 0x3F) << 6) | (uint32_t)((unsigned char)cp[3] & 0x3F);
}

// True for the scripts this path exists to serve: Japanese and Chinese, plus
// the punctuation and vowel-extension marks that are part of a repeated unit
// ("はい、" / "あ〜"). Deliberately NOT Hangul, Thai or Lao — Korean is written
// with spaces so the word path already covers it, and the others are untested
// here; widening the gate needs its own fixtures.
inline bool is_cjk_codepoint(uint32_t cp) {
    return (cp >= 0x3000 && cp <= 0x303F) || // CJK symbols & punctuation 、。〜
           (cp >= 0x3040 && cp <= 0x309F) || // Hiragana
           (cp >= 0x30A0 && cp <= 0x30FF) || // Katakana (incl. ー U+30FC)
           (cp >= 0x3400 && cp <= 0x4DBF) || // CJK Unified Ext A
           (cp >= 0x4E00 && cp <= 0x9FFF) || // CJK Unified
           (cp >= 0xF900 && cp <= 0xFAFF) || // CJK Compatibility Ideographs
           (cp >= 0xFF01 && cp <= 0xFF9F) || // Fullwidth forms, halfwidth katakana
           (cp >= 0x20000 && cp <= 0x2A6DF); // CJK Unified Ext B
}

// A token gets the code-point path only if it is long enough to carry evidence
// of a loop AND majority CJK. Both halves matter: the length floor keeps short
// natural reduplication ("ええ", "はいはい") intact, and the script test keeps
// Latin — where doubled letters are ordinary spelling — on the word path.
inline bool wants_codepoint_collapse(const std::vector<std::string>& cps) {
    // Below this many code points there is not enough text to distinguish a
    // decode loop from ordinary emphatic repetition.
    constexpr size_t MIN_CPS = 8;
    // Fraction of code points that must be CJK for the token to count as CJK.
    constexpr double MIN_CJK_RATIO = 0.6;

    if (cps.size() < MIN_CPS)
        return false;
    size_t cjk = 0;
    for (const auto& cp : cps)
        if (is_cjk_codepoint(decode_codepoint(cp)))
            cjk++;
    return (double)cjk >= MIN_CJK_RATIO * (double)cps.size();
}

// Collapse repeated code-point n-grams in a single whitespace-free token,
// using the same passes and rep limits as the word path. Returns `token`
// unchanged when the gate does not fire, so it is safe to call on anything.
inline std::string fix_loops_codepoints(const std::string& token, int max_n = 8) {
    if (loopfix_disabled())
        return token;
    const std::vector<std::string> cps = split_codepoints(token);
    if (!wants_codepoint_collapse(cps))
        return token;

    std::vector<int> idx((int)cps.size());
    for (size_t i = 0; i < cps.size(); i++)
        idx[i] = (int)i;
    for (int n = max_n; n >= 1; n--)
        idx = collapse_indices(cps, idx, n, n == 1 ? 3 : 2);

    std::string out;
    for (int i : idx)
        out += cps[i];
    return out;
}

// Split `text` on whitespace, collapse repeated n-grams from `max_n` down to 1
// (unigrams kept up to 3 reps, longer n-grams up to 2), and re-join with single
// spaces. Returns cleaned text.
//
// Each surviving token then gets a second, code-point-level collapse if it is
// majority CJK (§W1) — that is where a Japanese/Chinese loop lives, since the
// whitespace split leaves the whole segment as one token. Note this rewrites
// token *content*, which is why `fix_loops_keep_indices` below is left alone:
// it reports membership for parallel per-word arrays, and membership is
// unaffected. Word-level output on CJK is therefore still uncollapsed; on
// those scripts a whitespace "word" is not a linguistic unit anyway.
inline std::string fix_loops(const std::string& text, int max_n = 16) {
    std::vector<std::string> words = split_words(text);
    std::vector<int> keep = fix_loops_keep_indices(words, max_n);
    std::string out;
    for (size_t k = 0; k < keep.size(); k++) {
        if (k)
            out += ' ';
        out += fix_loops_codepoints(words[keep[k]]);
    }
    return out;
}

} // namespace core_ngram

// crispasr_watermark_stats.h — how to read a spread-spectrum watermark score.
//
// The detector (crispasr_watermark.h) is a SIGN-AGREEMENT TEST: it checks
// whether each of N pseudo-random spectral bins deviates from its neighbours in
// the direction the embedder would have pushed it, and returns the fraction of
// bins that agree. On audio with no watermark each bin is a coin flip, so the
// score is Binomial(N, 0.5) / N — mean 0.5, and NOT "0 means clean".
//
// That is the whole reason this header exists. The verdict used to be
// `confidence > 0.65 => "AI-GENERATED WATERMARK DETECTED"`. With N = 32 that
// threshold is 21/32 agreements, which unwatermarked audio reaches by chance
// 5.5% of the time — so roughly one in eighteen clean files was reported as
// watermarked, in the confident past tense. Measured on real speech
// (samples/*.wav, 55 clips): 4.8% false positives, matching the theory.
//
// A score is meaningless without its N. Report the p-value instead: the exact
// probability that unwatermarked audio would score this high or higher.
//
// WHAT THIS CANNOT FIX
// -------------------
// The instrument is weak, and moving the threshold only trades one error for
// the other. Measured true-positive rate on freshly watermarked speech:
//
//   clip length   >0.65 (p=5.5%)   >0.71875 (p=1.0%)
//        1.0 s        78 %              18 %
//        2.5 s        86 %              43 %
//        5.0 s        80 %              40 %
//       10.0 s       100 %              60 %
//
// So a threshold tight enough to stop crying wolf misses most short clips.
// The honest response is a three-way verdict with the p-value attached and the
// inconclusive band named as such, rather than a binary claim the data does not
// support. A NEGATIVE result is especially weak evidence: it does not mean the
// audio is human-made. AudioSeal (`--watermark-model auto`) is the sensitive
// detector; this one is a convenience check.
//
// SCOPE — two things this does not cover:
//
//   * AudioSeal. `--watermark-model auto` routes detection to a neural model
//     whose output is a probability, not a sign-agreement count. The binomial
//     null does not apply to it; callers must not run these bands over an
//     AudioSeal score. crispasr_wm_dispatch::get_ctx() says which detector ran.
//   * Partial bin sets. The detector skips bins that fall outside the spectrum
//     or sit in silence, and returns only the fraction, not how many survived.
//     Callers pass CRISPASR_WATERMARK_NBINS; if fewer bins were actually
//     scored, the true p-value is slightly LARGER (weaker) than reported.
//
// Weight-free and IO-free so the bands are unit-tested without audio
// (tests/test-voice-clone-policy.cpp), like the other compliance headers.

#pragma once

#include <string>

namespace crispasr_wm_stats {

// Exact upper-tail probability P(X >= k) for X ~ Binomial(n, 1/2): the chance
// that audio with NO watermark scores at least k/n by luck alone.
// Computed in double via Pascal's rule; n = 32 here, so no overflow concern.
inline double null_tail_probability(int k, int n) {
    if (n <= 0)
        return 1.0;
    if (k <= 0)
        return 1.0;
    if (k > n)
        return 0.0;
    // Row of binomial coefficients, scaled by 2^-n as we go to avoid overflow.
    double total = 0.0;
    double c = 1.0; // C(n, 0)
    const double scale = 1.0 / (double)(1ull << (n > 62 ? 62 : n));
    for (int i = 0; i <= n; ++i) {
        if (i >= k)
            total += c;
        // C(n, i+1) = C(n, i) * (n - i) / (i + 1)
        c = c * (double)(n - i) / (double)(i + 1);
    }
    return total * scale;
}

// p-value for a score in [0,1] measured over `n_bins` bins.
inline double p_value(float score, int n_bins) {
    if (n_bins <= 0)
        return 1.0;
    // Recover the agreement count the score was derived from.
    const int k = (int)((double)score * (double)n_bins + 0.5);
    return null_tail_probability(k, n_bins);
}

enum class Verdict {
    Detected,     // p < 0.01 — would happen by chance under 1 in 100
    Inconclusive, // p < 0.20 — leaning positive, but not evidence
    NotDetected,  // consistent with unwatermarked audio
};

// Significance bands. `kDetectedP` is the bar for asserting a watermark is
// present; 1% is chosen so the claim survives being made routinely, and is
// ~5x stricter than the 0.65 score threshold it replaces.
inline constexpr double kDetectedP = 0.01;
inline constexpr double kInconclusiveP = 0.20;

inline Verdict classify(float score, int n_bins) {
    // A score at or below chance is never evidence, whatever the tail says.
    if (score <= 0.5f)
        return Verdict::NotDetected;
    const double p = p_value(score, n_bins);
    if (p < kDetectedP)
        return Verdict::Detected;
    if (p < kInconclusiveP)
        return Verdict::Inconclusive;
    return Verdict::NotDetected;
}

// ── Per-frame statistic (crispasr_watermark_detect_frames_impl) ────────────
//
// Everything above is about a BIN COUNT: score = k/n, null = Binomial(n, 1/2),
// so a p-value is available exactly. The per-frame detector returns something
// different — a calibrated confidence squashed from the binding of two
// standardised statistics (consistency across frames, and specificity against
// 15 decoy sign patterns). There is no k, and running p_value() over it would
// invent a bin count that was never scored. Hence a separate band set rather
// than an overload that silently reuses the wrong null.
//
// The squash is built so the decision point — both bars exactly met — lands on
// kFramesDetected, which is the same 0.65 the CLI and docs already speak in.
//
// Measured on 1265 one-second clips of genuinely unmarked human speech
// (VoxConverse dev + JFK, native 16 kHz, see tools/watermark_detect_ab.cpp),
// scored against the same clips after the UNCHANGED embedder marked them:
//
//   clip    sign FP/TP @0.65      frames FP/TP @0.65
//   1.0 s      5.2% / 68.6%          0.9% / 96.8%
//   2.5 s      5.1% / 79.8%          1.2% / 99.6%
//   5.0 s      4.0% / 88.0%          1.6% / 99.6%
//  10.0 s      4.9% / 100.0%         3.3% / 100.0%
//
// Better on BOTH error rates at every clip length, which is why this is the
// default and the sign test is the fallback rather than the other way round.
inline constexpr float kFramesDetected = 0.65f;
inline constexpr float kFramesInconclusive = 0.5f;

inline Verdict classify_frames(float score) {
    if (score > kFramesDetected)
        return Verdict::Detected;
    if (score > kFramesInconclusive)
        return Verdict::Inconclusive;
    return Verdict::NotDetected;
}

inline const char* verdict_line(Verdict v) {
    switch (v) {
    case Verdict::Detected:
        return "AI-GENERATED WATERMARK DETECTED";
    case Verdict::Inconclusive:
        return "INCONCLUSIVE - consistent with a watermark, but not statistically significant";
    case Verdict::NotDetected:
        return "No watermark detected (this does NOT mean the audio is human-made)";
    }
    return "No watermark detected";
}

} // namespace crispasr_wm_stats

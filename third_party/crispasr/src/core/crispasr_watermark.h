// crispasr_watermark.h — spread-spectrum audio watermark for AI-generated speech.
//
// Embeds an imperceptible pseudorandom pattern into the frequency domain
// of synthesized PCM audio. The watermark survives common transformations
// (re-encoding, moderate compression, volume normalization) because it
// is spread across many frequency bins and frames.
//
// Header-only so the unit tests can exercise it without linking the
// server translation unit.
//
// Algorithm:
//   1. Divide audio into overlapping frames (hop = frame/2).
//   2. For each frame, compute a real DFT via the Danielson-Lanczos
//      radix-2 FFT (no external dependency).
//   3. A PRNG seeded with CRISPASR_WATERMARK_KEY selects which frequency
//      bins to modulate and the sign (+/-) of each nudge.
//   4. Each selected bin's magnitude is nudged by `alpha` (default 0.005,
//      ~-46 dB below full scale — inaudible for speech).
//   5. Inverse FFT + overlap-add reconstructs the watermarked signal.
//
// Detection:
//   Same PRNG sequence → same bin selection. Correlate the sign pattern
//   against actual magnitude deltas across frames. High positive
//   correlation → watermark present.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// Watermark key — a fixed 64-bit seed that defines the pseudorandom bin
// selection. Changing this key produces a different (incompatible)
// watermark. In a production deployment you'd want this to be
// configurable; for now it's compiled-in.
#ifndef CRISPASR_WATERMARK_KEY
#define CRISPASR_WATERMARK_KEY 0x437269737041535FULL // "CrispASR" in hex-ish
#endif

// Number of frequency bins to modulate per frame. More bins = more
// robust detection but marginally higher distortion. 32 is a good
// balance for 1024-sample frames at 24 kHz.
#ifndef CRISPASR_WATERMARK_NBINS
#define CRISPASR_WATERMARK_NBINS 32
#endif

// Which detection statistic is the DEFAULT when CRISPASR_WATERMARK_DETECT is
// unset. 1 = the per-frame t + decoy statistic, which beat the sign test on
// BOTH false positives and true positives at every clip length measured
// (tools/watermark_detect_ab.cpp; table in crispasr_watermark_stats.h). The
// sign test stays reachable with CRISPASR_WATERMARK_DETECT=sign — it is the
// regression-bisection path, not dead code.
#ifndef CRISPASR_WATERMARK_DETECT_FRAMES_DEFAULT
#define CRISPASR_WATERMARK_DETECT_FRAMES_DEFAULT 1
#endif

namespace crispasr_wm {

// Simple xoshiro128+ PRNG — deterministic, fast, no external deps.
struct prng {
    uint64_t s[2];

    explicit prng(uint64_t seed) {
        // SplitMix64 to initialize state from a single seed
        s[0] = splitmix(seed);
        s[1] = splitmix(s[0]);
    }

    uint64_t next() {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = s0 + s1;
        s1 ^= s0;
        s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
        s[1] = (s1 << 36) | (s1 >> 28);
        return result;
    }

    // Uniform in [0, bound)
    uint32_t next_u32(uint32_t bound) { return (uint32_t)(next() % bound); }

private:
    static uint64_t splitmix(uint64_t& x) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

// In-place radix-2 Cooley-Tukey FFT. `re` and `im` are arrays of
// length `n` (must be a power of 2). `inverse` = true for IFFT.
inline void fft_radix2(float* re, float* im, int n, bool inverse) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    const float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= n; len <<= 1) {
        const float angle = sign * 2.0f * 3.14159265358979323846f / (float)len;
        const float wre = std::cos(angle);
        const float wim = std::sin(angle);
        for (int i = 0; i < n; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                const float tre = re[i + j + len / 2] * ure - im[i + j + len / 2] * uim;
                const float tim = re[i + j + len / 2] * uim + im[i + j + len / 2] * ure;
                re[i + j + len / 2] = re[i + j] - tre;
                im[i + j + len / 2] = im[i + j] - tim;
                re[i + j] += tre;
                im[i + j] += tim;
                const float new_ure = ure * wre - uim * wim;
                uim = ure * wim + uim * wre;
                ure = new_ure;
            }
        }
    }
    if (inverse) {
        const float inv_n = 1.0f / (float)n;
        for (int i = 0; i < n; i++) {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

// Generate the watermark bin indices and signs for one frame.
// Returns (bin_index, sign) pairs. `n_fft` is the FFT size.
// Only bins in [lo_bin, hi_bin) are candidates (avoid DC, Nyquist,
// and very low frequencies that would cause audible artifacts).
struct wm_bin {
    int index;
    int sign; // +1 or -1
};

// Watermark band + strength.
//
// v2 (issue #260): the comb lives in the SPEECH band (~1.5–4 kHz) where speech
// formants mask it. The original v1 spread the same 32-bin comb across
// ~1.5–11.7 kHz; ~20 of those bins landed at 4–11.8 kHz where clean TTS speech
// is near-silent, so the comb painted an audible "tinny" tone (a fixed set of
// horizontal lines on the spectrogram) over the high-frequency near-silence.
// Keeping the comb inside the speech band + lowering alpha makes it inaudible
// while detection stays well above threshold on real speech.
//
// CRISPASR_WATERMARK_LEGACY=1 restores the pre-#260 wideband/loud behavior so
// the two paths can be A/B'd and old watermarks re-detected. Both embed and
// detect read the same env, so they always agree on the band.
inline void wm_params(int n_fft, int& lo_bin, int& hi_bin, float& default_alpha) {
    lo_bin = n_fft / 16; // skip lowest ~6% (sub-bass) — ~1.5 kHz @ 24 kHz
    if (std::getenv("CRISPASR_WATERMARK_LEGACY")) {
        hi_bin = n_fft / 2 - 1; // ~11.7 kHz — audible comb (legacy A/B only)
        default_alpha = 0.08f;
    } else {
        // ~4.8 kHz: caps the comb below 5 kHz, removing the entire 5–12 kHz
        // "tinny" region that was the audible complaint, while reaching just
        // into the fricative/sibilance band so detection stays robust on tonal
        // voiced speech (~0.88 on real qwen3-tts vs 0.94 legacy).
        hi_bin = n_fft / 5;
        default_alpha = 0.05f;
    }
}

inline std::vector<wm_bin> generate_bin_pattern(uint64_t key, int n_fft, int n_bins, int lo_bin, int hi_bin) {
    prng rng(key);
    const int range = hi_bin - lo_bin;
    if (range <= 0 || n_bins <= 0)
        return {};

    std::vector<wm_bin> bins(n_bins);
    for (int i = 0; i < n_bins; i++) {
        bins[i].index = lo_bin + (int)rng.next_u32((uint32_t)range);
        bins[i].sign = (rng.next() & 1) ? 1 : -1;
    }
    return bins;
}

} // namespace crispasr_wm

// ---------------------------------------------------------------------------
// Inline implementation: embed watermark into float32 PCM
// ---------------------------------------------------------------------------

// Embed the CrispASR watermark into `pcm` (float32 mono, any sample rate).
// Modifies the samples in-place. `alpha` controls watermark strength:
//   0.08  = reliable detection on speech (~38 dB SNR, imperceptible)
//   0.05  = conservative (lower confidence on tonal speech)
//   0.005 = legacy (too faint for reliable detection on real speech)
//
// Industry standard: AudioSeal/WavMark use 38-42 dB SNR. Human perception
// threshold for speech masking is ~20 dB; 38 dB is 18 dB below perception.
//
// The function is a no-op for very short audio (< 1 FFT frame).
// alpha < 0 selects the band-appropriate default (0.05 speech-band, 0.08 legacy).
// alpha == 0 is an explicit zero-strength no-op (leaves the signal unchanged);
// alpha > 0 uses that exact strength.
inline void crispasr_watermark_embed_impl(float* pcm, int n_samples, float alpha = -1.0f) {
    const int n_fft = 1024;
    const int hop = n_fft / 2; // 50% overlap
    if (n_samples < n_fft)
        return;

    int lo_bin, hi_bin;
    float default_alpha;
    crispasr_wm::wm_params(n_fft, lo_bin, hi_bin, default_alpha);
    if (alpha < 0.0f)
        alpha = default_alpha;

    const auto bins =
        crispasr_wm::generate_bin_pattern(CRISPASR_WATERMARK_KEY, n_fft, CRISPASR_WATERMARK_NBINS, lo_bin, hi_bin);
    if (bins.empty())
        return;

    // Hann window
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; i++)
        window[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f * (float)i / (float)(n_fft - 1)));

    // Overlap-add buffers
    std::vector<float> out(n_samples, 0.0f);
    std::vector<float> norm(n_samples, 0.0f);
    std::vector<float> re(n_fft), im(n_fft);

    for (int start = 0; start + n_fft <= n_samples; start += hop) {
        // Window the frame
        for (int i = 0; i < n_fft; i++) {
            re[i] = pcm[start + i] * window[i];
            im[i] = 0.0f;
        }

        // Forward FFT
        crispasr_wm::fft_radix2(re.data(), im.data(), n_fft, false);

        // Compute RMS magnitude across the spectrum to set an absolute
        // watermark energy level. This way even bins with near-zero
        // natural energy (common for tonal signals) receive a detectable
        // nudge proportional to the overall frame energy.
        double rms_mag = 0.0;
        for (int k = 1; k < n_fft / 2; k++)
            rms_mag += (double)re[k] * re[k] + (double)im[k] * im[k];
        rms_mag = std::sqrt(rms_mag / (double)(n_fft / 2 - 1));
        const float nudge = alpha * (float)rms_mag;

        // Modulate the *magnitude* of selected bins while preserving
        // phase. This ensures the sign pattern is detectable via
        // magnitude measurement — adding to re alone would lose the
        // sign for near-zero bins.
        for (const auto& b : bins) {
            float mag = std::sqrt(re[b.index] * re[b.index] + im[b.index] * im[b.index]);
            float new_mag = mag + nudge * (float)b.sign;
            if (new_mag < 0.0f)
                new_mag = 0.0f;
            float scale = (mag > 1e-15f) ? (new_mag / mag) : 0.0f;
            // For bins with no energy, inject at 0-phase so detection
            // can still see the magnitude
            if (mag < 1e-15f && b.sign > 0) {
                re[b.index] = nudge;
                im[b.index] = 0.0f;
            } else {
                re[b.index] *= scale;
                im[b.index] *= scale;
            }
            // Mirror the conjugate half for real-valued signal
            int mirror = n_fft - b.index;
            if (mirror != b.index && mirror > 0 && mirror < n_fft) {
                if (mag < 1e-15f && b.sign > 0) {
                    re[mirror] = nudge;
                    im[mirror] = 0.0f;
                } else {
                    re[mirror] *= scale;
                    im[mirror] *= scale;
                }
            }
        }

        // Inverse FFT
        crispasr_wm::fft_radix2(re.data(), im.data(), n_fft, true);

        // Overlap-add with window
        for (int i = 0; i < n_fft; i++) {
            out[start + i] += re[i] * window[i];
            norm[start + i] += window[i] * window[i];
        }
    }

    // Normalize and write back only the watermark delta. Near the first and
    // last Hann-window edges the overlap-add normalization can be tiny; using
    // a short boundary ramp avoids turning that into an audible impulse.
    for (int i = 0; i < n_samples; i++) {
        if (norm[i] > 1e-4f) {
            float watermarked = out[i] / norm[i];
            float delta = watermarked - pcm[i];
            float ramp_in = std::min(1.0f, (float)i / (float)n_fft);
            float ramp_out = std::min(1.0f, (float)(n_samples - 1 - i) / (float)n_fft);
            pcm[i] += delta * std::min(ramp_in, ramp_out);
        }
    }
}

// ---------------------------------------------------------------------------
// Inline implementation: detect watermark in float32 PCM
// ---------------------------------------------------------------------------

// Returns a confidence score in [0, 1]. Values above 0.65 strongly
// indicate the CrispASR watermark is present. Values below 0.4 indicate
// no watermark (or a different key).
//
// Uses averaged-spectrum detection: computes the mean magnitude spectrum
// across all frames, then correlates the watermark bin pattern against
// the averaged spectrum. This is significantly more robust on tonal/speech
// signals than per-frame detection because frame-level noise averages out.
inline float crispasr_watermark_detect_impl(const float* pcm, int n_samples) {
    const int n_fft = 1024;
    const int hop = n_fft / 2;
    if (n_samples < n_fft)
        return 0.0f;

    int lo_bin, hi_bin;
    float default_alpha_unused;
    crispasr_wm::wm_params(n_fft, lo_bin, hi_bin, default_alpha_unused);
    const auto bins =
        crispasr_wm::generate_bin_pattern(CRISPASR_WATERMARK_KEY, n_fft, CRISPASR_WATERMARK_NBINS, lo_bin, hi_bin);
    if (bins.empty())
        return 0.0f;

    // Hann window
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; i++)
        window[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f * (float)i / (float)(n_fft - 1)));

    const int n_fft_half = n_fft / 2;
    std::vector<float> re(n_fft), im(n_fft);

    // Phase 1: Accumulate magnitude spectra across all frames
    std::vector<double> avg_mags(n_fft_half, 0.0);
    int n_frames = 0;

    for (int start = 0; start + n_fft <= n_samples; start += hop) {
        for (int i = 0; i < n_fft; i++) {
            re[i] = pcm[start + i] * window[i];
            im[i] = 0.0f;
        }
        crispasr_wm::fft_radix2(re.data(), im.data(), n_fft, false);

        for (int b = 0; b < n_fft_half; b++)
            avg_mags[b] += std::sqrt((double)re[b] * re[b] + (double)im[b] * im[b]);
        n_frames++;
    }

    if (n_frames == 0)
        return 0.0f;

    // Phase 2: Average (cancels per-frame noise, preserves watermark)
    for (int b = 0; b < n_fft_half; b++)
        avg_mags[b] /= (double)n_frames;

    // Phase 3: Correlate watermark pattern against averaged spectrum
    double correlation = 0.0;
    int valid_bins = 0;

    for (const auto& b : bins) {
        if (b.index >= n_fft_half)
            continue;
        double local_mean = 0.0;
        int count = 0;
        for (int d = -2; d <= 2; d++) {
            int nb = b.index + d;
            if (nb >= 1 && nb < n_fft_half && d != 0) {
                local_mean += avg_mags[nb];
                count++;
            }
        }
        if (count == 0)
            continue;
        local_mean /= (double)count;
        if (local_mean < 1e-12 && avg_mags[b.index] < 1e-12)
            continue;
        double ref = std::max(local_mean, (double)1e-12);
        double delta = (avg_mags[b.index] - local_mean) / ref;
        correlation += (delta > 0 ? 1.0 : -1.0) * (double)b.sign;
        valid_bins++;
    }

    if (valid_bins == 0)
        return 0.0f;

    double score = (correlation / (double)valid_bins + 1.0) / 2.0;
    if (score < 0.0)
        score = 0.0;
    if (score > 1.0)
        score = 1.0;
    return (float)score;
}

// ---------------------------------------------------------------------------
// Per-frame detector (ported from CrispTTS Phase 28)
// ---------------------------------------------------------------------------
//
// The detector above collapses the whole file to ONE averaged spectrum and then
// scores 32 bins by the SIGN of their excess, discarding the size. Under the
// null that is a coin flip per bin: mean 0.5, sd sqrt(32)/(2*32) = 0.088, so
// the 0.65 threshold sits 1.7 sigma above chance. crispasr_watermark_stats.h
// answers that honestly with an exact binomial p-value — but a p-value can only
// TRADE the two error rates, and docs/eu-ai-act.md 6.7 prices the trade: at
// p < 0.01 the true-positive rate on 1 s clips is 18%.
//
// CrispTTS replaced the STATISTIC instead and measured FP 1.9% / TP 99.4%
// against 8.6% / 97.0% — better in both directions at once, because it fixes
// the two things that made the old one weak:
//
//   1. It keeps the MAGNITUDE of each bin's excess, not just its sign.
//   2. Its sample count is the number of FRAMES (hundreds to thousands), not
//      32. The null barely moves as clips lengthen while a real mark grows with
//      the evidence available.
//
// Two questions, and a mark must answer BOTH:
//
//   t  — is the comb's excess consistent across frames at all?
//   z  — is that specific to OUR pattern, or would any pattern score as well on
//        this audio? Measured against 15 decoy sign patterns over the SAME bins,
//        from keys we never embed with, standardised by their median and MAD.
//
// Neither alone works. On a stationary tone a raw t of 11.4 means nothing
// because every decoy scores just as extremely — only z separates them. But z
// alone rejects real marks at 44.1 kHz, where the comb sits in a low-energy
// region and the decoy spread grows.
//
// THE EMBED IS UNTOUCHED. That is deliberate and load-bearing: audio marked by
// any CrispASR, CrispTTS or Susurrus release still verifies through this, and
// the three projects can read each other's marks. Do not "improve" the embed to
// suit the detector.
//
// Selected by CRISPASR_WATERMARK_DETECT=frames|sign (see detect_uses_frames()).
namespace crispasr_wm {

inline constexpr int kDetectDecoys = 15;
inline constexpr int kDetectMinFrames = 20;
inline constexpr double kDetectMinScale = 0.75; // floor on the decoy MAD scale
inline constexpr double kDetectTMin = 3.0;      // consistency bar
inline constexpr double kDetectZMin = 1.0;      // specificity bar
// The real pattern must also out-score the single STRONGEST decoy, not merely
// the decoy median — the condition that removes the stationary-tone false
// positive the median comparison lets through. CrispTTS measured FP
// 1.89% -> 0.00% with TP unchanged at 99.37%, i.e. no true positive pays for
// it. 0.70 is the midpoint of the observed gap (tone 0.59, weakest real mark
// 0.84), so neither margin is thin.
// Overridable so the term can be A/B'd without editing the header. ⚠ Setting it
// near zero makes it non-binding only while `t_true` is POSITIVE: the ratio
// carries t_true's sign, so a tiny divisor amplifies a negative into a large
// negative and the term binds harder, not less. That never flips a verdict
// (both readings sit far below the bar), but do not read a near-zero override
// as "disabled" — it is "disabled for marks, exaggerated for non-marks".
#ifndef CRISPASR_WM_MAX_DECOY_RATIO
#define CRISPASR_WM_MAX_DECOY_RATIO 0.70
#endif
inline constexpr double kDetectMaxDecoyRatio = CRISPASR_WM_MAX_DECOY_RATIO;
// Squash so the decision point (a ratio of 1.0 on the binding constraint) lands
// exactly on the 0.65 score threshold the docs and CLI already speak in.
inline constexpr double kTScale = 0.35;
inline constexpr double kTCentre = 1.0 - kTScale * 0.6190392; // ln(0.65/0.35)

inline double t_to_confidence(double t_stat) {
    const double z = (t_stat - kTCentre) / kTScale;
    if (z > 60.0)
        return 1.0;
    if (z < -60.0)
        return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

} // namespace crispasr_wm

// Per-frame t + decoy-specificity detector. Returns a confidence in [0, 1]
// calibrated so > 0.65 means present, matching the existing threshold.
inline float crispasr_watermark_detect_frames_impl(const float* pcm, int n_samples) {
    const int n_fft = 1024;
    const int hop = n_fft / 2;
    if (n_samples < n_fft)
        return 0.0f;

    int lo_bin, hi_bin;
    float alpha_unused;
    crispasr_wm::wm_params(n_fft, lo_bin, hi_bin, alpha_unused);
    const auto bins =
        crispasr_wm::generate_bin_pattern(CRISPASR_WATERMARK_KEY, n_fft, CRISPASR_WATERMARK_NBINS, lo_bin, hi_bin);
    if (bins.empty())
        return 0.0f;

    const int n_fft_half = n_fft / 2;
    std::vector<int> idx;
    std::vector<double> sgn;
    idx.reserve(bins.size());
    sgn.reserve(bins.size());
    for (const auto& b : bins) {
        if (b.index < n_fft_half) {
            idx.push_back(b.index);
            sgn.push_back((double)b.sign);
        }
    }
    const int nb = (int)idx.size();
    if (nb == 0)
        return 0.0f;

    int n_frames = 0;
    for (int start = 0; start + n_fft <= n_samples; start += hop)
        n_frames++;
    if (n_frames < crispasr_wm::kDetectMinFrames)
        return 0.0f;

    // Decoy sign patterns over the SAME bins, from keys never embedded with.
    // Mirrors CrispTTS: key ^ (0x9E3779B97F4A7C15 * (k + 1)).
    const int n_pat = 1 + crispasr_wm::kDetectDecoys;
    std::vector<double> pats((size_t)n_pat * nb);
    for (int j = 0; j < nb; j++)
        pats[j] = sgn[j];
    for (int k = 0; k < crispasr_wm::kDetectDecoys; k++) {
        const uint64_t dk = (uint64_t)CRISPASR_WATERMARK_KEY ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(k + 1));
        crispasr_wm::prng rng(dk);
        for (int j = 0; j < nb; j++)
            pats[(size_t)(k + 1) * nb + j] = (rng.next() & 1) ? 1.0 : -1.0;
    }

    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; i++)
        window[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f * (float)i / (float)(n_fft - 1)));

    // Per-frame correlation of the local-excess vector with each pattern.
    std::vector<double> sum((size_t)n_pat, 0.0), sumsq((size_t)n_pat, 0.0);
    std::vector<float> re(n_fft), im(n_fft);
    std::vector<double> mags(n_fft_half), excess(nb);

    for (int start = 0, f = 0; start + n_fft <= n_samples; start += hop, f++) {
        for (int i = 0; i < n_fft; i++) {
            re[i] = pcm[start + i] * window[i];
            im[i] = 0.0f;
        }
        crispasr_wm::fft_radix2(re.data(), im.data(), n_fft, false);
        for (int b = 0; b < n_fft_half; b++)
            mags[b] = std::sqrt((double)re[b] * re[b] + (double)im[b] * im[b]);

        // Local baseline: the same +-2 neighbours the sign test uses. NOT
        // excluding other comb bins from it — CrispTTS measured that worse
        // (separation +0.031 -> +0.017), because the comb's signs are random so
        // an opposite-signed neighbour raises the contrast rather than muddying
        // it. Widening to +-4 or +-6 was worse still.
        for (int j = 0; j < nb; j++) {
            double local = 0.0;
            int count = 0;
            for (int d = -2; d <= 2; d++) {
                if (d == 0)
                    continue;
                const int b = idx[j] + d;
                if (b >= 1 && b < n_fft_half) {
                    local += mags[b];
                    count++;
                }
            }
            local = count ? local / (double)count : 0.0;
            const double ref = std::max(local, 1e-12);
            excess[j] = (mags[idx[j]] - local) / ref;
        }

        for (int p = 0; p < n_pat; p++) {
            double c = 0.0;
            const double* pv = &pats[(size_t)p * nb];
            for (int j = 0; j < nb; j++)
                c += excess[j] * pv[j];
            c /= (double)nb;
            sum[p] += c;
            sumsq[p] += c * c;
        }
    }

    // One-sample t across frames, for the true pattern and every decoy.
    std::vector<double> t_all((size_t)n_pat, 0.0);
    const double nf = (double)n_frames;
    for (int p = 0; p < n_pat; p++) {
        const double mean = sum[p] / nf;
        double var = (sumsq[p] - nf * mean * mean) / (nf - 1.0);
        if (!(var > 0.0))
            var = 0.0;
        const double sd = std::max(std::sqrt(var), 1e-12);
        t_all[p] = mean / (sd / std::sqrt(nf));
    }

    std::vector<double> dec(t_all.begin() + 1, t_all.end());
    std::sort(dec.begin(), dec.end());
    const size_t m = dec.size();
    const double centre = (m % 2) ? dec[m / 2] : 0.5 * (dec[m / 2 - 1] + dec[m / 2]);
    std::vector<double> absdev(m);
    for (size_t i = 0; i < m; i++)
        absdev[i] = std::fabs(dec[i] - centre);
    std::sort(absdev.begin(), absdev.end());
    const double mad = (m % 2) ? absdev[m / 2] : 0.5 * (absdev[m / 2 - 1] + absdev[m / 2]);
    const double scale = std::max(1.4826 * mad, crispasr_wm::kDetectMinScale);
    const double z = (t_all[0] - centre) / scale;

    // Third condition: beat the STRONGEST decoy, not just the typical one. On a
    // stationary tone the real pattern scores high and so does every decoy — a
    // median comparison cannot see that, a maximum comparison can. The tone
    // scores 0.59 on this ratio (t_true 11.44 against a decoy maximum of 19.44:
    // every absent pattern beats the real one, which is the tell), while the
    // weakest genuine mark scores 0.84, so 0.70 sits mid-gap.
    double strongest_decoy = 0.0;
    for (size_t i = 1; i < t_all.size(); i++)
        strongest_decoy = std::max(strongest_decoy, std::fabs(t_all[i]));
    const double decoy_ratio = t_all[0] / std::max(strongest_decoy, 1e-9);

    // The BINDING constraint decides — a mark must clear all three bars.
    const double ratio = std::min({t_all[0] / crispasr_wm::kDetectTMin, z / crispasr_wm::kDetectZMin,
                                   decoy_ratio / crispasr_wm::kDetectMaxDecoyRatio});
    return (float)crispasr_wm::t_to_confidence(ratio);
}

// Which statistic `--detect-watermark` uses. The old sign test stays reachable
// for A/B and for re-reading a score the way an older release reported it:
//   CRISPASR_WATERMARK_DETECT=sign    -> averaged-spectrum sign agreement
//   CRISPASR_WATERMARK_DETECT=frames  -> per-frame t + decoy specificity
inline bool crispasr_watermark_detect_uses_frames() {
    const char* e = std::getenv("CRISPASR_WATERMARK_DETECT");
    if (!e || !*e)
        return CRISPASR_WATERMARK_DETECT_FRAMES_DEFAULT;
    return !(std::strcmp(e, "sign") == 0 || std::strcmp(e, "0") == 0);
}

// The ONE entry point every surface must call. There are two spread-spectrum
// call sites (the CLI dispatch and the session C-ABI) and they have to agree on
// which statistic ran, or the same file reads differently depending on which
// binding asked — the multi-surface trap this repo keeps re-learning. Selecting
// inside a shared function is what makes that structural instead of a
// convention two files have to remember.
inline float crispasr_watermark_detect_select(const float* pcm, int n_samples) {
    return crispasr_watermark_detect_uses_frames() ? crispasr_watermark_detect_frames_impl(pcm, n_samples)
                                                   : crispasr_watermark_detect_impl(pcm, n_samples);
}

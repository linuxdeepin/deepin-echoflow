// crispasr-f0-eval — measure and compare CrispASR's F0 backends.
//
//   crispasr-f0-eval --crepe crepe-tiny-f16.gguf --beatrice beatrice-pitch-f32.gguf
//   crispasr-f0-eval --beatrice <gguf> --wav samples/jfk.wav
//
// WHY THIS EXISTS. The Beatrice pitch port was validated for PARITY (it matches
// torch stage by stage), which says nothing about whether the model is accurate.
// Accuracy was originally measured in a throwaway Python script against the
// TORCH reference, and the claim that it transfers to C++ rested on parity --
// an inference, not a measurement. This drives the actual shipped C++ path.
//
// Synthetic tones are the ground truth: sawtooth, because it has the harmonic
// structure of voiced speech (a pure sine is an unfair softball for a pitch
// detector trained on voices).
//
// TEST FREQUENCIES ARE DELIBERATELY OFF-GRID. Beatrice quantises to 96
// bins/octave anchored at A1, i.e. 8 bins per equal-tempered semitone, so
// testing on musical notes makes every answer land exactly on a bin and reports
// a flattering ~0 cents. The first Python run made exactly that mistake.

#include "beatrice_pitch.h"
#include "crepe.h"

#include "core/wav_reader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kSR = 16000;

// Beatrice's bin -> Hz. Recovered empirically from octave pairs (220 Hz -> bin
// 192 and 440 Hz -> bin 288 differ by exactly 96); the trainer source never
// states it. 96 bins/octave anchored at A1 = 12.5 cents per bin.
double beatrice_bin_to_hz(int bin) {
    return 55.0 * std::pow(2.0, (double)bin / 96.0);
}

std::vector<float> sawtooth(double f0, double seconds, float amp = 0.3f) {
    const int n = (int)(seconds * kSR);
    std::vector<float> v((size_t)n);
    for (int i = 0; i < n; i++) {
        const double t = (double)i / kSR;
        const double p = t * f0;
        v[(size_t)i] = (float)(amp * 2.0 * (p - std::floor(0.5 + p)));
    }
    return v;
}

double cents(double got, double want) {
    return 1200.0 * std::log2(got / want);
}

struct Score {
    double median_cents = 0; // |error| at the median frame
    double stab_cents = 0;   // spread across frames (robust)
    double octave_pct = 0;   // frames off by ~an octave
    int n = 0;
};

// Median of |error|, plus how much the estimate wanders frame to frame. Both are
// medians rather than means: one bad frame should not dominate the verdict.
Score score(const std::vector<double>& hz, double truth) {
    Score s;
    std::vector<double> e, c;
    for (double h : hz) {
        if (h <= 0)
            continue; // unvoiced / no estimate
        const double d = cents(h, truth);
        c.push_back(d);
        e.push_back(std::fabs(d));
    }
    s.n = (int)e.size();
    if (e.empty())
        return s;
    std::vector<double> se = e;
    std::sort(se.begin(), se.end());
    s.median_cents = se[se.size() / 2];
    std::vector<double> dev;
    const double med = se[se.size() / 2];
    for (double v : e)
        dev.push_back(std::fabs(v - med));
    std::sort(dev.begin(), dev.end());
    s.stab_cents = dev[dev.size() / 2];
    int oct = 0;
    for (double v : c)
        if (std::fabs(std::fabs(v) - 1200.0) < 100.0)
            oct++;
    s.octave_pct = 100.0 * oct / (double)c.size();
    return s;
}

// Drop the first and last quarter: both detectors have edge transients, and an
// edge artifact is not what this is measuring.
template <typename T> std::vector<T> middle(const std::vector<T>& v) {
    if (v.size() < 8)
        return v;
    return std::vector<T>(v.begin() + (long)v.size() / 4, v.end() - (long)v.size() / 4);
}

// `unvoiced_thresh` > 0 gates on pitch_features[0] via beatrice_pitch_to_f0_hz,
// which writes 0.0 for unvoiced -- the rvc_svc_convert() convention. Without it
// the model reports a confident pitch for silence.
std::vector<double> run_beatrice(beatrice_pitch_context* ctx, const std::vector<float>& pcm, float unvoiced_thresh) {
    std::vector<double> out;
    beatrice_pitch_result* r = beatrice_pitch_estimate(ctx, pcm.data(), (int)pcm.size());
    if (!r)
        return out;
    std::vector<float> f0((size_t)r->n_frames);
    beatrice_pitch_to_f0_hz(r, unvoiced_thresh, f0.data());
    for (int i = 0; i < r->n_frames; i++)
        out.push_back(f0[(size_t)i] > 0.0f ? (double)f0[(size_t)i] : -1.0);
    beatrice_pitch_result_free(r);
    return middle(out);
}

std::vector<double> run_crepe(crepe_context* ctx, const std::vector<float>& pcm, double conf_min) {
    std::vector<double> out;
    const int nf = crepe_n_frames(ctx, (int)pcm.size(), 10.0f);
    if (nf <= 0)
        return out;
    std::vector<crepe_frame> fr((size_t)nf);
    const int got = crepe_compute_f0(ctx, pcm.data(), (int)pcm.size(), 10.0f, fr.data(), nf);
    for (int i = 0; i < got; i++)
        out.push_back(fr[(size_t)i].voiced_prob >= conf_min ? fr[(size_t)i].f0_hz : -1.0);
    return middle(out);
}

} // namespace

int main(int argc, char** argv) {
    std::string crepe_path, beatrice_path, wav_path;
    double conf_min = 0.0;          // CREPE reports every frame; do not silently drop any
    float unvoiced_thresh = -99.0f; // beatrice voicing gate: ENERGY (see beatrice_pitch.h)
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--crepe")
            crepe_path = next();
        else if (a == "--beatrice")
            beatrice_path = next();
        else if (a == "--wav")
            wav_path = next();
        else if (a == "--crepe-conf")
            conf_min = atof(next());
        else if (a == "--beatrice-energy")
            unvoiced_thresh = (float)atof(next());
        else {
            fprintf(stderr, "usage: crispasr-f0-eval [--crepe m.gguf] [--beatrice m.gguf] [--wav f.wav]\n");
            return 2;
        }
    }
    if (crepe_path.empty() && beatrice_path.empty()) {
        fprintf(stderr, "error: give at least one of --crepe / --beatrice\n");
        return 2;
    }

    crepe_context* cc = nullptr;
    beatrice_pitch_context* bc = nullptr;
    if (!crepe_path.empty()) {
        cc = crepe_init(crepe_path.c_str(), 4);
        if (!cc) {
            fprintf(stderr, "error: cannot load crepe model %s\n", crepe_path.c_str());
            return 2;
        }
    }
    if (!beatrice_path.empty()) {
        beatrice_pitch_params p = beatrice_pitch_default_params();
        p.n_threads = 4;
        p.use_gpu = false;
        bc = beatrice_pitch_init_from_file(beatrice_path.c_str(), p);
        if (!bc) {
            fprintf(stderr, "error: cannot load beatrice model %s\n", beatrice_path.c_str());
            return 2;
        }
    }

    // Off-grid on purpose (see the header): none of these is an equal-tempered
    // note, so neither backend's quantisation is flattered.
    const double freqs[] = {58.3,  71.9,  87.1,  101.7, 118.4, 143.9, 167.5, 187.3, 209.1, 241.7,
                            277.9, 311.2, 351.4, 419.6, 505.2, 601.3, 707.8, 833.1, 971.4};
    const int nf = (int)(sizeof(freqs) / sizeof(freqs[0]));

    printf("F0 backend comparison — sawtooth tones, 1.0 s, %d Hz, OFF-GRID frequencies\n", kSR);
    printf("median |error| in cents; 'stab' = median frame-to-frame deviation; 'oct%%' = octave errors\n\n");
    printf("%9s |", "true Hz");
    if (bc)
        printf("  %-28s|", "beatrice");
    if (cc)
        printf("  %-28s", "crepe");
    printf("\n%9s |", "");
    if (bc)
        printf("  %8s %7s %6s     |", "cents", "stab", "oct%");
    if (cc)
        printf("  %8s %7s %6s", "cents", "stab", "oct%");
    printf("\n");

    std::vector<double> b_err, c_err;
    for (int i = 0; i < nf; i++) {
        const double f0 = freqs[i];
        const std::vector<float> pcm = sawtooth(f0, 1.0);
        printf("%9.1f |", f0);
        if (bc) {
            const Score s = score(run_beatrice(bc, pcm, unvoiced_thresh), f0);
            printf("  %8.1f %7.1f %6.1f     |", s.median_cents, s.stab_cents, s.octave_pct);
            b_err.push_back(s.median_cents);
        }
        if (cc) {
            const Score s = score(run_crepe(cc, pcm, conf_min), f0);
            printf("  %8.1f %7.1f %6.1f", s.median_cents, s.stab_cents, s.octave_pct);
            c_err.push_back(s.median_cents);
        }
        printf("\n");
    }

    auto summarise = [&](const char* name, const std::vector<double>& e) {
        if (e.empty())
            return;
        int within10 = 0, within50 = 0;
        for (double v : e) {
            if (v < 10)
                within10++;
            if (v < 50)
                within50++;
        }
        std::vector<double> s = e;
        std::sort(s.begin(), s.end());
        printf("  %-10s median %6.1f cents | within 10c: %d/%zu | within 50c: %d/%zu\n", name, s[s.size() / 2],
               within10, e.size(), within50, e.size());
    };
    printf("\nsummary across all test tones:\n");
    summarise("beatrice", b_err);
    summarise("crepe", c_err);

    if (!wav_path.empty()) {
        printf("\nreal audio (%s): backends compared against EACH OTHER, not truth --\n", wav_path.c_str());
        printf("  there is no ground truth here, so this shows agreement, not accuracy.\n");
        std::vector<float> pcm;
        int sr = 0;
        if (!crispasr::core::read_wav_mono_pcm16(wav_path, pcm, sr)) {
            printf("  (could not read wav)\n");
        } else if (sr != kSR) {
            printf("  (wav is %d Hz; this tool needs %d Hz)\n", sr, kSR);
        } else if (bc && cc) {
            const std::vector<double> b = run_beatrice(bc, pcm, unvoiced_thresh);
            const std::vector<double> c = run_crepe(cc, pcm, conf_min);
            const size_t n = std::min(b.size(), c.size());
            std::vector<double> d;
            for (size_t i = 0; i < n; i++)
                if (b[i] > 0 && c[i] > 0)
                    d.push_back(std::fabs(cents(b[i], c[i])));
            if (!d.empty()) {
                std::sort(d.begin(), d.end());
                int w50 = 0, oct = 0;
                for (double v : d) {
                    if (v < 50)
                        w50++;
                    if (std::fabs(v - 1200.0) < 100.0)
                        oct++;
                }
                printf("  %zu frames: median disagreement %.1f cents | within 50c: %.1f%% | octave: %.1f%%\n", d.size(),
                       d[d.size() / 2], 100.0 * w50 / d.size(), 100.0 * oct / d.size());
            }
        } else {
            printf("  (need both --crepe and --beatrice to compare)\n");
        }
    }

    if (cc)
        crepe_free(cc);
    if (bc)
        beatrice_pitch_free(bc);
    return 0;
}

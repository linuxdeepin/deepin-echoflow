// tools/watermark_detect_ab.cpp — A/B the two spread-spectrum detectors.
//
// Produces the false-positive / true-positive table that decides which
// statistic `--detect-watermark` should default to. Both detectors read the
// SAME embed, so a clip is scored twice: once as it arrives (null) and once
// after crispasr_watermark_embed_impl() marks it (positive).
//
//   build/bin/watermark-detect-ab <wav>...
//
// CORPUS RULES — the two ways this measurement lies, both of which CrispTTS hit
// before we did (their first "null" corpus put the null maximum above every
// positive):
//
//   1. **No already-marked audio in the null set.** TTS output from any of the
//      three projects is watermarked by default, so "a directory of wavs I had
//      lying around" is not a null corpus. Use human recordings.
//   2. **No clips upsampled from a lower rate.** An upsampled 16 kHz file has a
//      hard spectral cliff at 8 kHz. At 44.1 kHz the comb spans bins 64..204 =
//      2756..8786 Hz, so the cliff lands INSIDE the band and reads as structure.
//      At the native 16 kHz of our corpus the comb is 1000..3188 Hz and the
//      cliff cannot reach it — which is why this harness reports the rate of
//      every input and refuses a mixed-rate run.
//
// The clip lengths mirror docs/eu-ai-act.md 6.7 so the new numbers drop into
// the same table as the old ones.

#include "core/crispasr_watermark.h"
#include "core/wav_reader.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// The two operating points the docs quote for the sign test, plus the single
// calibrated decision point of the per-frame statistic.
constexpr float kThr065 = 0.65f;    // p = 5.5% under the binomial null
constexpr float kThr072 = 0.71875f; // p = 1.0% — the "DETECTED" bar today

struct Rates {
    int n = 0, fp = 0;
    int tp065 = 0, tp072 = 0, fp065 = 0, fp072 = 0;
};

const double kClipSeconds[] = {1.0, 2.5, 5.0, 10.0};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <wav>...\n", argv[0]);
        return 2;
    }

    std::vector<std::vector<float>> files;
    int sr_common = 0;
    for (int i = 1; i < argc; i++) {
        std::vector<float> pcm;
        int sr = 0;
        if (!crispasr::core::read_wav_mono_pcm16(argv[i], pcm, sr)) {
            std::fprintf(stderr, "FAIL: cannot read %s\n", argv[i]);
            return 1;
        }
        if (sr_common == 0)
            sr_common = sr;
        if (sr != sr_common) {
            // Mixing rates would silently mix comb-band placements — refuse.
            std::fprintf(stderr, "FAIL: %s is %d Hz, expected %d Hz (mixed-rate corpus)\n", argv[i], sr, sr_common);
            return 1;
        }
        files.push_back(std::move(pcm));
    }
    const int lo_check = 64 * sr_common / 1024, hi_check = 204 * sr_common / 1024;
    std::printf("corpus: %zu file(s) @ %d Hz; comb band ~%d..%d Hz; Nyquist %d Hz\n", files.size(), sr_common, lo_check,
                hi_check, sr_common / 2);
    if (8000 >= lo_check && 8000 <= hi_check)
        std::printf("  WARNING: an 8 kHz upsampling cliff would land INSIDE the comb band at this rate\n");

    std::printf("\n%-8s %-7s %6s | %-22s | %-22s\n", "clip", "stat", "n", "false positives", "true positives");
    std::printf("%-8s %-7s %6s | %10s %10s | %10s %10s\n", "", "", "", ">0.65", ">0.71875", ">0.65", ">0.71875");

    for (double clip_s : kClipSeconds) {
        const int clip_n = (int)(clip_s * sr_common);
        Rates sign, frames;
        for (const auto& pcm : files) {
            for (size_t off = 0; off + (size_t)clip_n <= pcm.size(); off += (size_t)clip_n) {
                std::vector<float> clean(pcm.begin() + (std::ptrdiff_t)off,
                                         pcm.begin() + (std::ptrdiff_t)(off + clip_n));
                // Skip near-silence: neither detector claims anything about it
                // and it is not what a false positive means here.
                double rms = 0.0;
                for (float v : clean)
                    rms += (double)v * v;
                rms = std::sqrt(rms / (double)clip_n);
                if (rms < 1e-3)
                    continue;

                std::vector<float> marked = clean;
                crispasr_watermark_embed_impl(marked.data(), clip_n);

                const float s_null = crispasr_watermark_detect_impl(clean.data(), clip_n);
                const float s_mark = crispasr_watermark_detect_impl(marked.data(), clip_n);
                const float f_null = crispasr_watermark_detect_frames_impl(clean.data(), clip_n);
                const float f_mark = crispasr_watermark_detect_frames_impl(marked.data(), clip_n);

                sign.n++;
                sign.fp065 += s_null > kThr065;
                sign.fp072 += s_null > kThr072;
                sign.tp065 += s_mark > kThr065;
                sign.tp072 += s_mark > kThr072;

                frames.n++;
                frames.fp065 += f_null > kThr065;
                frames.fp072 += f_null > kThr072;
                frames.tp065 += f_mark > kThr065;
                frames.tp072 += f_mark > kThr072;
            }
        }
        if (sign.n == 0)
            continue;
        auto pct = [](int a, int b) { return b ? 100.0 * (double)a / (double)b : 0.0; };
        std::printf("%-8.1f %-7s %6d | %9.1f%% %9.1f%% | %9.1f%% %9.1f%%\n", clip_s, "sign", sign.n,
                    pct(sign.fp065, sign.n), pct(sign.fp072, sign.n), pct(sign.tp065, sign.n), pct(sign.tp072, sign.n));
        std::printf("%-8.1f %-7s %6d | %9.1f%% %9.1f%% | %9.1f%% %9.1f%%\n", clip_s, "frames", frames.n,
                    pct(frames.fp065, frames.n), pct(frames.fp072, frames.n), pct(frames.tp065, frames.n),
                    pct(frames.tp072, frames.n));
    }
    return 0;
}

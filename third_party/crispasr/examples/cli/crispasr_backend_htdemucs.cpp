// crispasr_backend_htdemucs.cpp — HTDemucs source-separation backend shim.
//
// Source separation is its OWN task (audio out, N named stems), not
// transcription. It is exposed through the `--separate` dispatcher
// (examples/cli/crispasr_separate_cli.cpp), which resolves + resamples the
// audio to the model's rate, runs htdemucs_separate / mel_band_roformer_separate,
// and writes one WAV per stem. See docs/source-separation-surface.md.
//
// This shim exists only so htdemucs still appears in --list-backends with the
// CAP_SEPARATE capability and so `--backend htdemucs` without `--separate`
// gives a clear redirect instead of trying to run separation through the
// transcribe() contract (which cannot carry audio stems). The earlier version
// ran htdemucs_separate inside transcribe(), fed it the pipeline's mono 16 kHz
// where the model needs stereo 44.1 kHz, and stashed the result in a field the
// CLI never read — so it produced no stems. The real, validated path is
// `--separate` (both htdemucs and mel-band-roformer).

#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

class HtdemucsBackend : public CrispasrBackend {
public:
    const char* name() const override { return "htdemucs"; }

    uint32_t capabilities() const override { return CAP_SEPARATE | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: htdemucs is a source-separation model — run it with --separate\n"
                        "  (e.g. `crispasr --separate -m <htdemucs.gguf> -f mix.wav`), not as a\n"
                        "  transcribe backend. Separation writes <input>_<stem>.wav.\n");
        return false;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    void shutdown() override {}
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_htdemucs_backend() {
    return std::make_unique<HtdemucsBackend>();
}

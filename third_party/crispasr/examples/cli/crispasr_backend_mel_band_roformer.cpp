// crispasr_backend_mel_band_roformer.cpp — Mel-Band RoFormer separation shim.
//
// Companion to crispasr_backend_htdemucs.cpp: source separation is its own task
// (audio out, N named stems), exposed through the `--separate` dispatcher.
// See docs/source-separation-surface.md.
//
// Why this arrived late, and what it fixes: `--separate` resolves the model and
// reads `general.architecture` from the GGUF in its OWN dispatcher, so
// mel-band-roformer worked on the CLI — as the DEFAULT separation backend, no
// less — while appearing in neither the CLI roster/factory nor either
// arch-detect table, and having no session arm at all. So `--list-backends`
// did not list it, `-m <mbr.gguf>` auto-detection could not route to it, and
// separation through the session C ABI was htdemucs-only: every binding
// (Dart/Flutter, Python, Go, wasm) could reach the 4-stem model and not the
// MIT-licensed vocals/instrumental one.
//
// That is the multi-surface dispatch trap in its purest form — a working CLI
// path masking a backend that no other surface could see. See
// docs/contributing.md section 7 for the full task-shaped wiring list.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

class MelBandRoformerBackend : public CrispasrBackend {
public:
    const char* name() const override { return "mel-band-roformer"; }

    uint32_t capabilities() const override { return CAP_SEPARATE | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: mel-band-roformer is a source-separation model — run it with --separate\n"
                        "  (e.g. `crispasr --separate -m <mel-band-roformer-vocals-f16.gguf> -f song.wav`),\n"
                        "  not as a transcribe backend. --separate writes one WAV per stem.\n");
        return false;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    void shutdown() override {}
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_mel_band_roformer_backend() {
    return std::make_unique<MelBandRoformerBackend>();
}

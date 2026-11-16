// crispasr_backend_btc.cpp — BTC chord-recognition backend shim.
//
// Chord recognition is its OWN task (a chord timeline out, not text), so it is
// not layered onto transcribe() — see docs/source-separation-surface.md for the
// precedent that settled this for stems, and crispasr_backend_crepe.cpp for the
// pitch equivalent. It is exposed through the `--chords` dispatcher
// (examples/cli/crispasr_chords_cli.cpp), which resolves the model, decodes the
// audio to BTC's 22.05 kHz mono, runs btc_chords_recognize and prints one span
// per line.
//
// This shim exists only so `btc-chords` appears in --list-backends with the
// CAP_CHORDS capability and so `--backend btc-chords` without `--chords` gives a
// clear redirect instead of trying to force a chord timeline through the
// transcribe() contract (which carries text segments and cannot represent it).
//
// It was added late: the runtime, the CLI dispatcher and the session C ABI all
// shipped first, and `btc` appeared in NEITHER the roster nor the arch-detect
// table in crispasr_backend.cpp — so `--list-backends` did not know the backend
// existed and the auto-generated docs/feature-matrix.md would have dropped any
// hand-written row for it. That is the multi-surface dispatch trap; the fix is
// this shim plus the four registration sites in crispasr_backend.cpp.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

class BtcChordsBackend : public CrispasrBackend {
public:
    const char* name() const override { return "btc-chords"; }

    uint32_t capabilities() const override { return CAP_CHORDS | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: btc-chords is a chord-recognition model — run it with --chords\n"
                        "  (e.g. `crispasr --chords -m <btc-chords-large-f16.gguf> -f song.wav`), not as a\n"
                        "  transcribe backend. --chords prints start/end/chord per span.\n"
                        "  NOTE: the BTC weights are CC-BY-NC-SA (non-commercial); downloading them\n"
                        "  needs --accept-license cc-by-nc-sa-4.0.\n");
        return false;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    void shutdown() override {}
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_btc_chords_backend() {
    return std::make_unique<BtcChordsBackend>();
}

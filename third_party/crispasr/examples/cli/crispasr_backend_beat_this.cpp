// crispasr_backend_beat_this.cpp — Beat This! beat-tracking backend shim.
//
// Beat tracking is its OWN task (a beat/downbeat grid out, not text), so it is
// not layered onto transcribe() — see docs/source-separation-surface.md for the
// precedent that settled this for stems, crispasr_backend_crepe.cpp for the
// pitch equivalent and crispasr_backend_btc.cpp for chords. It is exposed
// through the `--beats` dispatcher (examples/cli/crispasr_beats_cli.cpp), which
// resolves the model, decodes the audio to 22.05 kHz mono, runs
// beat_this_track and prints one event per line.
//
// This shim exists only so `beat-this` appears in --list-backends with the
// CAP_BEATS capability and so `--backend beat-this` without `--beats` gives a
// clear redirect instead of trying to force a beat grid through the
// transcribe() contract (which carries text segments and cannot represent it).
//
// It is written together with the four registration sites in
// crispasr_backend.cpp rather than after the fact: btc-chords shipped its
// runtime, CLI dispatcher and session C ABI first and appeared in NEITHER the
// roster nor the arch-detect table, so --list-backends did not know the backend
// existed. That is the multi-surface dispatch trap, and it is cheaper to avoid
// than to notice.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

class BeatThisBackend : public CrispasrBackend {
public:
    const char* name() const override { return "beat-this"; }

    uint32_t capabilities() const override { return CAP_BEATS | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: beat-this is a beat-tracking model — run it with --beats\n"
                        "  (e.g. `crispasr --beats -m <beat-this-f16.gguf> -f song.wav`), not as a\n"
                        "  transcribe backend. --beats prints time/beat-or-downbeat per line.\n");
        return false;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    void shutdown() override {}
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_beat_this_backend() {
    return std::make_unique<BeatThisBackend>();
}

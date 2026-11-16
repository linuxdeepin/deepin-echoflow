// crispasr_backend_tabcnn.cpp — TabCNN guitar-tablature backend shim.
//
// Tablature is its own task: the output is a per-frame, per-string grid of fret
// SCORES, which cannot be expressed as crispasr_segments. So it is not layered
// onto transcribe() — same reasoning as --separate (stems), --pitch (F0),
// --chords (chord timeline) and --beats. Routing happens in the `--tab`
// dispatcher (examples/cli/crispasr_tab_cli.cpp).
//
// This shim exists so `tabcnn` appears in --list-backends with CAP_TAB, and so
// `--backend tabcnn` without `--tab` gives a clear redirect rather than forcing
// a fret grid through the transcribe() contract.
//
// Registering it here is not optional bookkeeping. btc-chords shipped its
// runtime, CLI dispatcher and session C ABI first and appeared in NEITHER the
// roster nor the arch-detect table, so --list-backends did not know it existed
// and the generated docs/feature-matrix.md would have silently dropped any
// hand-written row for it. That is the multi-surface dispatch trap.
//
// ⚠️ What this backend emits is EMISSION SCORES, not a decided tablature. The
// constrained Viterbi/DP that turns them into a playable fingering (one note
// per string, fret range, capo, hand span) belongs to the caller. Anything that
// argmaxes the grid and calls the result "the tab" is ignoring every
// playability constraint.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

class TabCnnBackend : public CrispasrBackend {
public:
    const char* name() const override { return "tabcnn"; }

    uint32_t capabilities() const override { return CAP_TAB | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: tabcnn is a guitar-tablature model — run it with --tab\n"
                        "  (e.g. `crispasr --tab -m <tabcnn-f16.gguf> -f guitar.wav`), not as a\n"
                        "  transcribe backend. --tab prints per-frame string/fret scores.\n"
                        "  NOTE: the weights are CC BY 4.0 (EGSet12,\n"
                        "  https://zenodo.org/records/11406378) and\n"
                        "  and REQUIRE attribution when redistributed.\n");
        return false;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    void shutdown() override {}
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_tabcnn_backend() {
    return std::make_unique<TabCnnBackend>();
}

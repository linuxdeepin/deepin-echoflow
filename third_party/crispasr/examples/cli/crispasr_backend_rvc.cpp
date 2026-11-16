// crispasr_backend_rvc.cpp — RVC voice-conversion backend shim.
//
// Unlike --separate / --pitch / --chords / --piano, RVC gets NO CLI verb, and
// that is deliberate rather than an omission: its input is ContentVec features,
// which CrispASR does not produce. The consumer owns the content encoder, so a
// standalone command line has nothing to feed it. The real surface is the
// session C ABI (crispasr_session_convert*).
//
// This shim exists only so `rvc-svc` appears in --list-backends and so
// `--backend rvc-svc` explains that instead of failing obscurely.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

class RvcBackend : public CrispasrBackend {
public:
    const char* name() const override { return "rvc-svc"; }

    uint32_t capabilities() const override { return CAP_AUTO_DOWNLOAD; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: rvc-svc is a VOICE CONVERSION model and has no CLI verb.\n"
                        "  Its input is ContentVec features, which CrispASR does not produce — the\n"
                        "  caller owns the content encoder. Use the session C ABI instead:\n"
                        "    crispasr_session_convert(s, content, n_frames, f0_hz, speaker_id, NULL, NULL)\n"
                        "  See docs/music-transcription/SVC_RECORD_SHAPES.md for the wire contract.\n");
        return false;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    void shutdown() override {}
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_rvc_backend() {
    return std::make_unique<RvcBackend>();
}

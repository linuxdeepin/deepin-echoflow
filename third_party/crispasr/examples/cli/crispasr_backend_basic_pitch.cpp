// crispasr_backend_basic_pitch.cpp — CLI adapter for Spotify Basic Pitch.
//
// CAP_PIANO is the task marker (audio → note events), shared with
// piano-transcription; --piano is the real surface and dispatches on the
// GGUF's architecture (see crispasr_piano_cli.cpp). Basic Pitch is NOT
// piano-specific — it is polyphonic and instrument-agnostic — but the task and
// the output shape are the same, which is what the capability names.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "basic_pitch.h"

#include <cstdio>
#include <string>
#include <vector>

class BasicPitchBackend : public CrispasrBackend {
public:
    bool init(const whisper_params& p) override {
        auto bp = basic_pitch_default_params();
        bp.n_threads = p.n_threads;
        bp.verbosity = p.no_prints ? 0 : (p.verbose ? 2 : 1);
        bp.use_gpu = p.use_gpu;
        ctx_ = basic_pitch_init_from_file(p.model.c_str(), bp);
        return ctx_ != nullptr;
    }

    void shutdown() override {
        if (ctx_) {
            basic_pitch_free(ctx_);
            ctx_ = nullptr;
        }
    }

    const char* name() const override { return "basic-pitch"; }
    uint32_t capabilities() const override { return CAP_PIANO | CAP_TIMESTAMPS_NATIVE | CAP_AUTO_DOWNLOAD; }
    int input_sample_rate() const override { return 22050; }

    std::vector<crispasr_segment> transcribe(const float* pcm, int n_samples, int64_t /*t0_ms*/,
                                             const whisper_params& /*p*/) override {
        if (!ctx_ || !pcm || n_samples <= 0)
            return {};

        basic_pitch_result result = {};
        if (basic_pitch_transcribe(ctx_, pcm, n_samples, &result) != 0)
            return {};

        static const char* note_names[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
        std::vector<crispasr_segment> segs;
        segs.reserve(result.n_notes);
        for (int i = 0; i < result.n_notes; i++) {
            const auto& ev = result.notes[i];
            char buf[64];
            snprintf(buf, sizeof(buf), "%s%d v=%d", note_names[ev.midi_note % 12], (ev.midi_note / 12) - 1,
                     ev.velocity);
            crispasr_segment seg;
            seg.t0 = (int64_t)(ev.start_time * 1000);
            seg.t1 = (int64_t)(ev.end_time * 1000);
            seg.text = buf;
            segs.push_back(seg);
        }
        basic_pitch_result_free(&result);
        return segs;
    }

private:
    basic_pitch_ctx* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_create_basic_pitch_backend() {
    return std::make_unique<BasicPitchBackend>();
}

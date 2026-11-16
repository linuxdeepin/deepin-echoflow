// crispasr_backend_piano_transcription.cpp — CLI adapter for piano transcription.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "piano_transcription.h"

#include <cstdio>
#include <string>
#include <vector>

class PianoTranscriptionBackend : public CrispasrBackend {
public:
    bool init(const whisper_params& p) override {
        auto cp = piano_transcription_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : (p.verbose ? 2 : 1);
        cp.use_gpu = p.use_gpu;
        ctx_ = piano_transcription_init_from_file(p.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    void shutdown() override {
        if (ctx_) {
            piano_transcription_free(ctx_);
            ctx_ = nullptr;
        }
    }

    const char* name() const override { return "piano-transcription"; }
    // CAP_PIANO marks the task; --piano is the real surface (note events).
    // CAP_TIMESTAMPS_NATIVE stays because the legacy transcribe() path still
    // renders notes as timestamped segments for callers that used it.
    uint32_t capabilities() const override { return CAP_PIANO | CAP_TIMESTAMPS_NATIVE | CAP_AUTO_DOWNLOAD; }
    int input_sample_rate() const override { return 16000; }

    std::vector<crispasr_segment> transcribe(const float* pcm, int n_samples, int64_t /*t0_ms*/,
                                             const whisper_params& /*p*/) override {
        if (!ctx_ || !pcm || n_samples <= 0)
            return {};

        piano_transcription_result result = {};
        int rc = piano_transcription_transcribe(ctx_, pcm, n_samples, &result);
        if (rc != 0)
            return {};

        // Convert note events to segments for CLI display.
        // Each note becomes a segment with onset/offset time and
        // text like "C4 v=80" (note name + velocity).
        std::vector<crispasr_segment> segs;
        segs.reserve(result.n_notes);

        static const char* note_names[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

        for (int i = 0; i < result.n_notes; i++) {
            auto& ev = result.note_events[i];
            int midi = ev.midi_note;
            int octave = (midi / 12) - 1;
            const char* note = note_names[midi % 12];

            char buf[64];
            snprintf(buf, sizeof(buf), "%s%d v=%d", note, octave, ev.velocity);

            crispasr_segment seg;
            seg.t0 = (int64_t)(ev.onset_time * 1000);
            seg.t1 = (int64_t)(ev.offset_time * 1000);
            seg.text = buf;
            segs.push_back(seg);
        }

        piano_transcription_result_free(&result);
        return segs;
    }

private:
    piano_transcription_ctx* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_create_piano_transcription_backend() {
    return std::make_unique<PianoTranscriptionBackend>();
}

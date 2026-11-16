// crispasr_backend_higgs_stt.cpp — adapter for bosonai/higgs-audio-v3-stt.
//
// Whisper-large-v3 encoder + depthwise-temporal-conv MLP projector +
// Qwen3-1.7B decoder. The full mel -> encoder -> ChatML-prompt splice ->
// KV greedy-decode pipeline lives in higgs_stt_transcribe(); this adapter
// just wraps it as a single transcript segment.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "higgs_stt.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

class HiggsSttBackend : public CrispasrBackend {
public:
    HiggsSttBackend() = default;
    ~HiggsSttBackend() override { HiggsSttBackend::shutdown(); }

    const char* name() const override { return "higgs-stt"; }

    uint32_t capabilities() const override {
        // CAP_UNBOUNDED_INPUT | CAP_INTERNAL_CHUNKING: higgs-audio handles long
        // audio internally the same way the upstream blueprint does — split the
        // waveform into chunk_size_seconds (4 s) chunks, encode each through the
        // Whisper tower + projector, concatenate the audio embeds, and run ONE
        // LLM AR decode over the whole sequence (higgs_stt_transcribe). The
        // crispasr_run.cpp auto-chunk gate must NOT fire: a CLI-level window
        // split would give each window a fresh LLM context (cold-starting the
        // decoder at every boundary) and produce overlap-duplicated text —
        // diverging from the blueprint's single-pass whole-clip decode.
        return CAP_AUTO_DOWNLOAD | CAP_FLASH_ATTN | CAP_UNBOUNDED_INPUT | CAP_INTERNAL_CHUNKING | CAP_BEAM_SEARCH;
    }

    bool init(const whisper_params& p) override {
        auto cp = higgs_stt_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = higgs_stt_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[higgs-stt]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_ || !samples || n_samples <= 0)
            return out;
        // --ask overrides the task prompt; otherwise -l <lang> injects a
        // language hint. Empty restores the default transcription instruction.
        if (!params.ask.empty()) {
            higgs_stt_set_ask(ctx_, params.ask.c_str());
        } else if (!params.language.empty() && params.language != "auto") {
            const std::string instr = "Transcribe the speech in " + crispasr_iso_to_english_lang(params.language) +
                                      ". Output only the spoken words in lowercase with no punctuation.";
            higgs_stt_set_ask(ctx_, instr.c_str());
        } else {
            higgs_stt_set_ask(ctx_, nullptr);
        }
        higgs_stt_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);
        // #292: forward --max-new-tokens only when explicit; 0 keeps the backend default.
        higgs_stt_set_max_new_tokens(ctx_, params.max_new_tokens_explicit ? params.max_new_tokens : 0);
        char* text = higgs_stt_transcribe(ctx_, samples, n_samples);
        crispasr_segment seg;
        seg.text = text ? text : "";
        free(text);
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)n_samples * 100 / 16000;
        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            higgs_stt_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    higgs_stt_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_higgs_stt_backend() {
    return std::unique_ptr<CrispasrBackend>(new HiggsSttBackend());
}

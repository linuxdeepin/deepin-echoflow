// crispasr_backend_voxtral_tts.cpp — Voxtral-4B-TTS backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "voxtral_tts.h"
#include "whisper_params.h"

#include <cstdlib>
#include <string>
#include <vector>

class VoxtralTtsBackend : public CrispasrBackend {
public:
    VoxtralTtsBackend() = default;

    const char* name() const override { return "voxtral-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        return {};
    }

    bool init(const whisper_params& params) override {
        voxtral_tts_context_params cp = voxtral_tts_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.temperature = params.temperature;
        ctx_ = voxtral_tts_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_)
            return {};

        // Seed the flow-matching noise for reproducible acoustic sampling (--seed).
        if (params.seed != 0)
            voxtral_tts_set_seed(ctx_, (uint64_t)params.seed);

        const char* voice = params.tts_voice.empty() ? nullptr : params.tts_voice.c_str();
        int n_samples = 0;
        float* pcm = voxtral_tts_synthesize(ctx_, text.c_str(), voice, &n_samples);
        if (!pcm || n_samples <= 0)
            return {};

        std::vector<float> out(pcm, pcm + n_samples);
        voxtral_tts_pcm_free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return 24000; }

    void shutdown() override {
        if (ctx_) {
            voxtral_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~VoxtralTtsBackend() override { VoxtralTtsBackend::shutdown(); }

private:
    voxtral_tts_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_voxtral_tts_backend() {
    return std::make_unique<VoxtralTtsBackend>();
}

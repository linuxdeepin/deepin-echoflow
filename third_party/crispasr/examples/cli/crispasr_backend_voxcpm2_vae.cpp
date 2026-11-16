// VoxCPM2 AudioVAE speech upscaler: 16 kHz mono PCM -> 48 kHz mono PCM.
#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "voxcpm2_vae.h"
#include "whisper_params.h"

#include <cstdio>

namespace {

class VoxCPM2VAEBackend final : public CrispasrBackend {
public:
    const char* name() const override { return "voxcpm2-vae"; }
    uint32_t capabilities() const override { return CAP_S2S; }
    int tts_sample_rate() const override { return 48000; }

    bool init(const whisper_params& p) override {
        voxcpm2_vae_context_params cp = voxcpm2_vae_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = voxcpm2_vae_init_from_file(p.model.c_str(), cp);
        if (!ctx_)
            std::fprintf(stderr, "crispasr[voxcpm2-vae]: failed to load '%s'\n", p.model.c_str());
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    std::vector<float> speech_to_speech(const float* samples, int n_samples, std::string* out_text,
                                        const whisper_params&) override {
        if (out_text)
            out_text->clear();
        int n = 0;
        float* pcm = voxcpm2_vae_upscale(ctx_, samples, n_samples, &n);
        if (!pcm || n <= 0) {
            voxcpm2_vae_pcm_free(pcm);
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        voxcpm2_vae_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            voxcpm2_vae_free(ctx_);
            ctx_ = nullptr;
        }
    }
    ~VoxCPM2VAEBackend() override { VoxCPM2VAEBackend::shutdown(); }

private:
    voxcpm2_vae_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_voxcpm2_vae_backend() {
    return std::make_unique<VoxCPM2VAEBackend>();
}

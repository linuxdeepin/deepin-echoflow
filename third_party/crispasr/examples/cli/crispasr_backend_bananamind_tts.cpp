// crispasr_backend_bananamind_tts.cpp -- adapter for BananaMind-TTS-V2.1.
//
// Tacotron-lite + HiFi-GAN TTS, character-based, en-us / de-de.
// 22050 Hz mono output.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "bananamind_tts.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

class BananaMindTTSBackend : public CrispasrBackend {
public:
    ~BananaMindTTSBackend() override { BananaMindTTSBackend::shutdown(); }

    const char* name() const override { return "bananamind-tts"; }

    uint32_t capabilities() const override { return CAP_TTS; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[bananamind-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        bananamind_tts_params bp = bananamind_tts_default_params();
        bp.n_threads = p.n_threads;
        bp.use_gpu = p.use_gpu;
        bp.verbosity = p.no_prints ? 0 : 1;

        ctx_ = bananamind_tts_init_from_file(p.model.c_str(), bp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[bananamind-tts]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& /*p*/) override {
        if (!ctx_)
            return {};

        float* pcm = nullptr;
        int sr = 0;
        int n = bananamind_tts_synthesize(ctx_, text.c_str(), &pcm, &sr);
        if (n <= 0 || !pcm)
            return {};

        std::vector<float> result(pcm, pcm + n);
        free(pcm);
        return result;
    }

    int tts_sample_rate() const override { return ctx_ ? bananamind_tts_sample_rate(ctx_) : 22050; }

    void shutdown() override {
        if (ctx_) {
            bananamind_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    bananamind_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_bananamind_tts_backend() {
    return std::make_unique<BananaMindTTSBackend>();
}

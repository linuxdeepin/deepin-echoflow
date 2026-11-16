// crispasr_backend_confucius4_tts.cpp — CLI adapter for Confucius4-TTS (§377).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_voice_provenance.h"
#include "whisper_params.h"
#include "confucius4_tts.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

class Confucius4TtsBackend : public CrispasrBackend {
public:
    const char* name() const override { return "confucius4-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_VOICE_CLONING | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE; }

    bool init(const whisper_params& p) override {
        auto cp = confucius4_tts_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        // whisper_params.temperature defaults to 0.0 (greedy), which is the right
        // default for ASR but not for this T2S stage -- the reference samples at
        // 0.8 and a greedy decode degenerates into a repeat loop that runs to the
        // 1520-token cap.  Only override the backend default when the user asked.
        if (p.temperature > 0.0f)
            cp.temperature = p.temperature;
        cp.seed = p.seed;
        if (p.tts_steps > 0)
            cp.ode_steps = p.tts_steps;

        ctx_ = confucius4_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[confucius4-tts]: failed to load T2S model '%s'\n", p.model.c_str());
            return false;
        }

        // Load the S2A companion model (--codec-model or auto-resolved sibling —
        // the registry downloads confucius4-tts-s2a-q4_k.gguf next to the T2S)
        std::string s2a_path = p.tts_codec_model;
        if (s2a_path.empty()) {
            std::string dir = p.model.substr(0, p.model.find_last_of("/\\") + 1);
            for (const char* pat :
                 {"confucius4-tts-s2a-q4_k.gguf", "confucius4-tts-s2a-q8_0.gguf", "confucius4-tts-s2a-f16.gguf"}) {
                std::string path = dir + pat;
                FILE* f = fopen(path.c_str(), "rb");
                if (f) {
                    fclose(f);
                    s2a_path = path;
                    break;
                }
            }
        }
        if (!s2a_path.empty()) {
            if (confucius4_tts_set_s2a_path(ctx_, s2a_path.c_str()) != 0) {
                fprintf(stderr, "crispasr[confucius4-tts]: failed to load S2A model '%s'\n", s2a_path.c_str());
                // Non-fatal: T2S works without S2A (generates codes but no audio)
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[confucius4-tts]: WARNING: no S2A model (--codec-model) — no audio\n");
        }

        // Auto-discover BigVGAN vocoder: look for *bigvgan* sibling next to T2S or S2A model
        auto try_vocoder = [&](const std::string& base) {
            if (base.empty())
                return;
            std::string dir = base.substr(0, base.find_last_of("/\\") + 1);
            for (const char* pat : {"confucius4-tts-bigvgan-22k-f16.gguf", "confucius4-tts-bigvgan-22k-q8_0.gguf"}) {
                std::string path = dir + pat;
                FILE* f = fopen(path.c_str(), "rb");
                if (f) {
                    fclose(f);
                    if (confucius4_tts_set_vocoder_path(ctx_, path.c_str()) == 0)
                        return;
                }
            }
        };
        try_vocoder(s2a_path);
        try_vocoder(p.model);

        // w2v-BERT encoder (optional sibling): enables the fully native T2S
        // condition_emb for --voice. Absent → S2A-side conditioning only
        // unless CRISPASR_CONFUCIUS4_COND_DIR supplies the features.
        {
            std::string dir = p.model.substr(0, p.model.find_last_of("/\\") + 1);
            for (const char* pat : {"confucius4-tts-w2v-f16.gguf", "confucius4-tts-w2v-q8_0.gguf"}) {
                std::string path = dir + pat;
                FILE* f = fopen(path.c_str(), "rb");
                if (f) {
                    fclose(f);
                    if (confucius4_tts_set_w2v_path(ctx_, path.c_str()) == 0)
                        break;
                }
            }
        }

        // Voice cloning (--voice ref.wav): native CAMPPlus style + prompt mel
        // (needs the campplus.* bake in the S2A GGUF). The T2S condition_emb
        // additionally needs w2v-BERT features via CRISPASR_CONFUCIUS4_COND_DIR
        // until that model is ported natively.
        if (!p.tts_voice.empty()) {
            // Bare voice names resolve against --voice-dir: the server passes
            // `voice` through verbatim by design and documents that the adapter
            // owns the interpretation, so an unresolved adapter treats the literal
            // name as a path and cloning silently fails over HTTP (#384). Shared
            // resolver, so this agrees with the server's provenance gate about
            // which file a name means; a name that is not a file there is
            // returned unchanged.
            const std::string voice = p.tts_voice_dir.empty()
                                          ? p.tts_voice
                                          : crispasr_voice::resolve_voice_path(p.tts_voice, p.tts_voice_dir);
            if (confucius4_tts_set_voice_path(ctx_, voice.c_str()) != 0) {
                fprintf(stderr, "crispasr[confucius4-tts]: WARNING: failed to apply voice prompt '%s'\n",
                        voice.c_str());
            } else if (!p.no_prints) {
                fprintf(stderr, "crispasr[confucius4-tts]: voice = '%s'\n", voice.c_str());
            }
        }

        return true;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_)
            return {};

        // "-l auto" is the CLI default; there is no LID for TTS input text, and
        // "auto" must not leak into the LANGUAGE_TOKEN_MAP prompt.
        const std::string lang = (params.language.empty() || params.language == "auto") ? "en" : params.language;
        int n_samples = 0;
        float* pcm = confucius4_tts_synthesize(ctx_, text.c_str(), lang.c_str(), &n_samples);
        if (!pcm || n_samples <= 0)
            return {};

        std::vector<float> out(pcm, pcm + n_samples);
        confucius4_tts_pcm_free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return confucius4_tts_sample_rate(ctx_); }

    void shutdown() override {
        if (ctx_) {
            confucius4_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~Confucius4TtsBackend() override { Confucius4TtsBackend::shutdown(); }

private:
    confucius4_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_confucius4_tts_backend() {
    return std::make_unique<Confucius4TtsBackend>();
}

// crispasr_backend_mini_omni2.cpp — adapter for Mini-Omni2.
//
// Whisper-small encoder + whisperMLP adapter + Qwen2-0.5B LLM.
// ASR (audio→text), TTS (text→audio via SNAC), S2S (audio→audio).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "mini_omni2.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

static bool file_exists(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f)
        return false;
    fclose(f);
    return true;
}

static std::string discover_snac(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "snac-24khz.gguf",
        "snac_24khz.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

class MiniOmni2Backend : public CrispasrBackend {
public:
    MiniOmni2Backend() = default;
    ~MiniOmni2Backend() override { MiniOmni2Backend::shutdown(); }

    const char* name() const override { return "mini-omni2"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_S2S | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE; }

    bool init(const whisper_params& p) override {
        mini_omni2_context_params mp = mini_omni2_context_default_params();
        mp.n_threads = p.n_threads;
        mp.verbosity = p.no_prints ? 0 : 1;
        mp.use_gpu = crispasr_backend_should_use_gpu(p);
        if (p.temperature > 0.0f)
            mp.temperature = p.temperature;

        ctx_ = mini_omni2_init_from_file(p.model.c_str(), mp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[mini-omni2]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // SNAC codec for TTS/S2S
        std::string codec_path = p.tts_codec_model;
        if (!codec_path.empty() && codec_path != "auto" && codec_path != "default") {
            codec_path = crispasr_resolve_model_cli(codec_path, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                                    p.tts_codec_quant);
        } else {
            codec_path.clear();
        }
        if (codec_path.empty())
            codec_path = discover_snac(p.model);
        if (codec_path.empty()) {
            // Try model registry companion
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                codec_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                        p.auto_download, p.tts_codec_quant);
            }
        }
        if (!codec_path.empty()) {
            if (mini_omni2_load_snac(ctx_, codec_path.c_str())) {
                snac_loaded_ = true;
                if (!p.no_prints)
                    fprintf(stderr, "crispasr[mini-omni2]: SNAC codec loaded from '%s'\n", codec_path.c_str());
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[mini-omni2]: no SNAC codec found. "
                            "Pass --codec-model snac-24khz.gguf for TTS/S2S.\n");
        }

        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // #292: forward --max-new-tokens only when explicit; 0 keeps the backend default.
        mini_omni2_set_max_new_tokens(ctx_, params.max_new_tokens_explicit ? params.max_new_tokens : 0);
        char* text = mini_omni2_transcribe(ctx_, samples, n_samples);
        if (!text)
            return out;

        crispasr_segment seg;
        seg.text = text;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples * 100.0 / 16000.0);
        free(text);

        out.push_back(std::move(seg));
        return out;
    }

    std::vector<float> speech_to_speech(const float* samples, int n_samples, std::string* out_text,
                                        const whisper_params& /*params*/) override {
        if (!ctx_)
            return {};
        if (!snac_loaded_) {
            fprintf(stderr, "crispasr[mini-omni2]: SNAC codec required for S2S. "
                            "Pass --codec-model snac-24khz.gguf\n");
            return {};
        }
        char* text = nullptr;
        int out_n = 0;
        float* pcm = mini_omni2_speech_to_speech(ctx_, samples, n_samples, &text, &out_n);
        if (text && out_text)
            *out_text = text;
        if (text)
            free(text);
        if (!pcm || out_n <= 0)
            return {};
        std::vector<float> out(pcm, pcm + out_n);
        free(pcm);
        return out;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};
        if (!snac_loaded_) {
            fprintf(stderr, "crispasr[mini-omni2]: SNAC codec required for TTS. "
                            "Pass --codec-model snac-24khz.gguf\n");
            return {};
        }

        int n_samples = 0;
        float* pcm = mini_omni2_synthesize(ctx_, text.c_str(), &n_samples);
        if (!pcm || n_samples <= 0)
            return {};

        std::vector<float> result(pcm, pcm + n_samples);
        free(pcm);
        return result;
    }

    void shutdown() override {
        if (ctx_) {
            mini_omni2_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    mini_omni2_context* ctx_ = nullptr;
    bool snac_loaded_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_mini_omni2_backend() {
    return std::make_unique<MiniOmni2Backend>();
}

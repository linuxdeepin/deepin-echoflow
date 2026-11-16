// crispasr_backend_irodori_tts.cpp — adapter for Aratako/Irodori-TTS.
//
// Flow-matching RF-DiT TTS with zero-shot voice cloning via DAC-VAE latents.
// Japanese-focused, 48 kHz output. Requires DAC-VAE decoder companion GGUF.
//
// Usage:
//   crispasr --backend irodori-tts -m auto --tts "こんにちは" -o out.wav

#include "crispasr_backend.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "crispasr_voice_provenance.h"
#include "whisper_params.h"

#include "core/wav_reader.h"

#include "irodori_tts.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}


// Look for dacvae GGUF next to the model.
static std::string discover_dacvae(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto pos = p.find_last_of("/\\");
        return (pos != std::string::npos) ? p.substr(0, pos) : ".";
    };
    std::string dir = dir_of(model_path);
    for (const char* name : {"dacvae-ja-32dim-f16.gguf", "dacvae-ja-32dim-q8_0.gguf"}) {
        std::string candidate = dir + "/" + name;
        if (file_exists(candidate))
            return candidate;
    }
    return {};
}

class IrodoriTTSBackend : public CrispasrBackend {
public:
    IrodoriTTSBackend() = default;
    ~IrodoriTTSBackend() override { IrodoriTTSBackend::shutdown(); }

    const char* name() const override { return "irodori-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_VOICE_CLONING | CAP_AUTO_DOWNLOAD; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        return {}; // TTS-only backend
    }

    bool init(const whisper_params& p) override {
        irodori_tts_params fp = irodori_tts_default_params();
        fp.n_threads = p.n_threads;
        fp.verbosity = p.no_prints ? 0 : 1;
        fp.seed = p.seed;
        fp.use_gpu = p.use_gpu;

        ctx_ = irodori_tts_init_from_file(p.model.c_str(), fp);
        if (!ctx_) {
            std::fprintf(stderr, "crispasr[irodori-tts]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // ── Diffusion knobs (#241) ──
        // The generic --tts-steps / --tts-cfg-scale flags map to the flow-matching
        // ODE step count and the (primary) text CFG scale. Irodori also carries a
        // separate speaker CFG scale that the single --tts-cfg-scale can't reach, so
        // expose all three via env overrides too. The setters mutate the context and
        // synthesize() reads ode_steps/cfg_scale live, so they take effect on the
        // next call regardless of ordering relative to codec loading below.
        if (p.tts_num_steps > 0)
            irodori_tts_set_ode_steps(ctx_, p.tts_num_steps);
        if (p.tts_cfg_scale >= 0.0f)
            irodori_tts_set_cfg_scale_text(ctx_, p.tts_cfg_scale);
        if (const char* e = std::getenv("CRISPASR_IRODORI_ODE_STEPS"))
            irodori_tts_set_ode_steps(ctx_, std::atoi(e));
        if (const char* e = std::getenv("CRISPASR_IRODORI_CFG_TEXT"))
            irodori_tts_set_cfg_scale_text(ctx_, (float)std::atof(e));
        if (const char* e = std::getenv("CRISPASR_IRODORI_CFG_SPEAKER"))
            irodori_tts_set_cfg_scale_speaker(ctx_, (float)std::atof(e));
        if (const char* e = std::getenv("CRISPASR_IRODORI_CFG_CAPTION"))
            irodori_tts_set_cfg_scale_caption(ctx_, (float)std::atof(e));

        // ── Codec companion loading (3-tier: --codec-model → sibling → registry auto-download) ──
        std::string codec_path = p.tts_codec_model;
        if (!codec_path.empty() && codec_path != "auto" && codec_path != "default") {
            codec_path = crispasr_resolve_model_cli(codec_path, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                                    p.tts_codec_quant);
        } else {
            codec_path.clear();
        }
        if (codec_path.empty()) {
            codec_path = discover_dacvae(p.model);
        }
        if (codec_path.empty()) {
            CrispasrRegistryEntry entry;
            std::string quant;
            if (crispasr_registry_lookup(p.backend, entry, quant) && !entry.companion_filename.empty()) {
                codec_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                        p.auto_download, p.tts_codec_quant);
            }
        }
        if (!codec_path.empty()) {
            irodori_tts_set_codec_path(ctx_, codec_path.c_str());
            if (!p.no_prints) {
                std::fprintf(stderr, "crispasr[irodori-tts]: codec path = '%s'\n", codec_path.c_str());
            }
        } else if (!p.no_prints) {
            std::fprintf(stderr, "crispasr[irodori-tts]: no DAC-VAE codec found. Pass --codec-model PATH or place "
                                 "dacvae-ja-32dim-f16.gguf next to the model. (output will be silent)\n");
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& p) override {
        if (!ctx_ || text.empty())
            return {};

        // Caption text (VoiceDesign): --instruct flag or env-gated.
        if (!p.tts_instruct.empty())
            irodori_tts_set_caption(ctx_, p.tts_instruct.c_str());
        else if (const char* cap = std::getenv("CRISPASR_IRODORI_CAPTION"))
            irodori_tts_set_caption(ctx_, cap);

        // Per-call voice gating. The reference lives on the context (stateful
        // set_reference), so we sync it to params.tts_voice on every call:
        //   - voice set    → encode the reference for zero-shot cloning
        //   - voice empty   → clear it, so the neutral-voice disclaimer synth
        //                    (which passes tts_voice="") is NOT cloned
        // The DAC-VAE encode is cached by path so chunked long text doesn't
        // re-encode the reference for every chunk.
        apply_reference(p);

        float* pcm = nullptr;
        int sr = 0;
        int n = irodori_tts_synthesize(ctx_, text.c_str(), &pcm, &sr);
        if (n <= 0 || !pcm)
            return {};

        std::vector<float> out(pcm, pcm + n);
        std::free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return ctx_ ? irodori_tts_sample_rate(ctx_) : 48000; }

    void shutdown() override {
        if (ctx_) {
            irodori_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    // Sync the context's cloning reference to params.tts_voice. Caches the
    // last-applied voice path so repeated calls (e.g. per chunk) don't re-load
    // the same reference. The expensive encode is cached in the runtime
    // (irodori_tts_set_reference, content-addressed), so it's skipped across
    // runs and shared with every other consumer (C ABI, server, wrappers).
    void apply_reference(const whisper_params& p) {
        // Resolve a BARE voice name against --voice-dir before anything else.
        // The server passes `voice` through verbatim by design and documents
        // that the adapter owns the interpretation, so an unresolved adapter
        // gets the literal name as a path: /v1/audio/speech with
        // {"voice": "reference"} listed fine under /v1/voices but failed to
        // load, while the CLI worked because it was handed a full path (#384).
        // Same shared resolver the server's provenance gate uses, so the two
        // surfaces agree on WHICH file a name means.
        const std::string voice = p.tts_voice.empty() || p.tts_voice_dir.empty()
                                      ? p.tts_voice
                                      : crispasr_voice::resolve_voice_path(p.tts_voice, p.tts_voice_dir);
        // Cache on the RESOLVED path: a name and its path are the same
        // reference, and re-encoding per chunk is the expensive part.
        if (voice == ref_path_)
            return; // already in the desired state (including both empty)
        ref_path_ = voice;

        if (voice.empty()) {
            irodori_tts_clear_reference(ctx_);
            return;
        }

        std::vector<float> pcm;
        int sr = 0;
        if (!crispasr::core::read_wav_mono_pcm16(voice, pcm, sr) || pcm.empty() || sr <= 0) {
            std::fprintf(stderr, "crispasr[irodori-tts]: failed to load reference audio '%s' (cloning disabled)\n",
                         voice.c_str());
            irodori_tts_clear_reference(ctx_);
            // Keep ref_path_ set so we don't re-warn for every chunk of the
            // same request; a fixed voice path won't become readable mid-run.
            return;
        }
        // irodori_tts_set_reference resamples to 48 kHz, loudness-normalizes,
        // runs the DAC-VAE encoder (skipped on a cache hit), and stores the
        // speaker latent. It fails (-1) only when the loaded codec GGUF has no
        // encoder tensors — set_reference prints the actionable message, so
        // just fall back quietly.
        if (irodori_tts_set_reference(ctx_, pcm.data(), (int)pcm.size(), sr) != 0) {
            irodori_tts_clear_reference(ctx_);
            // Keep ref_path_ set: don't retry (and re-warn) for every chunk.
            return;
        }
        if (!p.no_prints) {
            std::fprintf(stderr, "crispasr[irodori-tts]: reference voice '%s' (%d samples, %d Hz)\n", voice.c_str(),
                         (int)pcm.size(), sr);
        }
    }

    irodori_tts_context* ctx_ = nullptr;
    std::string ref_path_; // last voice path applied to ctx_ ("" = cleared)
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_irodori_tts_backend() {
    return std::make_unique<IrodoriTTSBackend>();
}

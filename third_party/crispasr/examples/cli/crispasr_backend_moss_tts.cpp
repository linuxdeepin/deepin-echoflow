// crispasr_backend_moss_tts.cpp — CLI adapter for MOSS-TTS-v1.5 (MossTTSDelay).
//
// Two-GGUF runtime: the Qwen3-8B backbone (from --model) plus a companion
// transformer RVQ codec (from --codec-model, or a sibling "<stem>-codec.gguf",
// or the registry companion). Voice cloning: `--voice ref.wav` encodes the
// reference through the codec encoder and clones that speaker.

#include "crispasr_backend.h"
#include "crispasr_speech_window.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "crispasr_voice_provenance.h"
#include "whisper_params.h"

#include "core/audio_resample.h"
#include "core/wav_reader.h"
#include "moss_tts.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static std::string dir_of(const std::string& p) {
    auto sep = p.find_last_of("/\\");
    return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
}

// Look for a sibling codec GGUF next to the backbone. The converter writes
// "<stem>-codec.gguf"; the auto-download path drops both into the same dir.
static std::string discover_codec(const std::string& model_path) {
    // Derived name: insert "-codec" before the ".gguf" suffix.
    const std::string ext = ".gguf";
    if (model_path.size() > ext.size() && model_path.compare(model_path.size() - ext.size(), ext.size(), ext) == 0) {
        std::string derived = model_path.substr(0, model_path.size() - ext.size()) + "-codec.gguf";
        if (file_exists(derived))
            return derived;
    }
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "moss-tts-codec.gguf",
        "moss-tts-v1.5-codec.gguf",
        "moss-audio-tokenizer.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

class MossTtsBackend : public CrispasrBackend {
public:
    MossTtsBackend() = default;
    ~MossTtsBackend() override { MossTtsBackend::shutdown(); }

    const char* name() const override { return "moss-tts"; }

    uint32_t capabilities() const override {
        return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN | CAP_VOICE_CLONING;
    }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[moss-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        moss_tts_context_params cp = moss_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.flash_attn = p.flash_attn;
        ctx_ = moss_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[moss-tts]: failed to load backbone '%s'\n", p.model.c_str());
            return false;
        }

        // Resolve the companion codec GGUF.
        std::string codec_path = p.tts_codec_model;
        if (!codec_path.empty() && codec_path != "auto" && codec_path != "default") {
            codec_path = crispasr_resolve_model_cli(codec_path, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                                    p.tts_codec_quant);
        } else {
            codec_path.clear();
        }
        if (codec_path.empty())
            codec_path = discover_codec(p.model);
        if (codec_path.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                codec_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                        p.auto_download, p.tts_codec_quant);
            }
        }
        if (codec_path.empty()) {
            fprintf(stderr, "crispasr[moss-tts]: no codec model found. Pass --codec-model PATH or place "
                            "<stem>-codec.gguf next to the backbone.\n");
            return false;
        }
        if (!moss_tts_set_codec_path(ctx_, codec_path.c_str())) {
            fprintf(stderr, "crispasr[moss-tts]: failed to load codec '%s'\n", codec_path.c_str());
            return false;
        }
        if (!p.no_prints)
            fprintf(stderr, "crispasr[moss-tts]: codec loaded from '%s'\n", codec_path.c_str());
        return true;
    }

    // Load + encode the reference WAV (voice cloning), re-loading only when the
    // path changes so single-shot and server callers pay the encode once.
    void prepare_voice(const whisper_params& params) {
        // Bare voice names resolve against --voice-dir: the server passes
        // `voice` verbatim by design and the adapter owns the
        // interpretation, so an unresolved adapter treats the literal name
        // as a path and cloning silently fails over HTTP (#384). Shared
        // resolver, so this agrees with the provenance gate; a name that
        // is not a file there is returned unchanged.
        const std::string voice = params.tts_voice.empty() || params.tts_voice_dir.empty()
                                      ? params.tts_voice
                                      : crispasr_voice::resolve_voice_path(params.tts_voice, params.tts_voice_dir);
        if (voice == last_voice_)
            return;
        last_voice_ = voice;
        if (voice.empty()) {
            moss_tts_set_reference_wav(ctx_, nullptr, 0); // plain TTS
            return;
        }
        std::vector<float> ref;
        int sr = 0;
        if (!crispasr::core::read_wav_mono_pcm16(voice, ref, sr) || ref.empty()) {
            fprintf(stderr, "crispasr[moss-tts]: failed to load reference audio '%s'\n", params.tts_voice.c_str());
            last_voice_.clear();
            return;
        }
        const int target = moss_tts_sampling_rate(ctx_);
        if (sr != target && sr > 0) {
            ref = core_audio::resample_polyphase(ref.data(), (int)ref.size(), sr, target);
            if (!params.no_prints)
                fprintf(stderr, "crispasr[moss-tts]: resampled reference %d->%d Hz\n", sr, target);
        }
        if (!moss_tts_set_reference_wav(ctx_, ref.data(), (int)ref.size())) {
            fprintf(stderr, "crispasr[moss-tts]: reference encode failed for '%s'\n", params.tts_voice.c_str());
            last_voice_.clear();
        } else if (!params.no_prints) {
            fprintf(stderr, "crispasr[moss-tts]: cloning voice from '%s'\n", params.tts_voice.c_str());
        }
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};
        prepare_voice(params);
        moss_tts_synth_params sp = moss_tts_synth_default_params();
        sp.seed = params.seed;
        // A single --temperature knob scales both text and audio sampling when set.
        if (params.temperature > 0.0f) {
            sp.text_temperature = params.temperature;
            sp.audio_temperature = params.temperature;
        }
        // Only when the user EXPLICITLY set --max-new-tokens: the whisper_params
        // default is 512, but moss-tts's own default is 4096, so `> 0` (always
        // true) would silently shrink every synthesis to 512 frames (~41 s). Same
        // trap as #292 — gate on the explicit flag, not the value.
        if (params.max_new_tokens_explicit)
            sp.max_new_tokens = params.max_new_tokens;
        // Duration control: max_speech_tokens maps to MOSS-TTS max_audio_frames.
        // 1s ≈ 12.5 tokens, so max_audio_frames is the token-level AR cap.
        if (params.tts_max_speech_tokens >= 0)
            sp.max_audio_frames = params.tts_max_speech_tokens;
        // min_speech_tokens maps to min_audio_frames: when both min and max are
        // set to the same value, the model is forced to generate exactly that
        // many frames — exact-duration synthesis (lip-sync / game dubbing).
        if (params.tts_min_speech_tokens >= 0) {
            sp.min_audio_frames = params.tts_min_speech_tokens;
            // #330: the floor is not the only bound — the decode loop stops at
            // max_new_frames regardless, so an unreachable floor silently
            // yields SHORTER audio than asked for. For a feature whose whole
            // point is exact duration, say so rather than let it pass.
            crispasr_speech_window::Window win;
            win.min_frames = sp.min_audio_frames;
            win.max_frames = sp.max_audio_frames > 0 ? sp.max_audio_frames : -1;
            // This backend bounds its loop on max_new_tokens (moss_tts.cpp:955),
            // not max_new_frames. One AR step emits one delayed frame, so the
            // token bound is the frame bound to within the delay offset — close
            // enough to warn on, and the warning is advisory either way.
            win.frame_cap = sp.max_new_tokens > 0 ? sp.max_new_tokens : crispasr_speech_window::kDefaultFrameCap;
            const std::string warn = crispasr_speech_window::diagnose(win);
            if (!warn.empty())
                fprintf(stderr, "crispasr[moss-tts]: warning: %s\n", warn.c_str());
        }
        if (params.tts_top_p >= 0.0f)
            sp.audio_top_p = params.tts_top_p;
        if (params.tts_top_k >= 0)
            sp.audio_top_k = params.tts_top_k;
        if (params.tts_repetition_penalty >= 0.0f)
            sp.audio_repetition_penalty = params.tts_repetition_penalty;
        std::string lang;
        if (!params.language.empty() && params.language != "auto") {
            lang = crispasr_iso_to_english_lang(params.language);
            sp.language = lang.c_str();
        }
        if (!params.tts_instruct.empty())
            sp.instruction = params.tts_instruct.c_str();

        int n = 0;
        float* pcm = moss_tts_synthesize(ctx_, text.c_str(), &sp, &n);
        if (!pcm || n <= 0) {
            free(pcm);
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return ctx_ ? moss_tts_sampling_rate(ctx_) : 24000; }

    void shutdown() override {
        if (ctx_) {
            moss_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    moss_tts_context* ctx_ = nullptr;
    std::string last_voice_; // cache key for the loaded reference voice
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_moss_tts_backend() {
    return std::unique_ptr<CrispasrBackend>(new MossTtsBackend());
}

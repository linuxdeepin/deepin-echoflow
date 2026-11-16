// crispasr_backend_cosyvoice3.cpp — adapter for FunAudioLLM/Fun-CosyVoice3-0.5B-2512 TTS.
//
// Four-GGUF runtime: LLM (-m), flow (sibling), CAMPPlus (sibling), HiFT
// (sibling), plus a voices.gguf carrying baked voice-clone bundles. The
// flow path can be overridden via --codec-model; CAMPPlus / HiFT / voices
// auto-discover as siblings of the LLM (or via the
// COSYVOICE3_CAMPPLUS_PATH / COSYVOICE3_HIFT_PATH /
// COSYVOICE3_VOICES_PATH env vars).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_tts_ref_text.h" // #334 auto-transcribe a --voice WAV
#include "whisper_params.h"

#include "cosyvoice3_tts.h"

#include "core/audio_resample.h"
#include "core/wav_reader.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>
#include "core/crispasr_env.h"

namespace {

bool ends_with_ci(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size())
        return false;
    for (size_t i = 0; i < suffix.size(); i++) {
        char a = (char)std::tolower((unsigned char)s[s.size() - suffix.size() + i]);
        char b = (char)std::tolower((unsigned char)suffix[i]);
        if (a != b)
            return false;
    }
    return true;
}

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

std::string dir_of(const std::string& path) {
    auto sep = path.find_last_of("/\\");
    return (sep == std::string::npos) ? std::string(".") : path.substr(0, sep);
}

std::string discover_sibling(const std::string& base_dir, const std::vector<const char*>& candidates) {
    for (const char* name : candidates) {
        std::string p = base_dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

class CosyVoice3TtsBackend : public CrispasrBackend {
public:
    CosyVoice3TtsBackend() = default;
    ~CosyVoice3TtsBackend() override { CosyVoice3TtsBackend::shutdown(); }

    const char* name() const override { return "cosyvoice3-tts"; }

    uint32_t capabilities() const override {
        // #304: CAP_SRC_TGT_LANGUAGE signals cross-lingual synthesis — pass a
        // target language (-l/--language, or -tl) that differs from the voice's
        // own language and CV3 drops the reference transcript to avoid the
        // cross-lingual accent (SubtitleEdit surfaces this as a language menu).
        return CAP_TTS | CAP_VOICE_CLONING | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN |
               CAP_SRC_TGT_LANGUAGE;
    }

    int tts_sample_rate() const override { return 24000; }

    // The voices.gguf this backend resolved at load. `--voice <name>` selects an
    // entry inside it, so this is the only path the clone gate can read.
    std::string voice_bank_path() const override { return voices_path_; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[cosyvoice3-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        cosyvoice3_tts_context_params cp = cosyvoice3_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        // CV3 trains with `ras_sampling(top_p=0.8, top_k=25, ...)`; greedy
        // (temperature=0) falls into the documented "silent_tokens"
        // loop within ~5 steps. Default to a non-zero temperature so
        // `cosyvoice3_tts_sample_ras` engages, but honour an explicit
        // user override (including --temperature 0 for diff testing).
        cp.temperature = p.temperature > 0.0f ? p.temperature : 0.8f;
        cp.seed = (uint64_t)p.seed;
        cp.max_tokens = p.max_new_tokens;

        ctx_ = cosyvoice3_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load LLM '%s'\n", p.model.c_str());
            return false;
        }

        const std::string base_dir = dir_of(p.model);

        // ---- Flow ----
        std::string flow_path = p.tts_codec_model;
        if (flow_path.empty() || flow_path == "auto" || flow_path == "default") {
            flow_path = discover_sibling(base_dir, {
                                                       "cosyvoice3-flow-f16.gguf",
                                                       "cosyvoice3-flow-q8_0.gguf",
                                                       "cosyvoice3-flow.gguf",
                                                   });
        }
        if (flow_path.empty()) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: no flow GGUF found. Place "
                            "cosyvoice3-flow-f16.gguf next to the LLM, or pass --codec-model PATH.\n");
            return false;
        }
        if (cosyvoice3_tts_init_flow_from_file(ctx_, flow_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load flow '%s'\n", flow_path.c_str());
            return false;
        }

        // ---- CAMPPlus ----
        std::string campplus_path;
        const char* env_campplus = crispasr_env::get("CRISPASR_COSYVOICE3_CAMPPLUS_PATH");
        if (env_campplus && env_campplus[0])
            campplus_path = env_campplus;
        if (campplus_path.empty()) {
            campplus_path = discover_sibling(base_dir, {
                                                           "cosyvoice3-campplus-f16.gguf",
                                                           "cosyvoice3-campplus.gguf",
                                                       });
        }
        campplus_path_ = std::move(campplus_path);

        // ---- HiFT ----
        std::string hift_path;
        const char* env_hift = crispasr_env::get("CRISPASR_COSYVOICE3_HIFT_PATH");
        if (env_hift && env_hift[0])
            hift_path = env_hift;
        if (hift_path.empty()) {
            hift_path = discover_sibling(base_dir, {
                                                       "cosyvoice3-hift-f16.gguf",
                                                       "cosyvoice3-hift.gguf",
                                                   });
        }
        if (hift_path.empty()) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: no HiFT GGUF found. Place "
                            "cosyvoice3-hift-f16.gguf next to the LLM, or set "
                            "COSYVOICE3_HIFT_PATH.\n");
            return false;
        }
        if (cosyvoice3_tts_init_hift_from_file(ctx_, hift_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load HiFT '%s'\n", hift_path.c_str());
            return false;
        }

        // ---- speech_tokenizer_v3 ----
        s3tok_path_ = discover_sibling(base_dir, {
                                                     "cosyvoice3-s3tok-f16.gguf",
                                                     "cosyvoice3-s3tok.gguf",
                                                 });

        // ---- Voices ----
        std::string voices_path;
        const char* env_voices = crispasr_env::get("CRISPASR_COSYVOICE3_VOICES_PATH");
        if (env_voices && env_voices[0])
            voices_path = env_voices;
        if (voices_path.empty()) {
            voices_path = discover_sibling(base_dir, {
                                                         "cosyvoice3-voices.gguf",
                                                         "voices.gguf",
                                                     });
        }
        if (voices_path.empty()) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: no voices GGUF found. Run "
                            "models/convert-cosyvoice3-voices-to-gguf.py and place the "
                            "output next to the LLM, or set COSYVOICE3_VOICES_PATH.\n");
            return false;
        }
        if (cosyvoice3_tts_init_voices_from_file(ctx_, voices_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load voices '%s'\n", voices_path.c_str());
            return false;
        }
        // Hand the bundle to the voice-clone gate. Every entry in here was baked
        // from a reference recording, and --voice names one of them rather than
        // a file, so without this the gate sees an unresolvable bare name and
        // calls a zero-shot voice clone a preset. See voice_bank_path().
        voices_path_ = voices_path;
        if (!p.no_prints) {
            int nv = cosyvoice3_tts_n_voices(ctx_);
            fprintf(stderr, "crispasr[cosyvoice3-tts]: %d voice(s) available:", nv);
            for (int i = 0; i < nv; i++) {
                fprintf(stderr, " %s", cosyvoice3_tts_voice_name(ctx_, i));
            }
            fprintf(stderr, "\n");
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};
        cosyvoice3_tts_set_temperature(ctx_, params.temperature > 0.0f ? params.temperature : 0.8f);
        cosyvoice3_tts_set_seed(ctx_, (uint64_t)params.seed);

        // #304 cross-lingual: the requested synthesis language — prefer -tl,
        // else -l (mirrors OmniVoice's language knob that SE already surfaces).
        // When it differs from the reference voice's language, synth drops the
        // reference transcript to avoid the cross-lingual accent (#304).
        std::string tgt_lang = !params.target_lang.empty()
                                   ? params.target_lang
                                   : (params.language != "auto" ? params.language : std::string());
        cosyvoice3_tts_set_target_language(ctx_, tgt_lang.c_str());
        // #329: --source-lang names the language of the REFERENCE clip. Without
        // it the backend infers one from the voice-bank name or the reference
        // transcript, which cannot always answer — and when it cannot, the
        // target language is silently ignored and the clone keeps the
        // reference's accent (exactly the report in #329).
        cosyvoice3_tts_set_reference_language(ctx_, params.source_lang.c_str());

        int n = 0;
        float* pcm = nullptr;
        if (ends_with_ci(params.tts_voice, ".wav")) {
            // #334: the talker is conditioned on the (reference transcript,
            // reference speech) pair and reads the speaker's rate off it, so a
            // transcript that does not match the clip is the single most
            // damaging input here — it stops the decode dead or rushes the
            // requested line. Refusing to run without one only pushed that job
            // onto the caller, who typically approximates. Transcribe it
            // ourselves instead (cached beside the clip), exactly as f5-tts
            // already does; an explicit --ref-text still wins.
            const std::string ref_text = resolve_ref_text(params);
            if (ref_text.empty()) {
                fprintf(stderr,
                        "crispasr[cosyvoice3-tts]: --voice is a WAV and --ref-text was not set, and the reference "
                        "could not be auto-transcribed. Pass --ref-text \"<exact transcript of the clip>\".\n");
                return {};
            }
            if (!ensure_cloning_models())
                return {};
            pcm = cosyvoice3_tts_synth_from_wav(ctx_, text.c_str(), params.tts_voice.c_str(), ref_text.c_str(), &n);
        } else {
            const char* voice = params.tts_voice.empty() ? nullptr : params.tts_voice.c_str();
            pcm = cosyvoice3_tts_synth(ctx_, text.c_str(), voice, &n);
        }
        if (!pcm || n <= 0) {
            if (pcm)
                free(pcm);
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            cosyvoice3_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    // An explicit --ref-text wins; otherwise read the clip, resample to 16 kHz
    // and hand it to the ASR backend behind the shared ref-text cache. Returns
    // "" only when both are unavailable.
    std::string resolve_ref_text(const whisper_params& p) {
        if (!p.tts_ref_text.empty())
            return p.tts_ref_text;
        std::vector<float> pcm;
        int sr = 0;
        if (!crispasr::core::read_wav_mono_pcm16(p.tts_voice, pcm, sr) || pcm.empty() || sr <= 0) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: could not read '%s' as PCM16 WAV for auto-transcription\n",
                    p.tts_voice.c_str());
            return "";
        }
        if (sr != 16000)
            pcm = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 16000);
        return crispasr_ref_text::resolve_cached(p.tts_voice, pcm, p, "crispasr[cosyvoice3-tts]",
                                                 crispasr_ref_cache::kCv3RefTextSuffix);
    }

    bool ensure_cloning_models() {
        std::lock_guard<std::mutex> lock(cloning_models_mutex_);
        if (cloning_models_loaded_)
            return true;
        if (campplus_path_.empty() || s3tok_path_.empty()) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: WAV cloning requires cosyvoice3-campplus-f16.gguf and "
                            "cosyvoice3-s3tok-f16.gguf beside the LLM (or COSYVOICE3_CAMPPLUS_PATH).\n");
            return false;
        }
        if (cosyvoice3_tts_init_campplus_from_file(ctx_, campplus_path_.c_str()) != 0) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load CAMPPlus '%s'\n", campplus_path_.c_str());
            return false;
        }
        if (cosyvoice3_tts_init_s3tok_from_file(ctx_, s3tok_path_.c_str()) != 0) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load s3tok '%s'\n", s3tok_path_.c_str());
            return false;
        }
        cloning_models_loaded_ = true;
        return true;
    }

    struct cosyvoice3_tts_context* ctx_ = nullptr;
    std::string campplus_path_;
    std::string s3tok_path_;
    std::string voices_path_;
    bool cloning_models_loaded_ = false;
    std::mutex cloning_models_mutex_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_cosyvoice3_tts_backend() {
    return std::unique_ptr<CrispasrBackend>(new CosyVoice3TtsBackend());
}

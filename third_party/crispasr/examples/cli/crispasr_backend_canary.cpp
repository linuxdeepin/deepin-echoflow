// crispasr_backend_canary.cpp — adapter for nvidia/canary-1b-v2.
//
// Wraps canary_init_from_file + canary_transcribe_ex. Canary supports
// explicit source/target language pairs (for speech translation) and a
// punctuation toggle, so this backend reads params.source_lang,
// params.target_lang, and params.punctuation from whisper_params.
//
// When source_lang is empty it defaults to params.language. When target_lang
// is empty it defaults to source_lang (ASR rather than translation). When
// params.translate is true and target_lang is unset, target_lang is forced
// to "en" — matching the semantics of whisper's --translate flag.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "canary.h"
#include "core/script_mismatch.h" // #419 wrong-script warning

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "core/crispasr_env.h"

namespace {

class CanaryBackend : public CrispasrBackend {
public:
    CanaryBackend() = default;
    ~CanaryBackend() override { CanaryBackend::shutdown(); }

    const char* name() const override { return "canary"; }

    uint32_t capabilities() const override {
        // CAP_INTERNAL_CHUNKING: same FastConformer encoder as parakeet —
        // the 30 s auto-chunk causes z-norm drift and content loss.  Let
        // the backend handle full audio in a single encoder pass (safe up
        // to ~60 s; for longer audio, parakeet_transcribe_streamed-style
        // chunking can be added later if needed).  Issue #89 follow-up.
        return CAP_TIMESTAMPS_NATIVE | CAP_TIMESTAMPS_CTC | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_TRANSLATE |
               CAP_SRC_TGT_LANGUAGE | CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN | CAP_TEMPERATURE | CAP_BEAM_SEARCH |
               CAP_DIARIZE | CAP_PARALLEL_PROCESSORS | CAP_AUTO_DOWNLOAD | CAP_UNBOUNDED_INPUT | CAP_INTERNAL_CHUNKING;
    }

    bool init(const whisper_params& p) override {
        canary_context_params cp = canary_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = canary_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[canary]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    void warmup() override {
        if (!ctx_)
            return;
        std::vector<float> silence(8000, 0.0f);
        canary_result* r = canary_transcribe_ex(ctx_, silence.data(), (int)silence.size(), "en", "en", true, 0);
        if (r)
            canary_result_free(r);
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Sticky decode-time sampling controls.
        canary_set_temperature(ctx_, params.temperature, params.seed);
        canary_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);
        // #292: forward --max-new-tokens only when explicit; 0 keeps the default.
        canary_set_max_new_tokens(ctx_, params.max_new_tokens_explicit ? params.max_new_tokens : 0);

        // Resolve src/tgt language with the fallback chain:
        //   source_lang -> language
        //   target_lang -> source_lang (ASR) or "en" (--translate)
        // Canary's prompt embeds the language token LITERALLY (e.g.
        // "<|en|>"); it has no "auto" token. If the dispatcher's LID
        // step failed and left params.language="auto", or the caller
        // never set one, fall back to "en" with a stderr note. This is
        // a defensive layer — the dispatcher should already have
        // resolved this — but we keep it so canary degrades gracefully
        // even if a future code path skips LID resolution.
        std::string src = params.source_lang.empty() ? params.language : params.source_lang;
        if (src == "auto" || src.empty()) {
            if (!params.no_prints) {
                fprintf(stderr,
                        "canary: no source language set (got '%s'); "
                        "defaulting to 'en'. Pass `-l <lang>` or "
                        "`--source-lang <lang>` to set explicitly.\n",
                        src.c_str());
            }
            src = "en";
        }
        std::string tgt = params.target_lang;
        if (tgt.empty() || tgt == "auto") {
            tgt = params.translate ? std::string("en") : src;
        }

        // Issue #89 / #140: canary-1b-v2 is trained on 25 European
        // languages (per the NVIDIA model card). The BPE vocab includes
        // every ISO-639 `<|xx|>` token, but only the 25 below have
        // training signal — anything else produces hallucinated output.
        // clang-format off
        static const char* kSupportedLangs[] = {
            "en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr",
            "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt",
            "ro", "sk", "sl", "es", "sv", "ru", "uk",
        };
        // clang-format on
        auto is_supported = [&](const std::string& lang) {
            for (const char* s : kSupportedLangs)
                if (lang == s)
                    return true;
            return false;
        };
        if (!is_supported(src) || !is_supported(tgt)) {
            fprintf(stderr,
                    "canary: src='%s' tgt='%s' — not in canary-1b-v2's "
                    "trained language set (25 European languages). "
                    "For Japanese/Mandarin use --backend parakeet; for "
                    "the broader multilingual set use --backend qwen3 "
                    "or --backend voxtral.\n",
                    src.c_str(), tgt.c_str());
            return out;
        }

        // Issue #419: print the EFFECTIVE language conditioning once. Canary
        // conditioned on the wrong language transliterates rather than
        // failing (Russian through <|en|> → "vikingi, otvazhnye voyny"), and
        // the reported case pointed at a frontend whose -l/--source-lang may
        // never have reached this process — one stderr line makes that
        // diagnosable from any log.
        static bool s_lang_logged = false;
        if (!params.no_prints && !s_lang_logged) {
            s_lang_logged = true;
            fprintf(stderr, "canary: languages src='%s' tgt='%s'\n", src.c_str(), tgt.c_str());
        }

        // Long-form handling follows canary-1b-v2's own `.transcribe()`
        // dynamic chunking (blueprint port in src/canary.cpp): audio that
        // fits one 40 s chunk is a single pass; longer audio is split into
        // dynamically sized 30..40 s raw-waveform chunks with a 1 s overlap
        // and merged by the reference's LCS alignment. Passing 0 / -1 lets
        // the library pick the reference's sizes.
        // CRISPASR_CANARY_STREAM_THRESHOLD_S=N forces single-pass for
        // inputs ≤ N seconds (debug/A-B); CRISPASR_CANARY_LEGACY_STREAM=1
        // selects the pre-blueprint 8 s / 2 s machinery inside the library.
        int stream_threshold_s = 0;
        if (const char* e = crispasr_env::get("CRISPASR_CANARY_STREAM_THRESHOLD_S")) {
            stream_threshold_s = std::max(0, atoi(e));
        }
        const bool force_single = stream_threshold_s > 0 && n_samples <= stream_threshold_s * 16000;

        canary_result* r = force_single ? canary_transcribe_ex(ctx_, samples, n_samples, src.c_str(), tgt.c_str(),
                                                               params.punctuation, t_offset_cs)
                                        : canary_transcribe_streamed(ctx_, samples, n_samples, src.c_str(), tgt.c_str(),
                                                                     params.punctuation, t_offset_cs, 0, -1);
        if (!r)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs;
        seg.text = r->text ? r->text : "";

        seg.words.reserve(r->n_words);
        for (int i = 0; i < r->n_words; i++) {
            const auto& w = r->words[i];
            crispasr_word cw;
            cw.text = w.text;
            cw.t0 = w.t0;
            cw.t1 = w.t1;
            seg.words.push_back(std::move(cw));
        }

        seg.tokens.reserve(r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            const auto& t = r->tokens[i];
            crispasr_token ct;
            ct.text = t.text;
            ct.id = t.id;
            ct.t0 = t.t0;
            ct.t1 = t.t1;
            ct.confidence = t.p;
            seg.tokens.push_back(std::move(ct));
        }

        if (!seg.words.empty()) {
            seg.t0 = seg.words.front().t0;
            seg.t1 = seg.words.back().t1;
        } else if (!seg.tokens.empty()) {
            seg.t0 = seg.tokens.front().t0;
            seg.t1 = seg.tokens.back().t1;
        }

        canary_result_free(r);

        // Issue #419: canary conditioned on the WRONG language renders speech
        // in the wrong SCRIPT — Russian decoded as <|en|> comes out as Latin
        // transliteration ("vikingi, otvazhnye voyny"), silently. With the
        // language tokens actually set to ru, every backend produces proper
        // Cyrillic (verified CPU / CUDA / Vulkan, single + streamed + short
        // slices). So a script contradiction means the conditioning did not
        // arrive — say so, name the effective langs and the model, and give
        // the user something actionable instead of translit subtitles.
        if (!params.no_prints && core_script::mismatch(tgt, seg.text)) {
            core_script::ScriptCounts sc;
            core_script::mismatch(tgt, seg.text, &sc);
            fprintf(stderr,
                    "canary: WARNING: target language '%s' expects %s but the transcript is "
                    "Latin-dominated (%d Latin vs %d %s letters). The model was likely "
                    "conditioned on the wrong language: verify -l/--source-lang actually reach "
                    "this process (frontends sometimes drop them) and that the model resolved to "
                    "canary-1b-v2 (run with -v to see the path). See issue #419.\n",
                    tgt.c_str(), tgt == "el" ? "Greek" : "Cyrillic", sc.latin, tgt == "el" ? sc.greek : sc.cyrillic,
                    tgt == "el" ? "Greek" : "Cyrillic");
        }

        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            canary_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    canary_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_canary_backend() {
    return std::unique_ptr<CrispasrBackend>(new CanaryBackend());
}

// crispasr_backend_gigaam.cpp — adapter for ai-sage/GigaAM-v3 (Russian ASR).
//
// Wraps gigaam_init_from_file + gigaam_transcribe_ex and converts the native
// gigaam_result into std::vector<crispasr_segment>.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "gigaam.h"

#include <cstdio>
#include <cstring>

namespace {

class GigaamBackend : public CrispasrBackend {
public:
    GigaamBackend() = default;
    ~GigaamBackend() override { GigaamBackend::shutdown(); }

    const char* name() const override { return "gigaam"; }

    uint32_t capabilities() const override {
        // CAP_PUNCTUATION_NATIVE is declared for EVERY revision, for two
        // different reasons:
        //   - the `e2e_*` ones carry punctuation + casing + inverse text
        //     normalization in their SentencePiece vocabulary, so a
        //     restoration pass on top would double-punctuate (#308);
        //   - the charwise `ctc` / `rnnt` ones emit bare lowercase Cyrillic,
        //     but the auto-enabled restorer is FireRedPunc, which is trained
        //     on Chinese + English and injects FULL-WIDTH CJK punctuation into
        //     Russian ("надеждой， сладкой ... зеленый。"). Suppressing the
        //     auto pass is strictly better than that.
        // An explicit `--punc-model` still applies either way — the flag only
        // gates crispasr_should_auto_enable_punctuation().
        return CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN |
               CAP_PUNCTUATION_NATIVE | CAP_AUTO_DOWNLOAD;
    }

    // Russian-only, so `-l auto` should not download and run a whisper-tiny
    // LID pass to "detect" a language the model cannot change (#227).
    const char* sole_language() const override { return "ru"; }

    bool init(const whisper_params& p) override {
        gigaam_context_params cp = gigaam_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = gigaam_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[gigaam]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        // GigaAM-v3 is Russian-only; a language flag cannot steer it.
        if (!p.language.empty() && p.language != "auto" && p.language != "ru") {
            fprintf(stderr, "crispasr[gigaam]: '%s' ignored — GigaAM-v3 is a Russian-only model\n", p.language.c_str());
        }
        return true;
    }

    void warmup() override {
        if (!ctx_)
            return;
        std::vector<float> silence(8000, 0.0f);
        gigaam_result* r = gigaam_transcribe_ex(ctx_, silence.data(), (int)silence.size(), 0);
        if (r)
            gigaam_result_free(r);
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // The transducer's per-frame symbol cap is the only decode knob this
        // model has; forward it only when the user set --max-new-tokens
        // explicitly so the CLI's global default never shrinks it (#292).
        gigaam_set_max_symbols(ctx_, params.max_new_tokens_explicit ? params.max_new_tokens : 0);

        gigaam_result* r = gigaam_transcribe_ex(ctx_, samples, n_samples, t_offset_cs);
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

        gigaam_result_free(r);
        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            gigaam_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    gigaam_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_gigaam_backend() {
    return std::unique_ptr<CrispasrBackend>(new GigaamBackend());
}

// crispasr_backend_firered_asr.cpp — FireRedASR2-AED backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "firered_asr.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/crispasr_env.h"
#include "core/ngram_loop_fix.h" // core_ngram::fix_loops (PLAN.md §W1b)

class FireredAsrBackend : public CrispasrBackend {
public:
    FireredAsrBackend() = default;

    const char* name() const override { return "firered-asr"; }

    uint32_t capabilities() const override {
        // firered-asr's encoder uses relative positional encoding with
        // pe_maxlen=5000 (~200 s of post-subsample frames) and an O(T²)
        // self-attention, so feeding a long file in one pass is both very
        // slow and risks running off the PE center window (issue #125: >50 s
        // audio hung indefinitely on CUDA).
        //
        // Declare CAP_UNBOUNDED_INPUT so the issue #89 long-audio dispatch
        // (crispasr_run.cpp) auto-chunks at 30 s when the user passes neither
        // --vad nor --chunk-seconds. Without the flag, should_auto_chunk_long()
        // returns early and the backend receives the whole file in one pass —
        // the earlier "do NOT declare it" reasoning was inverted (the
        // dispatcher only chunks unbounded-input backends; bounded ones get
        // the full audio). Short files still take the single-pass path; the
        // pe_maxlen guard in firered_asr.cpp remains as a backstop.
        return CAP_UNBOUNDED_INPUT | CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_BEAM_SEARCH | CAP_TOKEN_CONFIDENCE |
               CAP_FLASH_ATTN | CAP_DIARIZE;
    }

    bool init(const whisper_params& params) override {
        firered_asr_context_params cp = firered_asr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        if (getenv("CRISPASR_VERBOSE") || crispasr_env::get("CRISPASR_FIRERED_BENCH"))
            cp.verbosity = 2;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.beam_size = params.beam_size > 0 ? params.beam_size : 3;

        // FireRedASR2-AED auto-detects the spoken language (built-in LID) among
        // the languages it was trained on: Chinese (+ ~20 Chinese dialects),
        // English, and Cantonese. It has no token for any other language and
        // offers no per-request language override, so `-l <lang>` cannot steer
        // it. Warn instead of silently ignoring an unsupported request — issue
        // #199: `-l ja` on this Mandarin/English model produced hallucinations
        // because there is no Japanese in the model at all. zh/en/yue are
        // handled by auto-detection regardless of the flag, so only flag the
        // genuinely unsupported codes.
        if (!params.no_prints && !params.language.empty() && params.language != "auto") {
            const std::string& l = params.language;
            const bool firered_lang = (l == "zh" || l == "en" || l == "yue" || l == "zh-en" || l == "zh_en" ||
                                       l == "chinese" || l == "english" || l == "cantonese");
            if (!firered_lang) {
                fprintf(stderr,
                        "crispasr[firered-asr]: WARNING: this model only transcribes Chinese (+ Chinese "
                        "dialects), English, and Cantonese; '-l %s' is not supported and will be ignored "
                        "(expect hallucinations for non-Chinese/English audio). For %s, use a native backend "
                        "— e.g. reazonspeech / parakeet (parakeet-tdt-0.6b-ja) / fastconformer-ctc / funasr "
                        "(zh,yue,en,ja,ko).\n",
                        l.c_str(), crispasr_iso_to_english_lang(l).c_str());
            }
        }

        ctx_ = firered_asr_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        (void)params;
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        firered_asr_result* r = firered_asr_transcribe_with_probs(ctx_, samples, n_samples);
        if (!r || !r->text)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = r->text;

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        // Per-token confidence: one entry per emitted (non-special) token from
        // the winning beam. Timestamps are not available at token granularity
        // here (firered's AR decoder doesn't emit time-aligned outputs), so
        // we leave t0/t1 unset (-1).
        seg.tokens.reserve((size_t)r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            crispasr_token tok;
            tok.id = r->token_ids[i];
            tok.confidence = r->token_probs[i];
            const char* piece = firered_asr_token_text(ctx_, r->token_ids[i]);
            if (piece) {
                // Convert ▁ (U+2581, 0xE2 0x96 0x81) to space — same convention as
                // firered_asr_transcribe_impl.
                std::string p = piece;
                std::string decoded;
                for (size_t ci = 0; ci < p.size(); ci++) {
                    if ((unsigned char)p[ci] == 0xE2 && ci + 2 < p.size() && (unsigned char)p[ci + 1] == 0x96 &&
                        (unsigned char)p[ci + 2] == 0x81) {
                        decoded += ' ';
                        ci += 2;
                    } else {
                        decoded += p[ci];
                    }
                }
                tok.text = std::move(decoded);
            }
            seg.tokens.push_back(std::move(tok));
        }

        firered_asr_result_free(r);

        // PLAN.md §W1b: this adapter had NO core_ngram::fix_loops. The only
        // guard was the decode-time core_repeat::tail_is_repetition break in
        // firered_asr.cpp, which is wired into the beam_size==1 branch only —
        // so a beam run had nothing at all, and even a greedy run keeps
        // whatever residue the break stopped at. firered is Mandarin + 20+
        // Chinese dialects, i.e. exactly the script the collapse was blind to
        // until §W1 gave it a code-point path.
        {
            std::vector<std::string> tok_texts;
            tok_texts.reserve(seg.tokens.size());
            for (const auto& t : seg.tokens)
                tok_texts.push_back(t.text);
            const std::vector<int> keep = core_ngram::fix_loops_keep_indices(tok_texts);
            if (keep.size() < seg.tokens.size()) {
                std::vector<crispasr_token> kept;
                kept.reserve(keep.size());
                for (int ki : keep)
                    if (ki >= 0 && ki < (int)seg.tokens.size())
                        kept.push_back(seg.tokens[(size_t)ki]);
                seg.tokens = std::move(kept);
            }
            seg.text = core_ngram::fix_loops(seg.text);
        }

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            firered_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~FireredAsrBackend() override { FireredAsrBackend::shutdown(); }

private:
    firered_asr_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_firered_asr_backend() {
    return std::make_unique<FireredAsrBackend>();
}

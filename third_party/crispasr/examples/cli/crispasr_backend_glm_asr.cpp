// crispasr_backend_glm_asr.cpp — GLM-ASR-Nano backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "glm_asr.h"
#include "whisper_params.h"
#include "core/bpe.h"
#include "core/ngram_loop_fix.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class GlmAsrBackend : public CrispasrBackend {
public:
    GlmAsrBackend() = default;

    const char* name() const override { return "glm-asr"; }

    uint32_t capabilities() const override {
        // CAP_LANGUAGE_DETECT intentionally NOT declared: glm-asr has no
        // native LID. Declaring it would disable the framework pre-step
        // — see crispasr_backend_parakeet.cpp for the same reasoning.
        return CAP_TIMESTAMPS_CTC | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_AUTO_DOWNLOAD | CAP_TOKEN_CONFIDENCE |
               CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN | CAP_DIARIZE | CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE;
    }

    bool init(const whisper_params& params) override {
        glm_asr_context_params cp = glm_asr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.temperature = params.temperature;
        cp.beam_size = params.beam_size > 0 ? params.beam_size : 1;
        cp.translate = params.translate;
        if (!params.target_lang.empty())
            tgt_lang_ = params.target_lang;
        cp.target_lang = tgt_lang_.empty() ? nullptr : tgt_lang_.c_str();

        // GLM-ASR-Nano-2512 is trained on Chinese (+ Chinese dialects), English,
        // and Cantonese only (model card metadata: language: [en, zh]; prose adds
        // Cantonese + "other dialects"). For any other `-l <lang>` the transcribe
        // path still sends a "Please transcribe in <lang>." hint, but the model
        // has no such language and will hallucinate (#199: `-l ja`). Warn instead
        // of silently mistranscribing; the hint may still help steer zh/en/yue,
        // so those stay silent. Skips the warning under --translate (target_lang
        // is the output language there, a separate feature).
        if (!params.no_prints && !params.translate && !params.language.empty() && params.language != "auto") {
            const std::string& l = params.language;
            const bool glm_lang = (l == "zh" || l == "en" || l == "yue" || l == "zh-en" || l == "zh_en" ||
                                   l == "chinese" || l == "english" || l == "cantonese");
            if (!glm_lang) {
                fprintf(stderr,
                        "crispasr[glm-asr]: WARNING: this model only transcribes Chinese (+ Chinese "
                        "dialects), English, and Cantonese; '-l %s' is not supported and likely produces "
                        "hallucinations. For %s, use a native backend — e.g. reazonspeech / parakeet "
                        "(parakeet-tdt-0.6b-ja) / fastconformer-ctc / funasr (zh,yue,en,ja,ko).\n",
                        l.c_str(), crispasr_iso_to_english_lang(l).c_str());
            }
        }

        ctx_ = glm_asr_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Propagate ask / language instruction to the C library. "en" (the
        // CLI default) and "auto" use the blueprint's default transcription
        // prompt — a language instruction is only injected for an explicit
        // non-English hint (English IS the default prompt's behaviour;
        // silently replacing the blueprint prompt on every run would be
        // off-distribution).
        if (!params.ask.empty()) {
            glm_asr_set_ask(ctx_, params.ask.c_str());
        } else if (!params.language.empty() && params.language != "auto" && params.language != "en" &&
                   !params.translate) {
            const std::string instr = "Please transcribe in " + crispasr_iso_to_english_lang(params.language) + ".";
            glm_asr_set_ask(ctx_, instr.c_str());
        } else {
            glm_asr_set_ask(ctx_, nullptr);
        }
        // #292: forward --max-new-tokens only when explicit; 0 keeps the backend default.
        glm_asr_set_max_new_tokens(ctx_, params.max_new_tokens_explicit ? params.max_new_tokens : 0);

        // Best-of-N: when temperature > 0 and best_of > 1, run N seeded
        // decodes (process-global libc rand reseeded per run) and keep the
        // highest mean prob.
        const int n_runs = (params.temperature > 0.0f && params.best_of > 1) ? params.best_of : 1;
        glm_asr_result* r = nullptr;
        double best_score = -1.0;
        for (int run = 0; run < n_runs; run++) {
            if (n_runs > 1)
                glm_asr_set_seed(ctx_, (unsigned int)(params.seed ^ (run * 0x9E3779B9u + 1u)));
            glm_asr_result* cand = glm_asr_transcribe_with_probs(ctx_, samples, n_samples);
            if (!cand)
                continue;
            double sum = 0.0;
            int cnt = 0;
            for (int i = 0; i < cand->n_tokens; i++) {
                sum += (double)cand->token_probs[i];
                cnt++;
            }
            double score = (cnt > 0) ? (sum / cnt) : 0.0;
            if (!r || score > best_score) {
                if (r)
                    glm_asr_result_free(r);
                r = cand;
                best_score = score;
            } else {
                glm_asr_result_free(cand);
            }
        }
        if (!r || !r->text)
            return out;
        if (!params.no_prints && n_runs > 1)
            fprintf(stderr, "crispasr[glm-asr]: best-of-%d picked score=%.4f\n", n_runs, best_score);

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = r->text;

        // Trim leading/trailing whitespace
        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        // Apply fix_loops to text; token dedup done after token construction below (#218)

        // GPT-2 byte-level BPE decoder — full table (handles CJK, etc.).

        // Per-token confidence; no per-token timestamps (GLM-ASR's LLM
        // decoder isn't time-aligned).
        std::vector<crispasr_token> all_tokens;
        all_tokens.reserve((size_t)r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            crispasr_token tok;
            tok.id = r->token_ids[i];
            tok.confidence = r->token_probs[i];
            const char* raw = glm_asr_token_text(ctx_, r->token_ids[i]);
            tok.text = raw ? core_bpe::token_bytes_to_utf8(raw) : "";
            all_tokens.push_back(std::move(tok));
        }
        glm_asr_result_free(r);

        // Apply fix_loops to both text and tokens (#218)
        std::vector<std::string> tok_texts;
        for (auto& tk : all_tokens)
            tok_texts.push_back(tk.text);
        const std::vector<int> keep = core_ngram::fix_loops_keep_indices(tok_texts);
        seg.text = core_ngram::fix_loops(seg.text);
        for (int ki : keep) {
            if (ki >= 0 && ki < (int)all_tokens.size())
                seg.tokens.push_back(std::move(all_tokens[ki]));
        }

        // --no-punctuation: strip ASCII punctuation from segment text and per-token
        // pieces. GLM-ASR's LLM produces punctuated, capitalised English by
        // default; this matches the historical CTC-style "lowercase, no punc"
        // output users expect when they pass the toggle.
        if (!params.punctuation) {
            crispasr_strip_ascii_punctuation(seg.text);
            crispasr_lowercase_ascii(seg.text);
            for (auto& tok : seg.tokens) {
                crispasr_strip_ascii_punctuation(tok.text);
                crispasr_lowercase_ascii(tok.text);
            }
        }

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void transcribe_streaming(const float* samples, int n_samples, int64_t /*t_offset_cs*/,
                              const whisper_params& params, crispasr_stream_callback on_text) override {
        if (!ctx_ || params.beam_size > 1 || params.temperature > 0.0f) {
            CrispasrBackend::transcribe_streaming(samples, n_samples, 0, params, on_text);
            return;
        }

        // 1. Mel
        int n_mels = 0, T_mel = 0;
        float* mel = glm_asr_compute_mel(ctx_, samples, n_samples, &n_mels, &T_mel);
        if (!mel)
            return;

        // Pad/truncate to fixed 3000-frame window (same as glm_asr_transcribe_impl)
        const int T_target = 3000;
        if (T_mel != T_target) {
            std::vector<float> padded((size_t)n_mels * T_target, 0.0f);
            const int T_copy = std::min(T_mel, T_target);
            for (int m = n_mels - 1; m >= 0; m--)
                memcpy(padded.data() + (size_t)m * T_target, mel + (size_t)m * T_mel, (size_t)T_copy * sizeof(float));
            free(mel);
            mel = (float*)malloc(padded.size() * sizeof(float));
            if (!mel)
                return;
            memcpy(mel, padded.data(), padded.size() * sizeof(float));
            T_mel = T_target;
        }

        // 2. Encoder
        int N_enc = 0, enc_dim = 0;
        float* audio_embeds = glm_asr_run_encoder(ctx_, mel, n_mels, T_mel, &N_enc, &enc_dim);
        free(mel);
        if (!audio_embeds)
            return;

        // 3. Prompt (mirrors glm_asr_transcribe_impl's blueprint template:
        //    <|user|>\n<|begin_of_audio|>…<|end_of_audio|><|user|>\n
        //    Please transcribe this audio into text<|assistant|>\n).
        //    glm_asr_tokenize is specials-only, so free-text instructions
        //    fall back to the default transcription prompt like the impl.
        static const int32_t kNewline = 10;
        static const int32_t kDefaultInstruction[] = {14215, 1700, 8091, 643, 14812, 1636, 2815};
        std::vector<int32_t> ids;
        ids.push_back(59253); // <|user|>
        ids.push_back(kNewline);
        ids.push_back(59261); // <|begin_of_audio|>
        for (int i = 0; i < N_enc; i++)
            ids.push_back(59260); // <|pad|>
        ids.push_back(59262);     // <|end_of_audio|>
        ids.push_back(59253);     // <|user|>
        {
            std::string instr;
            if (!params.ask.empty()) {
                instr = "\n" + params.ask;
            } else if (!tgt_lang_.empty()) {
                char buf[256];
                snprintf(buf, sizeof(buf), "\nPlease translate the speech to %s.", tgt_lang_.c_str());
                instr = buf;
            } else if (!params.language.empty() && params.language != "auto" && params.language != "en") {
                instr = "\nPlease transcribe in " + crispasr_iso_to_english_lang(params.language) + ".";
            }
            bool emitted = false;
            if (!instr.empty()) {
                int n_instr = 0;
                int32_t* itoks = glm_asr_tokenize(ctx_, instr.c_str(), &n_instr);
                if (itoks && n_instr > 0) {
                    for (int i = 0; i < n_instr; i++)
                        ids.push_back(itoks[i]);
                    emitted = true;
                }
                free(itoks);
            }
            if (!emitted) {
                ids.push_back(kNewline);
                for (int32_t id : kDefaultInstruction)
                    ids.push_back(id);
            }
        }
        ids.push_back(59254); // <|assistant|>
        ids.push_back(kNewline);

        // 4. Embed + splice audio
        float* text_embeds = glm_asr_embed_tokens(ctx_, ids.data(), (int)ids.size());
        if (!text_embeds) {
            free(audio_embeds);
            return;
        }
        int spliced = 0;
        for (size_t i = 0; i < ids.size() && spliced < N_enc; i++) {
            if (ids[i] == 59260) {
                memcpy(text_embeds + i * enc_dim, audio_embeds + (size_t)spliced * enc_dim,
                       (size_t)enc_dim * sizeof(float));
                spliced++;
            }
        }
        free(audio_embeds);

        // 5. KV init + prefill (sized to prompt + decode budget; kv_init
        //    grows when a larger request arrives)
        if (!glm_asr_kv_init(ctx_, std::max(4096, (int)ids.size() + 512 + 16))) {
            free(text_embeds);
            return;
        }
        glm_asr_kv_reset(ctx_);

        int n_tok = 0, vocab = 0;
        float* logits = glm_asr_run_llm_kv(ctx_, text_embeds, (int)ids.size(), 0, &n_tok, &vocab);
        free(text_embeds);
        if (!logits)
            return;

        auto is_eos = [](int id) { return id == 59246 || id == 59253 || id == 59255; };
        auto argmax = [](const float* L, int n) {
            int best = 0;
            for (int i = 1; i < n; i++)
                if (L[i] > L[best])
                    best = i;
            return best;
        };
        auto decode_bpe = [](const char* raw) -> std::string {
            std::string out;
            if (!raw)
                return out;
            for (size_t ci = 0; raw[ci] != '\0';) {
                unsigned char c = (unsigned char)raw[ci];
                if (c == 0xC4 && raw[ci + 1] != '\0') {
                    unsigned char c2 = (unsigned char)raw[ci + 1];
                    if (c2 == 0xA0) {
                        out += ' ';
                        ci += 2;
                        continue;
                    }
                    if (c2 == 0x8A) {
                        out += '\n';
                        ci += 2;
                        continue;
                    }
                }
                out += (char)c;
                ci++;
            }
            return out;
        };

        // 6. Greedy decode with per-token streaming
        int next = argmax(logits, vocab);
        free(logits);

        std::string accumulated;
        bool first_tok = true;
        int n_past = (int)ids.size();

        for (int step = 0; step < 512; step++) {
            if (is_eos(next))
                break;
            const char* raw = glm_asr_token_text(ctx_, next);
            if (raw) {
                std::string piece = decode_bpe(raw);
                if (first_tok) {
                    size_t sp = 0;
                    while (sp < piece.size() && (piece[sp] == ' ' || piece[sp] == '\n'))
                        sp++;
                    piece = piece.substr(sp);
                    if (!piece.empty())
                        first_tok = false;
                }
                if (!params.punctuation) {
                    crispasr_strip_ascii_punctuation(piece);
                    crispasr_lowercase_ascii(piece);
                }
                accumulated += piece;
                if (!accumulated.empty())
                    on_text(accumulated.c_str(), false);
            }
            float* emb = glm_asr_embed_tokens(ctx_, &next, 1);
            if (!emb)
                break;
            float* lg = glm_asr_run_llm_kv(ctx_, emb, 1, n_past, nullptr, nullptr);
            free(emb);
            if (!lg)
                break;
            n_past++;
            next = argmax(lg, vocab);
            free(lg);
        }
        on_text(accumulated.c_str(), true);
    }

    void shutdown() override {
        if (ctx_) {
            glm_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~GlmAsrBackend() override { GlmAsrBackend::shutdown(); }

private:
    glm_asr_context* ctx_ = nullptr;
    std::string tgt_lang_;
};

std::unique_ptr<CrispasrBackend> crispasr_make_glm_asr_backend() {
    return std::make_unique<GlmAsrBackend>();
}

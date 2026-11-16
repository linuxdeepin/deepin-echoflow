// crispasr_backend_kyutai_stt.cpp — Kyutai STT backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "kyutai_stt.h"
#include "whisper_params.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class KyutaiSttBackend : public CrispasrBackend {
public:
    KyutaiSttBackend() = default;

    const char* name() const override { return "kyutai-stt"; }

    uint32_t capabilities() const override {
        // PLAN #61c: kyutai's "delayed-streams" architecture aligns each
        // emitted text token to its source audio frame at zero extra cost,
        // so both segment and word timestamps are native (no DTW or CTC
        // aligner needed).
        // CAP_DIARIZE: framework post-step works on the segment list.
        return CAP_AUTO_DOWNLOAD | CAP_TOKEN_CONFIDENCE | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_TIMESTAMPS_NATIVE |
               CAP_WORD_TIMESTAMPS | CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN | CAP_DIARIZE;
    }

    // Kyutai's Mimi codec operates at 24 kHz.
    int input_sample_rate() const override { return 24000; }

    bool init(const whisper_params& params) override {
        kyutai_stt_context_params cp = kyutai_stt_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.temperature = params.temperature;
        cp.beam_size = params.beam_size > 0 ? params.beam_size : 1;
        cp.input_sample_rate = 24000;
        ctx_ = kyutai_stt_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // #366: this warned "English-only model" for EVERY kyutai-stt GGUF,
        // including stt-1b-en_fr — which is the default the registry
        // downloads, and which does support French. Ask the model instead of
        // assuming, and name what it actually supports.
        if (!params.language.empty() && !kyutai_stt_supports_language(ctx_, params.language.c_str()))
            fprintf(stderr, "crispasr[kyutai-stt]: model supports %s; language='%s' ignored\n",
                    kyutai_stt_languages(ctx_), params.language.c_str());

        // PLAN #125 P6b: kyutai's batch path scales superlinearly with
        // n_samples (~14 s/s on the 50 min file vs 1.36 s/s on JFK, per
        // reports 05 + 11). KV grows O(N) per emitted token, attention
        // turns memory-bound past L1, so total cost is O(N^2). Bound
        // per-call N by chunking at 30 s = 480 000 samples; each chunk
        // produces its own segment and the silence-tail flush from P6a
        // applies per-chunk so chunk boundaries close cleanly. Linear
        // wallclock + a predictable progress for callers.
        const int sr = input_sample_rate();
        const int kChunkSamples = 30 * sr;
        if (n_samples > kChunkSamples) {
            for (int start = 0; start < n_samples; start += kChunkSamples) {
                const int this_n = std::min(kChunkSamples, n_samples - start);
                const int64_t chunk_offset_cs = t_offset_cs + (int64_t)((double)start / (double)sr * 100.0);
                auto chunk_segs = transcribe_one(samples + start, this_n, chunk_offset_cs, params);
                for (auto& s : chunk_segs)
                    out.push_back(std::move(s));
            }
            return out;
        }
        return transcribe_one(samples, n_samples, t_offset_cs, params);
    }

    std::vector<crispasr_segment> transcribe_one(const float* samples, int n_samples, int64_t t_offset_cs,
                                                 const whisper_params& params) {
        std::vector<crispasr_segment> out;
        const int sr = input_sample_rate();

        // PLAN #61c: use the _ex API to get per-token + word-level
        // timestamps. The kyutai LM emits one text token per Mimi frame
        // (12.5 Hz = 8 cs/frame) with the audio_delay correction baked in.
        // Best-of-N: when temperature > 0 and best_of > 1, run N seeded
        // decodes (process-global libc rand reseeded per run) and keep the
        // highest mean prob across the per-token probs in the result.

        // PLAN #125 P6a: kyutai's causal LM needs a tail of audio to flush
        // its final-token state. The batch wrapper passes raw samples and
        // stops, so the model emits the partial token id for the last word
        // (e.g. "c" for "country") but never receives the audio it needs
        // to commit "ountry." + EOS. Append silence equal to the model's
        // total lookahead (audio_delay_seconds + silence_prefix_seconds)
        // so the LM walks forward and finishes all pending words.
        // Tokens emitted during the silence-tail keep their t_offset_cs
        // arithmetic and simply land a few cs past the original input end
        // — acceptable for word-timestamps.
        // Default 500 ms for models where the accessor isn't available yet;
        // stt-2.6b-en uses 2.5+1.0 = 3.5s → 56000 samples.
        const float lookahead_s = kyutai_stt_total_lookahead_seconds(ctx_);
        const int kTailSilenceSamples = std::max(sr / 2, (int)(lookahead_s * (float)sr));
        std::vector<float> padded;
        padded.reserve((size_t)n_samples + kTailSilenceSamples);
        padded.assign(samples, samples + n_samples);
        padded.resize((size_t)n_samples + kTailSilenceSamples, 0.0f);
        const float* pad_ptr = padded.data();
        const int pad_n = (int)padded.size();

        const int n_runs = (params.temperature > 0.0f && params.best_of > 1) ? params.best_of : 1;
        kyutai_stt_result_ex* r = nullptr;
        double best_score = -1.0;
        for (int run = 0; run < n_runs; run++) {
            if (n_runs > 1)
                kyutai_stt_set_seed(ctx_, (unsigned int)(params.seed ^ (run * 0x9E3779B9u + 1u)));
            kyutai_stt_result_ex* cand = kyutai_stt_transcribe_ex(ctx_, pad_ptr, pad_n, t_offset_cs);
            if (!cand)
                continue;
            double sum = 0.0;
            int cnt = 0;
            for (int i = 0; i < cand->n_tokens; i++) {
                sum += (double)cand->tokens[i].p;
                cnt++;
            }
            double score = (cnt > 0) ? (sum / cnt) : 0.0;
            if (!r || score > best_score) {
                if (r)
                    kyutai_stt_result_ex_free(r);
                r = cand;
                best_score = score;
            } else {
                kyutai_stt_result_ex_free(cand);
            }
        }
        if (!r || !r->text)
            return out;
        if (!params.no_prints && n_runs > 1)
            fprintf(stderr, "crispasr[kyutai-stt]: best-of-%d picked score=%.4f\n", n_runs, best_score);

        crispasr_segment seg;
        seg.text = r->text;
        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        // Segment span: prefer the actual word range; fall back to the
        // full audio buffer when no words emitted.
        if (r->n_words > 0) {
            seg.t0 = r->words[0].t0;
            seg.t1 = r->words[r->n_words - 1].t1;
        } else {
            seg.t0 = t_offset_cs;
            seg.t1 = t_offset_cs + (int64_t)((double)n_samples / (double)sr * 100.0);
        }

        seg.tokens.reserve((size_t)r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            const kyutai_stt_token_data& src = r->tokens[i];
            crispasr_token tok;
            tok.id = src.id;
            tok.confidence = src.p;
            tok.t0 = src.t0;
            tok.t1 = src.t1;
            tok.text = src.text;
            seg.tokens.push_back(std::move(tok));
        }

        seg.words.reserve((size_t)r->n_words);
        for (int i = 0; i < r->n_words; i++) {
            crispasr_word w;
            w.text = r->words[i].text;
            w.t0 = r->words[i].t0;
            w.t1 = r->words[i].t1;
            seg.words.push_back(std::move(w));
        }

        kyutai_stt_result_ex_free(r);

        // --no-punctuation: post-strip ASCII punctuation + lowercase. Kyutai
        // emits punctuated mixed-case English by default; the toggle gives
        // the historical CTC-style "lowercase, no punc" surface.
        if (!params.punctuation) {
            crispasr_strip_ascii_punctuation(seg.text);
            crispasr_lowercase_ascii(seg.text);
            for (auto& tok : seg.tokens) {
                crispasr_strip_ascii_punctuation(tok.text);
                crispasr_lowercase_ascii(tok.text);
            }
            for (auto& w : seg.words) {
                crispasr_strip_ascii_punctuation(w.text);
                crispasr_lowercase_ascii(w.text);
            }
        }

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void transcribe_streaming(const float* samples, int n_samples, int64_t /*t_offset_cs*/,
                              const whisper_params& params, crispasr_stream_callback on_text) override {
        if (!ctx_) {
            CrispasrBackend::transcribe_streaming(samples, n_samples, 0, params, on_text);
            return;
        }
        std::string accumulated;
        bool first_tok = true;
        // emit_token in kyutai_stt_transcribe_impl already filters padding tokens
        // and fires on_tok only for valid non-pad text tokens.
        auto cb = [&](int tok_id, float /*prob*/, void* /*ud*/) {
            const char* raw = kyutai_stt_token_text(ctx_, tok_id);
            if (!raw || !*raw)
                return;
            std::string piece(raw);
            size_t pos = 0;
            while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
                piece.replace(pos, 3, " ");
                pos++;
            }
            if (first_tok) {
                size_t sp = 0;
                while (sp < piece.size() && (piece[sp] == ' ' || piece[sp] == '\n'))
                    sp++;
                piece = piece.substr(sp);
                if (!piece.empty())
                    first_tok = false;
            }
            accumulated += piece;
            if (!accumulated.empty())
                on_text(accumulated.c_str(), false);
        };
        auto cb_fn = [](int id, float p, void* ud) { (*static_cast<decltype(cb)*>(ud))(id, p, nullptr); };
        kyutai_stt_transcribe_cb(ctx_, samples, n_samples, cb_fn, &cb);
        on_text(accumulated.c_str(), true);
    }

    void shutdown() override {
        if (ctx_) {
            kyutai_stt_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~KyutaiSttBackend() override { KyutaiSttBackend::shutdown(); }

private:
    kyutai_stt_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_kyutai_stt_backend() {
    return std::make_unique<KyutaiSttBackend>();
}

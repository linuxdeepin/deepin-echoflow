// crispasr_backend_canary_qwen.cpp — adapter for nvidia/canary-qwen-2.5b.
//
// SALM: FastConformer encoder + linear projection + Qwen3-1.7B LLM decoder.
// English ASR only. Wraps canary_qwen_init_from_file + canary_qwen_transcribe_ex.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "core/ngram_loop_fix.h"

#include "canary_qwen.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class CanaryQwenBackend : public CrispasrBackend {
public:
    CanaryQwenBackend() = default;
    ~CanaryQwenBackend() override { CanaryQwenBackend::shutdown(); }

    const char* name() const override { return "canary-qwen"; }

    // NO CAP_INTERNAL_CHUNKING (#290). The flag asserts that the backend slices
    // long audio itself, and src/canary_qwen.cpp does not: it has no chunker at
    // all, unlike src/parakeet.cpp and src/canary.cpp (which take a real
    // chunk_seconds). Declaring it disabled BOTH dispatcher safety nets, so a
    // long clip went through the encoder in a single pass:
    //   - with --chunk-seconds: the #257 gate in crispasr_run.cpp handed the
    //     whole clip over, trusting the internal chunker that isn't there;
    //   - without it: should_auto_chunk_long() bailed at "backend handles its
    //     own chunking", so the 30 s long-audio fallback never armed.
    // canary-qwen also does not prefers_vad(), so the auto-VAD safeguard was out
    // too — every net keyed off this one flag. Reported as no chunking, sparse
    // output, and RSS growing 384 MiB -> 6.3 GiB -> 10.2 GiB with clip length
    // (O(T^2) attention over an unchunked clip).
    // Overlap-save context stays blocked separately via the kBlocked list in
    // crispasr_chunk_context_gate.h (#218) — that gate is correct and unrelated.
    uint32_t capabilities() const override {
        return CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_AUTO_DOWNLOAD |
               CAP_UNBOUNDED_INPUT;
    }

    bool init(const whisper_params& p) override {
        canary_qwen_context_params cp = canary_qwen_context_default_params();
        cp.n_threads = p.n_threads;
        cp.flash_attn = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = canary_qwen_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[canary-qwen]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    void warmup() override {
        if (!ctx_)
            return;
        std::vector<float> silence(8000, 0.0f);
        canary_qwen_result* r = canary_qwen_transcribe_ex(ctx_, silence.data(), (int)silence.size());
        if (r)
            canary_qwen_result_free(r);
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        canary_qwen_set_temperature(ctx_, params.temperature, params.seed);
        canary_qwen_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);
        // #292: forward --max-new-tokens only when explicit; 0 keeps the backend default.
        canary_qwen_set_max_new_tokens(ctx_, params.max_new_tokens_explicit ? params.max_new_tokens : 0);

        canary_qwen_result* r = canary_qwen_transcribe_ex(ctx_, samples, n_samples);
        if (!r)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)(100.0 * n_samples / 16000.0);

        // Build token texts for dedup, then apply fix_loops to both text and tokens (#218)
        std::vector<std::string> tok_texts;
        for (int i = 0; i < r->n_tokens; i++)
            tok_texts.push_back(r->tokens[i].text ? r->tokens[i].text : "");
        const std::vector<int> keep = core_ngram::fix_loops_keep_indices(tok_texts);
        seg.text = core_ngram::fix_loops(r->text ? r->text : "");
        for (int ki : keep) {
            if (ki < 0 || ki >= r->n_tokens)
                continue;
            crispasr_token tok;
            tok.id = r->tokens[ki].id;
            tok.text = r->tokens[ki].text;
            tok.confidence = r->tokens[ki].p;
            tok.t0 = seg.t0;
            tok.t1 = seg.t1;
            seg.tokens.push_back(tok);
        }

        out.push_back(std::move(seg));
        canary_qwen_result_free(r);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            canary_qwen_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    canary_qwen_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_canary_qwen_backend() {
    return std::make_unique<CanaryQwenBackend>();
}

// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: MIT

// Shared Qwen3 prefix-stream model call for CLI and session bindings.
#pragma once
#include "qwen3_asr.h"
#include "crisp_audio.h"
#include "core/bpe.h"
#include "core/qwen3_prompt.h"
#include "core/qwen3_stream_metrics.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
namespace core_qwen3_stream_model {
inline int special_id(qwen3_asr_context* ctx, const char* token) {
    int n = 0;
    int32_t* ids = qwen3_asr_tokenize(ctx, token, &n);
    const int id = ids && n == 1 ? ids[0] : -1;
    free(ids);
    return id;
}
// Only cache tokens before the first audio placeholder. Their causal KV states
// are invariant across updates; audio and transcript-dependent positions are
// deliberately recomputed for every cumulative-audio decode.
struct PrefixKvCache {
    qwen3_asr_context* context = nullptr;
    int max_ctx = 0;
    uint64_t revision = 0;
    std::vector<int32_t> prefix_ids;
    bool ready = false;
};

inline int stable_prefix_length(const std::vector<int32_t>& ids, int audio_pad_id) {
    const auto first_audio = std::find(ids.begin(), ids.end(), audio_pad_id);
    return first_audio == ids.end() ? 0 : static_cast<int>(first_audio - ids.begin());
}

inline bool can_reuse_prefix_kv(const PrefixKvCache& cache, qwen3_asr_context* context,
                                int required_ctx, const std::vector<int32_t>& ids,
                                int prefix_len, uint64_t revision = 0) {
    return prefix_len > 0 && prefix_len <= static_cast<int>(ids.size())
        && cache.ready && cache.context == context && cache.max_ctx >= required_ctx
        && cache.revision == revision
        && cache.prefix_ids == std::vector<int32_t>(ids.begin(), ids.begin() + prefix_len);
}

inline std::string generate(qwen3_asr_context* ctx_, bool raon_, const std::vector<float>& audio,
                            const std::string& assistant_suffix, int max_new, bool* success = nullptr,
                            PrefixKvCache* prefix_cache = nullptr, DecodeMetrics* metrics = nullptr,
                            bool split_prefill = false, crisp_audio_encoder_cache* encoder_cache = nullptr,
                            const std::string& hotwords = {}) {
    DecodeProfile profile(metrics, static_cast<std::int64_t>(audio.size()));
    const auto run_llm = [&](const float* embeds, int count, int past, int* tokens, int* vocab) {
        auto* result = qwen3_asr_run_llm_kv(ctx_, embeds, count, past, tokens, vocab);
        const auto timing = qwen3_asr_last_llm_metrics(ctx_);
        profile.metrics().llm_graph_ms += timing.graph_ms;
        profile.metrics().llm_alloc_ms += timing.alloc_ms;
        profile.metrics().llm_compute_ms += timing.compute_ms;
        profile.metrics().llm_transfer_ms += timing.transfer_ms;
        return result;
    };
    if (success)
        *success = false;
    if (!ctx_ || audio.empty())
        return {};
    int N_enc = 0, pdim = 0;
    float* audio_embeds = nullptr;
    if (raon_) {
        profile.phase(&DecodeMetrics::encoder_ms);
        audio_embeds = qwen3_asr_raon_encode(ctx_, audio.data(), (int)audio.size(), &N_enc, &pdim);
    } else {
        int nm = 0, nt = 0;
        profile.phase(&DecodeMetrics::mel_ms);
        float* mel = qwen3_asr_compute_mel(ctx_, audio.data(), (int)audio.size(), &nm, &nt);
        if (mel) {
            profile.phase(&DecodeMetrics::encoder_ms);
            audio_embeds = encoder_cache
                ? qwen3_asr_run_encoder_cached(ctx_, encoder_cache, mel, nm, nt, &N_enc, &pdim)
                : qwen3_asr_run_encoder(ctx_, mel, nm, nt, &N_enc, &pdim);
            if (encoder_cache) {
                const auto stats = crisp_audio_encoder_cache_get_stats(encoder_cache);
                profile.metrics().stem_reused = stats.reused_stem_chunks;
                profile.metrics().stem_computed = stats.computed_stem_chunks;
                profile.metrics().encoder_regions_reused = stats.reused_encoder_regions;
                profile.metrics().encoder_regions_computed = stats.computed_encoder_regions;
                profile.metrics().encoder_window_chunks = stats.window_chunks;
                profile.metrics().encoder_cached_bytes = stats.cached_bytes;
            }
            free(mel);
        }
    }
    profile.phase(&DecodeMetrics::prepare_ms);
    profile.metrics().audio_frames = N_enc;
    if (!audio_embeds)
        return {};
    std::string text;
    if (raon_) {
        text = "<|im_start|>user\n<|audio_start|>";
        for (int i = 0; i < N_enc; ++i)
            text += "<|audio_pad|>";
        text += "<|audio_end|>Transcribe the audio into text<|im_end|>\n<|im_start|>assistant\n" + assistant_suffix;
    } else {
        text = core_qwen3_prompt::build(N_enc, {}, {}, assistant_suffix, hotwords);
    }
    int n_prompt = 0;
    int32_t* raw_ids = qwen3_asr_tokenize(ctx_, text.c_str(), &n_prompt);
    if (!raw_ids) {
        free(audio_embeds);
        return {};
    }
    std::vector<int32_t> ids(raw_ids, raw_ids + n_prompt);
    free(raw_ids);
    const int audio_pad_id = special_id(ctx_, "<|audio_pad|>");
    const int prompt_len = (int)ids.size();
    profile.metrics().prompt_tokens = prompt_len;
    const int prefix_len = prefix_cache && audio_pad_id >= 0
        ? stable_prefix_length(ids, audio_pad_id) : 0;
    const int required_ctx = std::max(4096, prompt_len + max_new + 16);
    if (audio_pad_id < 0 || !qwen3_asr_kv_init(ctx_, required_ctx)) {
        free(audio_embeds);
        return {};
    }

    const bool reuse_prefix = prefix_cache && can_reuse_prefix_kv(
        *prefix_cache, ctx_, required_ctx, ids, prefix_len, qwen3_asr_kv_revision(ctx_));
    profile.metrics().prefix_reused = reuse_prefix;
    const int embedding_start = reuse_prefix ? prefix_len : 0;
    float* emb = qwen3_asr_embed_tokens(ctx_, ids.data() + embedding_start,
                                        prompt_len - embedding_start);
    if (!emb) {
        free(audio_embeds);
        return {};
    }
    int spliced = 0;
    for (int i = embedding_start; i < prompt_len && spliced < N_enc; ++i) {
        if (ids[i] == audio_pad_id) {
            std::memcpy(emb + (size_t)(i - embedding_start) * pdim,
                        audio_embeds + (size_t)(spliced++) * pdim, pdim * sizeof(float));
        }
    }
    free(audio_embeds);

    int n_t = 0, vocab = 0;
    float* logits = nullptr;
    profile.phase(&DecodeMetrics::prefill_ms);
    if (reuse_prefix) {
        logits = run_llm(emb, prompt_len - prefix_len, prefix_len, &n_t, &vocab);
    } else {
        if (prefix_cache) prefix_cache->ready = false;
        qwen3_asr_kv_reset(ctx_);
        // The full causal prefill also populates the fixed-prefix KV. Avoid a
        // second forward just to create it. Keep split prefill for parity/A/B.
        if (split_prefill && prefix_cache && prefix_len > 0) {
            logits = run_llm(emb, prefix_len, 0, &n_t, &vocab);
            if (!logits) {
                free(emb);
                return {};
            }
            free(logits);
            logits = run_llm(emb + static_cast<size_t>(prefix_len) * pdim,
                             prompt_len - prefix_len, prefix_len, &n_t, &vocab);
        } else {
            logits = run_llm(emb, prompt_len, 0, &n_t, &vocab);
        }
        if (logits && prefix_cache && prefix_len > 0) {
            prefix_cache->context = ctx_;
            prefix_cache->max_ctx = required_ctx;
            prefix_cache->prefix_ids.assign(ids.begin(), ids.begin() + prefix_len);
            prefix_cache->ready = true;
        }
    }
    free(emb);
    if (!logits)
        return {};
    profile.phase(&DecodeMetrics::generate_ms);
    // generation_config.eos_token_id = [<|endoftext|>, <|im_end|>]
    const int eos_a = special_id(ctx_, "<|im_end|>"), eos_b = special_id(ctx_, "<|endoftext|>");
    auto argmax = [vocab](const float* row) {
        int best = 0;
        for (int v = 1; v < vocab; v++)
            if (row[v] > row[best])
                best = v;
        return best;
    };
    std::string bytes;
    static const bool trace_ids =
        getenv("CRISPASR_QWEN3_STREAM_TRACE") && atoi(getenv("CRISPASR_QWEN3_STREAM_TRACE")) >= 2;
    if (trace_ids)
        fprintf(stderr, "QWEN3_STREAM_IDS n_audio=%zu n_enc=%d prompt_len=%d ids=", audio.size(), N_enc, prompt_len);
    int id = argmax(logits + (size_t)(n_t - 1) * vocab);
    free(logits);
    int n_past = prompt_len;
    for (int step = 0; step < max_new; step++) {
        if (trace_ids)
            fprintf(stderr, "%d,", id);
        if (id == eos_a || id == eos_b)
            break;
        ++profile.metrics().generated_tokens;
        const char* piece = qwen3_asr_token_text(ctx_, id);
        // skip_special_tokens: the special-flagged added tokens are the
        // <|...|> family; <asr_text> / <non_speech> are NOT special and stay.
        if (piece && !(piece[0] == '<' && piece[1] == '|'))
            bytes += core_bpe::token_bytes_to_utf8(piece);
        if (step + 1 == max_new)
            break;
        float* te = qwen3_asr_embed_tokens(ctx_, &id, 1);
        if (!te)
            return {};
        int nt2 = 0, v2 = 0;
        float* lg = run_llm(te, 1, n_past, &nt2, &v2);
        free(te);
        if (!lg)
            return {};
        n_past++;
        id = argmax(lg);
        free(lg);
    }
    if (trace_ids)
        fprintf(stderr, "\n");
    if (success)
        *success = true;
    profile.metrics().success = true;
    if (prefix_cache && prefix_cache->ready) {
        prefix_cache->revision = qwen3_asr_kv_revision(ctx_);
    }
    return bytes;
}

} // namespace core_qwen3_stream_model

// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace core_qwen3_stream_model {

struct DecodeMetrics {
    double mel_ms = 0;
    double encoder_ms = 0;
    double prepare_ms = 0;
    double prefill_ms = 0;
    double generate_ms = 0;
    double total_ms = 0;
    double llm_graph_ms = 0;
    double llm_alloc_ms = 0;
    double llm_compute_ms = 0;
    double llm_transfer_ms = 0;
    std::int64_t audio_samples = 0;
    int audio_frames = 0;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int stem_reused = 0;
    int stem_computed = 0;
    int encoder_regions_reused = 0;
    int encoder_regions_computed = 0;
    int encoder_window_chunks = 0;
    size_t encoder_cached_bytes = 0;
    bool prefix_reused = false;
    bool success = false;
};

// One profile per actual generation, including unchanged text and failures.
// No transcript or audio is written to the diagnostic log.
class DecodeProfile {
public:
    using Clock = std::chrono::steady_clock;
    explicit DecodeProfile(DecodeMetrics* output, std::int64_t samples)
        : output_(output ? output : &local_), started_(Clock::now()), phaseStarted_(started_)
    {
        *output_ = {};
        output_->audio_samples = samples;
        phase_ = &output_->prepare_ms;
    }
    ~DecodeProfile()
    {
        const auto now = Clock::now();
        finishPhase(now);
        output_->total_ms = elapsed(started_, now);
        const char* enabled = std::getenv("CRISPASR_QWEN3_STREAM_PROFILE");
        if (enabled && std::atoi(enabled) != 0) {
            std::fprintf(stderr,
                "QWEN3_STREAM_PERF audio_ms=%.3f audio_frames=%d prompt_tokens=%d "
                "generated_tokens=%d mel_ms=%.3f encoder_ms=%.3f prepare_ms=%.3f "
                "prefill_ms=%.3f generate_ms=%.3f total_ms=%.3f llm_graph_ms=%.3f "
                "llm_alloc_ms=%.3f llm_compute_ms=%.3f llm_transfer_ms=%.3f "
                "stem_reused=%d stem_computed=%d encoder_regions_reused=%d encoder_regions_computed=%d "
                "encoder_window_chunks=%d encoder_cached_bytes=%zu prefix_reused=%d success=%d\n",
                output_->audio_samples / 16.0, output_->audio_frames, output_->prompt_tokens,
                output_->generated_tokens, output_->mel_ms, output_->encoder_ms, output_->prepare_ms,
                output_->prefill_ms, output_->generate_ms, output_->total_ms,
                output_->llm_graph_ms, output_->llm_alloc_ms, output_->llm_compute_ms,
                output_->llm_transfer_ms, output_->stem_reused, output_->stem_computed,
                output_->encoder_regions_reused, output_->encoder_regions_computed,
                output_->encoder_window_chunks, output_->encoder_cached_bytes,
                output_->prefix_reused, output_->success);
        }
    }
    DecodeProfile(const DecodeProfile&) = delete;
    DecodeProfile& operator=(const DecodeProfile&) = delete;
    DecodeMetrics& metrics() { return *output_; }
    void phase(double DecodeMetrics::*member)
    {
        const auto now = Clock::now();
        finishPhase(now);
        phase_ = &(output_->*member);
        phaseStarted_ = now;
    }
private:
    static double elapsed(Clock::time_point start, Clock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    void finishPhase(Clock::time_point now) { *phase_ += elapsed(phaseStarted_, now); }
    DecodeMetrics local_;
    DecodeMetrics* output_;
    Clock::time_point started_;
    Clock::time_point phaseStarted_;
    double* phase_;
};

} // namespace core_qwen3_stream_model

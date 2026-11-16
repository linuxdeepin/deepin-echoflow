// crispasr_watermark_dispatch.h — unified watermark dispatch.
//
// Routes watermark embed/detect to either:
//   1. AudioSeal neural watermark (when --watermark-model is set and GGUF loaded)
//   2. Built-in spread-spectrum watermark (fallback)
//
// The dispatcher is initialized once at startup. Thread-safe after init.

#pragma once

#include "audioseal.h"
#include "core/crispasr_watermark.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace crispasr_wm_dispatch {

// Global AudioSeal context (null = not loaded → use spread-spectrum)
inline audioseal_ctx*& get_ctx() {
    static audioseal_ctx* ctx = nullptr;
    return ctx;
}

// Operator opt-out flag, set once at startup from the --no-watermark CLI flag.
// The CRISPASR_NO_WATERMARK env var is an equivalent opt-out (both are honored;
// neither is more "official" than the other). Either path disables watermark
// embedding for the process.
inline bool& disabled_flag() {
    static bool disabled = false;
    return disabled;
}

// Set from the --no-watermark CLI flag at startup.
inline void set_disabled(bool value) {
    disabled_flag() = value;
}

// Watertight override. When set, is_disabled() always returns false — even
// against --no-watermark / CRISPASR_NO_WATERMARK. The CLI sets this for outputs
// that cannot carry a C2PA manifest (raw ADTS .aac / Ogg .opus, and the raw PCM
// --tts-stream), where the audio watermark is the ONLY robust machine-readable
// AI mark. This guarantees no CLI path can ever emit a fully unmarked AI file:
// the watermark opt-out is honored only when C2PA still marks the output.
inline bool& forced_flag() {
    static bool forced = false;
    return forced;
}

// Force watermarking on for this process regardless of the opt-out.
inline void set_forced(bool value) {
    forced_flag() = value;
}

// True if watermarking has been turned off via the CLI flag or the env var,
// UNLESS a watertight override forces it on.
//
// `force` is the per-call equivalent of set_forced(), for callers that decide
// per output rather than per process. The server needs that: its response
// format is chosen per request, so which outputs can carry a C2PA manifest —
// and therefore which ones the watermark is the ONLY mark on — varies request
// to request, and mutating the process-global flag would race across
// --server-workers threads.
inline bool is_disabled(bool force = false) {
    if (force || forced_flag())
        return false;
    return disabled_flag() || std::getenv("CRISPASR_NO_WATERMARK") != nullptr;
}

// Initialize AudioSeal from GGUF path. Call once at startup.
// Returns true if loaded, false if not (falls back to spread-spectrum).
inline bool init(const std::string& model_path) {
    if (model_path.empty())
        return false;

    auto params = audioseal_default_params();
    params.verbosity = 1;
    get_ctx() = audioseal_init_from_file(model_path.c_str(), params);
    if (!get_ctx()) {
        fprintf(stderr,
                "crispasr: warning: could not load AudioSeal model '%s'; "
                "falling back to spread-spectrum watermark\n",
                model_path.c_str());
        return false;
    }
    return true;
}

inline void shutdown() {
    if (get_ctx()) {
        audioseal_free(get_ctx());
        get_ctx() = nullptr;
    }
}

// Embed watermark into float32 PCM. Modifies in-place.
// If AudioSeal is loaded, resamples to 16kHz if needed, embeds, and
// resamples back. Otherwise uses spread-spectrum.
//
// Watermarking is on by default. It can be turned off with the --no-watermark
// CLI flag or the CRISPASR_NO_WATERMARK env var (equivalent opt-outs). Either
// way we log a one-time warning: disabling the mark shifts the AI-content
// disclosure/marking duty onto the operator (see docs/issue-260/PLAN.md for the
// regulatory background, incl. EU AI Act Art. 50, which we intentionally do NOT
// name at runtime — the obligation is jurisdiction-specific and the operator,
// not this binary, is the party bound by it).
//
// Pass force=true when this particular output cannot carry a C2PA manifest, so
// stripping the watermark would leave it with no machine-readable AI mark at
// all. That is the watertight floor; see crispasr_enforce_cli_watermark_floor()
// for the CLI's process-wide equivalent.
inline void embed(float* pcm, int n_samples, int sample_rate = 24000, bool force = false) {
    if (is_disabled(force)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "crispasr: warning: watermarking disabled. "
                            "AI usage marking responsibility rests with the operator.\n");
        }
        return; // opt-out honored; warning emitted once per process
    }
    if (get_ctx()) {
        // AudioSeal operates at 16 kHz. If audio is at a different rate,
        // resample down, embed, resample back.
        if (sample_rate == 16000) {
            // Direct: embed in-place
            float* watermarked = audioseal_embed(get_ctx(), pcm, n_samples, nullptr);
            if (watermarked) {
                std::memcpy(pcm, watermarked, (size_t)n_samples * sizeof(float));
                std::free(watermarked);
                return;
            }
            // Fall through to spread-spectrum on failure
        } else {
            // Resample to 16 kHz
            int n_16k = (int)((int64_t)n_samples * 16000 / sample_rate);
            std::vector<float> resampled(n_16k);
            for (int i = 0; i < n_16k; i++) {
                float pos = (float)i * (float)sample_rate / 16000.0f;
                int s0 = (int)pos;
                int s1 = (s0 + 1 < n_samples) ? s0 + 1 : s0;
                float frac = pos - (float)s0;
                resampled[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
            }

            // Embed at 16 kHz
            float* watermarked = audioseal_embed(get_ctx(), resampled.data(), n_16k, nullptr);
            if (watermarked) {
                // Compute watermark delta at 16 kHz
                for (int i = 0; i < n_16k; i++)
                    resampled[i] = watermarked[i] - resampled[i];
                std::free(watermarked);

                // Upsample the watermark delta back to original rate and add
                for (int i = 0; i < n_samples; i++) {
                    float pos = (float)i * 16000.0f / (float)sample_rate;
                    int s0 = (int)pos;
                    int s1 = (s0 + 1 < n_16k) ? s0 + 1 : s0;
                    float frac = pos - (float)s0;
                    float delta = resampled[s0] * (1.0f - frac) + resampled[s1] * frac;
                    pcm[i] += delta;
                }
                return;
            }
        }
    }
    // Fallback: spread-spectrum watermark
    crispasr_watermark_embed_impl(pcm, n_samples);
}

// Detect watermark in float32 PCM.
// Returns confidence in [0, 1]. If AudioSeal is loaded, resamples and
// uses neural detection. Otherwise uses spread-spectrum.
inline float detect(const float* pcm, int n_samples, int sample_rate = 24000) {
    if (get_ctx()) {
        // Resample to 16 kHz if needed
        std::vector<float> pcm_16k;
        const float* detect_pcm = pcm;
        int detect_n = n_samples;
        if (sample_rate != 16000) {
            detect_n = (int)((int64_t)n_samples * 16000 / sample_rate);
            pcm_16k.resize(detect_n);
            for (int i = 0; i < detect_n; i++) {
                float pos = (float)i * (float)sample_rate / 16000.0f;
                int s0 = (int)pos;
                int s1 = (s0 + 1 < n_samples) ? s0 + 1 : s0;
                float frac = pos - (float)s0;
                pcm_16k[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
            }
            detect_pcm = pcm_16k.data();
        }
        int n_frames = 0;
        float* probs = audioseal_detect(get_ctx(), detect_pcm, detect_n, &n_frames, nullptr);
        if (probs && n_frames > 0) {
            // Average detection probability across frames
            double avg = 0.0;
            for (int i = 0; i < n_frames; i++)
                avg += probs[i];
            avg /= (double)n_frames;
            std::free(probs);
            return (float)avg;
        }
        if (probs)
            std::free(probs);
    }
    // Fallback: spread-spectrum. Goes through the selector so this surface and
    // the session C-ABI cannot end up on different statistics.
    return crispasr_watermark_detect_select(pcm, n_samples);
}

} // namespace crispasr_wm_dispatch

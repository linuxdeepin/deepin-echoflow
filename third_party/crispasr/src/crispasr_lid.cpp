// crispasr_lid.cpp — shared LID implementation.
// See crispasr_lid.h.
//
// Extracted from examples/cli/crispasr_lid.cpp. The whisper-tiny and
// native silero paths are algorithmic — they belong in the library.
// The sherpa-onnx subprocess fallback + model auto-download stay in
// the CLI shim.

#include "crispasr_lid.h"
#include "ecapa_lid.h"
#include "firered_lid.h"
#include "silero_lid.h"
#include "crispasr.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Process-lifetime cache: keep the whisper LID context around between
// invocations so batch runs (multiple files, or multiple slices of one
// long input) don't re-load the 75 MB ggml-tiny every time. Cache is
// keyed on the model path + relevant cparams — if those change we free
// the old context and load fresh.
struct WhisperLidCache {
    whisper_context* ctx = nullptr;
    std::string model_path;
    bool use_gpu = false;
    int gpu_device = 0;
    bool flash_attn = true;
};

WhisperLidCache& whisper_lid_cache() {
    static WhisperLidCache c;
    return c;
}

// Same process-lifetime caching for the silero / firered / ecapa LID models —
// previously each was init'd + freed on every detect() call, so a server with
// `language=auto` reloaded the model per request (#165). Keyed on model path +
// thread count; freed by crispasr_lid_free_cache() at shutdown.
struct SileroLidCache {
    silero_lid_context* ctx = nullptr;
    std::string model_path;
    int n_threads = 0;
};
struct FireredLidCache {
    firered_lid_context* ctx = nullptr;
    std::string model_path;
    int n_threads = 0;
};
struct EcapaLidCache {
    ecapa_lid_context* ctx = nullptr;
    std::string model_path;
    int n_threads = 0;
};
SileroLidCache& silero_lid_cache() {
    static SileroLidCache c;
    return c;
}
FireredLidCache& firered_lid_cache() {
    static FireredLidCache c;
    return c;
}
EcapaLidCache& ecapa_lid_cache() {
    static EcapaLidCache c;
    return c;
}

bool detect_whisper(const float* samples, int n_samples, const CrispasrLidOptions& opts, CrispasrLidResult& out) {
    if (opts.model_path.empty())
        return false;

    WhisperLidCache& c = whisper_lid_cache();
    const bool cache_miss = (c.ctx == nullptr) || (c.model_path != opts.model_path) || (c.use_gpu != opts.use_gpu) ||
                            (c.gpu_device != opts.gpu_device) || (c.flash_attn != opts.flash_attn);

    if (cache_miss) {
        if (c.ctx) {
            whisper_free(c.ctx);
            c.ctx = nullptr;
        }
        whisper_context_params cp = whisper_context_default_params();
        cp.use_gpu = opts.use_gpu;
        cp.gpu_device = opts.gpu_device;
        cp.flash_attn = opts.flash_attn;

        c.ctx = whisper_init_from_file_with_params(opts.model_path.c_str(), cp);
        if (!c.ctx) {
            if (opts.verbose)
                fprintf(stderr, "crispasr[lid]: failed to load '%s'\n", opts.model_path.c_str());
            return false;
        }

        if (!whisper_is_multilingual(c.ctx)) {
            if (opts.verbose)
                fprintf(stderr, "crispasr[lid]: model '%s' is English-only\n", opts.model_path.c_str());
            whisper_free(c.ctx);
            c.ctx = nullptr;
            return false;
        }

        c.model_path = opts.model_path;
        c.use_gpu = opts.use_gpu;
        c.gpu_device = opts.gpu_device;
        c.flash_attn = opts.flash_attn;
    }

    whisper_context* ctx = c.ctx;

    // Whisper's encoder expects exactly 30 s (480 000 samples). Pad with
    // zeros when shorter; truncate when longer. LID only looks at the
    // first 30 s anyway.
    constexpr int SR = 16000;
    constexpr int NEED = SR * 30;
    std::vector<float> pcm((size_t)NEED, 0.0f);
    const int n_use = std::min(n_samples, NEED);
    std::memcpy(pcm.data(), samples, (size_t)n_use * sizeof(float));

    if (whisper_pcm_to_mel(ctx, pcm.data(), NEED, opts.n_threads) != 0) {
        if (opts.verbose)
            fprintf(stderr, "crispasr[lid]: pcm_to_mel failed\n");
        return false;
    }
    if (whisper_encode(ctx, 0, opts.n_threads) != 0) {
        if (opts.verbose)
            fprintf(stderr, "crispasr[lid]: encode failed\n");
        return false;
    }

    const int n_langs = whisper_lang_max_id() + 1;
    std::vector<float> probs((size_t)n_langs, 0.0f);
    const int lang_id = whisper_lang_auto_detect(ctx, /*offset_ms=*/0, opts.n_threads, probs.data());
    if (lang_id < 0 || lang_id >= n_langs) {
        if (opts.verbose)
            fprintf(stderr, "crispasr[lid]: whisper_lang_auto_detect failed\n");
        return false;
    }

    out.lang_code = whisper_lang_str(lang_id);
    out.confidence = probs[lang_id];
    out.source = "whisper";

    if (opts.verbose) {
        fprintf(stderr, "crispasr[lid]: detected '%s' (p=%.3f) via whisper\n", out.lang_code.c_str(), out.confidence);
    }
    return true;
}

bool detect_silero(const float* samples, int n_samples, const CrispasrLidOptions& opts, CrispasrLidResult& out) {
    if (opts.model_path.empty())
        return false;

    SileroLidCache& sc = silero_lid_cache();
    if (!sc.ctx || sc.model_path != opts.model_path || sc.n_threads != opts.n_threads) {
        if (sc.ctx) {
            silero_lid_free(sc.ctx);
            sc.ctx = nullptr;
        }
        sc.ctx = silero_lid_init(opts.model_path.c_str(), opts.n_threads);
        if (!sc.ctx) {
            if (opts.verbose)
                fprintf(stderr, "crispasr[lid]: silero_lid_init('%s') failed\n", opts.model_path.c_str());
            return false;
        }
        sc.model_path = opts.model_path;
        sc.n_threads = opts.n_threads;
    }
    silero_lid_context* lid = sc.ctx;

    float conf = 0.0f;
    const char* lang = silero_lid_detect(lid, samples, n_samples, &conf);
    std::string code = lang ? lang : "";

    if (code.empty()) {
        if (opts.verbose)
            fprintf(stderr, "crispasr[lid]: silero_lid_detect returned no code\n");
        return false;
    }

    // Silero labels are "xx, Name" (e.g. "de, German") — extract ISO code only.
    auto comma = code.find(',');
    if (comma != std::string::npos)
        code.resize(comma);

    out.lang_code = code;
    out.confidence = conf;
    out.source = "silero";
    if (opts.verbose) {
        fprintf(stderr, "crispasr[lid]: detected '%s' (p=%.3f) via silero\n", out.lang_code.c_str(), out.confidence);
    }
    return true;
}

bool detect_firered(const float* samples, int n_samples, const CrispasrLidOptions& opts, CrispasrLidResult& out) {
    if (opts.model_path.empty())
        return false;

    FireredLidCache& fc = firered_lid_cache();
    if (!fc.ctx || fc.model_path != opts.model_path || fc.n_threads != opts.n_threads) {
        if (fc.ctx) {
            firered_lid_free(fc.ctx);
            fc.ctx = nullptr;
        }
        fc.ctx = firered_lid_init(opts.model_path.c_str(), opts.n_threads);
        if (!fc.ctx) {
            if (opts.verbose)
                fprintf(stderr, "crispasr[lid]: firered_lid_init('%s') failed\n", opts.model_path.c_str());
            return false;
        }
        fc.model_path = opts.model_path;
        fc.n_threads = opts.n_threads;
    }
    firered_lid_context* lid = fc.ctx;

    float conf = 0.0f;
    const char* lang = firered_lid_detect(lid, samples, n_samples, &conf);
    std::string code = lang ? lang : "";

    if (code.empty()) {
        if (opts.verbose)
            fprintf(stderr, "crispasr[lid]: firered_lid_detect returned no code\n");
        return false;
    }

    out.lang_code = code;
    out.confidence = conf;
    out.source = "firered";
    if (opts.verbose) {
        fprintf(stderr, "crispasr[lid]: detected '%s' (p=%.3f) via firered\n", out.lang_code.c_str(), out.confidence);
    }
    return true;
}

bool detect_ecapa(const float* samples, int n_samples, const CrispasrLidOptions& opts, CrispasrLidResult& out) {
    if (opts.model_path.empty())
        return false;

    EcapaLidCache& ec = ecapa_lid_cache();
    if (!ec.ctx || ec.model_path != opts.model_path || ec.n_threads != opts.n_threads) {
        if (ec.ctx) {
            ecapa_lid_free(ec.ctx);
            ec.ctx = nullptr;
        }
        ec.ctx = ecapa_lid_init(opts.model_path.c_str(), opts.n_threads);
        if (!ec.ctx) {
            if (opts.verbose)
                fprintf(stderr, "crispasr[lid]: ecapa_lid_init('%s') failed\n", opts.model_path.c_str());
            return false;
        }
        ec.model_path = opts.model_path;
        ec.n_threads = opts.n_threads;
    }
    ecapa_lid_context* lid = ec.ctx;

    float conf = 0.0f;
    const char* lang = ecapa_lid_detect(lid, samples, n_samples, &conf);
    std::string code = lang ? lang : "";

    if (code.empty()) {
        if (opts.verbose)
            fprintf(stderr, "crispasr[lid]: ecapa_lid_detect returned no code\n");
        return false;
    }

    out.lang_code = code;
    out.confidence = conf;
    out.source = "ecapa";
    if (opts.verbose)
        fprintf(stderr, "crispasr[lid]: detected '%s' (p=%.3f) via ecapa\n", out.lang_code.c_str(), out.confidence);
    return true;
}

} // namespace

bool crispasr_detect_language(const float* samples, int n_samples, const CrispasrLidOptions& opts,
                              CrispasrLidResult& out) {
    if (!samples || n_samples <= 0)
        return false;
    // Truncate to 15 s — LID models don't benefit from longer audio, and
    // processing the full file wastes time on long recordings.
    constexpr int kLidMaxSamples = 16000 * 15;
    if (n_samples > kLidMaxSamples)
        n_samples = kLidMaxSamples;
    switch (opts.method) {
    case CrispasrLidMethod::Whisper:
        return detect_whisper(samples, n_samples, opts, out);
    case CrispasrLidMethod::Silero:
        return detect_silero(samples, n_samples, opts, out);
    case CrispasrLidMethod::Firered:
        return detect_firered(samples, n_samples, opts, out);
    case CrispasrLidMethod::Ecapa:
        return detect_ecapa(samples, n_samples, opts, out);
    }
    return false;
}

std::vector<float> crispasr_lid_speech_prefix(const std::vector<float>& pcm,
                                              const std::vector<crispasr_audio_slice>& slices) {
    constexpr size_t kMaxSamples = 16000u * 15u;
    constexpr size_t kGapSamples = 1600u;
    std::vector<float> out;
    out.reserve(std::min(kMaxSamples, pcm.size()));

    for (const auto& slice : slices) {
        const int start = std::max(0, slice.start);
        const int end = std::min((int)pcm.size(), slice.end);
        if (end <= start || out.size() >= kMaxSamples)
            continue;
        if (!out.empty())
            out.insert(out.end(), std::min(kGapSamples, kMaxSamples - out.size()), 0.0f);
        const size_t remaining = kMaxSamples - out.size();
        const size_t count = std::min(remaining, (size_t)(end - start));
        out.insert(out.end(), pcm.begin() + start, pcm.begin() + start + count);
    }
    return out;
}

void crispasr_lid_free_cache() {
    WhisperLidCache& c = whisper_lid_cache();
    if (c.ctx) {
        whisper_free(c.ctx);
        c.ctx = nullptr;
        c.model_path.clear();
    }
    SileroLidCache& sc = silero_lid_cache();
    if (sc.ctx) {
        silero_lid_free(sc.ctx);
        sc.ctx = nullptr;
        sc.model_path.clear();
    }
    FireredLidCache& fc = firered_lid_cache();
    if (fc.ctx) {
        firered_lid_free(fc.ctx);
        fc.ctx = nullptr;
        fc.model_path.clear();
    }
    EcapaLidCache& ec = ecapa_lid_cache();
    if (ec.ctx) {
        ecapa_lid_free(ec.ctx);
        ec.ctx = nullptr;
        ec.model_path.clear();
    }
}

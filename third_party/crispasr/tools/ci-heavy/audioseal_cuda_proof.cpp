// Standalone #482 proof: requires a real CUDA device and refuses CPU fallback.
#include "audioseal.h"
#include "core/gpu_backend_pref.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

using Buffer = std::unique_ptr<float, decltype(&std::free)>;
using Context = std::unique_ptr<audioseal_ctx, decltype(&audioseal_free)>;

static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

static std::vector<float> signal(int seconds) {
    std::vector<float> pcm(seconds * 16000);
    for (size_t i = 0; i < pcm.size(); ++i)
        pcm[i] = 0.3f * std::sin(2.0f * 3.14159265f * 440.0f * (float)i / 16000.0f);
    return pcm;
}

static Context load(const char* path, bool gpu) {
    auto p = audioseal_default_params();
    p.use_gpu = gpu;
    Context ctx(audioseal_init_from_file(path, p), audioseal_free);
    require(bool(ctx), "model load failed");
    return ctx;
}

static double detect(audioseal_ctx* ctx, const float* pcm, int count, bool message) {
    int n = 0;
    uint8_t bits[16] = {};
    Buffer probs(audioseal_detect(ctx, pcm, count, &n, message ? bits : nullptr), std::free);
    require(bool(probs), "detect returned null");
    require(n == count, "detection length mismatch");
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        require(std::isfinite(probs.get()[i]) && probs.get()[i] >= 0 && probs.get()[i] <= 1,
                "invalid detection probability");
        sum += probs.get()[i];
    }
    return sum / n;
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: audioseal-cuda-proof audioseal.gguf");
        // Explicitly exercise the CUDA API before the model loader's fallback.
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        require(backend && ggml_backend_is_cuda(backend), "CUDA device unavailable");
        printf("CUDA backend: %s\n", ggml_backend_name(backend));
        ggml_backend_free(backend);
        crispasr_set_gpu_backend_pref("cuda");

        // One short CPU reference is part of the GPU numerical comparison.
        auto short_pcm = signal(1);
        auto cpu = load(argv[1], false);
        Buffer reference(audioseal_embed(cpu.get(), short_pcm.data(), (int)short_pcm.size(), nullptr), std::free);
        require(bool(reference), "CPU reference embed failed");
        cpu.reset();

        auto gpu = load(argv[1], true);
        for (int seconds : {1, 4, 10, 1}) {
            auto pcm = signal(seconds);
            // Build/grow the detector first, then the larger generator, on one context.
            const double clean = detect(gpu.get(), pcm.data(), (int)pcm.size(), true);
            require(clean < 0.5, "clean audio classified as watermarked");
            Buffer wm(audioseal_embed(gpu.get(), pcm.data(), (int)pcm.size(), nullptr), std::free);
            require(bool(wm), "GPU embed returned null");
            double energy = 0;
            for (size_t i = 0; i < pcm.size(); ++i) {
                require(std::isfinite(wm.get()[i]), "non-finite watermark output");
                const double d = wm.get()[i] - pcm[i];
                energy += d * d;
            }
            require(energy > 0, "empty watermark");
            const double score = detect(gpu.get(), wm.get(), (int)pcm.size(), false);
            require(score > 0.9, "watermark not recovered");
            if (seconds == 1) {
                double dot = 0, cpu_norm = 0, gpu_norm = 0;
                for (size_t i = 0; i < pcm.size(); ++i) {
                    const double a = reference.get()[i] - pcm[i];
                    const double b = wm.get()[i] - pcm[i];
                    dot += a * b;
                    cpu_norm += a * a;
                    gpu_norm += b * b;
                }
                require(cpu_norm > 0 && gpu_norm > 0, "zero watermark norm");
                const double cosine = dot / std::sqrt(cpu_norm * gpu_norm);
                const double ratio = std::sqrt(gpu_norm / cpu_norm);
                printf("short watermark parity: cosine=%.9f norm_ratio=%.9f\n", cosine, ratio);
                require(cosine > 0.999 && std::abs(ratio - 1) < 0.01, "CPU/GPU watermark parity failed");
            }
            printf("duration=%ds samples=%zu clean=%.6f watermarked=%.6f PASS\n", seconds, pcm.size(), clean, score);
            fflush(stdout);
        }
        puts("AUDIOSEAL_CUDA_PASS");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "AUDIOSEAL_CUDA_FAIL: %s\n", e.what());
        return 1;
    }
}

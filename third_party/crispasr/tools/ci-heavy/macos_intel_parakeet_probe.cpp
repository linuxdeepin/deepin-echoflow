// Stage and warmed encoder proof for the Intel release ISA flags (#484).
#include "parakeet.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4)
        return 2;
    FILE* f = fopen(argv[2], "rb");
    if (!f)
        return 2;
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    rewind(f);
    std::vector<float> pcm(bytes / sizeof(float));
    if (fread(pcm.data(), sizeof(float), pcm.size(), f) != pcm.size())
        return 2;
    fclose(f);
    auto params = parakeet_context_default_params();
    params.n_threads = 4;
    params.use_gpu = false;
    params.verbosity = 0;
    auto* ctx = parakeet_init_from_file(argv[1], params);
    if (!ctx)
        return 1;
    printf("ISA avx2=%d fma=%d f16c=%d bmi2=%d avx512=%d\n", ggml_cpu_has_avx2(), ggml_cpu_has_fma(),
           ggml_cpu_has_f16c(), ggml_cpu_has_bmi2(), ggml_cpu_has_avx512());
    int nm = 0, tm = 0, te = 0, d = 0;
    float* mel = parakeet_compute_mel(ctx, pcm.data(), (int)pcm.size(), &nm, &tm);
    if (!mel)
        return 1;
    for (int i = 0; i < 4; ++i) {
        const auto start = std::chrono::steady_clock::now();
        float* enc = parakeet_run_encoder(ctx, mel, nm, tm, &te, &d);
        if (!enc || te <= 0 || d <= 0)
            return 1;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        printf("ENCODER iteration=%d ms=%.3f T=%d D=%d\n", i, ms, te, d);
        if (i == 3) {
            f = fopen(argv[3], "wb");
            if (!f || fwrite(enc, sizeof(float), (size_t)te * d, f) != (size_t)te * d)
                return 1;
            fclose(f);
            auto* result = parakeet_decode_frames(ctx, enc, te, d, 0);
            if (!result || !result->text || !*result->text)
                return 1;
            printf("TRANSCRIPT %s\n", result->text);
            parakeet_result_free(result);
        }
        free(enc);
    }
    free(mel);
    parakeet_free(ctx);
    return 0;
}

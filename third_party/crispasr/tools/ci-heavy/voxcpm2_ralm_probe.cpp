// Internal CPU proof for #478: prefill states, KV, causal isolation and continuation.
#include "../../src/voxcpm2_tts.cpp"

#include <stdexcept>

static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

static bool parity_failed = false;

static void compare(const char* name, const std::vector<float>& ref, const std::vector<float>& actual) {
    require(ref.size() == actual.size() && !ref.empty(), "comparison shape mismatch");
    double dot = 0, aa = 0, bb = 0, err = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        require(std::isfinite(ref[i]) && std::isfinite(actual[i]), "non-finite output");
        dot += (double)ref[i] * actual[i];
        aa += (double)ref[i] * ref[i];
        bb += (double)actual[i] * actual[i];
        const double delta = actual[i] - ref[i];
        err += delta * delta;
    }
    require(aa > 0 && bb > 0, "zero output norm");
    const double cosine = dot / std::sqrt(aa * bb), ratio = std::sqrt(bb / aa), relative = std::sqrt(err / aa);
    printf("%s cosine=%.9f norm_ratio=%.9f relative_error=%.9f\n", name, cosine, ratio, relative);
    if (!(cosine > 0.9999 && std::abs(ratio - 1) < 0.005 && relative < 0.01)) {
        parity_failed = true;
        puts("PARITY_LIMIT_EXCEEDED");
    }
}

static std::vector<float> prefix_kv(voxcpm2_context* ctx, int T) {
    std::vector<float> result;
    const size_t count = (size_t)T * ctx->hp.ralm_n_kv * ctx->hp.ralm_head_dim;
    for (size_t l = 0; l < ctx->hp.ralm_n_layers; ++l) {
        result.insert(result.end(), ctx->ralm_kv.k_cache[l].begin(), ctx->ralm_kv.k_cache[l].begin() + count);
        result.insert(result.end(), ctx->ralm_kv.v_cache[l].begin(), ctx->ralm_kv.v_cache[l].begin() + count);
    }
    return result;
}

int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: voxcpm2-ralm-probe model.gguf dump-prefix");
        auto params = voxcpm2_context_default_params();
        params.use_gpu = false;
        params.n_threads = 4;
        params.verbosity = 0;
        std::unique_ptr<voxcpm2_context, decltype(&voxcpm2_free)> ctx(voxcpm2_init_from_file(argv[1], params),
                                                                      voxcpm2_free);
        require(bool(ctx), "model load failed");
        ggml_backend_t cpu = get_cpu_backend();
        g_cpu_n_threads = params.n_threads;
        core_cpu_backend::set_n_threads(cpu, params.n_threads);
        const int d = (int)ctx->hp.ralm_d_model;
        auto dump = [&](const char* suffix, const std::vector<float>& data) {
            const std::string path = std::string(argv[2]) + suffix;
            FILE* file = fopen(path.c_str(), "wb");
            require(file != nullptr, "dump open failed");
            const size_t wrote = fwrite(data.data(), sizeof(float), data.size(), file);
            fclose(file);
            require(wrote == data.size(), "dump write failed");
        };
        FILE* meta = fopen((std::string(argv[2]) + ".json").c_str(), "w");
        require(meta != nullptr, "metadata open failed");
        fprintf(meta, "{\"d\":%d,\"heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,\"layers\":%u,\"eps\":%.9g}\n", d,
                ctx->hp.ralm_n_heads, ctx->hp.ralm_n_kv, ctx->hp.ralm_head_dim, ctx->hp.ralm_n_layers,
                ctx->hp.rms_norm_eps);
        fclose(meta);
        for (int T : {1, 10, 62, 249, 4}) {
            std::vector<float> input((size_t)T * d), next(d);
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = 0.2f * std::sin((float)i * 0.013f) + 0.1f * std::cos((float)i * 0.027f);
            for (int i = 0; i < d; ++i)
                next[i] = 0.2f * std::cos((float)i * 0.019f);
            setenv("CRISPASR_VOXCPM2_RALM_PREFILL_BATCH", "0", 1);
            double start = vox_now_ms();
            auto reference = ralm_prefill_multi(ctx.get(), input.data(), T, cpu);
            const double eager_ms = vox_now_ms() - start;
            auto kv = prefix_kv(ctx.get(), T);
            auto continuation = ralm_step_graph(ctx.get(), next.data(), T);
            setenv("CRISPASR_VOXCPM2_RALM_PREFILL_BATCH", "1", 1);
            start = vox_now_ms();
            auto actual = ralm_prefill_multi(ctx.get(), input.data(), T, cpu);
            const double batch_ms = vox_now_ms() - start;
            require(ctx->ralm_kv_synced && ctx->ralm_kv.n_past == T, "batched path fell back or has wrong position");
            compare("hidden", reference, actual);
            compare("kv", kv, prefix_kv(ctx.get(), T));
            compare("continuation", continuation, ralm_step_graph(ctx.get(), next.data(), T));
            printf("T=%d eager_ms=%.3f batched_ms=%.3f\n", T, eager_ms, batch_ms);
            if (T == 10) {
                dump(".input.f32", input);
                dump(".eager.f32", reference);
                dump(".batched.f32", actual);
                dump(".kv.f32", kv);
            }
            if (T == 249) {
                // Cold results above are diagnostic; acceptance timings are
                // medians of three warmed back-to-back pairs on this host.
                std::vector<double> eager_times, batch_times;
                for (int repeat = 0; repeat < 3; ++repeat) {
                    setenv("CRISPASR_VOXCPM2_RALM_PREFILL_BATCH", "0", 1);
                    start = vox_now_ms();
                    ralm_prefill_multi(ctx.get(), input.data(), T, cpu);
                    eager_times.push_back(vox_now_ms() - start);
                    setenv("CRISPASR_VOXCPM2_RALM_PREFILL_BATCH", "1", 1);
                    start = vox_now_ms();
                    ralm_prefill_multi(ctx.get(), input.data(), T, cpu);
                    batch_times.push_back(vox_now_ms() - start);
                    require(ctx->ralm_kv_synced, "timing arm fell back");
                }
                std::sort(eager_times.begin(), eager_times.end());
                std::sort(batch_times.begin(), batch_times.end());
                printf("WARM T=249 eager_ms=%.3f batched_ms=%.3f speedup=%.3f\n", eager_times[1], batch_times[1],
                       eager_times[1] / batch_times[1]);
            }
            // Altering the future must not change any prefix hidden or KV value.
            if (T > 1) {
                auto altered = input;
                for (size_t i = (size_t)(T - 1) * d; i < altered.size(); ++i)
                    altered[i] += 1.0f;
                auto future = ralm_prefill_multi(ctx.get(), altered.data(), T, cpu);
                compare("causal_prefix", std::vector<float>(actual.begin(), actual.begin() + (size_t)(T - 1) * d),
                        std::vector<float>(future.begin(), future.begin() + (size_t)(T - 1) * d));
            }
            if (T == 4) {
                const bool expected_batch = ralm_prefill_batch_default(ctx.get());
                unsetenv("CRISPASR_VOXCPM2_RALM_PREFILL_BATCH");
                auto selected = ralm_prefill_multi(ctx.get(), input.data(), T, cpu);
                require(ctx->ralm_kv_synced == expected_batch, "wrong default prefill selection");
                require(selected == (expected_batch ? actual : reference), "default differs from its explicit arm");
                setenv("CRISPASR_VOXCPM2_RALM_PREFILL_BATCH", "1", 1);
                setenv("CRISPASR_VOXCPM2_USE_GRAPH", "0", 1);
                auto disabled = ralm_prefill_multi(ctx.get(), input.data(), T, cpu);
                require(!ctx->ralm_kv_synced && disabled == reference, "USE_GRAPH=0 did not retain eager prefill");
                setenv("CRISPASR_VOXCPM2_USE_GRAPH", "1", 1);
                printf("RALM_DEFAULT_PASS selected=%d\n", expected_batch ? 1 : 0);
            }
            fflush(stdout);
        }
        puts(parity_failed ? "RALM_PREFILL_PARITY_FAIL" : "RALM_PREFILL_PASS");
        return parity_failed ? 1 : 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "RALM_PREFILL_FAIL: %s\n", e.what());
        return 1;
    }
}

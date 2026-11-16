// PR #480: exercise actual CPU-to-Vulkan copies, including changed inputs on
// cached computes and graph metadata recycled before the next scheduler reset.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    ggml_backend_t vk = ggml_backend_vk_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    require(vk && cpu, "both Vulkan and CPU must initialize");
    std::printf("GPU backend: %s\n", ggml_backend_name(vk));
    ggml_backend_t backends[] = {vk, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false, true);
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(128, false);
    ip.no_alloc = true;
    auto inputs = ggml_init(ip);
    auto arena = ggml_init(ip);
    require(inputs && arena && sched, "contexts and scheduler allocated");
    auto a = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, 32);
    auto b = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, 32);
    ggml_set_input(a);
    ggml_set_input(b);
    auto input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
    require(input_buffer, "CPU input buffer allocated");
    float av[32], bv[32], result[32];
    auto graph = [&](bool reverse) {
        auto out = ggml_sub(arena, reverse ? b : a, reverse ? a : b);
        ggml_set_output(out);
        auto gf = ggml_new_graph_custom(arena, 128, false);
        ggml_build_forward_expand(gf, out);
        ggml_backend_sched_set_tensor_backend(sched, out, vk);
        require(ggml_backend_sched_alloc_graph(sched, gf), "graph allocated");
        require(ggml_backend_sched_get_tensor_backend(sched, out) == vk, "output placed on Vulkan");
        for (int iteration = 0; iteration < 5; ++iteration) {
            std::printf("compute reverse=%d iteration=%d\n", reverse, iteration);
            std::fflush(stdout);
            for (int i = 0; i < 32; ++i) {
                av[i] = float(i + iteration * 7);
                bv[i] = float(2 * i - iteration * 3);
            }
            ggml_backend_tensor_set(a, av, 0, sizeof(av));
            ggml_backend_tensor_set(b, bv, 0, sizeof(bv));
            require(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS, "compute succeeds");
            ggml_backend_sched_synchronize(sched);
            ggml_backend_tensor_get(out, result, 0, sizeof(result));
            for (int i = 0; i < 32; ++i) {
                float expected = reverse ? bv[i] - av[i] : av[i] - bv[i];
                require(std::fabs(result[i] - expected) <= 1e-6f, "changed-input output equals independent reference");
            }
            require(out->src[0] == (reverse ? b : a) && out->src[1] == (reverse ? a : b),
                    "compute restores original source pointers");
        }
    };
    graph(false);
    // The old mutation records point into this arena. Build reversed sources
    // BEFORE reset, exactly as callers recycling their graph metadata do.
    ggml_reset(arena);
    auto recycled = ggml_sub(arena, b, a);
    ggml_backend_sched_reset(sched);
    require(recycled->src[0] == b && recycled->src[1] == a, "reset must not overwrite recycled metadata");
    ggml_reset(arena);
    graph(true);
    ggml_backend_sched_reset(sched);
    ggml_reset(arena);
    auto uncomputed = ggml_sub(arena, a, b);
    auto gf = ggml_new_graph_custom(arena, 128, false);
    ggml_build_forward_expand(gf, uncomputed);
    ggml_backend_sched_set_tensor_backend(sched, uncomputed, vk);
    require(ggml_backend_sched_alloc_graph(sched, gf), "uncomputed graph allocated");
    ggml_backend_sched_reset(sched);
    require(uncomputed->src[0] == a && uncomputed->src[1] == b, "reset restores uncomputed graph sources");
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(arena);
    ggml_free(inputs);
    ggml_backend_free(vk);
    ggml_backend_free(cpu);
    std::puts("SCHEDULER_REPLAY_VULKAN_PASS changed_inputs=10 restored_sources=10 recycled_graph=1 uncomputed_reset=1");
}

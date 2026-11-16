// chatterbox_attn_policy.h — T3 GPT-2 attention-path selection (issue #402).
//
// On AMD Radeon 780M / RADV, the Vulkan FLASH_ATTN_EXT pipeline crashes for
// the T3 GPT-2 geometry (hd=64, 12 MHA heads, repeated T=1 AR decode) with
// both F16 and Q4_K weights, while the explicit softmax(QK^T)V path runs a
// full 431-step synthesis correctly (reporter-verified; first token
// divergence vs CPU at step 3 is consistent with backend float ordering, not
// a broken attention). We have no RADV box to bisect the shader, so the
// policy is: on a Vulkan backend the naive path is the DEFAULT for the T3
// GPT-2 graph; every other backend keeps ggml_flash_attn_ext.
//
// Env gates (never removed — regression bisection):
//   CRISPASR_CHATTERBOX_NAIVE_ATTN=1  → naive attention on EVERY backend
//                                       (pre-existing debug gate)
//   CRISPASR_CHATTERBOX_FLASH_ATTN=1  → flash_attn_ext even on Vulkan
//                                       (opt back in when a driver fix lands)
//
// Weight-free and header-only so tests/test-chatterbox-attn-policy.cpp can
// pin the decision table without loading a model.

#pragma once

#include <cstring>

namespace chatterbox_attn {

// "set and not 0" — matches the env_set() convention in chatterbox.cpp.
inline bool env_truthy(const char* v) {
    return v && *v && std::strcmp(v, "0") != 0;
}

// Decide whether the T3 GPT-2 path uses the explicit softmax(QK^T)V
// attention instead of ggml_flash_attn_ext.
//   backend_name — ggml_backend_name() of the T3 backend ("Vulkan0", "CUDA0",
//                  "Metal", "CPU", ...); nullptr is treated as CPU.
//   env_naive    — value of CRISPASR_CHATTERBOX_NAIVE_ATTN (getenv result)
//   env_flash    — value of CRISPASR_CHATTERBOX_FLASH_ATTN (getenv result)
inline bool use_naive_t3(const char* backend_name, const char* env_naive, const char* env_flash) {
    if (env_truthy(env_naive))
        return true; // explicit debug override wins everywhere
    if (env_truthy(env_flash))
        return false; // explicit flash opt-in wins over the Vulkan default
    return backend_name && std::strncmp(backend_name, "Vulkan", 6) == 0;
}

} // namespace chatterbox_attn

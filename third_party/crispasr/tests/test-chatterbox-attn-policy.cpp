// test-chatterbox-attn-policy.cpp — issue #402: T3 GPT-2 attention-path
// decision table.
//
// Hermetic: no model, no GPU. The policy picks naive softmax(QK^T)V
// attention on Vulkan backends (RADV 780M crashes in the FLASH_ATTN_EXT
// pipeline) and flash_attn_ext everywhere else, with env overrides in both
// directions. A wrong default here is invisible to every numeric check —
// on non-Vulkan boxes both paths produce near-identical output — so the
// decision table gets its own test.

#include "chatterbox_attn_policy.h"

#include <catch2/catch_test_macros.hpp>

using chatterbox_attn::use_naive_t3;

TEST_CASE("t3 attn policy: Vulkan defaults to naive, others to flash", "[unit][tts][chatterbox][attn-policy]") {
    // Vulkan backend names carry a device index suffix.
    CHECK(use_naive_t3("Vulkan0", nullptr, nullptr));
    CHECK(use_naive_t3("Vulkan1", nullptr, nullptr));
    CHECK(use_naive_t3("Vulkan", nullptr, nullptr));

    CHECK_FALSE(use_naive_t3("CUDA0", nullptr, nullptr));
    CHECK_FALSE(use_naive_t3("Metal", nullptr, nullptr));
    CHECK_FALSE(use_naive_t3("CPU", nullptr, nullptr));
    CHECK_FALSE(use_naive_t3("BLAS", nullptr, nullptr));
    // No backend (defensive) behaves like CPU.
    CHECK_FALSE(use_naive_t3(nullptr, nullptr, nullptr));
}

TEST_CASE("t3 attn policy: CRISPASR_CHATTERBOX_NAIVE_ATTN forces naive everywhere",
          "[unit][tts][chatterbox][attn-policy]") {
    CHECK(use_naive_t3("CUDA0", "1", nullptr));
    CHECK(use_naive_t3("Metal", "1", nullptr));
    CHECK(use_naive_t3("CPU", "1", nullptr));
    // The debug override outranks the flash opt-in: someone bisecting with
    // NAIVE_ATTN=1 must get the naive graph no matter what else is set.
    CHECK(use_naive_t3("Vulkan0", "1", "1"));
    // "0" means unset, matching chatterbox.cpp's env_set() convention.
    CHECK_FALSE(use_naive_t3("CPU", "0", nullptr));
}

TEST_CASE("t3 attn policy: CRISPASR_CHATTERBOX_FLASH_ATTN opts Vulkan back into flash",
          "[unit][tts][chatterbox][attn-policy]") {
    CHECK_FALSE(use_naive_t3("Vulkan0", nullptr, "1"));
    // ...and is a no-op where flash is already the default.
    CHECK_FALSE(use_naive_t3("CUDA0", nullptr, "1"));
    CHECK_FALSE(use_naive_t3("CPU", nullptr, "1"));
    // "0" does not opt in.
    CHECK(use_naive_t3("Vulkan0", nullptr, "0"));
}

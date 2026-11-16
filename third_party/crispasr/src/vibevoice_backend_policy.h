// vibevoice_backend_policy.h — σ-VAE tokenizer-encoder backend selection (#418).
//
// Issue #418 (ImCodingCat, Intel Arc B580): VibeVoice-ASR crashes on Vulkan in
// ggml_vk_dispatch_pipeline — the acoustic/semantic tokenizer encoders build
// conv dispatches over the raw 24 kHz waveform whose workgroup counts exceed
// `maxComputeWorkGroupCount` (65535 per dimension on Intel drivers and on
// llvmpipe/lavapipe; NVIDIA reports 2^31 for dim 0 and never trips). The σ-VAE
// DECODER got a CPU fallback for the same class of device in issue #52
// (`vibevoice_vae_should_use_cpu`); the ENCODERS — the ASR direction — missed
// it, exactly as the reporter guessed.
//
// The decision is a pure function of three strings so the table is testable
// without a Vulkan device (tests/test-vibevoice-enc-policy.cpp):
//
//   CRISPASR_VIBEVOICE_ENC_BACKEND=cpu  → encoders always on CPU
//   CRISPASR_VIBEVOICE_ENC_BACKEND=gpu  → always on the active backend, even
//                                         where known to abort (the repro arm)
//   unset / auto                        → CPU when the active backend is
//                                         Vulkan on a low-workgroup-limit
//                                         device (Intel iGPU/dGPU, llvmpipe)
//
// Unlike the decoder's policy this does NOT divert Metal: the encoders are the
// downsampling half (no 3200x transposed-conv upsample), well under Apple's
// GPU-watchdog horizon, and diverting them would regress working Mac setups.

#pragma once

#include <cstring>

namespace vibevoice_policy {

// Device names with maxComputeWorkGroupCount[0] == 65535, where the encoder
// dispatches over long audio overflow (#418). NVIDIA (2^31) and AMD RADV
// (2^32-1) do not trip. False positives only cost running the (cheap,
// downsampling) encoders on CPU.
inline bool low_workgroup_limit_device(const char* device_name) {
    if (!device_name)
        return false;
    return std::strstr(device_name, "Intel") != nullptr || std::strstr(device_name, "llvmpipe") != nullptr;
}

inline bool enc_use_cpu(const char* env, const char* backend_name, const char* device_name) {
    if (env && std::strcmp(env, "cpu") == 0)
        return true;
    if (env && std::strcmp(env, "gpu") == 0)
        return false;
    const bool vulkan = backend_name && std::strncmp(backend_name, "Vulkan", 6) == 0;
    return vulkan && low_workgroup_limit_device(device_name);
}

} // namespace vibevoice_policy

// test-crisp-audio-mapping-released.cpp — crisp_audio must not outlive its
// weight mapping either.
//
// crisp_audio's load_model() takes its backend buffer from
// core_gguf::load_weights and moves it into crisp_audio_context::model_buf.
// crisp_audio_free() destroyed that handle with ggml_backend_buffer_free(),
// which on a device advertising `buffer_from_host_ptr` releases the device-side
// view and leaves the host mmap in place — the loader's side-map entry is never
// taken and the pages are never unmapped. Two shipped consumers reach this
// path: src/qwen3_asr.cpp and src/higgs_stt.cpp both link crisp_audio, and
// CrispEmbed's BidirLM-Omni audio path links the same library.
//
// test-gguf-mapping-released.cpp pins the loader's own release. This file pins
// the consumer: that crisp_audio's handle actually reaches
// core_gguf::release_weight_buffer, through the public C API rather than
// through the loader.
//
// The oracle is the same exact one — after the free, no region of this process
// may name the GGUF. Which leg pins what:
//
//   * The GPU leg is the one that fails without the fix. It self-skips where no
//     device advertises buffer_from_host_ptr, because the leaking path does not
//     exist there.
//   * The CPU leg cannot fail on this bug — the CPU mmap path unmaps through
//     the buffer's own free callback whichever entry point releases it. It is
//     here so the case is not a pure skip on Linux/Windows CI, and it does
//     guard the weaker claim that crisp_audio releases at all.
//
// No model file, no download: a synthetic one-layer tower reaches the same
// loader branch a multi-gigabyte encoder does.

#include <catch2/catch_test_macros.hpp>

#include "test-region-probe.h"

#include "crisp_audio.h"

#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>

namespace {

using test_region::absolute_path_of;
using test_region::count_regions_backed_by;
using test_region::region_probe_available;

// Portable env helper (Windows has no POSIX setenv).
void test_setenv(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    ::setenv(k, v, 1);
#endif
}

// The dimensions the fixture declares. Small enough that the positional
// embedding crisp_audio precomputes at init costs nothing; d_model must stay
// above 2, since the sinusoid divides by (d_model/2 - 1).
constexpr uint32_t kLayers = 1;
constexpr uint32_t kDModel = 8;

// Every tensor load_model() looks up by name. Listed rather than derived: a
// rename in audio_tower.cpp should fail this fixture loudly at load time, not
// leave it silently loading a tower with no weights.
const char* kTowerTensors[] = {
    "audio.conv.1.weight",  "audio.conv.1.bias",  "audio.conv.2.weight",   "audio.conv.2.bias",
    "audio.conv.3.weight",  "audio.conv.3.bias",  "audio.conv_out.weight", "audio.conv_out.bias",
    "audio.ln_post.weight", "audio.ln_post.bias", "audio.proj1.weight",    "audio.proj1.bias",
    "audio.proj2.weight",   "audio.proj2.bias",
};

const char* kBlockSuffixes[] = {
    "attn_norm.weight", "attn_norm.bias", "attn_q.weight",   "attn_q.bias",   "attn_k.weight",   "attn_k.bias",
    "attn_v.weight",    "attn_v.bias",    "attn_out.weight", "attn_out.bias", "ffn_norm.weight", "ffn_norm.bias",
    "ffn_up.weight",    "ffn_up.bias",    "ffn_down.weight", "ffn_down.bias",
};

// Weights wide enough that the mapping is a region of its own rather than
// something the kernel might fold into a neighbour. One tensor carries the
// bulk; the rest only need to exist under the right name.
constexpr int kBulkElems = 1 << 20; // 4 MiB of f32
constexpr int kSmallElems = 4;

void add_tensor(ggml_context* ctx, gguf_context* g, const char* name, int elems) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elems);
    REQUIRE(t != nullptr);
    ggml_set_name(t, name);
    float* d = (float*)t->data;
    for (int i = 0; i < elems; i++)
        d[i] = (float)i;
    gguf_add_tensor(g, t);
}

// A GGUF crisp_audio loads successfully: hparams under the default
// `crisp_audio.` key prefix plus the full tensor set for a one-layer tower.
void write_tower_gguf(const std::string& path) {
    const size_t n_tensors = std::size(kTowerTensors) + kLayers * std::size(kBlockSuffixes);
    const size_t mem = (size_t)kBulkElems * sizeof(float) +
                       n_tensors * ((size_t)kSmallElems * sizeof(float) + ggml_tensor_overhead()) + 8192;
    ggml_init_params ip = {/*mem_size=*/mem, /*mem_buffer=*/nullptr, /*no_alloc=*/false};
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);

    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "crisp_audio_test");
    // d_model is the key load_model probes to decide the metadata prefix, so
    // it has to be present for the rest of these to be read at all.
    gguf_set_val_u32(g, "crisp_audio.d_model", kDModel);
    gguf_set_val_u32(g, "crisp_audio.n_layers", kLayers);
    gguf_set_val_u32(g, "crisp_audio.n_heads", 1);
    gguf_set_val_u32(g, "crisp_audio.head_dim", kDModel);
    gguf_set_val_u32(g, "crisp_audio.ff_dim", kDModel);
    gguf_set_val_u32(g, "crisp_audio.conv_channels", kDModel);
    gguf_set_val_u32(g, "crisp_audio.max_source_pos", 8);
    gguf_set_val_u32(g, "crisp_audio.output_dim", kDModel);

    bool bulk_used = false;
    for (const char* name : kTowerTensors) {
        add_tensor(ctx, g, name, bulk_used ? kSmallElems : kBulkElems);
        bulk_used = true;
    }
    for (uint32_t i = 0; i < kLayers; i++) {
        for (const char* suffix : kBlockSuffixes) {
            char name[160];
            std::snprintf(name, sizeof(name), "audio.blk.%u.%s", i, suffix);
            add_tensor(ctx, g, name, kSmallElems);
        }
    }

    REQUIRE(gguf_write_to_file(g, path.c_str(), /*only_meta=*/false));
    gguf_free(g);
    ggml_free(ctx);
}

// A GGUF the loader maps but crisp_audio then rejects: valid hparams, no tower
// tensors. load_model() returns false after load_weights() has already mapped
// the file, and crisp_audio_init_from_file cleans up through crisp_audio_free.
void write_rejected_gguf(const std::string& path) {
    const size_t mem = (size_t)kBulkElems * sizeof(float) + ggml_tensor_overhead() + 4096;
    ggml_init_params ip = {/*mem_size=*/mem, /*mem_buffer=*/nullptr, /*no_alloc=*/false};
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);

    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "crisp_audio_test");
    gguf_set_val_u32(g, "crisp_audio.d_model", kDModel);
    gguf_set_val_u32(g, "crisp_audio.n_layers", kLayers);
    gguf_set_val_u32(g, "crisp_audio.max_source_pos", 8);
    add_tensor(ctx, g, "audio.not_a_tower_tensor", kBulkElems);

    REQUIRE(gguf_write_to_file(g, path.c_str(), /*only_meta=*/false));
    gguf_free(g);
    ggml_free(ctx);
}

struct Fixture {
    std::string rel;
    std::string abs;
    Fixture(const char* name, bool loadable) : rel(name) {
        if (loadable)
            write_tower_gguf(rel);
        else
            write_rejected_gguf(rel);
        abs = absolute_path_of(rel);
    }
    ~Fixture() { std::remove(rel.c_str()); }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
};

// True when this machine has the GPU device crisp_audio_init_from_file would
// pick AND that device hands host pointers to the backend — the two conditions
// that together select the leaking loader branch.
bool host_ptr_gpu_available() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev)
        return false;
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(dev, &props);
    return props.caps.buffer_from_host_ptr;
}

crisp_audio_params quiet_params(bool use_gpu) {
    crisp_audio_params p = crisp_audio_params_default();
    p.verbosity = 0;
    p.use_gpu = use_gpu;
    return p;
}

} // namespace

TEST_CASE("crisp_audio_free unmaps the zero-copy GPU path's weight region", "[unit][crisp-audio-mapping]") {
    if (!region_probe_available()) {
        SUCCEED("region enumeration unavailable on this platform");
        return;
    }
    if (!host_ptr_gpu_available()) {
        SUCCEED("no GPU device advertising buffer_from_host_ptr — the leaking path does not exist here");
        return;
    }
    test_setenv("CRISPASR_GGUF_MMAP", "1");

    Fixture fx("crispasr_test_crisp_audio_gpu.gguf", /*loadable=*/true);
    REQUIRE(count_regions_backed_by(fx.abs) == 0);

    crisp_audio_params p = quiet_params(/*use_gpu=*/true);
    crisp_audio_context* ctx = crisp_audio_init_from_file(fx.rel.c_str(), &p);
    REQUIRE(ctx != nullptr);
    // Positive control: crisp_audio really took the zero-copy branch. Without
    // it a fall-through to the legacy alloc+copy loader would satisfy the
    // absence assertion below having never created the mapping under test.
    // Not pinned to 1 — a kernel may report one mapping as several adjacent
    // regions; the repeated-cycles case is what pins accumulation.
    REQUIRE(count_regions_backed_by(fx.abs) >= 1);

    crisp_audio_free(ctx);
    REQUIRE(count_regions_backed_by(fx.abs) == 0);
}

TEST_CASE("repeated crisp_audio init/free cycles leave no mapping behind", "[unit][crisp-audio-mapping]") {
    if (!region_probe_available()) {
        SUCCEED("region enumeration unavailable on this platform");
        return;
    }
    if (!host_ptr_gpu_available()) {
        SUCCEED("no GPU device advertising buffer_from_host_ptr — the leaking path does not exist here");
        return;
    }
    test_setenv("CRISPASR_GGUF_MMAP", "1");

    // Each init maps the file again and records a separate region, so a
    // release that handled only one of them accumulates the rest. Five cycles
    // make that a count of five rather than an ambiguous one.
    Fixture fx("crispasr_test_crisp_audio_loop.gguf", /*loadable=*/true);
    crisp_audio_params p = quiet_params(/*use_gpu=*/true);

    for (int i = 0; i < 5; i++) {
        crisp_audio_context* ctx = crisp_audio_init_from_file(fx.rel.c_str(), &p);
        REQUIRE(ctx != nullptr);
        crisp_audio_free(ctx);
    }
    REQUIRE(count_regions_backed_by(fx.abs) == 0);
}

TEST_CASE("a crisp_audio load rejected after mapping leaves no region", "[unit][crisp-audio-mapping]") {
    if (!region_probe_available()) {
        SUCCEED("region enumeration unavailable on this platform");
        return;
    }
    if (!host_ptr_gpu_available()) {
        SUCCEED("no GPU device advertising buffer_from_host_ptr — the leaking path does not exist here");
        return;
    }
    test_setenv("CRISPASR_GGUF_MMAP", "1");

    // load_weights succeeds and maps the file; the missing tower tensors are
    // what fail, one step later. crisp_audio_init_from_file's own cleanup call
    // is the release path here, and it is the same one the success case uses.
    Fixture fx("crispasr_test_crisp_audio_reject.gguf", /*loadable=*/false);
    REQUIRE(count_regions_backed_by(fx.abs) == 0);

    crisp_audio_params p = quiet_params(/*use_gpu=*/true);
    REQUIRE(crisp_audio_init_from_file(fx.rel.c_str(), &p) == nullptr);
    REQUIRE(count_regions_backed_by(fx.abs) == 0);

    // Positive control, after the fact: the same file on the same backend does
    // map. Without it the zero above would also be satisfied by a loader that
    // rejected the GGUF before ever mapping it.
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    REQUIRE(dev != nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    REQUIRE(backend != nullptr);
    core_gguf::WeightLoad wl;
    REQUIRE(core_gguf::load_weights(fx.rel.c_str(), backend, "test-crisp-audio-reject", wl));
    REQUIRE(count_regions_backed_by(fx.abs) >= 1);
    core_gguf::free_weights(wl);
    REQUIRE(count_regions_backed_by(fx.abs) == 0);
    ggml_backend_free(backend);
}

TEST_CASE("crisp_audio_free unmaps the CPU mmap path's weight region", "[unit][crisp-audio-mapping]") {
    if (!region_probe_available()) {
        SUCCEED("region enumeration unavailable on this platform");
        return;
    }
    test_setenv("CRISPASR_GGUF_MMAP", "1");

    // This leg passed before the fix too — the CPU mmap path unmaps through
    // the buffer's own free callback whichever entry point releases it. It
    // runs everywhere, and it is what keeps this file from being a pure skip
    // on hosts without a host-pointer GPU.
    Fixture fx("crispasr_test_crisp_audio_cpu.gguf", /*loadable=*/true);
    REQUIRE(count_regions_backed_by(fx.abs) == 0);

    crisp_audio_params p = quiet_params(/*use_gpu=*/false);
    crisp_audio_context* ctx = crisp_audio_init_from_file(fx.rel.c_str(), &p);
    REQUIRE(ctx != nullptr);
    REQUIRE(count_regions_backed_by(fx.abs) >= 1);

    crisp_audio_free(ctx);
    REQUIRE(count_regions_backed_by(fx.abs) == 0);
}

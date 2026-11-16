// test-companion-resolve.cpp — unit tests for companion model resolution.
//
// Exercises the companion pre-download guard logic from issues #146 / #148:
//   - skip when --codec-model is set
//   - skip when companion sits next to the model file (sibling)
//   - skip when companion is in the cache dir
//   - correct companion size in download prompt
//
// Pure filesystem operations — no network, no models.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_cache.h"
#include "crispasr_model_registry.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#include <windows.h>
static std::string make_temp_dir() {
    char buf[MAX_PATH];
    GetTempPathA(MAX_PATH, buf);
    std::string base = buf;
    if (!base.empty() && (base.back() == '\\' || base.back() == '/'))
        base.pop_back();
    std::string dir = base + "/crispasr_companion_test_" + std::to_string(_getpid());
    _mkdir(dir.c_str());
    return dir;
}
static void remove_file(const std::string& path) {
    DeleteFileA(path.c_str());
}
static void remove_dir(const std::string& path) {
    _rmdir(path.c_str());
}
#else
#include <sys/stat.h>
#include <unistd.h>
static std::string make_temp_dir() {
    const char* env = std::getenv("CRISPASR_SCRATCH_DIR");
    std::string base = (env && *env) ? env : ".scratch";
    mkdir(base.c_str(), 0755);
    std::string pattern = base + "/crispasr_companion_XXXXXX";
    std::string writable = pattern;
    char* buf = writable.data();
    return mkdtemp(buf) ? std::string(buf) : base;
}
static void remove_file(const std::string& path) {
    ::unlink(path.c_str());
}
static void remove_dir(const std::string& path) {
    ::rmdir(path.c_str());
}
#endif

static void write_dummy(const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fprintf(f, "dummy");
        fclose(f);
    }
}

// ── Sibling-discovery simulation ─────────────────────────────────────
//
// The dispatcher checks for a companion file next to the model file
// before triggering the download prompt. These tests verify the sibling
// check logic in isolation.

TEST_CASE("companion: sibling file found next to model skips resolve", "[unit][companion]") {
    const std::string dir = make_temp_dir();

    // Create a dummy model file and its companion sibling.
    const std::string model_path = dir + "/mimo-asr-q4_k.gguf";
    const std::string companion_path = dir + "/mimo-tokenizer-q4_k.gguf";
    write_dummy(model_path);
    write_dummy(companion_path);

    // Simulate the dispatcher's sibling check.
    CrispasrRegistryEntry entry;
    REQUIRE(crispasr_registry_lookup("mimo-asr", entry));

    bool companion_found = false;
    {
        const auto sep = model_path.find_last_of("/\\");
        REQUIRE(sep != std::string::npos);
        const std::string sibling = model_path.substr(0, sep + 1) + entry.companion_filename;
        FILE* f = fopen(sibling.c_str(), "rb");
        if (f) {
            fclose(f);
            companion_found = true;
        }
    }
    REQUIRE(companion_found);

    remove_file(companion_path);
    remove_file(model_path);
    remove_dir(dir);
}

TEST_CASE("companion: sibling file absent triggers resolve path", "[unit][companion]") {
    const std::string dir = make_temp_dir();

    // Only the model file exists — no companion sibling.
    const std::string model_path = dir + "/mimo-asr-q4_k.gguf";
    write_dummy(model_path);

    CrispasrRegistryEntry entry;
    REQUIRE(crispasr_registry_lookup("mimo-asr", entry));

    bool companion_found = false;
    {
        const auto sep = model_path.find_last_of("/\\");
        REQUIRE(sep != std::string::npos);
        const std::string sibling = model_path.substr(0, sep + 1) + entry.companion_filename;
        FILE* f = fopen(sibling.c_str(), "rb");
        if (f) {
            fclose(f);
            companion_found = true;
        }
    }
    REQUIRE_FALSE(companion_found);

    remove_file(model_path);
    remove_dir(dir);
}

// ── Cache-directory probe ────────────────────────────────────────────
//
// When no sibling is found, the dispatcher probes the cache dir via
// crispasr_cache::probe_cached_file before triggering resolve.

TEST_CASE("companion: cached companion in cache_dir skips resolve", "[unit][companion]") {
    const std::string cache_dir = make_temp_dir();

    // Plant the companion in the cache directory.
    const std::string cached_companion = cache_dir + "/mimo-tokenizer-q4_k.gguf";
    write_dummy(cached_companion);

    CrispasrRegistryEntry entry;
    REQUIRE(crispasr_registry_lookup("mimo-asr", entry));

    const std::string found = crispasr_cache::probe_cached_file(entry.companion_filename, cache_dir);
    REQUIRE(!found.empty());

    remove_file(cached_companion);
    remove_dir(cache_dir);
}

TEST_CASE("companion: empty cache_dir means companion not found", "[unit][companion]") {
    const std::string cache_dir = make_temp_dir();

    CrispasrRegistryEntry entry;
    REQUIRE(crispasr_registry_lookup("mimo-asr", entry));

    // Probe for a name that CANNOT be on this machine, not the real companion.
    // probe_cached_file searches more than the directory it is handed — the
    // CRISPASR_MODELS_DIR env var, the platform default cache and other
    // well-known model locations. So asking for the genuine
    // companion asserts "this developer has never downloaded mimo-asr", which
    // is not the contract under test and fails on any populated machine (it
    // passes in CI only because a runner's cache is empty). The contract is
    // "an empty cache_dir yields no hit", and a unique suffix tests exactly
    // that, on every machine.
    const std::string absent = std::string(entry.companion_filename) + ".absent-9d3f1c07";
    const std::string found = crispasr_cache::probe_cached_file(absent, cache_dir);
    REQUIRE(found.empty());

    remove_dir(cache_dir);
}

// ── Companion size correctness ───────────────────────────────────────
//
// When the download prompt fires, the size shown should be the
// companion's actual size, not the LM's.

TEST_CASE("companion: mimo-asr companion size is ~395 MB, not ~4.2 GB (#148)", "[unit][companion]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("mimo-asr", e));
    REQUIRE(e.companion_approx_size == "~395 MB");
    REQUIRE(e.approx_size == "~4.2 GB"); // LM size, for contrast
}

TEST_CASE("companion: qwen3-tts companion size is ~60 MB, not ~986 MB (#146)", "[unit][companion]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("qwen3-tts", e));
    REQUIRE(e.companion_approx_size == "~60 MB");
    REQUIRE(e.approx_size == "~986 MB");
}

TEST_CASE("companion: lookup_by_filename shows companion size in approx_size (#146)", "[unit][companion]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup_by_filename("mimo-tokenizer-q4_k.gguf", e));
    // When resolving a companion by filename, approx_size should be the
    // companion's own size — this is what the "Available for download"
    // message prints.
    REQUIRE(e.approx_size == "~395 MB");
}

TEST_CASE("companion: snac-24khz lookup_by_filename shows codec size", "[unit][companion]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup_by_filename("snac-24khz.gguf", e));
    REQUIRE(e.approx_size == "~80 MB");
    REQUIRE(e.approx_size.find("GB") == std::string::npos);
}

TEST_CASE("companion: chatterbox-s3gen lookup_by_filename shows vocoder size", "[unit][companion]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup_by_filename("chatterbox-s3gen-q8_0.gguf", e));
    REQUIRE(e.approx_size == "~627 MB");
}

// #393: `--model-quant f16` rewrote the filename and URL to the f16 artifact
// but kept the registry row's approx_size, so a 4.7 GB download was announced
// as "~1.3 GB" (the q4_k figure). Per-quant sizes are not in the registry and
// a ratio-scaled guess would be an invention — this quantizer keeps norms and
// embeddings at F16, so the ratio is model-dependent — so a substituted quant
// reports NO size rather than a wrong one. The un-substituted lookup must keep
// reporting its size, or this "fix" would just delete information.
TEST_CASE("quant: a substituted quant reports no size instead of the wrong one (#393)", "[unit][companion]") {
    CrispasrRegistryEntry base;
    REQUIRE(crispasr_registry_lookup("qwen3-1.7b", base));
    const std::string default_name = base.filename;
    REQUIRE_FALSE(base.approx_size.empty()); // the default row still has its size

    CrispasrRegistryEntry f16;
    REQUIRE(crispasr_registry_lookup("qwen3-1.7b", f16, "f16"));
    if (f16.filename == default_name)
        SKIP("registry default for this backend is already f16");

    INFO("default " << default_name << " (" << base.approx_size << ") vs f16 " << f16.filename);
    CHECK(f16.url.find(f16.filename) != std::string::npos); // filename/URL did get rewritten
    CHECK(f16.approx_size.empty());                         // and the stale size is not reported
}

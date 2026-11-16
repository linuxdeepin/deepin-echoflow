// test-crispasr-cache.cpp — unit tests for crispasr_cache helpers.
//
// Covers pure filesystem operations (file_present, dir, ensure_cached_file
// happy path) without making any network requests so the suite stays fast
// and hermetic.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_cache.h"

#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#include <sys/stat.h>
#include <windows.h>
// MinGW's mkdir takes only a path — no mode — so the POSIX two-argument calls
// below are a hard error there ("too many arguments to function
// 'int mkdir(const char*)'"). Same shim src/crispasr_cache.cpp already carries
// for exactly this reason; MSVC does not declare mkdir at all, so it needs it
// too. Only build.yml's msbuild ALL_BUILD compiles the tests on Windows, which
// is why this went unnoticed.
#define mkdir(d, m) _mkdir(d)
static std::string make_temp_dir() {
    char buf[MAX_PATH];
    GetTempPathA(MAX_PATH, buf);
    // buf ends with backslash; trim it for consistent path joining
    std::string base = buf;
    if (!base.empty() && (base.back() == '\\' || base.back() == '/'))
        base.pop_back();
    std::string dir = base + "/crispasr_unit_" + std::to_string(_getpid());
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
    std::string pattern = base + "/crispasr_unit_XXXXXX";
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

static void write_file(const std::string& path, const char* content) {
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fputs(content, f);
        fclose(f);
    }
}

static void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

// ─── file_present() ──────────────────────────────────────────────────────────

TEST_CASE("file_present: nonexistent path returns false", "[unit]") {
    REQUIRE_FALSE(crispasr_cache::file_present("/this/absolutely/does/not/exist.bin"));
}

TEST_CASE("file_present: empty file (0-byte zombie) returns false", "[unit]") {
    const std::string tmp = make_temp_dir() + "/empty.bin";
    write_file(tmp, "");
    REQUIRE_FALSE(crispasr_cache::file_present(tmp));
    remove_file(tmp);
}

TEST_CASE("file_present: non-empty file returns true", "[unit]") {
    const std::string tmp = make_temp_dir() + "/nonempty.bin";
    write_file(tmp, "fake model bytes");
    REQUIRE(crispasr_cache::file_present(tmp));
    remove_file(tmp);
}

// #393: a cached GGUF larger than 2 GiB was reported MISSING on Windows, so
// `-m auto` re-downloaded a model that was already on disk. MSVC's `stat` is
// `_stat64i32` — a 32-bit st_size — and it fails outright past 2 GiB, which is
// every model this cache exists to hold. The file is created SPARSE (seek past
// the end, write one byte), so it costs a few KiB on NTFS/ext4/APFS rather than
// 2 GiB; a filesystem that refuses that just skips the test rather than
// filling the disk.
TEST_CASE("file_present: a file larger than 2 GiB is found (#393)", "[unit]") {
    const std::string tmp = make_temp_dir() + "/large.bin";
    constexpr long long kOver2GiB = 2LL * 1024 * 1024 * 1024 + 4096;

    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f)
        SKIP("could not create the temp file");
#ifdef _WIN32
    const bool seek_ok = _fseeki64(f, kOver2GiB - 1, SEEK_SET) == 0;
#else
    const bool seek_ok = fseeko(f, (off_t)(kOver2GiB - 1), SEEK_SET) == 0;
#endif
    const bool wrote = seek_ok && fputc(0, f) != EOF;
    const bool closed = fclose(f) == 0;
    if (!wrote || !closed) {
        remove_file(tmp);
        SKIP("filesystem would not take a sparse 2 GiB file");
    }

    // Guard the guard: if the file did not actually reach that size, the test
    // would pass for the wrong reason on the very platform it targets.
    std::error_code ec;
    const std::uintmax_t actual = std::filesystem::file_size(std::filesystem::path(tmp), ec);
    if (ec || actual < (std::uintmax_t)kOver2GiB) {
        remove_file(tmp);
        SKIP("sparse file did not reach 2 GiB");
    }

    CHECK(crispasr_cache::file_present(tmp));
    remove_file(tmp);
}

// ─── dir() ───────────────────────────────────────────────────────────────────

TEST_CASE("dir: override path is returned unchanged", "[unit]") {
    const std::string base = make_temp_dir();
    const std::string override_dir = base + "/my_cache";

    const std::string result = crispasr_cache::dir(override_dir);
    REQUIRE(result == override_dir);
}

TEST_CASE("dir: override creates the leaf directory", "[unit]") {
    const std::string base = make_temp_dir();
    const std::string override_dir = base + "/my_cache2";

    crispasr_cache::dir(override_dir);

    struct stat st {};
    REQUIRE(stat(override_dir.c_str(), &st) == 0);
#ifndef _WIN32
    REQUIRE(S_ISDIR(st.st_mode));
#endif
}

TEST_CASE("dir: empty override returns a non-empty platform default", "[unit]") {
    REQUIRE_FALSE(crispasr_cache::dir("").empty());
}

TEST_CASE("dir: environment and CLI destination precedence", "[unit]") {
    const char* old_cache = std::getenv("CRISPASR_CACHE_DIR");
    const char* old_models = std::getenv("CRISPASR_MODELS_DIR");
    const std::string saved_cache = old_cache ? old_cache : "";
    const std::string saved_models = old_models ? old_models : "";
    const bool had_cache = old_cache != nullptr;
    const bool had_models = old_models != nullptr;

    const std::string base = make_temp_dir();
    const std::string cache_env = base + "/cache_env";
    const std::string models_env = base + "/models_env";
    const std::string cli_override = base + "/cli_override";

    set_env("CRISPASR_CACHE_DIR", cache_env.c_str());
    set_env("CRISPASR_MODELS_DIR", models_env.c_str());
    REQUIRE(crispasr_cache::dir("") == cache_env);
    REQUIRE(crispasr_cache::dir(cli_override) == cli_override);

    set_env("CRISPASR_CACHE_DIR", nullptr);
    REQUIRE(crispasr_cache::dir("") == models_env);

    set_env("CRISPASR_CACHE_DIR", had_cache ? saved_cache.c_str() : nullptr);
    set_env("CRISPASR_MODELS_DIR", had_models ? saved_models.c_str() : nullptr);

    remove_dir(cli_override);
    remove_dir(models_env);
    remove_dir(cache_env);
    remove_dir(base);
}

// ─── ensure_cached_file() ────────────────────────────────────────────────────

TEST_CASE("ensure_cached_file: returns existing file path without fetching", "[unit]") {
    const std::string cache_dir = make_temp_dir();
    const std::string filename = "model.bin";
    const std::string full_path = cache_dir + "/" + filename;

    // Pre-populate the cache so no download is triggered.
    write_file(full_path, "fake model data");

    const std::string result =
        crispasr_cache::ensure_cached_file(filename,
                                           "https://example.invalid/model.bin", // URL — must not be reached
                                           /*quiet=*/true, "test", cache_dir);

    REQUIRE(result == full_path);
    remove_file(full_path);
}

// ─── ensure_cached_file: source integrity (issue #250) ───────────────────────

TEST_CASE("ensure_cached_file: matching source sidecar reuses the cached file", "[unit]") {
    const std::string cache_dir = make_temp_dir();
    const std::string filename = "tokenizer.bin";
    const std::string path = cache_dir + "/" + filename;
    const std::string url = "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/tokenizer.bin";

    write_file(path, "MOONSHINE-EN-TOKENIZER");
    write_file(path + ".src", url.c_str()); // records the origin of the cached bytes

    // Same url → cached file is source-consistent → returned without fetching.
    const std::string result = crispasr_cache::ensure_cached_file(filename, url, /*quiet=*/true, "test", cache_dir);
    REQUIRE(result == path);

    remove_file(path + ".src");
    remove_file(path);
}

TEST_CASE("ensure_cached_file: cross-repo same-basename file is NOT reused (issue #250)", "[unit]") {
    // The bug: a `tokenizer.bin` cached for moonshine-tiny (EN) was handed to a
    // moonshine-tiny-de request because the cache keyed on basename only. With
    // the source sidecar, the EN file must NOT be returned for the DE url.
    const std::string cache_dir = make_temp_dir();
    const std::string filename = "tokenizer.bin";
    const std::string path = cache_dir + "/" + filename;
    const std::string url_en = "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/tokenizer.bin";
    const std::string url_de = "https://huggingface.co/cstr/moonshine-tiny-de-fidoriel-GGUF/resolve/main/tokenizer.bin";

    write_file(path, "MOONSHINE-EN-TOKENIZER");
    write_file(path + ".src", url_en.c_str());

    // Request the DE url. The EN cached file is a different source → not reused.
    // The DE url below does not exist (file:// to a missing path) so the fetch
    // fails locally; the point is that the EN file was NOT returned.
    const std::string result =
        crispasr_cache::ensure_cached_file(filename, "file:///crispasr-test-nonexistent/tokenizer.bin",
                                           /*quiet=*/true, "test", cache_dir);
    REQUIRE(result != path); // the wrong-source EN file was never handed back

    // And the EN file + sidecar were left intact (not corrupted by the failed fetch).
    REQUIRE(crispasr_cache::file_present(path));
    remove_file(path + ".src");
    remove_file(path);
    (void)url_de;
}

TEST_CASE("ensure_cached_file: wrong-source hit is skipped in favor of a correct-source dir", "[unit]") {
    // Canonical cache holds the EN tokenizer; a second search dir
    // (CRISPASR_MODELS_DIR) holds the DE tokenizer with a matching sidecar.
    // A DE request must skip the canonical EN file and return the DE one —
    // fully offline, no fetch.
    const char* old_models = std::getenv("CRISPASR_MODELS_DIR");
    const std::string saved_models = old_models ? old_models : "";
    const bool had_models = old_models != nullptr;

    const std::string base = make_temp_dir();
    const std::string canonical = base + "/canonical";
    const std::string models_dir = base + "/models";
    mkdir(canonical.c_str(), 0755);
    mkdir(models_dir.c_str(), 0755);

    const std::string filename = "tokenizer.bin";
    const std::string url_en = "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/tokenizer.bin";
    const std::string url_de = "https://huggingface.co/cstr/moonshine-tiny-de-fidoriel-GGUF/resolve/main/tokenizer.bin";

    write_file(canonical + "/" + filename, "EN");
    write_file(canonical + "/" + filename + ".src", url_en.c_str());
    write_file(models_dir + "/" + filename, "DE");
    write_file(models_dir + "/" + filename + ".src", url_de.c_str());

    set_env("CRISPASR_MODELS_DIR", models_dir.c_str());
    const std::string result = crispasr_cache::ensure_cached_file(filename, url_de, /*quiet=*/true, "test",
                                                                  /*cache_dir_override=*/canonical);
    set_env("CRISPASR_MODELS_DIR", had_models ? saved_models.c_str() : nullptr);

    REQUIRE(result == models_dir + "/" + filename); // DE file returned, EN skipped
}

TEST_CASE("ensure_cached_file: no-sidecar legacy flat file is still trusted (backward compat)", "[unit]") {
    const std::string cache_dir = make_temp_dir();
    const std::string filename = "codec.gguf";
    const std::string path = cache_dir + "/" + filename;

    write_file(path, "legacy bytes, no sidecar"); // pre-#250 cache: no .src

    const std::string result = crispasr_cache::ensure_cached_file(filename, "https://example.invalid/codec.gguf",
                                                                  /*quiet=*/true, "test", cache_dir);
    REQUIRE(result == path); // trusted as-is, no fetch attempted
    remove_file(path);
}

TEST_CASE("ensure_cached_file: a leftover .part temp is never treated as a cache hit", "[unit]") {
    // Atomic downloads write to `<file>.part.<pid>.<n>` then rename; an
    // interrupted transfer leaves only the temp, which must not satisfy a
    // lookup for `<file>` (issue #250 Q10).
    const std::string cache_dir = make_temp_dir();
    write_file(cache_dir + "/model.gguf.part.999.0", "partial bytes");
    REQUIRE(crispasr_cache::probe_cached_file("model.gguf", cache_dir).empty());
    remove_file(cache_dir + "/model.gguf.part.999.0");
}

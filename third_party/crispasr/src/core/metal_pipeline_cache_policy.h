#pragma once
// metal_pipeline_cache_policy.h — bound the cost of ggml-metal's persistent
// MTLBinaryArchive pipeline cache before the Metal device is initialised.
//
// WHY (CrispEmbed T18/G4, measured 2026-08-05, M1 16 GB; logic synced from
// CrispEmbed's copy of this header — pcs.cpp rule, logic not bytes):
//   ggml carries our patch (PLAN #88) that opens a per-device MTLBinaryArchive
//   at `~/Library/Caches/ggml-metal/<device>.archive` so PSO creation can hit a
//   serialised pipeline instead of the shader compiler. That archive is
//   APPEND-ONLY across every engine and every binary that ever ran on the box
//   — CrispASR binaries exit normally, so unlike CrispEmbed's `_exit()`ing
//   one-shots they flush on EVERY CLI run and grow it. On the shared dev box
//   it had reached 683 MB; opening it costs ~1 ms/MB of fixed init, and the
//   full-size archive bought nothing measurable on first encode (macOS keeps
//   its own system-level shader cache underneath).
//
// WHAT:
//   Before any Metal device init, look at the archives in the cache directory
//   ggml would use. If the largest exceeds the cap, set
//   GGML_METAL_PIPELINE_CACHE_DISABLE=1 so ggml skips opening it. Skipping the
//   open also leaves `dev->binary_archive` nil, so the flush at device free
//   becomes a no-op — an oversized archive stops growing too. Small archives
//   are left alone: the pathology is unbounded growth, not the mechanism.
//
// GATES:
//   CRISPASR_METAL_PIPELINE_CACHE_MAX_MB=<N>  cap in MB (default 64).
//                                             0 = uncapped = legacy behaviour.
//   GGML_METAL_PIPELINE_CACHE=<dir>           ggml's own cache-dir override,
//                                             honoured here too.
//   GGML_METAL_PIPELINE_CACHE_DISABLE=1       ggml's own kill switch; if the
//                                             caller already set it we do not
//                                             touch it.
//
// The archive path is resolved WITHOUT touching Metal (the device name is not
// known until MTLCreateSystemDefaultDevice, which is part of what we are trying
// to avoid paying for), so this scans every `*.archive` in the directory and
// judges by the largest.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Everything below the includes is Apple-only: the archive lives at
// ~/Library/Caches/ggml-metal and the scan uses POSIX dirent/stat, neither of
// which MSVC has. Guard the whole apparatus and leave `apply()` a no-op
// elsewhere — the policy is meaningless without Metal.
#if defined(__APPLE__)
#include <dirent.h>
#include <sys/stat.h>
#endif

namespace core_metal_cache {

#if defined(__APPLE__)

inline long long cap_bytes() {
    const char* e = std::getenv("CRISPASR_METAL_PIPELINE_CACHE_MAX_MB");
    long long mb = 64; // default cap
    if (e && e[0]) {
        char* end = nullptr;
        const long long v = std::strtoll(e, &end, 10);
        if (end != e && v >= 0)
            mb = v;
    }
    return mb * 1024LL * 1024LL;
}

inline std::string cache_dir() {
    const char* e = std::getenv("GGML_METAL_PIPELINE_CACHE");
    if (e && e[0])
        return std::string(e);
    const char* home = std::getenv("HOME");
    if (!home || !home[0])
        return std::string();
    return std::string(home) + "/Library/Caches/ggml-metal";
}

// Largest `*.archive` in the cache dir, or 0 when there is none / unreadable.
inline long long largest_archive_bytes(std::string* out_path = nullptr) {
    const std::string dir = cache_dir();
    if (dir.empty())
        return 0;
    DIR* d = opendir(dir.c_str());
    if (!d)
        return 0;
    long long best = 0;
    while (dirent* ent = readdir(d)) {
        const size_t n = std::strlen(ent->d_name);
        if (n < 9 || std::strcmp(ent->d_name + n - 8, ".archive") != 0)
            continue;
        const std::string full = dir + "/" + ent->d_name;
        struct stat st {};
        if (stat(full.c_str(), &st) != 0)
            continue;
        if ((long long)st.st_size > best) {
            best = (long long)st.st_size;
            if (out_path)
                *out_path = full;
        }
    }
    closedir(d);
    return best;
}

#endif // __APPLE__

// Apply the policy. Idempotent, safe to call from every backend init.
// Returns true when it disabled the cache on this call.
inline bool apply() {
#if !defined(__APPLE__)
    return false;
#else
    static bool done = false;
    static bool disabled = false;
    if (done)
        return disabled;
    done = true;

    // Respect an explicit caller decision either way.
    if (const char* dis = std::getenv("GGML_METAL_PIPELINE_CACHE_DISABLE"); dis && dis[0] && dis[0] != '0') {
        return false;
    }
    const long long cap = cap_bytes();
    if (cap == 0)
        return false; // uncapped = legacy behaviour

    std::string path;
    const long long sz = largest_archive_bytes(&path);
    if (sz <= cap)
        return false;

    setenv("GGML_METAL_PIPELINE_CACHE_DISABLE", "1", 1);
    disabled = true;
    fprintf(stderr,
            "crispasr: Metal pipeline-cache archive is %.0f MB (> %.0f MB cap) — skipping it; opening it costs "
            "~1 ms/MB of fixed init. Delete %s to reclaim the disk, or set "
            "CRISPASR_METAL_PIPELINE_CACHE_MAX_MB=0 to use it anyway.\n",
            (double)sz / (1024.0 * 1024.0), (double)cap / (1024.0 * 1024.0),
            path.empty() ? "the archive" : path.c_str());
    return true;
#endif
}

} // namespace core_metal_cache

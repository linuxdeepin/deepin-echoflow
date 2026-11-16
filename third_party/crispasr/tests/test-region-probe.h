// test-region-probe.h — count this process's mapped regions backed by a file.
//
// Test-only helper. The GGUF loader hands a host mmap to the backend on the
// zero-copy path, and the only exact way to assert that the mapping was
// released is to ask the kernel which regions still name the file. A footprint
// or RSS delta would need a threshold and a settling window and would be the
// flaky case in this suite; a region count needs neither.
//
// The count matters, not just presence: load_weights maps the file once per
// *load*, so repeated loads accumulate separate regions and a release that
// dropped only the most recent one would still pass a presence-only check.
//
// count_regions_backed_by() returns (size_t)-1 where the platform offers no
// region enumeration, which callers treat as "cannot assert here".

#pragma once

#include <string>

#if defined(__APPLE__)
#include <climits>
#include <cstdlib>
#include <libproc.h>
#include <sys/proc_info.h>
#include <unistd.h>
#elif defined(__linux__)
#include <climits>
#include <cstdlib>
#include <fstream>
#else
#include <cstdlib>
#endif

namespace test_region {

inline size_t count_regions_backed_by(const std::string& path) {
#if defined(__APPLE__)
    // PROC_PIDREGIONPATHINFO returns the region *and* its backing path in one
    // record, so the path is always the one belonging to the region reported.
    // proc_regionfilename() alone is not usable here: asked about the base of
    // an anonymous region it answers with the file of the next region above,
    // which counts an unrelated neighbour as a mapping of the weight file.
    size_t n = 0;
    const pid_t pid = getpid();
    uint64_t addr = 0;
    for (;;) {
        struct proc_regionwithpathinfo rpi;
        const int got = proc_pidinfo(pid, PROC_PIDREGIONPATHINFO, addr, &rpi, sizeof(rpi));
        if (got != (int)sizeof(rpi))
            break; // no region at or above `addr`
        if (rpi.prp_vip.vip_path[0] != '\0' && path == rpi.prp_vip.vip_path)
            n++;
        const uint64_t next = rpi.prp_prinfo.pri_address + rpi.prp_prinfo.pri_size;
        if (next <= addr)
            break; // no forward progress; stop rather than spin
        addr = next;
    }
    return n;
#elif defined(__linux__)
    size_t n = 0;
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        // The path is the last field and may contain spaces, so take
        // everything from the field separator rather than tokenizing on space.
        const size_t slash = line.find(" /");
        if (slash == std::string::npos)
            continue;
        if (line.substr(slash + 1) == path)
            n++;
    }
    return n;
#else
    (void)path;
    return (size_t)-1;
#endif
}

// True when this platform can answer the question at all.
inline bool region_probe_available() {
    return count_regions_backed_by("/") != (size_t)-1;
}

// The probe reports absolute paths, so a fixture written under a relative name
// has to be resolved before it can be compared.
inline std::string absolute_path_of(const std::string& path) {
#if defined(_WIN32)
    return path;
#else
    char resolved[PATH_MAX] = {0};
    if (!realpath(path.c_str(), resolved))
        return path;
    return std::string(resolved);
#endif
}

} // namespace test_region

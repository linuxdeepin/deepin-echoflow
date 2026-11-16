// test-issue-239-popen-mode.cpp — regression guard for issue #239.
//
// On Linux/POSIX, popen() only accepts "r" or "w"; passing "rb" returns
// NULL with errno=EINVAL (glibc), silently breaking the ffmpeg fallback
// path for .opus (and other ffmpeg-only formats). The fix strips 'b' in
// the POSIX branch of crispasr_popen before calling ::popen().
//
// These tests verify crispasr_popen can successfully open a real
// subprocess with mode "rb" and "r", and that the normalised mode reaches
// the shell (the command echoes $0 so we can confirm it ran at all).

#include <catch2/catch_test_macros.hpp>

#include "crispasr_popen.h"

#include <cstdio>
#include <string>

TEST_CASE("crispasr_popen: mode 'r' opens a subprocess", "[unit][issue-239][popen]") {
    FILE* p = crispasr::crispasr_popen("echo hello", "r");
    REQUIRE(p != nullptr);
    char buf[64] = {};
    (void)fgets(buf, sizeof(buf), p);
    crispasr::crispasr_pclose(p);
    std::string out(buf);
    REQUIRE(out.find("hello") != std::string::npos);
}

TEST_CASE("crispasr_popen: mode 'rb' opens a subprocess on all platforms", "[unit][issue-239][popen]") {
    // Before the fix this returned nullptr on Linux (errno=EINVAL).
    FILE* p = crispasr::crispasr_popen("echo hello", "rb");
    REQUIRE(p != nullptr);
    char buf[64] = {};
    (void)fgets(buf, sizeof(buf), p);
    crispasr::crispasr_pclose(p);
    std::string out(buf);
    REQUIRE(out.find("hello") != std::string::npos);
}

TEST_CASE("crispasr_popen: 'wb' mode strips 'b' cleanly", "[unit][issue-239][popen]") {
    // Write a byte to /dev/null via popen to confirm 'w'/'wb' normalise the same way.
    FILE* p = crispasr::crispasr_popen("cat > /dev/null", "wb");
    REQUIRE(p != nullptr);
    (void)fputc('x', p);
    crispasr::crispasr_pclose(p);
}

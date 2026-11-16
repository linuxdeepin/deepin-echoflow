#pragma once

// Portable setenv/unsetenv for tests.
//
// setenv/unsetenv are POSIX. MSVC does not have them at all, and MinGW does
// not declare them either ("'setenv' was not declared in this scope"), so any
// test that manipulates the environment fails to compile on Windows. Ten test
// files did, and nothing reported it: ci.yml's Windows job builds only
// crispasr-cli plus three named test targets, and build.yml's msbuild
// ALL_BUILD — the only thing that compiles the whole test tree there — had
// been red since 2026-08-18 for unrelated reasons.
//
// _putenv_s always overwrites, which matches every call site in the tests
// (they all pass overwrite=1), and setting a variable to "" is how Windows
// removes it.

#ifdef _WIN32
#include <cstdlib>

static inline int portable_setenv(const char* name, const char* value, int overwrite) {
    (void)overwrite;
    return _putenv_s(name, value);
}
static inline int portable_unsetenv(const char* name) {
    return _putenv_s(name, "");
}
#define setenv portable_setenv
#define unsetenv portable_unsetenv
#endif

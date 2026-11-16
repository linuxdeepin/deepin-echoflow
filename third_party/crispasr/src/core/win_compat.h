// win_compat.h — portable shims for POSIX calls missing on MSVC.
//
// MSVC has no setenv/unsetenv (only _putenv_s). Provide free-function shims so
// the existing setenv("X","1",overwrite) / unsetenv("X") call sites compile and
// behave identically on Windows. No-op on POSIX (uses the real libc versions).
#pragma once

#ifdef _WIN32
#include <cstdlib>

static inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite) {
        size_t sz = 0;
        if (getenv_s(&sz, nullptr, 0, name) == 0 && sz != 0)
            return 0; // already set and overwrite==0
    }
    return _putenv_s(name, value);
}

static inline int unsetenv(const char* name) {
    return _putenv_s(name, ""); // MSVC: empty value removes the variable
}
#endif

// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_PROJECT_PATHS_H
#define ECHOFLOW_PROJECT_PATHS_H

#include <cstdlib>
#include <filesystem>

namespace echoflow {

// Existing model paths are relative to the config directory. Reuse the old
// config in place so upgrading does not move or duplicate downloaded weights.
inline std::filesystem::path projectConfigPath(const std::filesystem::path& configRoot)
{
    const auto current = configRoot / "deepin" / "deepin-echoflow" / "deepin-echoflow.conf";
    if (!std::filesystem::exists(current)) {
        const auto previous = configRoot / "deepin-echoflow" / "deepin-echoflow.conf";
        if (std::filesystem::is_regular_file(previous)) {
            return previous;
        }
        const auto legacy = configRoot / "echoflow" / "echoflow.conf";
        if (std::filesystem::is_regular_file(legacy)) {
            return legacy;
        }
    }
    return current;
}

inline std::filesystem::path defaultProjectConfigPath()
{
    if (const char* configHome = std::getenv("XDG_CONFIG_HOME"); configHome && configHome[0] == '/') {
        return projectConfigPath(configHome);
    }
    const char* home = std::getenv("HOME");
    return projectConfigPath(std::filesystem::path(home ? home : "/tmp") / ".config");
}

} // namespace echoflow

#endif

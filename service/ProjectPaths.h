// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_PROJECT_PATHS_H
#define ECHOFLOW_PROJECT_PATHS_H

#include <filesystem>

namespace echoflow {

// Existing model paths are relative to the config directory. Reuse the old
// config in place so upgrading does not move or duplicate downloaded weights.
inline std::filesystem::path projectConfigPath(const std::filesystem::path& configRoot)
{
    const auto current = configRoot / "deepin-echoflow" / "deepin-echoflow.conf";
    const auto legacy = configRoot / "echoflow" / "echoflow.conf";
    if (std::filesystem::exists(current) || !std::filesystem::exists(legacy)) return current;
    return legacy;
}

} // namespace echoflow

#endif

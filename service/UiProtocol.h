// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_UI_PROTOCOL_H
#define ECHOFLOW_UI_PROTOCOL_H

#include <filesystem>

namespace echoflow {

inline std::filesystem::path uiAudioLevelSocketPath(const std::filesystem::path& stateSocket)
{
    return stateSocket.parent_path()
        / (stateSocket.stem().string() + "-level" + stateSocket.extension().string());
}

} // namespace echoflow

#endif

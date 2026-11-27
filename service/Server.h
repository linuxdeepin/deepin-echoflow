// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_SERVER_H
#define ECHOFLOW_SERVER_H

#include "Config.h"
#include "VoiceSession.h"

#include <optional>

namespace echoflow {

class Server {
public:
    Server(Config cfg, VoiceSession& session,
           std::optional<ConfigFileSnapshot> startupConfig = std::nullopt);
    int run();
    int run(int activationFd, const std::filesystem::path& serverPath);

private:
    Config cfg_;
    VoiceSession& session_;
    std::optional<ConfigFileSnapshot> startupConfig_;
};

}  // namespace echoflow

#endif

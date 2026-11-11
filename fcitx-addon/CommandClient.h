// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstring>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace echoflow {

enum class CommandChannel { Control, Wakeup, Ui };

inline bool sendAddonCommand(const std::string &runtimeDir, CommandChannel channel,
                             std::string_view command) {
    const char *name = channel == CommandChannel::Wakeup ? "echoflow-wakeup.sock"
        : channel == CommandChannel::Ui ? "echoflow-ui.sock" : "echoflow-control.sock";
    const std::string path = runtimeDir + "/" + name;
    sockaddr_un server {};
    if (path.size() >= sizeof(server.sun_path)) {
        return false;
    }
    const int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) {
        return false;
    }
    server.sun_family = AF_UNIX;
    std::memcpy(server.sun_path, path.c_str(), path.size() + 1);
    // These notifications need no reply or temporary client socket. Never block
    // Fcitx's event loop if a receiver stops draining its datagram queue.
    const bool sent = sendto(fd, command.data(), command.size(), 0,
        reinterpret_cast<sockaddr *>(&server), sizeof(server)) >= 0;
    close(fd);
    return sent;
}

inline bool sendAddonBlur(const std::string &runtimeDir) {
    // Keep all focus updates on one channel so an old service-relayed BLUR
    // cannot arrive after a newer FOCUS. Cancel ASR only if it is running.
    const bool sent = sendAddonCommand(runtimeDir, CommandChannel::Ui, "BLUR");
    sendAddonCommand(runtimeDir, CommandChannel::Control, "BLUR ui-notified");
    return sent;
}

} // namespace echoflow

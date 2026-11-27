// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ModelUninstallGuard.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace echoflow {

ModelUseLock::Result acquireModelUninstallLock(const std::filesystem::path& controlPath,
                                               ModelUseLock& lock)
{
    using Result = ModelUseLock::Result;
    auto result = lock.tryLockForRemoval();
    if (result == Result::Error) {
        return result;
    }

    sockaddr_un server {};
    const auto path = controlPath.string();
    if (path.size() >= sizeof(server.sun_path)) {
        return Result::Error;
    }
    server.sun_family = AF_UNIX;
    std::memcpy(server.sun_path, path.c_str(), path.size() + 1);
    const int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return Result::Error;
    }
    if (connect(fd, reinterpret_cast<const sockaddr*>(&server), sizeof(server)) != 0) {
        const int error = errno;
        close(fd);
        if (error != ENOENT && error != ECONNREFUSED) {
            return Result::Error;
        }
        // A service may have exited between the first lock attempt and connect().
        return result == Result::Busy ? lock.tryLockForRemoval() : result;
    }
    if (result == Result::Acquired) {
        // A running service without a shared model lock predates this protocol.
        close(fd);
        return Result::Error;
    }

    sockaddr_un replyAddress {};
    replyAddress.sun_family = AF_UNIX;
    constexpr char kCommand[] = "RELEASE_MODELS";
    const bool sent = bind(fd, reinterpret_cast<const sockaddr*>(&replyAddress),
                           sizeof(replyAddress.sun_family)) == 0
        && send(fd, kCommand, sizeof(kCommand) - 1, 0) == sizeof(kCommand) - 1;
    pollfd replySocket{fd, POLLIN, 0};
    char reply[16];
    const auto size = sent && poll(&replySocket, 1, 400) == 1
        ? recv(fd, reply, sizeof(reply), 0) : -1;
    close(fd);
    if (size == 5 && std::memcmp(reply, "BUSY\n", 5) == 0) {
        return Result::Busy;
    }
    if (size != 3 || std::memcmp(reply, "OK\n", 3) != 0) {
        return sent && size < 0 ? Result::Busy : Result::Error;
    }

    using Clock = std::chrono::steady_clock;
    using namespace std::chrono_literals;
    const auto deadline = Clock::now() + 600ms;
    do {
        result = lock.tryLockForRemoval();
        if (result != Result::Busy) {
            return result;
        }
        std::this_thread::sleep_for(10ms);
    } while (Clock::now() < deadline);
    return Result::Busy;
}

} // namespace echoflow

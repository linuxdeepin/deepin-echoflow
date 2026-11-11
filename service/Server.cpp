// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Server.h"

#include "SingleInstance.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace echoflow {

namespace {

constexpr std::array<const char*, 10> kAllowedCommands = {
    "FOCUS", "REFOCUS", "CARET", "BLUR", "CTRL_DOWN", "TYPED", "CANCEL", "STATUS", "START", "STOP"};

std::string trim(const std::string& value)
{
    auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string upper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

bool isAllowed(const std::string& command)
{
    std::string trimmed = trim(command);
    auto space = trimmed.find(' ');
    std::string verb = upper(space == std::string::npos ? trimmed : trimmed.substr(0, space));
    return std::find(kAllowedCommands.begin(), kAllowedCommands.end(), verb) != kAllowedCommands.end();
}

// Focus, caret, typing reports and STATUS polls arrive continuously; logging
// every one would bury the session commands that explain a cancelled commit.
bool isSessionCommand(const std::string& verb)
{
    return verb != "FOCUS" && verb != "REFOCUS" && verb != "CARET" && verb != "TYPED"
        && verb != "STATUS";
}

int systemdActivationSocket()
{
    const char* pidValue = std::getenv("LISTEN_PID");
    const char* fdsValue = std::getenv("LISTEN_FDS");
    if (!pidValue || !fdsValue || std::strtol(pidValue, nullptr, 10) != getpid()
        || std::strtol(fdsValue, nullptr, 10) != 1) {
        return -1;
    }
    unsetenv("LISTEN_PID");
    unsetenv("LISTEN_FDS");
    constexpr int fd = 3;
    int type = 0;
    socklen_t typeSize = sizeof(type);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &typeSize) != 0
        || type != SOCK_DGRAM) {
        return -1;
    }
    return fd;
}

std::chrono::milliseconds idleTimeout()
{
    constexpr long kDefaultTimeoutMs = 10 * 60 * 1000;
    const char* value = std::getenv("ECHOFLOW_IDLE_TIMEOUT_MS");
    if (!value || !*value) {
        return std::chrono::milliseconds(kDefaultTimeoutMs);
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0
        ? std::chrono::milliseconds(parsed) : std::chrono::milliseconds(kDefaultTimeoutMs);
}

bool isActivityCommand(const std::string& verb)
{
    return verb == "START" || verb == "STOP" || verb == "CTRL_DOWN" || verb == "CANCEL";
}

}  // namespace

Server::Server(Config cfg, VoiceSession& session)
    : cfg_(std::move(cfg))
    , session_(session)
{
}

int Server::run()
{
    const int activationFd = systemdActivationSocket();
    if (activationFd < 0) {
        log("service must be started by echoflow.socket activation");
        return 1;
    }
    auto serverPath = controlSocketPath(cfg_);
    SingleInstanceGuard instance;
    std::string error;
    if (!instance.acquire(serverPath, &error)) {
        log(error);
        return 1;
    }
    const int fd = instance.fd();
    log("EchoFlow service listening on " + serverPath.string());

    using Clock = std::chrono::steady_clock;
    const auto timeout = idleTimeout();
    auto lastActivity = Clock::now();
    auto lastState = session_.state();
    char buf[4096];
    for (;;) {
        const auto now = Clock::now();
        const auto state = session_.state();
        if (state != lastState) {
            if (state == SessionState::Idle) {
                lastActivity = now;
            }
            lastState = state;
        }
        if (state == SessionState::Idle && now - lastActivity >= timeout) {
            log("service idle timeout reached; exiting normally");
            return 0;
        }

        int waitMs = 1000;
        if (state == SessionState::Idle) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                timeout - (now - lastActivity));
            waitMs = static_cast<int>(std::max<long long>(0,
                std::min<long long>(remaining.count(), waitMs)));
        }
        pollfd watched[2] = {{fd, POLLIN, 0}, {activationFd, POLLIN, 0}};
        const int ready = poll(watched, 2, waitMs);
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            log(std::string("poll failed: ") + std::strerror(errno));
            continue;
        }
        if (ready == 0) {
            continue;
        }

        for (const auto& source : watched) {
            if (!(source.revents & POLLIN)) {
                continue;
            }
            sockaddr_un peer {};
            socklen_t peerLen = sizeof(peer);
            const ssize_t n = recvfrom(source.fd, buf, sizeof(buf) - 1, 0,
                reinterpret_cast<sockaddr*>(&peer), &peerLen);
            if (n <= 0) {
                continue;
            }
            buf[n] = '\0';
            std::string command = trim(std::string(buf, n));
            auto space = command.find(' ');
            std::string verb = upper(space == std::string::npos ? command : command.substr(0, space));
            if (isSessionCommand(verb)) {
                log("control command: " + command);
            }
            std::string reply = isAllowed(command) ? session_.handleCommand(command) + "\n"
                                                   : "ERR unknown-command\n";
            if (peerLen > 0) {
                sendto(source.fd, reply.data(), reply.size(), 0,
                    reinterpret_cast<sockaddr*>(&peer), peerLen);
            }
            const auto after = session_.state();
            if (after != lastState) {
                if (after == SessionState::Idle) {
                    lastActivity = Clock::now();
                }
                lastState = after;
            } else if (after == SessionState::Idle && isActivityCommand(verb)) {
                lastActivity = Clock::now();
            }
        }
    }
}

}  // namespace echoflow

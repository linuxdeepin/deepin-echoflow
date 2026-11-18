// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Committer.h"

#include <chrono>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace echoflow {

Committer::Committer(std::filesystem::path fcitxSocket)
    : fcitxSocket_(std::move(fcitxSocket))
{
}

std::pair<bool, std::string> Committer::beginVoiceSession(std::uint64_t generation)
{
    return sendRequest("VOICE_BEGIN " + std::to_string(generation) + "\n");
}

std::pair<bool, std::string> Committer::cancelVoiceSession(std::uint64_t generation)
{
    return sendRequest("VOICE_CANCEL " + std::to_string(generation) + "\n");
}

std::pair<bool, std::string> Committer::commitText(const std::string& text)
{
    return sendRequest("COMMIT\n" + text);
}

std::pair<bool, std::string> Committer::commitText(
    const std::string& text, std::uint64_t generation)
{
    return sendRequest("COMMIT " + std::to_string(generation) + "\n" + text);
}

std::pair<bool, std::string> Committer::setPreedit(const std::string& text)
{
    return sendRequest("PREEDIT\n" + text, 20);
}

std::pair<bool, std::string> Committer::setPreedit(
    const std::string& text, std::uint64_t generation)
{
    return sendRequest("PREEDIT " + std::to_string(generation) + "\n" + text, 20);
}

std::pair<bool, std::string> Committer::setPreedit(
    const std::string& text, std::uint64_t generation, std::uint64_t revision)
{
    return sendRequest("PREEDIT " + std::to_string(generation) + " "
        + std::to_string(revision) + "\n" + text, 20);
}

std::pair<bool, std::string> Committer::clearPreedit()
{
    return sendRequest("CLEAR_PREEDIT\n");
}

std::pair<bool, std::string> Committer::clearPreedit(std::uint64_t generation)
{
    return sendRequest("CLEAR_PREEDIT " + std::to_string(generation) + "\n");
}

std::pair<bool, std::string> Committer::sendRequest(const std::string& payload, int timeoutMs)
{
    int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return {false, "socket"};
    }

    sockaddr_un clientAddr {};
    clientAddr.sun_family = AF_UNIX;
    std::string clientPath = "/tmp/deepin-echoflow-client-" + std::to_string(getpid()) + "-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".sock";
    unlink(clientPath.c_str());
    strncpy(clientAddr.sun_path, clientPath.c_str(), sizeof(clientAddr.sun_path) - 1);
    if (bind(fd, reinterpret_cast<sockaddr*>(&clientAddr), sizeof(clientAddr)) < 0) {
        close(fd);
        unlink(clientPath.c_str());
        return {false, "bind"};
    }

    sockaddr_un serverAddr {};
    serverAddr.sun_family = AF_UNIX;
    strncpy(serverAddr.sun_path, fcitxSocket_.c_str(), sizeof(serverAddr.sun_path) - 1);
    if (sendto(fd, payload.data(), payload.size(), 0,
               reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr))
        < 0) {
        close(fd);
        unlink(clientPath.c_str());
        return {false, "sendto"};
    }

    pollfd replyFd {fd, POLLIN, 0};
    int ready;
    do {
        ready = poll(&replyFd, 1, timeoutMs);
    } while (ready < 0 && errno == EINTR);
    char buf[256] = {};
    ssize_t n = ready > 0 ? recv(fd, buf, sizeof(buf) - 1, 0) : -1;
    close(fd);
    unlink(clientPath.c_str());
    if (n <= 0) {
        return {false, "timeout"};
    }

    std::string reply(buf, n);
    auto end = reply.find_last_not_of(" \t\r\n");
    reply = end == std::string::npos ? std::string() : reply.substr(0, end + 1);
    return {reply == "OK", reply};
}

}  // namespace echoflow

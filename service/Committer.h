// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_COMMITTER_H
#define ECHOFLOW_COMMITTER_H

#include "Interfaces.h"

#include <cstdint>
#include <filesystem>

namespace echoflow {

class Committer : public ICommitter {
public:
    explicit Committer(std::filesystem::path fcitxSocket);
    std::pair<bool, std::string> beginVoiceSession(std::uint64_t generation) override;
    std::pair<bool, std::string> cancelVoiceSession(std::uint64_t generation) override;
    std::pair<bool, std::string> commitText(const std::string& text) override;
    std::pair<bool, std::string> commitText(
        const std::string& text, std::uint64_t generation) override;
    std::pair<bool, std::string> setPreedit(const std::string& text) override;
    std::pair<bool, std::string> setPreedit(
        const std::string& text, std::uint64_t generation) override;
    std::pair<bool, std::string> setPreedit(
        const std::string& text, std::uint64_t generation,
        std::uint64_t revision) override;
    std::pair<bool, std::string> clearPreedit() override;
    std::pair<bool, std::string> clearPreedit(std::uint64_t generation) override;

private:
    std::pair<bool, std::string> sendRequest(const std::string& payload, int timeoutMs = 500);
    std::filesystem::path fcitxSocket_;
};

}  // namespace echoflow

#endif

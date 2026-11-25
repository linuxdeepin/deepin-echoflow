// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_ASR_HOTWORDS_H
#define ECHOFLOW_ASR_HOTWORDS_H

#include <string>
#include <vector>

namespace echoflow {

class AsrHotwords {
public:
    explicit AsrHotwords(const std::string& value = {});
    const std::string& prompt() const { return prompt_; }
    std::string filterEcho(std::string text) const;

private:
    std::string prompt_;
    std::vector<std::string> echoTerms_;
};

} // namespace echoflow

#endif

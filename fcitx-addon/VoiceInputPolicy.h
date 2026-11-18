// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_VOICE_INPUT_POLICY_H
#define ECHOFLOW_VOICE_INPUT_POLICY_H

#include <fcitx-utils/capabilityflags.h>
#include <cstdint>

namespace echoflow {

// VoiceSession numbers come from the system monotonic clock and advance
// across daemon restarts. Keep the watermark after cancel/commit.
class VoiceGenerations {
public:
    bool canBegin(std::uint64_t generation) const { return generation > latest_; }
    void begin(std::uint64_t generation) { latest_ = generation; }

private:
    std::uint64_t latest_ = 0;
};

inline bool isVoiceInputAllowed(fcitx::CapabilityFlags capabilities)
{
    return !capabilities.testAny(fcitx::CapabilityFlag::PasswordOrSensitive);
}

} // namespace echoflow

#endif

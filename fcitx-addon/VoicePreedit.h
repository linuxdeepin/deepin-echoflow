// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_VOICE_PREEDIT_H
#define ECHOFLOW_VOICE_PREEDIT_H

#include <fcitx/inputpanel.h>
#include <fcitx/text.h>

#include <string>

namespace echoflow {

class VoicePreedit {
public:
    void begin(const fcitx::InputPanel& panel)
    {
        originalClient_ = panel.clientPreedit();
        originalPanel_ = panel.preedit();
        saved_ = true;
    }

    void update(fcitx::InputPanel& panel, const std::string& text, bool inlinePreedit)
    {
        fcitx::Text preedit;
        if (!text.empty()) {
            preedit.append(text, fcitx::TextFormatFlag::Underline);
            preedit.setCursor(static_cast<int>(text.size()));
        }
        panel.setClientPreedit(inlinePreedit ? preedit : fcitx::Text());
        panel.setPreedit(inlinePreedit ? fcitx::Text() : preedit);
    }

    bool restore(fcitx::InputPanel& panel)
    {
        if (!saved_) return false;
        panel.setClientPreedit(originalClient_);
        panel.setPreedit(originalPanel_);
        reset();
        return true;
    }

    void reset()
    {
        originalClient_ = fcitx::Text();
        originalPanel_ = fcitx::Text();
        saved_ = false;
    }

private:
    fcitx::Text originalClient_;
    fcitx::Text originalPanel_;
    bool saved_ = false;
};

} // namespace echoflow

#endif

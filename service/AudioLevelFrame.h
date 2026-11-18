// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_AUDIO_LEVEL_FRAME_H
#define ECHOFLOW_AUDIO_LEVEL_FRAME_H

#include <array>

namespace echoflow {

// Bars describe seven frequency bands from one current audio window.
// One absolute loudness scale drives the frame; values are in [0, 1].
struct AudioLevelFrame {
    double level = 0;
    std::array<double, 7> bars{};
};

} // namespace echoflow

#endif

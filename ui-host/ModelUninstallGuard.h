// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "ModelUseLock.h"

namespace echoflow {

// Claims exclusive access after an idle service has released its models.
ModelUseLock::Result acquireModelUninstallLock(const std::filesystem::path& controlPath,
                                               ModelUseLock& lock);

} // namespace echoflow

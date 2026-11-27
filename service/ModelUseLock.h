// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>

namespace echoflow {

// Held by the service from config load until all model owners are destroyed.
// The UI holds it exclusively while switching the selection and removing weights.
class ModelUseLock {
public:
    enum class Result { Acquired, Busy, Error };

    explicit ModelUseLock(const std::filesystem::path& controlPath);
    ~ModelUseLock();

    ModelUseLock(const ModelUseLock&) = delete;
    ModelUseLock& operator=(const ModelUseLock&) = delete;

    Result lockForUse();
    Result tryLockForRemoval();

private:
    Result lock(int operation);
    int fd_ = -1;
};

} // namespace echoflow

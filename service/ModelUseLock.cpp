// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ModelUseLock.h"

#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace echoflow {

ModelUseLock::ModelUseLock(const std::filesystem::path& controlPath)
{
    auto lockPath = controlPath;
    lockPath += ".models-" + std::to_string(getuid()) + ".lock";
    fd_ = open(lockPath.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0600);
    if (fd_ >= 0) {
        struct stat info {};
        if (fstat(fd_, &info) != 0 || !S_ISREG(info.st_mode) || info.st_uid != getuid()) {
            close(fd_);
            fd_ = -1;
        }
    }
}

ModelUseLock::~ModelUseLock()
{
    if (fd_ >= 0) {
        close(fd_);
    }
}

ModelUseLock::Result ModelUseLock::lockForUse()
{
    return lock(LOCK_SH);
}

ModelUseLock::Result ModelUseLock::tryLockForRemoval()
{
    return lock(LOCK_EX | LOCK_NB);
}

ModelUseLock::Result ModelUseLock::lock(int operation)
{
    if (fd_ < 0) {
        return Result::Error;
    }
    while (flock(fd_, operation) != 0) {
        if (errno != EINTR) {
            return errno == EWOULDBLOCK ? Result::Busy : Result::Error;
        }
    }
    return Result::Acquired;
}

} // namespace echoflow

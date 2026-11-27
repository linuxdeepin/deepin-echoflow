// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cerrno>
#include <cstring>
#include <string>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace echoflow {

class CommitSocket {
public:
    enum class Result { Listening, Busy, Failed };

    CommitSocket() = default;
    CommitSocket(const CommitSocket &) = delete;
    CommitSocket &operator=(const CommitSocket &) = delete;

    ~CommitSocket() {
        if (fd_ < 0) return;
        struct stat current {};
        if (lstat(path_.c_str(), &current) == 0 && current.st_dev == identity_.st_dev
            && current.st_ino == identity_.st_ino) {
            unlink(path_.c_str());
        }
        close(fd_);
        close(lockFd_);
    }

    int fd() const { return fd_; }
    int error() const { return error_; }

    Result open(const std::string &path) {
        if (fd_ >= 0) return Result::Listening;
        sockaddr_un addr {};
        if (path.size() >= sizeof(addr.sun_path)) {
            error_ = ENAMETOOLONG;
            return Result::Failed;
        }
        path_ = path;
        // Keep the lock file across runs so waiters always lock the same inode.
        const auto lockPath = path + ".lock";
        int lockFd = ::open(lockPath.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (lockFd < 0) {
            error_ = errno;
            return Result::Failed;
        }
        if (flock(lockFd, LOCK_EX | LOCK_NB) != 0) {
            error_ = errno;
            close(lockFd);
            return error_ == EWOULDBLOCK ? Result::Busy : Result::Failed;
        }

        int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        const auto release = [&](Result result, int error) {
            if (fd >= 0) close(fd);
            close(lockFd);
            error_ = error;
            return result;
        };
        if (fd < 0) return release(Result::Failed, errno);
        addr.sun_family = AF_UNIX;
        std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
        const auto *address = reinterpret_cast<const sockaddr *>(&addr);
        if (bind(fd, address, sizeof(addr)) != 0) {
            if (errno != EADDRINUSE) return release(Result::Failed, errno);
            struct stat existing {};
            if (lstat(path.c_str(), &existing) != 0) {
                return release(errno == ENOENT ? Result::Busy : Result::Failed, errno);
            }
            if (!S_ISSOCK(existing.st_mode) || existing.st_uid != getuid()) {
                return release(Result::Failed, EADDRINUSE);
            }
            // A running addon from before the lock was introduced may own it.
            if (connect(fd, address, sizeof(addr)) == 0) return release(Result::Busy, 0);
            if (errno == ENOENT) return release(Result::Busy, errno);
            if (errno != ECONNREFUSED) return release(Result::Failed, errno);
            if (unlink(path.c_str()) != 0) {
                return release(errno == ENOENT ? Result::Busy : Result::Failed, errno);
            }
            if (bind(fd, address, sizeof(addr)) != 0) {
                return release(errno == EADDRINUSE ? Result::Busy : Result::Failed, errno);
            }
        }
        // The socket fd inode is on sockfs, not the runtime directory filesystem.
        if (lstat(path.c_str(), &identity_) != 0) return release(Result::Failed, errno);
        fd_ = fd;
        lockFd_ = lockFd;
        error_ = 0;
        return Result::Listening;
    }

private:
    std::string path_;
    struct stat identity_ {};
    int fd_ = -1;
    int lockFd_ = -1;
    int error_ = 0;
};

} // namespace echoflow

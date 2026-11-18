// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QtTest/QtTest>

#include "Committer.h"

#include <cerrno>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <vector>
#include <unistd.h>

using namespace echoflow;

class TestCommitter : public QObject {
    Q_OBJECT

private slots:
    void commitReturnsOkOnAck();
    void commitReturnsErrOnNack();
    void preeditCommandsUseAddonProtocol();
    void directCommitDoesNotSendClear();
    void preeditReturnsAddonRejection();
    void emptyPreeditUsesRevisionedProtocol();

private:
    std::string serverPath_;
    std::string setupError_;
    int setupServer();
};

int TestCommitter::setupServer()
{
    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) {
        setupError_ = "socket errno=" + std::to_string(errno);
        return fd;
    }
    timeval timeout {1, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    serverPath_ = std::string("/tmp/deepin-echoflow-test-") + std::to_string(getpid()) + ".sock";
    unlink(serverPath_.c_str());

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, serverPath_.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        setupError_ = "bind " + serverPath_ + " errno=" + std::to_string(errno);
        close(fd);
        return -1;
    }
    setupError_.clear();
    return fd;
}

struct ReplyResult {
    ssize_t received = -1;
    int receiveErrno = 0;
    int sent = -1;
    int sendErrno = 0;
    std::string payload;
};

static void recvAndReply(int fd, const std::string& reply, ReplyResult* result)
{
    char buf[512];
    sockaddr_un peer {};
    socklen_t peerLen = sizeof(peer);
    ssize_t n = recvfrom(fd, buf, sizeof(buf) - 1, 0, reinterpret_cast<sockaddr*>(&peer), &peerLen);
    result->received = n;
    result->receiveErrno = errno;
    if (n > 0) {
        result->payload.assign(buf, n);
        result->sent = sendto(fd, reply.data(), reply.size(), 0, reinterpret_cast<sockaddr*>(&peer), peerLen);
        result->sendErrno = errno;
    }
}

void TestCommitter::commitReturnsOkOnAck()
{
    int fd = setupServer();
    QVERIFY2(fd >= 0, setupError_.c_str());
    Committer committer(serverPath_);
    ReplyResult reply;
    std::thread server([fd, &reply] { recvAndReply(fd, "OK\n", &reply); });

    auto [ok, detail] = committer.commitText("hello");
    server.join();
    close(fd);
    unlink(serverPath_.c_str());

    QVERIFY2(reply.received > 0, std::string("server recv errno=" + std::to_string(reply.receiveErrno)).c_str());
    QCOMPARE(QString::fromStdString(reply.payload), QStringLiteral("COMMIT\nhello"));
    QVERIFY2(reply.sent > 0, std::string("server send errno=" + std::to_string(reply.sendErrno)).c_str());
    QVERIFY2(ok, detail.c_str());
    QCOMPARE(QString::fromStdString(detail), QStringLiteral("OK"));
}

void TestCommitter::commitReturnsErrOnNack()
{
    int fd = setupServer();
    QVERIFY2(fd >= 0, setupError_.c_str());
    Committer committer(serverPath_);
    ReplyResult reply;
    std::thread server([fd, &reply] { recvAndReply(fd, "NO\n", &reply); });

    auto [ok, detail] = committer.commitText("hello");
    server.join();
    close(fd);
    unlink(serverPath_.c_str());

    QVERIFY2(reply.received > 0, std::string("server recv errno=" + std::to_string(reply.receiveErrno)).c_str());
    QCOMPARE(QString::fromStdString(reply.payload), QStringLiteral("COMMIT\nhello"));
    QVERIFY2(reply.sent > 0, std::string("server send errno=" + std::to_string(reply.sendErrno)).c_str());
    QVERIFY(!ok);
    QCOMPARE(QString::fromStdString(detail), QStringLiteral("NO"));
}


void TestCommitter::preeditCommandsUseAddonProtocol()
{
    int fd = setupServer();
    QVERIFY2(fd >= 0, setupError_.c_str());
    Committer committer(serverPath_);
    std::vector<std::string> payloads;
    std::thread server([fd, &payloads] {
        for (int i = 0; i < 4; ++i) {
            char buf[512];
            sockaddr_un peer {};
            socklen_t peerLen = sizeof(peer);
            const ssize_t n = recvfrom(fd, buf, sizeof(buf), 0,
                reinterpret_cast<sockaddr*>(&peer), &peerLen);
            if (n > 0) {
                payloads.emplace_back(buf, static_cast<size_t>(n));
                const std::string reply = "OK\n";
                sendto(fd, reply.data(), reply.size(), 0,
                    reinterpret_cast<sockaddr*>(&peer), peerLen);
            }
        }
    });

    const auto [beginOk, beginDetail] = committer.beginVoiceSession(42);
    const auto [setOk, setDetail] = committer.setPreedit("partial text", 42, 1);
    const auto [clearOk, clearDetail] = committer.clearPreedit(42);
    const auto [commitOk, commitDetail] = committer.commitText("final text", 42);
    server.join();
    close(fd);
    unlink(serverPath_.c_str());

    QVERIFY2(beginOk, beginDetail.c_str());
    QVERIFY2(setOk, setDetail.c_str());
    QVERIFY2(clearOk, clearDetail.c_str());
    QVERIFY2(commitOk, commitDetail.c_str());
    QCOMPARE(payloads.size(), size_t(4));
    QCOMPARE(QString::fromStdString(payloads[0]), QStringLiteral("VOICE_BEGIN 42\n"));
    QCOMPARE(QString::fromStdString(payloads[1]), QStringLiteral("PREEDIT 42 1\npartial text"));
    QCOMPARE(QString::fromStdString(payloads[2]), QStringLiteral("CLEAR_PREEDIT 42\n"));
    QCOMPARE(QString::fromStdString(payloads[3]), QStringLiteral("COMMIT 42\nfinal text"));
}

void TestCommitter::directCommitDoesNotSendClear()
{
    const int fd = setupServer();
    QVERIFY2(fd >= 0, setupError_.c_str());
    Committer committer(serverPath_);
    std::vector<std::string> requests;
    std::thread server([&] {
        for (int i = 0; i < 3; ++i) {
            ReplyResult reply;
            recvAndReply(fd, "OK\n", &reply);
            requests.push_back(reply.payload);
        }
    });
    const auto begin = committer.beginVoiceSession(42);
    const auto preview = committer.setPreedit("前文", 42, 1);
    const auto commit = committer.commitText("前文和尾音", 42);
    server.join();
    close(fd);
    unlink(serverPath_.c_str());
    QVERIFY(begin.first);
    QVERIFY(preview.first);
    QVERIFY(commit.first);
    QCOMPARE(requests, (std::vector<std::string>{"VOICE_BEGIN 42\n", "PREEDIT 42 1\n前文",
                                                "COMMIT 42\n前文和尾音"}));
}

void TestCommitter::preeditReturnsAddonRejection()
{
    int fd = setupServer();
    QVERIFY2(fd >= 0, setupError_.c_str());
    Committer committer(serverPath_);
    ReplyResult reply;
    std::thread server([fd, &reply] {
        recvAndReply(fd, "ERR input-context-changed\n", &reply);
    });
    const auto [ok, detail] = committer.setPreedit("preview", 42, 1);
    server.join();
    close(fd);
    unlink(serverPath_.c_str());
    QVERIFY(!ok);
    QCOMPARE(QString::fromStdString(detail), QStringLiteral("ERR input-context-changed"));
}

void TestCommitter::emptyPreeditUsesRevisionedProtocol()
{
    const int fd = setupServer();
    QVERIFY2(fd >= 0, setupError_.c_str());
    Committer committer(serverPath_);
    ReplyResult reply;
    std::thread server([fd, &reply] { recvAndReply(fd, "OK\n", &reply); });
    const auto [ok, detail] = committer.setPreedit("", 42, 2);
    server.join();
    close(fd);
    unlink(serverPath_.c_str());
    QVERIFY2(ok, detail.c_str());
    QCOMPARE(reply.payload, std::string("PREEDIT 42 2\n"));
}

QTEST_GUILESS_MAIN(TestCommitter)
#include "test_committer.moc"

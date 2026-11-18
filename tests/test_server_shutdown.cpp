// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Server.h"

#include <QTemporaryDir>
#include <QtTest/QtTest>

#include <csignal>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace echoflow;

class Pipeline final : public ILiveVoicePipeline {
public:
    void start() override {}
    std::string finish() override { return {}; }
    void cancel() override { cancelled = true; }
    void setPartialTextCallback(std::function<void(const std::string&)>) override {}
    bool cancelled = false;
};

class CommitterProbe final : public ICommitter {
public:
    explicit CommitterProbe(int output) : output_(output) {}
    std::pair<bool, std::string> beginVoiceSession(std::uint64_t generation) override
    {
        generation_ = generation;
        return {write(output_, "B", 1) == 1, "begin"};
    }
    std::pair<bool, std::string> cancelVoiceSession(std::uint64_t generation) override
    {
        return {generation_ == generation && write(output_, "C", 1) == 1, "cancel"};
    }
    std::pair<bool, std::string> commitText(const std::string&) override { return {false, "unexpected commit"}; }

private:
    int output_;
    std::uint64_t generation_ = 0;
};

class Notifier final : public IUiNotifier {
public:
    void send(const std::string&) override {}
};

struct Child {
    pid_t pid;
    ~Child()
    {
        if (pid > 0) {
            kill(pid, SIGKILL);
            waitpid(pid, nullptr, 0);
        }
    }
};

char nextEvent(int fd)
{
    pollfd watched{fd, POLLIN, 0};
    char event = '\0';
    if (poll(&watched, 1, 3000) == 1 && read(fd, &event, 1) == 1) return event;
    return '\0';
}
} // namespace

class TestServerShutdown : public QObject {
    Q_OBJECT
private slots:
    void sigtermCancelsActiveRecording();
};

void TestServerShutdown::sigtermCancelsActiveRecording()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const auto path = directory.filePath("control.sock").toStdString();
    int activation[2];
    QCOMPARE(socketpair(AF_UNIX, SOCK_DGRAM, 0, activation), 0);
    int events[2];
    QCOMPARE(pipe(events), 0);
    Child child{fork()};
    QVERIFY(child.pid >= 0);
    if (child.pid == 0) {
        close(activation[0]);
        close(events[0]);
        Pipeline pipeline;
        CommitterProbe committer(events[1]);
        Notifier notifier;
        Config config;
        int status;
        {
            VoiceSession session(config, pipeline, committer, notifier);
            Server server(config, session);
            status = server.run(activation[1], path);
        }
        _exit(status == 0 && pipeline.cancelled ? 0 : 1);
    }
    close(activation[1]);
    close(events[1]);
    const auto sent = send(activation[0], "START", 5, 0);
    close(activation[0]);
    const char begin = nextEvent(events[0]);
    const auto signalled = kill(child.pid, SIGTERM);
    const char cancel = nextEvent(events[0]);
    const char closed = nextEvent(events[0]);
    close(events[0]);
    QCOMPARE(sent, ssize_t(5));
    QCOMPARE(begin, 'B');
    QCOMPARE(signalled, 0);
    QCOMPARE(cancel, 'C');
    QCOMPARE(closed, '\0');
    int status = 0;
    QCOMPARE(waitpid(child.pid, &status, 0), child.pid);
    child.pid = 0;
    QVERIFY(WIFEXITED(status));
    QCOMPARE(WEXITSTATUS(status), 0);
}

QTEST_GUILESS_MAIN(TestServerShutdown)
#include "test_server_shutdown.moc"

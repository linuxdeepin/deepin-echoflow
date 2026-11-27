// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Server.h"
#include "ModelUninstallGuard.h"

#include <QSaveFile>
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
    std::string finish() override
    {
        if (finishGate >= 0) {
            char proceed;
            if (write(events, "F", 1) == 1) {
                const auto bytes = read(finishGate, &proceed, 1);
                (void)bytes;
            }
        }
        return {};
    }
    void cancel() override { cancelled = true; }
    void setPartialTextCallback(std::function<void(const std::string&)>) override {}
    bool cancelled = false;
    int events = -1;
    int finishGate = -1;
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
    void releaseModelsWaitsForIdle();
    void settingsChangeDuringStartupPreservesWakeup();
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

void TestServerShutdown::releaseModelsWaitsForIdle()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const auto path = directory.filePath("control.sock").toStdString();
    int activation[2];
    QCOMPARE(socketpair(AF_UNIX, SOCK_DGRAM, 0, activation), 0);
    int events[2];
    QCOMPARE(pipe(events), 0);
    int finalize[2];
    QCOMPARE(pipe(finalize), 0);
    Child child{fork()};
    QVERIFY(child.pid >= 0);
    if (child.pid == 0) {
        close(activation[0]);
        close(events[0]);
        close(finalize[1]);
        int result = 1;
        {
            ModelUseLock modelLock(path);
            if (modelLock.lockForUse() == ModelUseLock::Result::Acquired) {
                Pipeline pipeline;
                pipeline.events = events[1];
                pipeline.finishGate = finalize[0];
                CommitterProbe committer(events[1]);
                Notifier notifier;
                Config config;
                VoiceSession session(config, pipeline, committer, notifier);
                Server server(config, session);
                result = server.run(activation[1], path);
            }
        }
        _exit(result);
    }
    close(activation[1]);
    close(events[1]);
    close(finalize[0]);
    QCOMPARE(send(activation[0], "START", 5, 0), ssize_t(5));
    QCOMPARE(nextEvent(events[0]), 'B');

    ModelUseLock uninstallLock(path);
    QCOMPARE(acquireModelUninstallLock(path, uninstallLock), ModelUseLock::Result::Busy);
    QCOMPARE(send(activation[0], "STOP", 4, 0), ssize_t(4));
    QCOMPARE(nextEvent(events[0]), 'F');
    QCOMPARE(acquireModelUninstallLock(path, uninstallLock), ModelUseLock::Result::Busy);
    QCOMPARE(write(finalize[1], "F", 1), ssize_t(1));
    close(finalize[1]);
    QCOMPARE(nextEvent(events[0]), 'C');
    QTRY_COMPARE(acquireModelUninstallLock(path, uninstallLock), ModelUseLock::Result::Acquired);
    close(activation[0]);
    QCOMPARE(nextEvent(events[0]), '\0');
    close(events[0]);
    int status = 0;
    QCOMPARE(waitpid(child.pid, &status, 0), child.pid);
    child.pid = 0;
    QVERIFY(WIFEXITED(status));
    QCOMPARE(WEXITSTATUS(status), 0);
}

void TestServerShutdown::settingsChangeDuringStartupPreservesWakeup()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString configPath = directory.filePath("config.ini");
    const auto writeConfig = [&configPath](const QByteArray& model) {
        QSaveFile file(configPath);
        const QByteArray contents = "[basic.model.model_name]\nvalue=" + model + '\n';
        return file.open(QIODevice::WriteOnly) && file.write(contents) == contents.size()
            && file.commit();
    };
    QVERIFY(writeConfig("qwen3-asr-0.6b"));
    ConfigFileSnapshot startupConfig(configPath.toStdString());
    Config config = startupConfig.load();
    QCOMPARE(config.modelName, "qwen3-asr-0.6b");
    QVERIFY(!startupConfig.changed());
    // Model initialization is still in progress. No control socket exists yet.
    QVERIFY(writeConfig("qwen3-asr-1.7b"));

    int activation[2];
    QCOMPARE(socketpair(AF_UNIX, SOCK_DGRAM, 0, activation), 0);
    QCOMPARE(send(activation[0], "START", 5, 0), ssize_t(5));
    close(activation[0]);
    Child child{fork()};
    QVERIFY(child.pid >= 0);
    if (child.pid == 0) {
        qputenv("ECHOFLOW_IDLE_TIMEOUT_MS", "100");
        Pipeline pipeline;
        CommitterProbe committer(-1);
        Notifier notifier;
        VoiceSession session(config, pipeline, committer, notifier);
        Server server(config, session, std::move(startupConfig));
        _exit(server.run(activation[1], directory.filePath("control.sock").toStdString()));
    }
    int status = 0;
    QCOMPARE(waitpid(child.pid, &status, 0), child.pid);
    child.pid = 0;
    char pending[16];
    const auto size = recv(activation[1], pending, sizeof(pending), MSG_DONTWAIT);
    close(activation[1]);
    QVERIFY(WIFEXITED(status));
    QCOMPARE(WEXITSTATUS(status), 0);
    QCOMPARE(size, ssize_t(5));
    QCOMPARE(QByteArray(pending, size), QByteArray("START"));
    QCOMPARE(ConfigFileSnapshot(configPath.toStdString()).load().modelName, "qwen3-asr-1.7b");
}

QTEST_GUILESS_MAIN(TestServerShutdown)
#include "test_server_shutdown.moc"

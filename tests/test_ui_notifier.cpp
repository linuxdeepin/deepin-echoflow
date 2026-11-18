// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "UiNotifier.h"
#include "AudioLevelMeter.h"
#include "UiProtocol.h"
#include "UiSocketServer.h"
#include "CapsuleController.h"
#include "VoiceSession.h"

#include <QTest>
#include <QTemporaryDir>
#include <QSignalSpy>
#include <QScopeGuard>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace echoflow;
using namespace std::chrono_literals;

namespace {
class UiInbox {
public:
    QTemporaryDir directory;
    std::string path;
    int fd = -1;
    std::vector<int> senders;
    explicit UiInbox(std::string socketPath = {})
    {
        path = socketPath.empty() ? directory.filePath("ui.sock").toStdString() : std::move(socketPath);
        fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (fd < 0) return;
        sockaddr_un address {};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
            close(fd);
            fd = -1;
        }
    }
    ~UiInbox()
    {
        for (int sender : senders) close(sender);
        if (fd >= 0) close(fd);
    }
    bool fill()
    {
        return fillQueue(path, senders, "x");
    }
    static bool fillQueue(const std::string& path, std::vector<int>& senders,
                          const std::string& payload)
    {
        sockaddr_un address {};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        // One sender can exhaust its own SO_SNDBUF before the receiver queue
        // is full. Keep senders alive and prove a fresh sender cannot enqueue.
        for (int peer = 0; peer < 64; ++peer) {
            const int sender = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
            if (sender < 0) return false;
            senders.push_back(sender);
            size_t sent = 0;
            for (int i = 0; i < 100000; ++i) {
                if (sendto(sender, payload.data(), payload.size(), 0,
                           reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
                    if (errno != EAGAIN && errno != EWOULDBLOCK) return false;
                    if (sent == 0) return true;
                    break;
                }
                ++sent;
            }
        }
        return false;
    }
};

struct Pipeline : ILiveVoicePipeline {
    std::function<void(const AudioLevelFrame&)> level;
    void setAudioLevelCallback(std::function<void(const AudioLevelFrame&)> value) override { level = std::move(value); }
    std::function<void(const std::string&)> partial;
    std::function<void(const std::string&)> confirmed;
    void start() override {}
    std::string finish() override { return "first"; }
    void cancel() override {}
    void setPartialTextCallback(std::function<void(const std::string&)> value) override { partial = std::move(value); }
    void setConfirmedTextCallback(std::function<void(const std::string&)> value) override { confirmed = std::move(value); }
};
struct Committer : ICommitter {
    std::atomic<int> commits{0};
    std::atomic<bool> polished{false};
    std::pair<bool, std::string> commitText(const std::string&) override { ++commits; return {true, "OK"}; }
    std::pair<bool, std::string> setPreedit(const std::string& text) override {
        if (text == "P:first") polished = true;
        return {true, "OK"};
    }
};
struct Polisher : ITextPolisher {
    std::string polish(const std::string& text, const std::function<bool()>&) override { return "P:" + text; }
};
} // namespace

class TestUiNotifier : public QObject {
    Q_OBJECT
private slots:
    void audioUsesIndependentQueue();
    void fullAudioQueueStillDeliversStates();
    void fullAudioQueueStillFinishesCapsule_data();
    void fullAudioQueueStillFinishesCapsule();
    void newRecordingDiscardsOldAudio();
    void audioQueueKeepsLatestLevel();
    void capturedWindowReachesWaveformInOneUpdate();
    void sendsDatagram();
    void fullReceiverDoesNotBlock();
    void busyUiCannotBlockStopOrCancel();
};

void TestUiNotifier::audioUsesIndependentQueue()
{
    UiInbox state;
    UiInbox audio(uiAudioLevelSocketPath(state.path).string());
    QVERIFY(state.fd >= 0 && audio.fd >= 0);
    UnixDatagramUiNotifier notifier(state.path);
    notifier.send("AUDIO_LEVEL 700");
    char buffer[128];
    const auto size = recv(audio.fd, buffer, sizeof(buffer), 0);
    QVERIFY(size > 0);
    QCOMPARE(std::string(buffer, size), std::string("AUDIO_LEVEL 700"));
    QCOMPARE(recv(state.fd, buffer, sizeof(buffer), 0), ssize_t(-1));
    QVERIFY(errno == EAGAIN || errno == EWOULDBLOCK);
    notifier.send("IDLE");
    const auto stateSize = recv(state.fd, buffer, sizeof(buffer), 0);
    QVERIFY(stateSize > 0);
    QCOMPARE(std::string(buffer, stateSize), std::string("IDLE"));
}

void TestUiNotifier::fullAudioQueueStillDeliversStates()
{
    UiInbox state;
    UiInbox audio(uiAudioLevelSocketPath(state.path).string());
    QVERIFY(state.fd >= 0 && audio.fd >= 0);
    QVERIFY(audio.fill());
    UnixDatagramUiNotifier notifier(state.path);
    const auto started = std::chrono::steady_clock::now();
    notifier.send("AUDIO_LEVEL 700");
    notifier.send("TRANSCRIBING");
    notifier.send("IDLE");
    QVERIFY(std::chrono::steady_clock::now() - started < 200ms);
    for (const auto& expected : {std::string("TRANSCRIBING"), std::string("IDLE")}) {
        char buffer[128];
        const auto size = recv(state.fd, buffer, sizeof(buffer), 0);
        QVERIFY(size > 0);
        QCOMPARE(std::string(buffer, size), expected);
    }
}

void TestUiNotifier::fullAudioQueueStillFinishesCapsule_data()
{
    QTest::addColumn<QString>("command");
    QTest::newRow("stop") << QStringLiteral("STOP");
    QTest::newRow("cancel") << QStringLiteral("CANCEL");
}

void TestUiNotifier::fullAudioQueueStillFinishesCapsule()
{
    QFETCH(QString, command);
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const auto statePath = directory.filePath("ui.sock");
    CapsuleController controller([](std::string_view) { return true; });
    UiSocketServer server(statePath, &controller);
    QString error;
    QVERIFY2(server.start(&error), qPrintable(error));
    UnixDatagramUiNotifier notifier(statePath.toStdString());
    Pipeline pipeline;
    Committer committer;
    VoiceSession session(Config::defaultConfig(), pipeline, committer, notifier);
    session.handleCommand("START");
    QTRY_VERIFY(!controller.idle());
    std::vector<int> senders;
    const auto closeSenders = qScopeGuard([&] { for (int fd : senders) close(fd); });
    // Pause UI dispatch and fill the real UI audio socket to EAGAIN.
    QVERIFY(UiInbox::fillQueue(uiAudioLevelSocketPath(statePath.toStdString()).string(),
                              senders, "AUDIO_LEVEL 900"));
    pipeline.level(AudioLevelFrame{0.7});
    session.handleCommand(command.toStdString());
    QTRY_COMPARE(session.state(), SessionState::Idle);
    QTRY_VERIFY(controller.idle());
    QCOMPARE(controller.audioLevel(), 0.0);
    QCOMPARE(committer.commits.load(), command == QStringLiteral("STOP") ? 1 : 0);
}

void TestUiNotifier::newRecordingDiscardsOldAudio()
{
    QTemporaryDir directory;
    CapsuleController controller([](std::string_view) { return true; });
    UiSocketServer server(directory.filePath("ui.sock"), &controller);
    QString error;
    QVERIFY2(server.start(&error), qPrintable(error));
    UnixDatagramUiNotifier notifier(directory.filePath("ui.sock").toStdString());
    notifier.send("AUDIO_LEVEL 900");
    notifier.send("RECORDING");
    QTRY_VERIFY(!controller.idle());
    QCoreApplication::processEvents();
    QCOMPARE(controller.audioLevel(), 0.0);
    notifier.send("AUDIO_LEVEL 300");
    QTRY_COMPARE(controller.audioLevel(), 0.3);
}

void TestUiNotifier::audioQueueKeepsLatestLevel()
{
    QTemporaryDir directory;
    CapsuleController controller([](std::string_view) { return true; });
    UiSocketServer server(directory.filePath("ui.sock"), &controller);
    QString error;
    QVERIFY2(server.start(&error), qPrintable(error));
    UnixDatagramUiNotifier notifier(directory.filePath("ui.sock").toStdString());
    notifier.send("RECORDING");
    QTRY_VERIFY(!controller.idle());
    QSignalSpy spy(&controller, &CapsuleController::audioLevelChanged);
    notifier.send("AUDIO_LEVEL 100");
    notifier.send("AUDIO_LEVEL 800");
    notifier.send("AUDIO_LEVEL 300");
    QTRY_COMPARE(controller.audioLevel(), 0.3);
    QCOMPARE(spy.count(), 1);
}

void TestUiNotifier::capturedWindowReachesWaveformInOneUpdate()
{
    QTemporaryDir directory;
    CapsuleController controller([](std::string_view) { return true; });
    UiSocketServer server(directory.filePath("ui.sock"), &controller);
    QString error;
    QVERIFY2(server.start(&error), qPrintable(error));
    UnixDatagramUiNotifier notifier(directory.filePath("ui.sock").toStdString());
    Pipeline pipeline;
    Committer committer;
    VoiceSession session(Config::defaultConfig(), pipeline, committer, notifier);
    session.handleCommand("START");
    QTRY_VERIFY(!controller.idle());
    QSignalSpy spy(&controller, &CapsuleController::audioLevelsChanged);
    AudioLevelMeter meter(16000, 1, [&](const AudioLevelFrame& frame) { pipeline.level(frame); });
    std::vector<int16_t> window(800);
    for (size_t i = 0; i < window.size(); ++i) {
        const double t = static_cast<double>(i) / 16000;
        window[i] = static_cast<int16_t>(32767 * (0.04 * std::sin(2 * 3.141592653589793 * 300 * t)
            + 0.12 * std::sin(2 * 3.141592653589793 * 2200 * t)));
    }
    meter.append(window.data(), window.size());
    QTRY_COMPARE(spy.count(), 1);
    const auto bars = controller.audioLevels();
    QVERIFY(bars[0].toDouble() < 0.01);
    QVERIFY(bars[6].toDouble() < 0.01);
    QVERIFY(bars[4].toDouble() > bars[1].toDouble() * 2);
    QVERIFY(bars[1].toDouble() > 0.1);
    session.handleCommand("CANCEL");
    QTRY_VERIFY(controller.idle());
    QCOMPARE(controller.audioLevels(), (QVariantList{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
}

void TestUiNotifier::sendsDatagram()
{
    UiInbox inbox;
    QVERIFY(inbox.fd >= 0);
    UnixDatagramUiNotifier notifier(inbox.path);
    notifier.send("STREAM_TEXT preview");
    char buffer[128];
    const ssize_t n = recv(inbox.fd, buffer, sizeof(buffer), 0);
    QVERIFY(n > 0);
    QCOMPARE(std::string(buffer, n), std::string("STREAM_TEXT preview"));
}

void TestUiNotifier::fullReceiverDoesNotBlock()
{
    UiInbox inbox;
    QVERIFY(inbox.fd >= 0);
    QVERIFY(inbox.fill());
    UnixDatagramUiNotifier notifier(inbox.path);
    const auto started = std::chrono::steady_clock::now();
    notifier.send("STREAM_TEXT preview");
    notifier.send("TRANSCRIBING");
    QVERIFY(std::chrono::steady_clock::now() - started < 200ms);
}

void TestUiNotifier::busyUiCannotBlockStopOrCancel()
{
    UiInbox inbox;
    QVERIFY(inbox.fd >= 0);
    QVERIFY(inbox.fill());
    UnixDatagramUiNotifier notifier(inbox.path);
    Pipeline pipeline;
    Committer committer;
    Polisher polisher;
    VoiceSession session(Config::defaultConfig(), pipeline, committer, notifier);
    session.setTextPolisher(&polisher);
    QCOMPARE(session.handleCommand("START"), std::string("RECORDING"));
    pipeline.partial("first");
    pipeline.confirmed("first");
    QTRY_VERIFY(committer.polished.load());
    const auto started = std::chrono::steady_clock::now();
    QCOMPARE(session.handleCommand("STOP"), std::string("TRANSCRIBING"));
    QTRY_COMPARE_WITH_TIMEOUT(session.state(), SessionState::Idle, 1000);
    QVERIFY(std::chrono::steady_clock::now() - started < 1s);
    QCOMPARE(committer.commits.load(), 1);
    QCOMPARE(session.handleCommand("START"), std::string("RECORDING"));
    pipeline.partial("first");
    const auto cancelling = std::chrono::steady_clock::now();
    QCOMPARE(session.handleCommand("CANCEL"), std::string("CANCELLED"));
    QVERIFY(std::chrono::steady_clock::now() - cancelling < 200ms);
}

QTEST_MAIN(TestUiNotifier)
#include "test_ui_notifier.moc"

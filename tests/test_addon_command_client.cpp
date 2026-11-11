// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../fcitx-addon/CommandClient.h"

#include <QtTest>
#include <QTemporaryDir>

class DatagramReceiver {
public:
    explicit DatagramReceiver(const QString &path) {
        fd_ = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        sockaddr_un address {};
        address.sun_family = AF_UNIX;
        const auto bytes = path.toLocal8Bit();
        if (bytes.size() >= static_cast<qsizetype>(sizeof(address.sun_path))) {
            if (fd_ >= 0) close(fd_);
            fd_ = -1;
            return;
        }
        std::memcpy(address.sun_path, bytes.constData(), bytes.size() + 1);
        if (fd_ >= 0 && bind(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
            close(fd_);
            fd_ = -1;
        }
    }
    ~DatagramReceiver() { if (fd_ >= 0) close(fd_); }
    bool valid() const { return fd_ >= 0; }
    QByteArray receive() const {
        char bytes[512];
        const ssize_t size = recv(fd_, bytes, sizeof(bytes), 0);
        return size < 0 ? QByteArray() : QByteArray(bytes, size);
    }
private:
    int fd_ = -1;
};

class TestAddonCommandClient : public QObject {
    Q_OBJECT
private slots:
    void focusWithoutService() {
        QTemporaryDir runtime;
        QVERIFY(runtime.isValid());
        DatagramReceiver ui(runtime.filePath("echoflow-ui.sock"));
        DatagramReceiver wakeup(runtime.filePath("echoflow-wakeup.sock"));
        QVERIFY(ui.valid());
        QVERIFY(wakeup.valid());
        for (const auto &message : {"FOCUS 10 20 1 18", "CARET 20 20 1 18", "TYPED", "REFOCUS 30 20 1 18"}) {
            QVERIFY(echoflow::sendAddonCommand(runtime.path().toStdString(),
                echoflow::CommandChannel::Ui, message));
            QCOMPARE(ui.receive(), QByteArray(message));
        }
        QVERIFY(echoflow::sendAddonBlur(runtime.path().toStdString()));
        QCOMPARE(ui.receive(), QByteArray("BLUR"));
        QVERIFY(wakeup.receive().isEmpty());
    }
    void blurCancelsRunningServiceWithoutRelayingFocus() {
        QTemporaryDir runtime;
        DatagramReceiver ui(runtime.filePath("echoflow-ui.sock"));
        DatagramReceiver control(runtime.filePath("echoflow-control.sock"));
        QVERIFY(ui.valid());
        QVERIFY(control.valid());
        QVERIFY(echoflow::sendAddonBlur(runtime.path().toStdString()));
        QVERIFY(echoflow::sendAddonCommand(runtime.path().toStdString(),
            echoflow::CommandChannel::Ui, "FOCUS 10 20 1 18"));
        QCOMPARE(ui.receive(), QByteArray("BLUR"));
        QCOMPARE(ui.receive(), QByteArray("FOCUS 10 20 1 18"));
        QCOMPARE(control.receive(), QByteArray("BLUR ui-notified"));
        QVERIFY(control.receive().isEmpty());
    }
    void recordingCommandsUseWakeupSocket() {
        QTemporaryDir runtime;
        DatagramReceiver wakeup(runtime.filePath("echoflow-wakeup.sock"));
        QVERIFY(wakeup.valid());
        for (const auto &message : {"START", "STOP"}) {
            QVERIFY(echoflow::sendAddonCommand(runtime.path().toStdString(),
                echoflow::CommandChannel::Wakeup, message));
            QCOMPARE(wakeup.receive(), QByteArray(message));
        }
    }
    void missingOrFullReceiverDoesNotBlock() {
        QTemporaryDir runtime;
        QVERIFY(!echoflow::sendAddonCommand(runtime.path().toStdString(),
            echoflow::CommandChannel::Ui, "FOCUS"));
        DatagramReceiver ui(runtime.filePath("echoflow-ui.sock"));
        QVERIFY(ui.valid());
        bool full = false;
        for (int i = 0; i < 1024; ++i) {
            if (!echoflow::sendAddonCommand(runtime.path().toStdString(),
                echoflow::CommandChannel::Ui, "CARET 10 20 1 18")) {
                full = true;
                break;
            }
        }
        QVERIFY(full);
    }
};

QTEST_GUILESS_MAIN(TestAddonCommandClient)
#include "test_addon_command_client.moc"

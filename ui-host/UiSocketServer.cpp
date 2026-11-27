// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "UiSocketServer.h"

#include "CapsuleController.h"
#include "UiProtocol.h"

#include <QDebug>
#include <QSocketNotifier>

#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>

UiSocketServer::UiSocketServer(QString path, CapsuleController* controller, QObject* parent)
    : QObject(parent), path_(std::move(path)),
      audioPath_(QString::fromStdString(echoflow::uiAudioLevelSocketPath(path_.toStdString()).string())),
      controller_(controller)
{
}

UiSocketServer::~UiSocketServer()
{
    audioNotifier_.reset();
    notifier_.reset();
    if (audioFd_ >= 0) close(audioFd_);
    if (fd_ >= 0) close(fd_);
    if (audioBound_) unlink(audioPath_.toLocal8Bit().constData());
    if (bound_) unlink(path_.toLocal8Bit().constData());
}

bool UiSocketServer::bindSocket(const QString& path, int& fd, bool& bound, QString* error)
{
    const QByteArray socketPath = path.toLocal8Bit();
    sockaddr_un addr {};
    if (socketPath.size() >= static_cast<int>(sizeof(addr.sun_path))) {
        *error = QStringLiteral("UI socket path is too long: %1").arg(path);
        return false;
    }
    fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        *error = QStringLiteral("failed to create UI socket: %1").arg(strerror(errno));
        return false;
    }
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socketPath.constData(), sizeof(addr.sun_path) - 1);
    unlink(socketPath.constData());
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        *error = QStringLiteral("failed to bind %1: %2").arg(path, strerror(errno));
        close(fd);
        fd = -1;
        return false;
    }
    bound = true;
    chmod(socketPath.constData(), S_IRUSR | S_IWUSR);
    return true;
}

bool UiSocketServer::start(QString* error)
{
    if (!bindSocket(path_, fd_, bound_, error)
        || !bindSocket(audioPath_, audioFd_, audioBound_, error)) return false;
    notifier_ = std::make_unique<QSocketNotifier>(fd_, QSocketNotifier::Read, this);
    connect(notifier_.get(), &QSocketNotifier::activated, this, &UiSocketServer::readPending);
    audioNotifier_ = std::make_unique<QSocketNotifier>(audioFd_, QSocketNotifier::Read, this);
    connect(audioNotifier_.get(), &QSocketNotifier::activated, this, &UiSocketServer::readAudioPending);
    return true;
}

void UiSocketServer::readPending()
{
    while (true) {
        char buffer[4096];
        const ssize_t size = recv(fd_, buffer, sizeof(buffer), 0);
        if (size < 0) {
            if (errno == EINTR) continue;
            return;
        }
        const auto message = QString::fromUtf8(buffer, static_cast<qsizetype>(size)).trimmed();
        // The preceding recording's meter queue must not animate a new session.
        // Losing the new session's first level is harmless: another arrives in 50 ms.
        if (message == QStringLiteral("RECORDING")) discardAudioPending();
        applyMessage(message);
    }
}

void UiSocketServer::discardAudioPending()
{
    char buffer[256];
    while (recv(audioFd_, buffer, sizeof(buffer), 0) >= 0 || errno == EINTR) {}
}

void UiSocketServer::readAudioPending()
{
    // State wins regardless of which QSocketNotifier Qt dispatches first.
    readPending();
    QString latest;
    while (true) {
        char buffer[256];
        const ssize_t size = recv(audioFd_, buffer, sizeof(buffer), 0);
        if (size < 0) {
            if (errno == EINTR) continue;
            break;
        }
        const auto message = QString::fromUtf8(buffer, static_cast<qsizetype>(size)).trimmed();
        if (message.startsWith(QStringLiteral("AUDIO_LEVEL "))) latest = message;
    }
    if (!latest.isEmpty()) controller_->applyServiceMessage(latest);
}

void UiSocketServer::applyMessage(const QString& message)
{
    if (message == QStringLiteral("MODEL_REQUIRED")) {
        emit modelRequired();
        return;
    }
    // Older services must not revive the focus-tracking capsule or place
    // metering back on the state channel.
    if (message.startsWith(QStringLiteral("SHOW_TOOLTIP"))
        || message.startsWith(QStringLiteral("AUDIO_LEVEL"))) return;
    const qsizetype separator = message.indexOf(QLatin1Char(' '));
    const QString type = separator < 0 ? message : message.first(separator);
    const qsizetype payloadCharacters = message.startsWith(QStringLiteral("STREAM_TEXT "))
        ? message.size() - 12 : 0;
    const bool accepted = controller_->applyServiceMessage(message);
    qInfo("deepin-echoflow UI message: type=%s, bytes=%lld, payload_chars=%lld, accepted=%s",
          qPrintable(type), static_cast<long long>(message.toUtf8().size()),
          static_cast<long long>(payloadCharacters), accepted ? "true" : "false");
}
